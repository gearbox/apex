"""Byte-level checks for the media preparation boundary and its hash input."""

from __future__ import annotations

import asyncio
import errno
import io
import json
import os
import shutil
import struct
import subprocess
import threading
import zlib
from itertools import pairwise
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest
from PIL import Image, ImageChops, JpegImagePlugin, PngImagePlugin
from structlog.testing import capture_logs

from src.api.services.image_normalization import ImageTooLargeError
from src.api.services.media_ingest import (
    ImageIngestPolicy,
    InvalidMediaError,
    MediaIngestService,
    MediaProcessingError,
    UnsupportedMediaError,
)
from src.api.services.media_ingest import service as ingest_service
from src.api.services.media_ingest.image_strip import strip_image_metadata
from src.api.services.media_ingest.pdq import pdq_from_image_bytes
from src.api.services.media_ingest.service import DurationSource, _VideoProbe
from src.api.services.media_tools import (
    MediaToolExitError,
    MediaToolNotFoundError,
    MediaToolTimeoutError,
)
from src.core.enums import MediaFormat

if TYPE_CHECKING:
    from pathlib import Path

    from src.core.media_hash import PdqHash

pytestmark = pytest.mark.unit
_FFMPEG = shutil.which("ffmpeg")


def _require_ffmpeg() -> str:
    if _FFMPEG is None:
        if os.environ.get("CI"):
            pytest.fail("ffmpeg is required for media ingest tests in CI")
        pytest.skip("ffmpeg is unavailable locally")
    return _FFMPEG


def _image_bytes(image_format: str, *, size: tuple[int, int] = (20, 12)) -> bytes:
    image = Image.new("RGB", size)
    for x in range(size[0]):
        for y in range(size[1]):
            image.putpixel((x, y), (x * 11 % 256, y * 19 % 256, (x + y) * 7 % 256))
    output = io.BytesIO()
    image.save(output, format=image_format)
    return output.getvalue()


def _decoded_rgb(data: bytes) -> Image.Image:
    with Image.open(io.BytesIO(data)) as image:
        image.load()
        return image.convert("RGB")


def _with_jpeg_comment(data: bytes, comment: bytes) -> bytes:
    return data[:2] + b"\xff\xfe" + struct.pack(">H", len(comment) + 2) + comment + data[2:]


def _png_chunk(kind: bytes, payload: bytes) -> bytes:
    return (
        struct.pack(">I", len(payload))
        + kind
        + payload
        + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF)
    )


def _with_webp_chunk(data: bytes, kind: bytes, payload: bytes) -> bytes:
    assert data[:4] == b"RIFF"
    assert data[8:12] == b"WEBP"
    chunk = (
        kind + struct.pack("<I", len(payload)) + payload + (b"\x00" if len(payload) % 2 else b"")
    )
    body = data[12:] + chunk
    return b"RIFF" + struct.pack("<I", len(body) + 4) + b"WEBP" + body


def _service() -> MediaIngestService:
    return MediaIngestService(
        max_image_megapixels=10,
        max_input_bytes=2 * 1024 * 1024,
        image_concurrency=1,
        video_concurrency=1,
    )


class TestImageStrip:
    def test_png_pixel_aspect_and_icc_are_retained(self) -> None:
        original = _image_bytes("PNG")
        phys = _png_chunk(b"pHYs", struct.pack(">IIB", 2000, 1000, 0))
        icc = _png_chunk(b"iCCP", b"sRGB\x00\x00" + zlib.compress(b"test-profile"))
        source = original[:33] + phys + icc + original[33:]
        stripped = strip_image_metadata(source, MediaFormat.PNG)
        assert phys in stripped
        assert icc in stripped

    def test_jfif_density_and_icc_survive_without_thumbnail(self) -> None:
        original = _image_bytes("JPEG")
        jfif = b"JFIF\x00\x01\x02\x00\x00\x02\x00\x01\x01\x01rgb"
        app0 = b"\xff\xe0" + struct.pack(">H", len(jfif) + 2) + jfif
        icc = b"ICC_PROFILE\x00\x01\x01test-profile"
        app2 = b"\xff\xe2" + struct.pack(">H", len(icc) + 2) + icc
        stripped = strip_image_metadata(original[:2] + app0 + app2 + original[2:], MediaFormat.JPEG)
        assert b"JFIF\x00\x01\x02\x00\x00\x02\x00\x01\x00\x00" in stripped
        assert icc in stripped
        assert b"rgb" not in stripped

    @pytest.mark.parametrize(
        "image_format,pillow_format",
        [(MediaFormat.PNG, "PNG"), (MediaFormat.JPEG, "JPEG"), (MediaFormat.WEBP, "WEBP")],
    )
    def test_stripping_is_idempotent(self, image_format: MediaFormat, pillow_format: str) -> None:
        source = _image_bytes(pillow_format)
        once = strip_image_metadata(source, image_format)
        assert strip_image_metadata(once, image_format) == once

    def test_png_drops_text_and_trailer_without_changing_pixels(self) -> None:
        source = Image.new("RGBA", (8, 5), (20, 30, 40, 128))
        info = PngImagePlugin.PngInfo()
        info.add_text("Comment", "descriptive-canary")
        output = io.BytesIO()
        source.save(output, format="PNG", pnginfo=info)
        original = output.getvalue()

        sanitized = strip_image_metadata(original + b"not-a-png-trailer", MediaFormat.PNG)

        assert b"descriptive-canary" not in sanitized
        assert not sanitized.endswith(b"not-a-png-trailer")
        assert (
            ImageChops.difference(_decoded_rgb(original), _decoded_rgb(sanitized)).getbbox() is None
        )

    def test_png_rejects_bad_crc(self) -> None:
        malformed = bytearray(_image_bytes("PNG"))
        malformed[-1] ^= 1

        with pytest.raises(InvalidMediaError, match="CRC"):
            strip_image_metadata(bytes(malformed), MediaFormat.PNG)

    def test_jpeg_drops_comment_and_preserves_entropy_payload(self) -> None:
        original = _image_bytes("JPEG")
        source = _with_jpeg_comment(original, b"descriptive-canary")

        sanitized = strip_image_metadata(source + b"trailer", MediaFormat.JPEG)

        assert b"descriptive-canary" not in sanitized
        assert not sanitized.endswith(b"trailer")
        assert (
            ImageChops.difference(_decoded_rgb(original), _decoded_rgb(sanitized)).getbbox() is None
        )

    def test_webp_drops_unapproved_top_level_chunk(self) -> None:
        original = _image_bytes("WEBP")
        source = _with_webp_chunk(original, b"XMP ", b"descriptive-canary")

        sanitized = strip_image_metadata(source, MediaFormat.WEBP)

        assert b"descriptive-canary" not in sanitized
        assert (
            ImageChops.difference(_decoded_rgb(original), _decoded_rgb(sanitized)).getbbox() is None
        )


class TestImagePreparation:
    @pytest.mark.parametrize("image_format", ["PNG", "JPEG", "WEBP"])
    async def test_all_descriptive_image_carriers_are_removed(self, image_format: str) -> None:
        canary = b"private-metadata-canary"
        source = _image_bytes(image_format)
        if image_format == "JPEG":
            carriers = b"".join(
                b"\xff" + bytes([marker]) + struct.pack(">H", len(canary) + 2) + canary
                for marker in [0xE1, 0xEB, 0xED, 0xFE, 0xE2]
            )
            source = source[:2] + carriers + source[2:] + canary
        elif image_format == "PNG":
            carriers = b"".join(
                _png_chunk(kind, payload)
                for kind, payload in [
                    (b"tEXt", b"Comment\x00" + canary),
                    (b"iTXt", b"Comment\x00\x00\x00\x00\x00" + canary),
                    (b"zTXt", b"Comment\x00\x00" + zlib.compress(canary)),
                    (b"eXIf", b"Exif\x00\x00" + canary),
                    (b"caBX", canary),
                ]
            )
            source = source[:33] + carriers + source[33:]
        else:
            for kind in [b"EXIF", b"XMP ", b"JUNK"]:
                source = _with_webp_chunk(source, kind, canary)

        prepared = await _service().prepare_image(source, policy=ImageIngestPolicy.UPLOAD)
        assert canary not in prepared.data

    async def test_prepared_image_is_byte_identical_on_reingest(self) -> None:
        service = _service()
        first = await service.prepare_image(_image_bytes("PNG"), policy=ImageIngestPolicy.UPLOAD)
        second = await service.prepare_image(first.data, policy=ImageIngestPolicy.UPLOAD)
        assert second.data == first.data

    async def test_hashes_the_exact_sanitized_upload_bytes(self) -> None:
        image = Image.new("RGB", (16, 12), (35, 80, 120))
        info = PngImagePlugin.PngInfo()
        info.add_text("Comment", "upload-canary")
        output = io.BytesIO()
        image.save(output, format="PNG", pnginfo=info)

        prepared = await _service().prepare_image(
            output.getvalue(), policy=ImageIngestPolicy.UPLOAD
        )

        assert b"upload-canary" not in prepared.data
        assert prepared.hash_set.profile_id == "pdq-image-rgb-white-v1"
        assert prepared.hash_set.samples[0].pdq == pdq_from_image_bytes(prepared.data)
        assert prepared.width == 16
        assert prepared.height == 12

    async def test_static_exif_orientation_is_baked_before_storage(self) -> None:
        image = Image.new("RGB", (20, 10), (20, 100, 200))
        exif = Image.Exif()
        exif[0x0112] = 6
        output = io.BytesIO()
        image.save(output, format="JPEG", exif=exif, dpi=(300, 100))

        prepared = await _service().prepare_image(
            output.getvalue(), policy=ImageIngestPolicy.UPLOAD
        )

        assert prepared.orientation_baked is True
        assert prepared.converted is True
        assert (prepared.width, prepared.height) == (10, 20)
        assert b"Exif" not in prepared.data
        with Image.open(io.BytesIO(prepared.data)) as stored:
            assert stored.info["dpi"] == (100, 300)

    async def test_jpeg_orientation_bake_preserves_subsampling(self) -> None:
        image = Image.new("RGB", (20, 10), (20, 100, 200))
        exif = Image.Exif()
        exif[0x0112] = 6
        output = io.BytesIO()
        image.save(output, format="JPEG", exif=exif, subsampling=0)
        prepared = await _service().prepare_image(
            output.getvalue(), policy=ImageIngestPolicy.UPLOAD
        )
        with Image.open(io.BytesIO(prepared.data)) as stored:
            assert JpegImagePlugin.get_sampling(stored) == 0

    @pytest.mark.parametrize("orientation", [0, 9])
    async def test_invalid_orientation_is_ignored(self, orientation: int) -> None:
        image = Image.new("RGB", (20, 10), (20, 100, 200))
        exif = Image.Exif()
        exif[0x0112] = orientation
        output = io.BytesIO()
        image.save(output, format="JPEG", exif=exif)
        prepared = await _service().prepare_image(
            output.getvalue(), policy=ImageIngestPolicy.UPLOAD
        )
        assert not prepared.orientation_baked
        assert (prepared.width, prepared.height) == (20, 10)
        assert b"Exif" not in prepared.data

    async def test_orientation_pixel_cap_precedes_full_decode(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        image = Image.new("RGB", (20, 20), (20, 100, 200))
        exif = Image.Exif()
        exif[0x0112] = 6
        output = io.BytesIO()
        image.save(output, format="JPEG", exif=exif)

        def forbidden_load(_image: Image.Image) -> None:
            pytest.fail("full image decode occurred before the pixel cap")

        monkeypatch.setattr(Image.Image, "load", forbidden_load)
        service = MediaIngestService(max_image_megapixels=0.0001, max_input_bytes=1024)
        with pytest.raises(ImageTooLargeError):
            await service.prepare_image(output.getvalue(), policy=ImageIngestPolicy.UPLOAD)

    async def test_garbage_exif_does_not_reject_valid_jpeg(self) -> None:
        original = _image_bytes("JPEG")
        payload = b"Exif\x00\x00garbage-orientation"
        app1 = b"\xff\xe1" + struct.pack(">H", len(payload) + 2) + payload
        prepared = await _service().prepare_image(
            original[:2] + app1 + original[2:], policy=ImageIngestPolicy.UPLOAD
        )
        assert not prepared.orientation_baked
        assert b"Exif" not in prepared.data

    async def test_provider_policy_rejects_non_supported_container(self) -> None:
        image = Image.new("RGB", (4, 4), (255, 0, 0))
        output = io.BytesIO()
        image.save(output, format="GIF")

        with pytest.raises(InvalidMediaError, match="provider image"):
            await _service().prepare_image(output.getvalue(), policy=ImageIngestPolicy.PROVIDER)

    def test_pdq_reference_vector_uses_big_endian_packing(self) -> None:
        image = Image.new("RGB", (16, 16), (20, 40, 60))
        output = io.BytesIO()
        image.save(output, format="PNG")

        fingerprint = pdq_from_image_bytes(output.getvalue())

        assert fingerprint.bits.hex() == (
            "113400002c4b00002c4b1134820000002c4b2c4b554b11340000000000001134"
        )
        assert fingerprint.quality == 0


async def test_subsecond_video_keeps_the_first_decoded_frame(tmp_path: Path) -> None:
    source = tmp_path / "short.mp4"
    cmd = [
        _require_ffmpeg(),
        "-y",
        "-f",
        "lavfi",
        "-i",
        "testsrc=size=64x48:rate=10",
        "-t",
        "0.2",
        "-an",
        "-c:v",
        "mpeg4",
        str(source),
    ]
    await asyncio.to_thread(lambda: subprocess.run(cmd, check=True, capture_output=True))  # noqa: S603

    prepared = await _service().prepare_video(source.read_bytes())

    assert (prepared.format, prepared.width, prepared.height, prepared.duration_ms) == (
        MediaFormat.MP4,
        64,
        48,
        200,
    )
    assert [sample.frame_timestamp_ms for sample in prepared.hash_set.samples] == [0]


@pytest.mark.parametrize("duration,expected_count", [(15, 15), (30, 30), (120, 60)])
async def test_video_sampling_has_all_pts_records(
    tmp_path: Path, duration: int, expected_count: int
) -> None:
    source = tmp_path / "sample.mp4"
    await asyncio.to_thread(
        subprocess.run,
        [
            _require_ffmpeg(),
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc=size=64x48:rate=24",
            "-t",
            str(duration),
            "-an",
            "-c:v",
            "libx264",
            "-g",
            "48",
            str(source),
        ],
        check=True,
        capture_output=True,
    )
    prepared = await _service().prepare_video(source.read_bytes())
    stamps = [
        timestamp
        for sample in prepared.hash_set.samples
        if (timestamp := sample.frame_timestamp_ms) is not None
    ]
    assert len(stamps) == expected_count
    if duration < 120:
        assert stamps == list(range(0, duration * 1000, 1000))
        assert prepared.hash_set.sampling_profile == "uniform-pts-v1"
    else:
        gaps = [b - a for a, b in pairwise(stamps)]
        assert min(gaps) >= 1990, gaps
        assert prepared.hash_set.sampling_profile == "uniform-pts-keyframes-v1"


async def test_webm_visual_duration_tag_is_used(tmp_path: Path) -> None:
    source = tmp_path / "source.webm"
    await asyncio.to_thread(
        subprocess.run,
        [
            _require_ffmpeg(),
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc=size=64x48:rate=12",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:sample_rate=48000",
            "-t",
            "3",
            "-c:v",
            "libvpx",
            "-c:a",
            "libopus",
            str(source),
        ],
        check=True,
        capture_output=True,
    )
    prepared = await _service().prepare_video(source.read_bytes())
    assert prepared.format is MediaFormat.WEBM
    assert abs(prepared.duration_ms - 3000) <= 50


@pytest.mark.parametrize("mutation", ["missing", "malformed"])
async def test_webm_visual_duration_falls_back_to_container(tmp_path: Path, mutation: str) -> None:
    source = tmp_path / "source.webm"
    await asyncio.to_thread(
        subprocess.run,
        [
            _require_ffmpeg(),
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc=size=64x48:rate=12",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:sample_rate=48000",
            "-t",
            "3",
            "-c:v",
            "libvpx",
            "-c:a",
            "libopus",
            str(source),
        ],
        check=True,
        capture_output=True,
    )
    data = source.read_bytes()
    assert b"DURATION" in data
    if mutation == "missing":
        data = data.replace(b"DURATION", b"NO_DURAT", 1)
    else:
        assert b"00:00:03.000000000" in data
        data = data.replace(b"00:00:03.000000000", b"garbage___________", 1)
    prepared = await _service().prepare_video(data)
    assert prepared.format is MediaFormat.WEBM
    assert abs(prepared.duration_ms - 3008) <= 50


async def test_truncated_webm_cluster_is_invalid(tmp_path: Path) -> None:
    source = tmp_path / "source.webm"
    await asyncio.to_thread(
        subprocess.run,
        [
            _require_ffmpeg(),
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc=size=64x48:rate=12",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:sample_rate=48000",
            "-t",
            "3",
            "-c:v",
            "libvpx",
            "-c:a",
            "libopus",
            str(source),
        ],
        check=True,
        capture_output=True,
    )
    data = source.read_bytes()
    with pytest.raises(InvalidMediaError, match="not decodable"):
        await _service().prepare_video(data[: len(data) // 2])


@pytest.mark.parametrize(
    "tag,expected",
    [
        ("00:00:03.000000000", (3.0, DurationSource.TAG)),
        ("garbage", (3.008, DurationSource.CONTAINER)),
        (None, (3.008, DurationSource.CONTAINER)),
    ],
)
def test_visual_duration_resolution(
    tag: str | None, expected: tuple[float, DurationSource]
) -> None:
    stream: dict[str, object] = {"tags": {"DURATION": tag} if tag else {}}
    assert MediaIngestService._stream_duration_seconds(stream, {"duration": "3.008"}) == expected


@pytest.mark.parametrize(
    "stream,expected",
    [
        ({"duration": "2.5"}, (2.5, DurationSource.STREAM)),
        ({"duration_ts": 2500, "time_base": "1/1000"}, (2.5, DurationSource.TIMELINE)),
        ({"duration": "N/A", "tags": {}}, (None, DurationSource.UNKNOWN)),
    ],
)
def test_visual_duration_provenance(
    stream: dict[str, object], expected: tuple[float | None, DurationSource]
) -> None:
    assert MediaIngestService._stream_duration_seconds(stream, {}) == expected


def _probe(duration_ms: int | None, source: DurationSource) -> _VideoProbe:
    return _VideoProbe(
        format=MediaFormat.WEBM,
        width=64,
        height=48,
        duration_ms=duration_ms,
        duration_source=source,
        video_stream_index=0,
        audio_stream_index=None,
    )


_STREAM_LEVEL = [DurationSource.STREAM, DurationSource.TIMELINE, DurationSource.TAG]


@pytest.mark.parametrize("source_kind", _STREAM_LEVEL)
@pytest.mark.parametrize("prepared_kind", _STREAM_LEVEL)
def test_remux_duration_loss_between_stream_sources_is_invalid(
    source_kind: DurationSource, prepared_kind: DurationSource
) -> None:
    with pytest.raises(InvalidMediaError, match="not decodable"):
        MediaIngestService._validate_remux_duration(
            _probe(4000, source_kind), _probe(3700, prepared_kind)
        )
    # Within tolerance is accepted.
    MediaIngestService._validate_remux_duration(
        _probe(4000, source_kind), _probe(3800, prepared_kind)
    )


@pytest.mark.parametrize(
    "source,prepared",
    [
        (_probe(4000, DurationSource.CONTAINER), _probe(3000, DurationSource.STREAM)),
        (_probe(4000, DurationSource.STREAM), _probe(3000, DurationSource.CONTAINER)),
        (_probe(None, DurationSource.UNKNOWN), _probe(3000, DurationSource.TAG)),
        (_probe(4000, DurationSource.TAG), _probe(None, DurationSource.UNKNOWN)),
        (_probe(None, DurationSource.UNKNOWN), _probe(None, DurationSource.UNKNOWN)),
    ],
)
def test_remux_duration_check_is_skipped_without_stream_level_sides(
    source: _VideoProbe, prepared: _VideoProbe
) -> None:
    MediaIngestService._validate_remux_duration(source, prepared)


def test_prepared_probe_duration_is_mandatory_and_capped() -> None:
    with pytest.raises(InvalidMediaError, match="duration is unavailable"):
        MediaIngestService._prepared_duration_ms(_probe(None, DurationSource.UNKNOWN), None)
    with pytest.raises(InvalidMediaError, match="duration exceeds"):
        MediaIngestService._prepared_duration_ms(_probe(3000, DurationSource.TAG), 2)
    assert MediaIngestService._prepared_duration_ms(_probe(3000, DurationSource.TAG), None) == 3000


async def _live_webm(
    tmp_path: Path, *, video_seconds: int, audio_seconds: int | None = None
) -> bytes:
    """Mux like browser ``MediaRecorder``: no stream or container duration at all."""
    source = tmp_path / "live.webm"
    cmd = [
        _require_ffmpeg(),
        "-y",
        "-f",
        "lavfi",
        "-i",
        f"testsrc=size=64x48:rate=12:duration={video_seconds}",
    ]
    if audio_seconds is not None:
        cmd += [
            "-f",
            "lavfi",
            "-i",
            f"sine=frequency=440:sample_rate=48000:duration={audio_seconds}",
        ]
    cmd += ["-c:v", "libvpx"]
    if audio_seconds is not None:
        cmd += ["-c:a", "libopus"]
    cmd += ["-live", "1", "-f", "webm", str(source)]
    await asyncio.to_thread(subprocess.run, cmd, check=True, capture_output=True)
    probe = await asyncio.to_thread(
        subprocess.run,
        [
            shutil.which("ffprobe") or "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "stream=duration:stream_tags=DURATION:format=duration",
            "-of",
            "json",
            str(source),
        ],
        check=True,
        capture_output=True,
    )
    decoded = json.loads(probe.stdout)
    # Guard the fixture shape itself: the regression is "no duration anywhere".
    assert "duration" not in decoded["format"]
    for stream in decoded["streams"]:
        assert "duration" not in stream
        assert "DURATION" not in (stream.get("tags") or {})
    return source.read_bytes()


async def test_live_webm_without_duration_metadata_is_prepared(tmp_path: Path) -> None:
    data = await _live_webm(tmp_path, video_seconds=4)
    prepared = await _service().prepare_video(data)
    assert prepared.format is MediaFormat.WEBM
    assert abs(prepared.duration_ms - 4000) <= 100
    timestamps = [sample.frame_timestamp_ms or 0 for sample in prepared.hash_set.samples]
    assert len(timestamps) == 4
    assert [round(ts / 1000) for ts in timestamps] == [0, 1, 2, 3]


async def test_live_webm_duration_is_visual_not_longer_audio(tmp_path: Path) -> None:
    data = await _live_webm(tmp_path, video_seconds=3, audio_seconds=5)
    prepared = await _service().prepare_video(data)
    assert prepared.format is MediaFormat.WEBM
    assert abs(prepared.duration_ms - 3000) <= 100


async def test_live_webm_over_duration_cap_is_rejected_after_remux(tmp_path: Path) -> None:
    data = await _live_webm(tmp_path, video_seconds=3)
    service = _service()
    remux = service._remux
    remuxed: list[bool] = []

    async def tracking_remux(*args: object, **kwargs: object) -> None:
        await remux(*args, **kwargs)  # type: ignore[arg-type]
        remuxed.append(True)

    with (
        patch.object(service, "_remux", side_effect=tracking_remux),
        pytest.raises(InvalidMediaError, match="duration exceeds"),
    ):
        await service.prepare_video(data, max_duration_seconds=2)
    # The source has no duration, so only the prepared-probe cap can fire.
    assert remuxed == [True]


async def test_failing_remux_logs_the_error_tail_not_the_banner(tmp_path: Path) -> None:
    data = await _live_webm(tmp_path, video_seconds=2)
    truncated = tmp_path / "truncated.webm"
    truncated.write_bytes(data[:200])
    service = _service()
    deadline = asyncio.get_running_loop().time() + 30
    with capture_logs() as logs, pytest.raises(InvalidMediaError, match="not decodable"):
        await service._remux(
            truncated,
            tmp_path / "prepared.webm",
            _probe(None, DurationSource.UNKNOWN),
            deadline,
        )
    (event,) = [log for log in logs if log["event"] == "media_ingest.video_not_decodable"]
    assert event["log_level"] == "warning"
    assert event["stage"] == "remux"
    assert "End of file" in event["stderr_excerpt"]
    assert "configuration:" not in event["stderr_excerpt"]


def test_visual_duration_over_container_is_invalid() -> None:
    with pytest.raises(InvalidMediaError, match="exceeds container"):
        MediaIngestService._stream_duration_seconds(
            {"tags": {"DURATION": "00:00:05.000000000"}}, {"duration": "3.008"}
        )


@pytest.mark.parametrize("brand", ["iso5", "dash"])
async def test_mp4_compatible_brands_are_accepted(tmp_path: Path, brand: str) -> None:
    source = tmp_path / "brand.mp4"
    cmd = [
        _require_ffmpeg(),
        "-y",
        "-f",
        "lavfi",
        "-i",
        "testsrc=size=64x48:rate=10",
        "-t",
        "1",
        "-an",
        "-c:v",
        "mpeg4",
        "-brand",
        brand,
    ]
    if brand == "dash":
        cmd.extend(["-movflags", "+frag_keyframe+empty_moov"])
    cmd.append(str(source))
    await asyncio.to_thread(subprocess.run, cmd, check=True, capture_output=True)
    prepared = await _service().prepare_video(source.read_bytes())
    assert prepared.format is MediaFormat.MP4


async def test_quicktime_mov_is_classified_from_probe(tmp_path: Path) -> None:
    source = tmp_path / "source.mov"
    await asyncio.to_thread(
        subprocess.run,
        [
            _require_ffmpeg(),
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc=size=64x48:rate=10",
            "-t",
            "1",
            "-an",
            "-c:v",
            "mpeg4",
            "-f",
            "mov",
            str(source),
        ],
        check=True,
        capture_output=True,
    )
    prepared = await _service().prepare_video(source.read_bytes())
    assert prepared.format is MediaFormat.MOV


async def test_matroska_is_not_webm(tmp_path: Path) -> None:
    source = tmp_path / "source.mkv"
    await asyncio.to_thread(
        subprocess.run,
        [
            _require_ffmpeg(),
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc=size=64x48:rate=10",
            "-t",
            "1",
            "-an",
            "-c:v",
            "libvpx",
            "-f",
            "matroska",
            str(source),
        ],
        check=True,
        capture_output=True,
    )
    with pytest.raises(UnsupportedMediaError):
        await _service().prepare_video(source.read_bytes())


def test_pure_3gp_brand_is_not_mp4() -> None:
    header = b"\x00\x00\x00\x14ftyp3gp4\x00\x00\x00\x003gp4"
    with pytest.raises(UnsupportedMediaError):
        MediaIngestService._detect_container(header, "mov,mp4,m4a,3gp,3g2,mj2")


async def test_remux_preserves_rotation_and_sar_but_removes_tags_and_extra_stream(
    tmp_path: Path,
) -> None:
    ffmpeg = _require_ffmpeg()
    base = tmp_path / "base.mp4"
    rotated = tmp_path / "rotated.mp4"
    source = tmp_path / "source.mp4"
    subtitle = tmp_path / "subtitle.srt"
    subtitle.write_text("1\n00:00:00,000 --> 00:00:01,000\nextra stream\n")
    commands = [
        [
            ffmpeg,
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc=size=64x48:rate=10",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:sample_rate=48000",
            "-t",
            "2",
            "-vf",
            "setsar=2/1",
            "-c:v",
            "mpeg4",
            "-c:a",
            "aac",
            "-metadata",
            "title=private-metadata-canary",
            "-metadata",
            "location=+12.34+56.78/",
            "-metadata:s:v:0",
            "handler_name=private-metadata-canary",
            str(base),
        ],
        [ffmpeg, "-y", "-display_rotation:v:0", "90", "-i", str(base), "-c", "copy", str(rotated)],
        [
            ffmpeg,
            "-y",
            "-i",
            str(rotated),
            "-i",
            str(subtitle),
            "-map",
            "0",
            "-map",
            "1",
            "-c",
            "copy",
            "-c:s",
            "mov_text",
            str(source),
        ],
    ]
    for command in commands:
        await asyncio.to_thread(subprocess.run, command, check=True, capture_output=True)
    # ffmpeg's data demuxer supplies codec=none, which the MP4 muxer rejects.
    # Convert the extra mov_text track's fixed-width sample/handler codes into
    # a GoPro metadata track so ffprobe sees a real data stream.
    source_bytes = source.read_bytes()
    assert source_bytes.count(b"tx3g") == 1
    assert source_bytes.count(b"sbtl") == 1
    source.write_bytes(source_bytes.replace(b"tx3g", b"gpmd").replace(b"sbtl", b"meta"))
    source_probe = await asyncio.to_thread(
        subprocess.run,
        ["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(source)],
        check=True,
        capture_output=True,
    )
    source_metadata = json.loads(source_probe.stdout)
    assert "location" in source_metadata["format"]["tags"]
    assert [stream["codec_type"] for stream in source_metadata["streams"]] == [
        "video",
        "audio",
        "data",
    ]
    prepared = await _service().prepare_video(source.read_bytes())
    assert b"private-metadata-canary" not in prepared.data
    output = tmp_path / "prepared.mp4"
    output.write_bytes(prepared.data)
    probe = await asyncio.to_thread(
        subprocess.run,
        ["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(output)],
        check=True,
        capture_output=True,
    )
    decoded = json.loads(probe.stdout)
    assert "location" not in decoded["format"].get("tags", {})
    assert [stream["codec_type"] for stream in decoded["streams"]] == ["video", "audio"]
    visual = decoded["streams"][0]
    assert visual["sample_aspect_ratio"] == "2:1"
    assert any(item.get("rotation") == 90 for item in visual.get("side_data_list", []))


async def test_truncated_mp4_is_invalid_media(tmp_path: Path) -> None:
    source = tmp_path / "video.mp4"
    await asyncio.to_thread(
        subprocess.run,
        [
            _require_ffmpeg(),
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc=size=128x72:rate=24",
            "-t",
            "5",
            "-an",
            "-c:v",
            "mpeg4",
            str(source),
        ],
        check=True,
        capture_output=True,
    )
    with pytest.raises(InvalidMediaError, match="not decodable"):
        await _service().prepare_video(source.read_bytes()[:4000])


@pytest.mark.parametrize(
    "failure",
    [
        MediaToolTimeoutError("timed out"),
        MediaToolNotFoundError("missing ffprobe"),
        MediaToolExitError("ffprobe", -9, "killed"),
    ],
)
async def test_operational_probe_failures_remain_retryable(failure: Exception) -> None:
    with (
        patch("src.api.services.media_ingest.service.run_media_command", side_effect=failure),
        pytest.raises(MediaProcessingError),
    ):
        await _service().prepare_video(b"any bytes")


async def test_temp_dir_creation_failure_is_operational() -> None:
    """ENOSPC/EACCES from ``mkdtemp`` is retryable, and nothing is cleaned up."""
    with (
        patch(
            "src.api.services.media_ingest.service.tempfile.mkdtemp",
            side_effect=OSError(errno.ENOSPC, "No space left on device"),
        ),
        patch("src.api.services.media_ingest.service.shutil.rmtree") as rmtree,
        pytest.raises(MediaProcessingError, match="failed operationally") as raised,
    ):
        await _service().prepare_video(b"any bytes")
    assert isinstance(raised.value.__cause__, OSError)
    rmtree.assert_not_called()


async def test_header_read_failure_is_operational_not_invalid(tmp_path: Path) -> None:
    source = tmp_path / "video.mp4"
    await asyncio.to_thread(
        subprocess.run,
        [
            _require_ffmpeg(),
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc=size=64x48:rate=10",
            "-t",
            "1",
            "-an",
            "-c:v",
            "mpeg4",
            str(source),
        ],
        check=True,
        capture_output=True,
    )
    with (
        patch.object(ingest_service, "_read_header", side_effect=PermissionError("denied")),
        pytest.raises(MediaProcessingError, match="failed operationally") as raised,
    ):
        await _service().prepare_video(source.read_bytes())
    assert isinstance(raised.value.__cause__, PermissionError)


async def test_video_frame_pdq_runs_off_the_event_loop(tmp_path: Path) -> None:
    source = tmp_path / "video.mp4"
    await asyncio.to_thread(
        subprocess.run,
        [
            _require_ffmpeg(),
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc=size=64x48:rate=10",
            "-t",
            "3",
            "-an",
            "-c:v",
            "mpeg4",
            str(source),
        ],
        check=True,
        capture_output=True,
    )
    loop_thread = threading.get_ident()
    callers: list[int] = []

    def recording_pdq(data: bytes) -> PdqHash:
        callers.append(threading.get_ident())
        return pdq_from_image_bytes(data)

    with patch.object(ingest_service, "pdq_from_image_bytes", side_effect=recording_pdq):
        prepared = await _service().prepare_video(source.read_bytes())

    assert len(callers) == len(prepared.hash_set.samples)
    assert len(callers) == 3
    assert loop_thread not in callers
