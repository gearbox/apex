"""Byte-level checks for the media preparation boundary and its hash input."""

from __future__ import annotations

import asyncio
import io
import shutil
import struct
import subprocess
from typing import TYPE_CHECKING

import pytest
from PIL import Image, ImageChops, PngImagePlugin

from src.api.services.media_ingest import (
    ImageIngestPolicy,
    InvalidMediaError,
    MediaIngestService,
)
from src.api.services.media_ingest.image_strip import strip_image_metadata
from src.api.services.media_ingest.pdq import pdq_from_image_bytes
from src.core.enums import MediaFormat

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.unit
_FFMPEG = shutil.which("ffmpeg")


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


def _with_webp_chunk(data: bytes, kind: bytes, payload: bytes) -> bytes:
    assert data[:4] == b"RIFF" and data[8:12] == b"WEBP"
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


@pytest.mark.skipif(_FFMPEG is None, reason="ffmpeg is required for video ingest coverage")
async def test_subsecond_video_keeps_the_first_decoded_frame(tmp_path: Path) -> None:
    source = tmp_path / "short.mp4"
    assert _FFMPEG is not None
    cmd = [
        _FFMPEG,
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
