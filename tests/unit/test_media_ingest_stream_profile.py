"""``VideoStreamProfile`` facts come from real ffprobe output on real tiny ffmpeg fixtures.

The profile is informational: it must never change what ``prepare_video`` accepts or
rejects, and a missing/odd optional probe fact degrades to ``None``/``0``.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
from typing import TYPE_CHECKING

import pytest

from src.api.services.media_ingest import MediaIngestService
from src.api.services.media_ingest.service import DurationSource
from src.api.services.media_tools import ProcessResult
from src.core.enums import MediaFormat

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.unit

_FFMPEG = shutil.which("ffmpeg")
_MP4_FORMAT_NAME = "mov,mp4,m4a,3gp,3g2,mj2"


def _require_ffmpeg() -> str:
    if _FFMPEG is None:
        if os.environ.get("CI"):
            pytest.fail("ffmpeg is required for media ingest tests in CI")
        pytest.skip("ffmpeg is unavailable locally")
    return _FFMPEG


def _service() -> MediaIngestService:
    return MediaIngestService(
        max_image_megapixels=10,
        max_input_bytes=2 * 1024 * 1024,
        image_concurrency=1,
        video_concurrency=1,
    )


async def _ffmpeg(*args: str) -> None:
    await asyncio.to_thread(
        subprocess.run,
        [_require_ffmpeg(), "-y", "-v", "error", *args],
        check=True,
        capture_output=True,
    )


_TESTSRC = ["-f", "lavfi", "-i", "testsrc=size=64x48:rate=10", "-t", "1"]


async def _h264_mp4(path: Path, *extra: str) -> bytes:
    await _ffmpeg(*_TESTSRC, "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", *extra, str(path))
    return await asyncio.to_thread(path.read_bytes)


async def test_h264_sdr_mp4_profile(tmp_path: Path) -> None:
    prepared = await _service().prepare_video(await _h264_mp4(tmp_path / "a.mp4"))

    profile = prepared.stream_profile
    assert profile.container is MediaFormat.MP4
    assert profile.codec == "h264"
    assert profile.pix_fmt == "yuv420p"
    assert profile.codec_profile is not None
    assert profile.rotation_degrees == 0
    assert profile.sample_aspect_ratio in {"1:1", None}
    assert profile.is_hdr is False
    assert profile.has_audio is False


async def test_profile_reports_audio(tmp_path: Path) -> None:
    source = tmp_path / "audio.mp4"
    await _ffmpeg(
        *_TESTSRC,
        "-f",
        "lavfi",
        "-i",
        "sine=frequency=440:sample_rate=48000",
        "-t",
        "1",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        str(source),
    )

    prepared = await _service().prepare_video(source.read_bytes())

    assert prepared.stream_profile.has_audio is True


@pytest.mark.parametrize("degrees", [90, 180, 270])
async def test_display_matrix_rotation_is_normalized(tmp_path: Path, degrees: int) -> None:
    base = tmp_path / "base.mp4"
    rotated = tmp_path / "rotated.mp4"
    await _h264_mp4(base)
    await _ffmpeg(
        "-display_rotation:v:0", str(degrees), "-i", str(base), "-c", "copy", str(rotated)
    )

    prepared = await _service().prepare_video(rotated.read_bytes())

    # ffprobe reports e.g. -90 for a 270° display matrix; the profile normalizes it into [0, 360).
    assert prepared.stream_profile.rotation_degrees == degrees


async def test_hdr_transfer_tags_are_reported_as_hdr(tmp_path: Path) -> None:
    data = await _h264_mp4(
        tmp_path / "hlg.mp4",
        # VUI colour tags written by the encoder itself; ffmpeg's generic -color_trc
        # output option is not applied by every build.
        "-x264-params",
        "colorprim=bt2020:transfer=arib-std-b67:colormatrix=bt2020nc",
    )

    prepared = await _service().prepare_video(data)

    profile = prepared.stream_profile
    assert profile.color_transfer == "arib-std-b67"
    assert profile.color_primaries == "bt2020"
    assert profile.is_hdr is True


async def test_vp9_webm_profile(tmp_path: Path) -> None:
    source = tmp_path / "v.webm"
    await _ffmpeg(*_TESTSRC, "-an", "-c:v", "libvpx-vp9", str(source))

    prepared = await _service().prepare_video(source.read_bytes())

    assert prepared.stream_profile.container is MediaFormat.WEBM
    assert prepared.stream_profile.codec == "vp9"
    assert prepared.stream_profile.is_hdr is False


async def test_profile_describes_the_prepared_stream_not_just_the_source(tmp_path: Path) -> None:
    prepared = await _service().prepare_video(await _h264_mp4(tmp_path / "a.mp4"))

    assert prepared.stream_profile.container is prepared.format


class TestOptionalFactsAreNeverAGate:
    """I12 — a missing or odd optional fact must not become ``InvalidMediaError``."""

    @staticmethod
    async def _probe(tmp_path: Path, stream: dict[str, object]) -> object:
        path = tmp_path / "real.mp4"
        await _h264_mp4(path)
        payload = {
            "streams": [
                {
                    "index": 0,
                    "codec_type": "video",
                    "codec_name": "h264",
                    "width": 64,
                    "height": 48,
                    "duration": "1.0",
                    **stream,
                }
            ],
            "format": {"format_name": _MP4_FORMAT_NAME, "duration": "1.0"},
        }
        service = _service()

        async def fake_run_stage(*_: object, **__: object) -> ProcessResult:
            return ProcessResult(stdout=json.dumps(payload).encode(), stderr=b"")

        service._run_stage = fake_run_stage  # type: ignore[method-assign]
        loop = asyncio.get_running_loop()
        return await service._probe_video(path, loop.time() + 60)

    async def test_all_optional_facts_missing(self, tmp_path: Path) -> None:
        probe = await self._probe(tmp_path, {})

        profile = probe.stream_profile()  # type: ignore[attr-defined]
        assert profile.codec == "h264"
        assert profile.codec_profile is None
        assert profile.pix_fmt is None
        assert profile.color_transfer is None
        assert profile.color_primaries is None
        assert profile.sample_aspect_ratio is None
        assert profile.rotation_degrees == 0
        assert profile.is_hdr is False
        assert probe.duration_source is DurationSource.STREAM  # type: ignore[attr-defined]

    @pytest.mark.parametrize(
        "odd",
        [
            {"profile": 5, "pix_fmt": ["x"], "color_transfer": {"a": 1}, "side_data_list": "x"},
            {"profile": "", "pix_fmt": "", "side_data_list": [None, 3, {"rotation": "junk"}]},
            {"side_data_list": [{"rotation": None}], "sample_aspect_ratio": 7},
            {"side_data_list": [{"displaymatrix": "x"}, {"rotation": "-90"}]},
        ],
    )
    async def test_odd_optional_facts_degrade_instead_of_rejecting(
        self, tmp_path: Path, odd: dict[str, object]
    ) -> None:
        probe = await self._probe(tmp_path, odd)

        profile = probe.stream_profile()  # type: ignore[attr-defined]
        assert profile.rotation_degrees in {0, 270}
        assert profile.is_hdr is False


@pytest.mark.parametrize(
    ("side_data", "expected"),
    [
        (None, 0),
        ([], 0),
        ([{"rotation": 90}], 90),
        ([{"rotation": -90}], 270),
        ([{"rotation": -180}], 180),
        ([{"rotation": 360}], 0),
        ([{"rotation": 450}], 90),
        ([{"rotation": 45}], 45),  # informational: not snapped to a right angle
        ([{"other": 1}, {"rotation": 90}], 90),  # first entry that carries rotation wins
        ([{"rotation": 90}, {"rotation": 180}], 90),
        ([{"rotation": "garbage"}], 0),
    ],
)
def test_rotation_normalization(side_data: object, expected: int) -> None:
    assert MediaIngestService._rotation_degrees(side_data) == expected
