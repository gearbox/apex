"""Unit tests for ffmpeg/ffprobe subprocess wrappers (frames.ffmpeg).

Mirrors tests/unit/test_thumbnail.py's convention: subprocess.run is fully
mocked (no real ffmpeg invocation) so tests are deterministic and portable
across dev machines / CI, matching FFMPEG_PATH/FFPROBE_PATH being fixed
Docker-only paths.
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.api.services.frames.ffmpeg import (
    FFMPEG_PATH,
    FFPROBE_PATH,
    FfmpegError,
    FfprobeError,
    VideoProbe,
    compute_uniform_timestamps,
    extract_frame,
    extract_preview_strip,
    probe,
)

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[2]

_PROBE_JSON = json.dumps(
    {
        "streams": [{"width": 1920, "height": 1080, "codec_name": "h264"}],
        "format": {"duration": "12.5"},
    }
).encode()

_PNG_BYTES = b"\x89PNG\r\n\x1a\nfakepngdata"
_WEBP_BYTES = b"RIFF\x00\x00\x00\x00WEBPfakewebpdata"


def _run_result(returncode: int = 0, stdout: bytes = b"", stderr: bytes = b"") -> MagicMock:
    r = MagicMock(spec=subprocess.CompletedProcess)
    r.returncode = returncode
    r.stdout = stdout
    r.stderr = stderr
    return r


class TestProbe:
    async def test_probe_returns_duration_and_dimensions(self) -> None:
        with patch("src.api.services.frames.ffmpeg.subprocess.run") as mock_run:
            mock_run.return_value = _run_result(returncode=0, stdout=_PROBE_JSON)
            result = await probe(Path("fake.mp4"), timeout_seconds=30)

        assert isinstance(result, VideoProbe)
        assert result.duration_ms == 12500
        assert result.width == 1920
        assert result.height == 1080
        assert result.codec == "h264"
        assert mock_run.call_args.args[0][0] == FFPROBE_PATH

    async def test_probe_rejects_non_video_bytes(self) -> None:
        with patch("src.api.services.frames.ffmpeg.subprocess.run") as mock_run:
            mock_run.return_value = _run_result(returncode=1, stderr=b"Invalid data found")
            with pytest.raises(FfprobeError):
                await probe(Path("fake.txt"), timeout_seconds=30)

    async def test_probe_rejects_no_video_stream(self) -> None:
        empty_streams = json.dumps({"streams": [], "format": {"duration": "1.0"}}).encode()
        with patch("src.api.services.frames.ffmpeg.subprocess.run") as mock_run:
            mock_run.return_value = _run_result(returncode=0, stdout=empty_streams)
            with pytest.raises(FfprobeError, match="No video stream"):
                await probe(Path("fake.mp4"), timeout_seconds=30)

    async def test_probe_rejects_malformed_json(self) -> None:
        with patch("src.api.services.frames.ffmpeg.subprocess.run") as mock_run:
            mock_run.return_value = _run_result(returncode=0, stdout=b"not json")
            with pytest.raises(FfprobeError, match="Could not parse"):
                await probe(Path("fake.mp4"), timeout_seconds=30)

    async def test_probe_raises_on_missing_binary(self) -> None:
        with patch("src.api.services.frames.ffmpeg.subprocess.run") as mock_run:
            mock_run.side_effect = FileNotFoundError("ffprobe not found")
            with pytest.raises(FfprobeError, match="not found"):
                await probe(Path("fake.mp4"), timeout_seconds=30)


class TestExtractFrame:
    async def test_extract_frame_at_timestamp_returns_png(self) -> None:
        with patch("src.api.services.frames.ffmpeg.subprocess.run") as mock_run:
            mock_run.return_value = _run_result(returncode=0, stdout=_PNG_BYTES)
            result = await extract_frame(
                Path("fake.mp4"), 2500, out_format="png", timeout_seconds=30
            )

        assert result == _PNG_BYTES
        args = mock_run.call_args.args[0]
        assert args[0] == FFMPEG_PATH
        assert "-ss" in args
        # -ss must precede -i (fast, frame-accurate seek).
        assert args.index("-ss") < args.index("-i")
        assert args[args.index("-ss") + 1] == "2.500"
        assert "-vf" not in args  # no scaling for full-res extract

    async def test_extract_frame_scaled_max_edge_for_preview(self) -> None:
        with patch("src.api.services.frames.ffmpeg.subprocess.run") as mock_run:
            mock_run.return_value = _run_result(returncode=0, stdout=_WEBP_BYTES)
            result = await extract_frame(
                Path("fake.mp4"),
                0,
                out_format="webp",
                max_edge=512,
                timeout_seconds=30,
            )

        assert result == _WEBP_BYTES
        args = mock_run.call_args.args[0]
        assert "-vf" in args
        scale_arg = args[args.index("-vf") + 1]
        assert "512" in scale_arg
        assert "libwebp" in args

    async def test_extract_frame_raises_on_nonzero_exit(self) -> None:
        with patch("src.api.services.frames.ffmpeg.subprocess.run") as mock_run:
            mock_run.return_value = _run_result(returncode=1, stderr=b"decode error")
            with pytest.raises(FfmpegError, match="decode error"):
                await extract_frame(Path("fake.mp4"), 0, out_format="png", timeout_seconds=30)

    async def test_extract_frame_raises_on_empty_output(self) -> None:
        with patch("src.api.services.frames.ffmpeg.subprocess.run") as mock_run:
            mock_run.return_value = _run_result(returncode=0, stdout=b"")
            with pytest.raises(FfmpegError, match="no output"):
                await extract_frame(Path("fake.mp4"), 0, out_format="png", timeout_seconds=30)

    async def test_extract_frame_rejects_unsupported_format(self) -> None:
        with pytest.raises(ValueError, match="Unsupported out_format"):
            await extract_frame(Path("fake.mp4"), 0, out_format="gif", timeout_seconds=30)

    async def test_ffmpeg_timeout_raises(self) -> None:
        with patch("src.api.services.frames.ffmpeg.subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.TimeoutExpired(cmd=FFMPEG_PATH, timeout=30)
            with pytest.raises(FfmpegError, match="timed out"):
                await extract_frame(Path("fake.mp4"), 0, out_format="png", timeout_seconds=30)


class TestExtractPreviewStrip:
    async def test_extracts_one_frame_per_timestamp(self) -> None:
        with patch("src.api.services.frames.ffmpeg.subprocess.run") as mock_run:
            mock_run.return_value = _run_result(returncode=0, stdout=_WEBP_BYTES)
            results = await extract_preview_strip(
                Path("fake.mp4"), [0, 1000, 2000], max_edge=512, timeout_seconds=30
            )

        assert results == [_WEBP_BYTES, _WEBP_BYTES, _WEBP_BYTES]
        assert mock_run.call_count == 3


class TestComputeUniformTimestamps:
    @pytest.mark.parametrize("frame_count", range(2, 61))
    def test_first_frame_at_zero_none_at_end(self, frame_count: int) -> None:
        duration_ms = 10_000
        timestamps = compute_uniform_timestamps(duration_ms, frame_count)

        assert len(timestamps) == frame_count
        assert timestamps[0] == 0
        assert all(ts < duration_ms for ts in timestamps)
        # Non-decreasing (uniform spacing).
        assert timestamps == sorted(timestamps)

    def test_spacing_is_uniform(self) -> None:
        timestamps = compute_uniform_timestamps(10_000, 5)
        assert timestamps == [0, 2000, 4000, 6000, 8000]


class TestFfmpegPathAssumption:
    """Guards the hardcoded FFMPEG_PATH/FFPROBE_PATH against silent image drift.

    Mirrors tests/unit/test_thumbnail.py::TestFfmpegPathAssumption.
    """

    @pytest.mark.parametrize("dockerfile_name", ["Dockerfile", "Dockerfile.dev"])
    def test_image_is_debian_based(self, dockerfile_name: str) -> None:
        contents = (_REPO_ROOT / dockerfile_name).read_text()
        from_lines = [line for line in contents.splitlines() if line.strip().startswith("FROM")]
        assert from_lines, f"{dockerfile_name} has no FROM line"
        for line in from_lines:
            assert re.search(r"-slim\b", line), (
                f"{dockerfile_name} base image changed away from a Debian '-slim' "
                f"variant ({line!r}). FFMPEG_PATH/FFPROBE_PATH assume apt-installed "
                "ffmpeg lands at /usr/bin/ — verify the new image's ffmpeg location."
            )

    @pytest.mark.parametrize("dockerfile_name", ["Dockerfile", "Dockerfile.dev"])
    def test_ffmpeg_installed_via_apt(self, dockerfile_name: str) -> None:
        contents = (_REPO_ROOT / dockerfile_name).read_text()
        joined_lines = re.sub(r"\\\n\s*", " ", contents).splitlines()
        apt_install_lines = [line for line in joined_lines if "apt-get install" in line]
        assert any("ffmpeg" in line for line in apt_install_lines), (
            f"{dockerfile_name} no longer installs ffmpeg via 'apt-get install'. "
            "FFMPEG_PATH/FFPROBE_PATH assume that install path."
        )
