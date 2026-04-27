"""Unit tests for video thumbnail extraction."""

from __future__ import annotations

import subprocess
from unittest.mock import MagicMock, patch

import pytest

from src.api.services.thumbnail import _extract_sync, extract_video_thumbnail

pytestmark = pytest.mark.unit

_FAKE_VIDEO = b"\x00\x01\x02\x03"
_FAKE_JPEG = b"\xff\xd8\xff\xe0jpeg"


class TestExtractVideoThumbnail:
    async def test_returns_jpeg_bytes_on_success(self) -> None:
        with patch("src.api.services.thumbnail.asyncio.to_thread") as mock_thread:
            mock_thread.return_value = _FAKE_JPEG
            result = await extract_video_thumbnail(_FAKE_VIDEO)
        assert result == _FAKE_JPEG

    async def test_returns_none_on_exception(self) -> None:
        with patch("src.api.services.thumbnail.asyncio.to_thread") as mock_thread:
            mock_thread.side_effect = RuntimeError("ffmpeg crashed")
            result = await extract_video_thumbnail(_FAKE_VIDEO)
        assert result is None


class TestExtractSync:
    def _run_result(self, returncode: int = 0, stdout: bytes = _FAKE_JPEG) -> MagicMock:
        r = MagicMock(spec=subprocess.CompletedProcess)
        r.returncode = returncode
        r.stdout = stdout
        r.stderr = b""
        return r

    def test_returns_jpeg_bytes_on_success(self) -> None:
        with patch("src.api.services.thumbnail.subprocess.run") as mock_run:
            mock_run.return_value = self._run_result(returncode=0, stdout=_FAKE_JPEG)
            result = _extract_sync(_FAKE_VIDEO)
        assert result == _FAKE_JPEG

    def test_returns_none_when_ffmpeg_nonzero_exit(self) -> None:
        with patch("src.api.services.thumbnail.subprocess.run") as mock_run:
            mock_run.return_value = self._run_result(returncode=1, stdout=b"")
            result = _extract_sync(_FAKE_VIDEO)
        assert result is None

    def test_returns_none_when_ffmpeg_empty_output(self) -> None:
        with patch("src.api.services.thumbnail.subprocess.run") as mock_run:
            mock_run.return_value = self._run_result(returncode=0, stdout=b"")
            result = _extract_sync(_FAKE_VIDEO)
        assert result is None

    def test_returns_none_when_ffmpeg_not_found(self) -> None:
        with patch("src.api.services.thumbnail.subprocess.run") as mock_run:
            mock_run.side_effect = FileNotFoundError("ffmpeg not found")
            result = _extract_sync(_FAKE_VIDEO)
        assert result is None

    def test_returns_none_on_timeout(self) -> None:
        with patch("src.api.services.thumbnail.subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.TimeoutExpired(cmd="ffmpeg", timeout=30)
            result = _extract_sync(_FAKE_VIDEO)
        assert result is None

    def test_temp_file_cleaned_up_on_success(self) -> None:
        with (
            patch("src.api.services.thumbnail.subprocess.run") as mock_run,
            patch("src.api.services.thumbnail.Path.unlink") as mock_unlink,
        ):
            mock_run.return_value = self._run_result()
            _extract_sync(_FAKE_VIDEO)
        mock_unlink.assert_called_once_with(missing_ok=True)
