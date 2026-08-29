"""Unit tests for video thumbnail extraction."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.api.services.thumbnail import FFMPEG_PATH, _extract_sync, extract_video_thumbnail

pytestmark = pytest.mark.unit

_FAKE_VIDEO = b"\x00\x01\x02\x03"
_FAKE_JPEG = b"\xff\xd8\xff\xe0jpeg"

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _read_dockerfile(dockerfile_name: str) -> str:
    """Read a build manifest when tests run from a source checkout.

    Application-image test runs package the Python sources under ``/app`` but
    intentionally omit Docker build manifests. The path invariant cannot be
    checked there, so skip it rather than treating the absent build context as
    a product failure.
    """
    path = _REPO_ROOT / dockerfile_name
    if not path.is_file():
        pytest.skip(f"{dockerfile_name} is not present in this test environment")
    return path.read_text()


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
        assert mock_run.call_args.args[0][0] == FFMPEG_PATH
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
            mock_run.side_effect = subprocess.TimeoutExpired(cmd=FFMPEG_PATH, timeout=30)
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


class TestFfmpegPathAssumption:
    """Guards the hardcoded FFMPEG_PATH against silent drift from image changes.

    FFMPEG_PATH is a fixed absolute path rather than a PATH lookup (to satisfy
    Ruff S607). That is only correct because both Dockerfiles use a
    Debian-based image and install ffmpeg via apt, which places the binary at
    /usr/bin/ffmpeg. If the base image or install method ever changes, ffmpeg
    could land somewhere else and extraction would fail silently (caught and
    logged as a warning, thumbnail just becomes None) instead of raising. This
    test fails loudly instead, so a Dockerfile change forces a check of
    FFMPEG_PATH.
    """

    @pytest.mark.parametrize("dockerfile_name", ["Dockerfile", "Dockerfile.dev"])
    def test_image_is_debian_based(self, dockerfile_name: str) -> None:
        contents = _read_dockerfile(dockerfile_name)
        from_lines = [line for line in contents.splitlines() if line.strip().startswith("FROM")]
        assert from_lines, f"{dockerfile_name} has no FROM line"
        for line in from_lines:
            assert re.search(r"-slim\b", line), (
                f"{dockerfile_name} base image changed away from a Debian '-slim' "
                f"variant ({line!r}). FFMPEG_PATH={FFMPEG_PATH!r} assumes apt-installed "
                "ffmpeg lands at /usr/bin/ffmpeg on Debian — verify the new image's "
                "ffmpeg location and update FFMPEG_PATH if needed."
            )

    @pytest.mark.parametrize("dockerfile_name", ["Dockerfile", "Dockerfile.dev"])
    def test_ffmpeg_installed_via_apt(self, dockerfile_name: str) -> None:
        contents = _read_dockerfile(dockerfile_name)
        assert "apk add" not in contents, (
            f"{dockerfile_name} installs packages via apk (Alpine) — ffmpeg would not "
            f"be at FFMPEG_PATH={FFMPEG_PATH!r}. Update FFMPEG_PATH to match."
        )
        # Collapse RUN line-continuations (`\` + newline) so a multi-line
        # `apt-get install ... \` block reads as one logical line.
        joined_lines = re.sub(r"\\\n\s*", " ", contents).splitlines()
        apt_install_lines = [line for line in joined_lines if "apt-get install" in line]
        assert any("ffmpeg" in line for line in apt_install_lines), (
            f"{dockerfile_name} no longer installs ffmpeg via 'apt-get install'. "
            f"FFMPEG_PATH={FFMPEG_PATH!r} assumes that install path — verify how ffmpeg "
            "is provided now and update FFMPEG_PATH if it moved."
        )
