"""Subprocess output and error classification contracts."""

from __future__ import annotations

import sys

import pytest

from src.api.services.media_tools import (
    MediaToolError,
    MediaToolExitError,
    run_media_command,
    run_media_command_sync,
)

pytestmark = pytest.mark.unit


async def test_success_stderr_capture_is_separate_from_error_excerpt() -> None:
    command = [sys.executable, "-c", "import sys; sys.stderr.write('x' * 5000)"]
    result = await run_media_command(command, timeout_seconds=5)
    assert len(result.stderr) == 5000
    with pytest.raises(MediaToolError, match="stderr capture limit"):
        await run_media_command(command, timeout_seconds=5, stderr_capture_limit=4096)


def test_sync_success_capture_and_error_excerpt_are_independent() -> None:
    success = [sys.executable, "-c", "import sys; sys.stderr.write('x' * 5000)"]
    result = run_media_command_sync(success, timeout_seconds=5)
    assert len(result.stderr) == 5000
    with pytest.raises(MediaToolError, match="stderr capture limit"):
        run_media_command_sync(success, timeout_seconds=5, stderr_capture_limit=4096)

    failure = [sys.executable, "-c", "import sys; sys.stderr.write('y' * 5000); sys.exit(7)"]
    with pytest.raises(MediaToolExitError) as caught:
        run_media_command_sync(failure, timeout_seconds=5)
    assert caught.value.returncode == 7
    assert caught.value.stderr_excerpt == "y" * 4096
