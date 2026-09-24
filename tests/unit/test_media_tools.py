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

    failure = [sys.executable, "-c", _BANNER_THEN_ERROR]
    with pytest.raises(MediaToolExitError) as caught:
        run_media_command_sync(failure, timeout_seconds=5)
    assert caught.value.returncode == 7
    assert caught.value.stderr_excerpt == "b" * 4086 + "fatal: EOF"


# ffmpeg prints its banner first and the fatal error last; the excerpt must
# keep the tail.
_BANNER_THEN_ERROR = "import sys; sys.stderr.write('b' * 5000 + 'fatal: EOF'); sys.exit(7)"
# Reading stdin must see EOF immediately rather than inherit the worker's stdin.
_READS_STDIN = "import sys; sys.stdout.write(repr(sys.stdin.read()))"


async def test_async_error_excerpt_is_the_stderr_tail() -> None:
    with pytest.raises(MediaToolExitError) as caught:
        await run_media_command([sys.executable, "-c", _BANNER_THEN_ERROR], timeout_seconds=5)
    assert caught.value.returncode == 7
    assert caught.value.stderr_excerpt.endswith("fatal: EOF")
    assert len(caught.value.stderr_excerpt) == 4096


async def test_async_stdin_is_devnull() -> None:
    result = await run_media_command([sys.executable, "-c", _READS_STDIN], timeout_seconds=5)
    assert result.stdout == b"''"


def test_sync_stdin_is_devnull() -> None:
    result = run_media_command_sync([sys.executable, "-c", _READS_STDIN], timeout_seconds=5)
    assert result.stdout == b"''"
