"""Small cancellable subprocess primitives shared by media services."""

from __future__ import annotations

import asyncio
import subprocess
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence


@dataclass(frozen=True, slots=True)
class ProcessResult:
    """Captured subprocess output, with stderr bounded for logs/errors."""

    stdout: bytes
    stderr: bytes


class MediaToolError(RuntimeError):
    """A media executable is absent, failed, or exceeded its deadline."""


class MediaToolNotFoundError(MediaToolError):
    """The requested executable is unavailable."""


class MediaToolTimeoutError(MediaToolError):
    """The process exceeded its deadline."""


class MediaToolExitError(MediaToolError):
    """The process exited unsuccessfully."""

    def __init__(self, executable: str, returncode: int, stderr_excerpt: str) -> None:
        self.returncode = returncode
        self.stderr_excerpt = stderr_excerpt
        super().__init__(f"{executable} exited {returncode}: {stderr_excerpt}")


def _successful_result(
    stdout: bytes, stderr: bytes, *, executable: str, stderr_capture_limit: int
) -> ProcessResult:
    if len(stderr) > stderr_capture_limit:
        raise MediaToolError(f"{executable} exceeded stderr capture limit")
    return ProcessResult(stdout=stdout, stderr=stderr)


def run_media_command_sync(
    args: Sequence[str],
    *,
    timeout_seconds: float,
    stderr_limit: int = 4_096,
    stderr_capture_limit: int = 1_048_576,
) -> ProcessResult:
    """Synchronous counterpart for legacy thread-worker media facades."""
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")
    try:
        process = subprocess.run(  # noqa: S603 - command is assembled only from service constants
            args,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise MediaToolTimeoutError(f"{args[0]} timed out after {timeout_seconds}s") from exc
    except FileNotFoundError as exc:
        raise MediaToolNotFoundError(f"media executable not found: {args[0]}") from exc
    if process.returncode != 0:
        # The tail carries the fatal error; the head is banner/preamble.
        stderr = process.stderr
        message = stderr[-stderr_limit:].decode("utf-8", errors="replace")
        raise MediaToolExitError(args[0], process.returncode, message)
    return _successful_result(
        process.stdout,
        process.stderr,
        executable=args[0],
        stderr_capture_limit=stderr_capture_limit,
    )


async def run_media_command(
    args: Sequence[str],
    *,
    timeout_seconds: float,
    stderr_limit: int = 4_096,
    stderr_capture_limit: int = 1_048_576,
) -> ProcessResult:
    """Run a command and ensure cancelled/timed-out children are reaped."""
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")
    try:
        process = await asyncio.create_subprocess_exec(
            *args,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except FileNotFoundError as exc:
        raise MediaToolNotFoundError(f"media executable not found: {args[0]}") from exc
    try:
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout_seconds)
    except (TimeoutError, asyncio.CancelledError) as exc:
        if process.returncode is None:
            process.kill()
        await process.wait()
        if isinstance(exc, TimeoutError):
            raise MediaToolTimeoutError(f"{args[0]} timed out after {timeout_seconds}s") from exc
        raise
    if process.returncode is not None and process.returncode != 0:
        # The tail carries the fatal error; the head is banner/preamble.
        message = stderr[-stderr_limit:].decode("utf-8", errors="replace")
        raise MediaToolExitError(args[0], process.returncode, message)
    return _successful_result(
        stdout, stderr, executable=args[0], stderr_capture_limit=stderr_capture_limit
    )
