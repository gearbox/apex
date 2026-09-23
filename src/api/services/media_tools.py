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


def run_media_command_sync(
    args: Sequence[str], *, timeout_seconds: float, stderr_limit: int = 4_096
) -> ProcessResult:
    """Synchronous counterpart for legacy thread-worker media facades."""
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")
    try:
        process = subprocess.run(  # noqa: S603 - command is assembled only from service constants
            args,
            capture_output=True,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise MediaToolError(f"{args[0]} timed out after {timeout_seconds}s") from exc
    except FileNotFoundError as exc:
        raise MediaToolError(f"media executable not found: {args[0]}") from exc
    if process.returncode != 0:
        message = process.stderr[:stderr_limit].decode("utf-8", errors="replace")
        raise MediaToolError(f"{args[0]} exited {process.returncode}: {message}")
    return ProcessResult(stdout=process.stdout, stderr=process.stderr[:stderr_limit])


async def run_media_command(
    args: Sequence[str], *, timeout_seconds: float, stderr_limit: int = 4_096
) -> ProcessResult:
    """Run a command and ensure cancelled/timed-out children are reaped."""
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")
    try:
        process = await asyncio.create_subprocess_exec(
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except FileNotFoundError as exc:
        raise MediaToolError(f"media executable not found: {args[0]}") from exc
    try:
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout_seconds)
    except (TimeoutError, asyncio.CancelledError):
        if process.returncode is None:
            process.kill()
        await process.wait()
        raise
    if process.returncode != 0:
        message = stderr[:stderr_limit].decode("utf-8", errors="replace")
        raise MediaToolError(f"{args[0]} exited {process.returncode}: {message}")
    return ProcessResult(stdout=stdout, stderr=stderr[:stderr_limit])
