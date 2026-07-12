"""ffmpeg/ffprobe subprocess wrappers for video frame extraction.

Pure functions over paths/bytes — no R2, no DB, no settings singleton (every
tunable is an explicit parameter). Unlike ``services/thumbnail.py`` (whose
swallow-and-return-None contract is correct for an optional poster-frame
derivative), every failure here is **fail loud**: a frame extraction job that
can't decode its source must be marked ``failed`` with the real ffmpeg/ffprobe
stderr, never silently degrade.

FFMPEG_PATH/FFPROBE_PATH are fixed absolute paths (not a PATH lookup, to
satisfy Ruff S607) — see ``tests/unit/test_frames_ffmpeg.py::TestFfmpegPathAssumption``,
which mirrors ``tests/unit/test_thumbnail.py``'s Dockerfile-consistency guard.
"""

from __future__ import annotations

import asyncio
import json
import subprocess
from dataclasses import dataclass
from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    from pathlib import Path

logger = structlog.get_logger(__name__)

FFMPEG_PATH = "/usr/bin/ffmpeg"
FFPROBE_PATH = "/usr/bin/ffprobe"

_STDERR_TRIM_BYTES = 500


class FfmpegError(Exception):
    """Raised when an ffmpeg invocation fails, times out, or is unavailable."""


class FfprobeError(Exception):
    """Raised when ffprobe fails to decode/probe a video, or is unavailable."""


@dataclass(frozen=True, slots=True)
class VideoProbe:
    """Result of probing a video file."""

    duration_ms: int
    width: int
    height: int
    codec: str


def _trimmed_stderr(stderr: bytes) -> str:
    return stderr[:_STDERR_TRIM_BYTES].decode("utf-8", errors="replace")


def _run(
    args: list[str],
    *,
    timeout_seconds: float,
    error_cls: type[Exception],
) -> subprocess.CompletedProcess[bytes]:
    """Run a subprocess, raising ``error_cls`` (chained) on any failure mode.

    Wall-clock timeout is enforced by ``subprocess.run`` itself, not an outer
    ``asyncio.wait_for`` — cancelling an awaited ``to_thread`` does not kill
    the child process, so the timeout must live here.
    """
    try:
        result = subprocess.run(  # noqa: S603
            args,
            capture_output=True,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired as e:
        raise error_cls(f"{args[0]} timed out after {timeout_seconds}s") from e
    except FileNotFoundError as e:
        raise error_cls(f"{args[0]} not found") from e

    if result.returncode != 0:
        raise error_cls(f"{args[0]} exited {result.returncode}: {_trimmed_stderr(result.stderr)}")
    return result


def _parse_probe_json(raw: bytes) -> VideoProbe:
    data = json.loads(raw)
    streams = data["streams"]
    if not streams:
        raise FfprobeError("No video stream found")
    stream = streams[0]
    duration_s = float(data["format"]["duration"])
    return VideoProbe(
        duration_ms=round(duration_s * 1000),
        width=int(stream["width"]),
        height=int(stream["height"]),
        codec=str(stream["codec_name"]),
    )


def _probe_sync(video_path: Path, *, timeout_seconds: float) -> VideoProbe:
    result = _run(
        [
            FFPROBE_PATH,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,codec_name",
            "-show_entries",
            "format=duration",
            "-of",
            "json",
            str(video_path),
        ],
        timeout_seconds=timeout_seconds,
        error_cls=FfprobeError,
    )

    try:
        return _parse_probe_json(result.stdout)
    except FfprobeError:
        raise
    except (KeyError, IndexError, ValueError, TypeError, json.JSONDecodeError) as e:
        raise FfprobeError(f"Could not parse ffprobe output: {e}") from e


async def probe(video_path: Path, *, timeout_seconds: float) -> VideoProbe:
    """Probe a video file for duration/dimensions/codec.

    Raises:
        FfprobeError: If the file isn't decodable, has no video stream, or
            ffprobe cannot be run at all (missing binary, timeout).
    """
    return await asyncio.to_thread(_probe_sync, video_path, timeout_seconds=timeout_seconds)


_SCALE_FILTER = "scale='min(iw,{edge})':'min(ih,{edge})':force_original_aspect_ratio=decrease"
_CODEC_BY_FORMAT = {"png": "png", "webp": "libwebp"}


def _extract_frame_sync(
    video_path: Path,
    timestamp_ms: int,
    *,
    out_format: str,
    max_edge: int | None,
    timeout_seconds: float,
) -> bytes:
    if out_format not in _CODEC_BY_FORMAT:
        raise ValueError(f"Unsupported out_format: {out_format!r}")

    # -ss before -i: fast, frame-accurate seek (pitfall: -ss after -i decodes
    # from the start, O(duration) per frame).
    args = [
        FFMPEG_PATH,
        "-y",
        "-ss",
        f"{timestamp_ms / 1000:.3f}",
        "-i",
        str(video_path),
        "-frames:v",
        "1",
    ]
    if max_edge is not None:
        args += ["-vf", _SCALE_FILTER.format(edge=max_edge)]
    args += [
        "-f",
        "image2pipe",
        "-vcodec",
        _CODEC_BY_FORMAT[out_format],
        "pipe:1",
    ]

    result = _run(args, timeout_seconds=timeout_seconds, error_cls=FfmpegError)
    if not result.stdout:
        raise FfmpegError(f"ffmpeg produced no output for timestamp {timestamp_ms}ms")
    return result.stdout


async def extract_frame(
    video_path: Path,
    timestamp_ms: int,
    *,
    out_format: str,
    max_edge: int | None = None,
    timeout_seconds: float,
) -> bytes:
    """Extract a single frame at ``timestamp_ms``.

    Args:
        video_path: Path to the source video file.
        timestamp_ms: Timestamp to seek to, in milliseconds.
        out_format: ``"png"`` (full-resolution extract) or ``"webp"`` (preview).
        max_edge: When set, scales the longest edge down to this many pixels
            (never upscales) — used for the preview strip.
        timeout_seconds: Wall-clock subprocess timeout.

    Returns:
        Encoded image bytes.

    Raises:
        FfmpegError: If ffmpeg fails, times out, or produces no output.
    """
    return await asyncio.to_thread(
        _extract_frame_sync,
        video_path,
        timestamp_ms,
        out_format=out_format,
        max_edge=max_edge,
        timeout_seconds=timeout_seconds,
    )


async def extract_preview_strip(
    video_path: Path,
    timestamps_ms: list[int],
    *,
    max_edge: int,
    timeout_seconds: float,
) -> list[bytes]:
    """Extract a WEBP frame at each timestamp.

    A loop of single-frame seeks rather than a single-pass ``select`` filter —
    simpler and gives an exact-timestamp guarantee the extract phase needs to
    round-trip; N <= 60 seeks on a short clip is fast enough (KISS).
    """
    return [
        await extract_frame(
            video_path,
            ts,
            out_format="webp",
            max_edge=max_edge,
            timeout_seconds=timeout_seconds,
        )
        for ts in timestamps_ms
    ]


def compute_uniform_timestamps(duration_ms: int, frame_count: int) -> list[int]:
    """Compute ``frame_count`` uniformly spaced timestamps over ``[0, duration_ms)``.

    First frame is always at 0; deliberately no frame at the exact end, where
    seeks are unreliable.
    """
    return [round(i * duration_ms / frame_count) for i in range(frame_count)]
