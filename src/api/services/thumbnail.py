"""Video thumbnail extraction.

Extracts the first frame of a video as a JPEG thumbnail using ffmpeg.
Called from ``GrokJobService._store_video_result()`` immediately after
the video is downloaded from xAI CDN.

If ffmpeg is unavailable the function logs a warning and returns None
rather than crashing the job — thumbnail is optional.
"""

from __future__ import annotations

import asyncio
import subprocess
import tempfile
from pathlib import Path

import structlog

logger = structlog.get_logger(__name__)


async def extract_video_thumbnail(video_bytes: bytes) -> bytes | None:
    """Extract the first frame of a video as JPEG bytes.

    Runs ffmpeg in a subprocess via ``asyncio.to_thread`` so it does not
    block the event loop. Uses a temp file for input because ffmpeg cannot
    always reliably seek a pipe for frame extraction.

    Args:
        video_bytes: Raw video file bytes (MP4, WebM, etc.).

    Returns:
        JPEG image bytes of the first frame, or ``None`` on any failure.
    """
    try:
        return await asyncio.to_thread(_extract_sync, video_bytes)
    except Exception:
        logger.warning("thumbnail.extraction_failed")
        return None


def _extract_sync(video_bytes: bytes) -> bytes | None:
    """Synchronous ffmpeg wrapper — runs in a thread pool."""
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_in:
        tmp_in.write(video_bytes)
        tmp_in_path = Path(tmp_in.name)

    try:
        result = subprocess.run(  # noqa: S603
            [
                "ffmpeg",
                "-y",  # overwrite output
                "-i",
                str(tmp_in_path),  # input file
                "-vframes",
                "1",  # extract exactly one frame
                "-q:v",
                "2",  # JPEG quality (2 = ~95%, good balance)
                "-f",
                "image2pipe",  # output to stdout
                "-vcodec",
                "mjpeg",
                "pipe:1",
            ],
            capture_output=True,
            timeout=30,
        )

        if result.returncode != 0:
            logger.warning(
                "thumbnail.ffmpeg_unavailable",
                code=result.returncode,
                stderr=result.stderr[:200].decode("utf-8", errors="replace"),
            )
            return None

        if not result.stdout:
            logger.warning("thumbnail.ffmpeg_empty_output")
            return None

        return result.stdout

    except FileNotFoundError:
        logger.warning("thumbnail.ffmpeg_not_found")
        return None
    except subprocess.TimeoutExpired:
        logger.warning("thumbnail.ffmpeg_timeout")
        return None
    finally:
        tmp_in_path.unlink(missing_ok=True)
