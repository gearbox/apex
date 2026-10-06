"""``media.video_profile`` — the one structured log of a stored video's stream facts.

Writers (not the ingest service, which owns no storage or database context) call this
*after* the original row exists, passing their origin and IDs. The facts are
informational: they let us measure how much stored video is HDR / rotated / non-H.264
before deciding whether any browser-compatibility conversion is worth building. The
event carries no user-authored data.
"""

from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    from uuid import UUID

    from src.api.services.media_ingest.types import VideoStreamProfile

logger = structlog.get_logger(__name__)


class VideoProfileOrigin(StrEnum):
    """Which writer stored the video."""

    UPLOAD = "upload"
    GROK_OUTPUT = "grok_output"


def log_video_profile(
    *,
    origin: VideoProfileOrigin,
    asset_ref: str,
    profile: VideoStreamProfile,
    width: int | None,
    height: int | None,
    duration_ms: int | None,
    job_id: UUID | None = None,
) -> None:
    """Emit one ``media.video_profile`` event for a stored video original."""
    extra = {"job_id": str(job_id)} if job_id is not None else {}
    logger.info(
        "media.video_profile",
        origin=origin.value,
        asset_ref=asset_ref,
        **extra,
        container=profile.container.value,
        codec=profile.codec,
        codec_profile=profile.codec_profile,
        pix_fmt=profile.pix_fmt,
        color_transfer=profile.color_transfer,
        color_primaries=profile.color_primaries,
        hdr=profile.is_hdr,
        rotation_degrees=profile.rotation_degrees,
        sample_aspect_ratio=profile.sample_aspect_ratio,
        has_audio=profile.has_audio,
        width=width,
        height=height,
        duration_ms=duration_ms,
    )
