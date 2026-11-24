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
    ALIBABA_OUTPUT = "alibaba_output"


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


class AudioProfileOrigin(StrEnum):
    """Which writer stored the audio."""

    UPLOAD = "upload"


def log_audio_profile(
    *,
    origin: AudioProfileOrigin,
    asset_ref: str,
    audio_format: str,
    codec: str,
    sample_rate_hz: int,
    channels: int,
    size_bytes: int,
    duration_ms: int,
) -> None:
    """Emit one ``media.audio_profile`` event for a stored audio original.

    Format, codec, sample rate, channels, size and duration only: never a tag value or a file
    name (the tags are what ingest strips, so they are never read here and never logged).
    """
    logger.info(
        "media.audio_profile",
        origin=origin.value,
        asset_ref=asset_ref,
        format=audio_format,
        codec=codec,
        sample_rate_hz=sample_rate_hz,
        channels=channels,
        size_bytes=size_bytes,
        duration_ms=duration_ms,
    )
