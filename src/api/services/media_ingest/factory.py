"""Explicit composition root for the process-local media ingest service."""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.api.services.media_ingest.service import MediaIngestService

if TYPE_CHECKING:
    from src.core.config import Settings


def build_media_ingest_service(settings: Settings) -> MediaIngestService:
    """Build one shared bounded service from validated application settings."""
    return MediaIngestService(
        max_image_megapixels=settings.image_max_input_megapixels,
        max_input_bytes=settings.max_upload_size_mb * 1024 * 1024,
        video_max_frames=settings.media_ingest_video_max_frames,
        video_max_edge=settings.media_ingest_video_max_edge,
        video_concurrency=settings.media_ingest_video_concurrency,
        image_concurrency=settings.media_ingest_image_concurrency,
        admission_wait_seconds=settings.media_ingest_admission_wait_seconds,
        video_deadline_seconds=settings.media_ingest_video_deadline_seconds,
        stage_timeout_seconds=settings.media_ingest_stage_timeout_seconds,
        max_animation_frames=settings.media_ingest_max_animation_frames,
    )
