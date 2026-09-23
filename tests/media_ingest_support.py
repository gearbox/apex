"""Consistent ingest configuration for direct service construction in tests."""

from src.api.services.media_ingest import MediaIngestService


def make_media_ingestor(*, max_image_megapixels: float = 100) -> MediaIngestService:
    return MediaIngestService(
        max_image_megapixels=max_image_megapixels,
        max_input_bytes=20 * 1024 * 1024,
    )
