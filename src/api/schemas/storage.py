from datetime import datetime

import msgspec

from src.api.schemas.media import MediaObject


class UploadResponse(msgspec.Struct, kw_only=True):
    """Response for successful image upload."""

    id: str
    filename: str
    created_at: datetime
    expires_at: datetime

    media: MediaObject
    """Media envelope for the uploaded image: original asset + sm/md WEBP variants."""


class StorageStatsResponse(msgspec.Struct, kw_only=True):
    """Response for storage statistics."""

    upload_count: int
    output_count: int
    total_bytes: int
    total_mb: float
