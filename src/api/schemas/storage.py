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


class ImageAccessResponse(msgspec.Struct, kw_only=True):
    """Response with presigned URL for image access."""

    id: str
    storage_key: str
    presigned_url: str
    content_type: str
    size_bytes: int
    expires_in_seconds: int


class ImageListItem(msgspec.Struct, kw_only=True):
    """Item in image list response."""

    id: str
    filename: str
    created_at: datetime
    expires_at: datetime

    media: MediaObject
    """Media envelope: original asset + sm/md WEBP preview variants."""


class OutputListItem(msgspec.Struct, kw_only=True):
    """Item in output list response."""

    id: str
    job_id: str
    output_index: int
    created_at: datetime
    expires_at: datetime

    media: MediaObject
    """Media envelope: original asset + sm/md WEBP preview variants."""


class StorageStatsResponse(msgspec.Struct, kw_only=True):
    """Response for storage statistics."""

    upload_count: int
    output_count: int
    total_bytes: int
    total_mb: float
