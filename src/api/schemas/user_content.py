from datetime import datetime
from uuid import UUID

import msgspec


class UploadedImage(msgspec.Struct, kw_only=True):
    """Result of uploading an image."""

    id: UUID
    storage_key: str
    filename: str
    content_type: str
    size_bytes: int
    created_at: datetime
    expires_at: datetime


class GeneratedImage(msgspec.Struct, kw_only=True):
    """Result of storing a generated image."""

    id: UUID
    job_id: UUID
    storage_key: str
    content_type: str
    size_bytes: int
    output_index: int
    created_at: datetime
    expires_at: datetime


class ImageAccess(msgspec.Struct, kw_only=True):
    """Access information for an image."""

    storage_key: str
    presigned_url: str
    content_type: str
    size_bytes: int
    expires_in_seconds: int
