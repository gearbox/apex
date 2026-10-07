from datetime import datetime
from uuid import UUID

import msgspec

from src.api.schemas.media import MediaObject


class UploadedImage(msgspec.Struct, kw_only=True):
    """Result of uploading an image."""

    id: UUID
    storage_key: str
    filename: str
    content_type: str
    size_bytes: int
    created_at: datetime
    expires_at: datetime
    media: MediaObject
