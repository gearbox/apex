"""Storage service DTOs using msgspec."""

from __future__ import annotations

__all__ = [
    "DownloadResult",
    "MaxFileSize",
    "MediaFormat",
    "StorageStats",
    "StorageType",
    "StoredFile",
    "UploadRequest",
    "UploadResult",
]

from datetime import datetime
from enum import Enum
from typing import Annotated
from uuid import UUID

import msgspec

from src.core.enums import MediaFormat


class StorageType(str, Enum):
    """Type of stored content."""

    UPLOAD = "upload"  # User-uploaded input images
    OUTPUT = "output"  # Generated output images


# Validation constraints
MaxFileSize = Annotated[int, msgspec.Meta(le=20 * 1024 * 1024)]  # 20MB


class StoredFile(msgspec.Struct, kw_only=True):
    """Represents a file stored in R2."""

    id: UUID
    user_id: UUID
    storage_type: StorageType
    storage_key: str  # Full R2 key: users/{user_id}/uploads/{uuid}.{ext}
    filename: str  # Original filename
    format: MediaFormat
    size_bytes: int
    content_type: str
    created_at: datetime
    expires_at: datetime | None = None  # For retention policy
    job_id: UUID | None = None  # Associated job (for outputs)


class UploadRequest(msgspec.Struct, kw_only=True):
    """Request to upload a file."""

    user_id: UUID
    filename: str
    content_type: str
    size_bytes: int


class UploadResult(msgspec.Struct, kw_only=True):
    """Result of a successful upload."""

    id: UUID
    storage_key: str
    presigned_url: str | None = None  # For direct access
    expires_at: datetime | None = None


class DownloadResult(msgspec.Struct, kw_only=True):
    """Result of requesting a file download."""

    storage_key: str
    presigned_url: str
    content_type: str
    size_bytes: int
    expires_in_seconds: int


class StorageStats(msgspec.Struct, kw_only=True):
    """Storage usage statistics for a user."""

    user_id: UUID
    total_uploads: int
    total_outputs: int
    total_size_bytes: int
    oldest_file: datetime | None = None
