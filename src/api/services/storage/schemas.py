"""Storage service DTOs using msgspec."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Annotated
from uuid import UUID

import msgspec


class StorageType(str, Enum):
    """Type of stored content."""

    UPLOAD = "upload"  # User-uploaded input images
    OUTPUT = "output"  # Generated output images


class ImageFormat(str, Enum):
    """Supported image formats (ComfyUI compatible)."""

    PNG = "png"
    JPEG = "jpeg"
    WEBP = "webp"

    @classmethod
    def from_content_type(cls, content_type: str) -> ImageFormat:
        """Get format from MIME type."""
        mapping = {
            "image/png": cls.PNG,
            "image/jpeg": cls.JPEG,
            "image/jpg": cls.JPEG,
            "image/webp": cls.WEBP,
        }
        fmt = mapping.get(content_type.lower())
        if fmt is None:
            raise ValueError(f"Unsupported content type: {content_type}")
        return fmt

    @classmethod
    def from_extension(cls, ext: str) -> ImageFormat:
        """Get format from file extension."""
        ext = ext.lower().lstrip(".")
        mapping = {
            "png": cls.PNG,
            "jpeg": cls.JPEG,
            "jpg": cls.JPEG,
            "webp": cls.WEBP,
        }
        fmt = mapping.get(ext)
        if fmt is None:
            raise ValueError(f"Unsupported extension: {ext}")
        return fmt

    @property
    def content_type(self) -> str:
        """Get MIME type for format."""
        return {
            self.PNG: "image/png",
            self.JPEG: "image/jpeg",
            self.WEBP: "image/webp",
        }[self]

    @property
    def extension(self) -> str:
        """Get file extension for format."""
        return self.value


# Validation constraints
MaxFileSize = Annotated[int, msgspec.Meta(le=20 * 1024 * 1024)]  # 20MB


class StoredFile(msgspec.Struct, kw_only=True):
    """Represents a file stored in R2."""

    id: UUID
    user_id: UUID
    storage_type: StorageType
    storage_key: str  # Full R2 key: users/{user_id}/uploads/{uuid}.{ext}
    filename: str  # Original filename
    format: ImageFormat
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
