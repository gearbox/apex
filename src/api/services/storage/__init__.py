"""Storage service module.

Provides abstracted file storage operations with Cloudflare R2 implementation.
"""

from .base import StorageService
from .exceptions import (
    StorageConnectionError,
    StorageDeleteError,
    StorageDownloadError,
    StorageError,
    StorageNotFoundError,
    StorageUploadError,
    StorageValidationError,
)
from .r2 import R2StorageService, R2StorageSettings
from .schemas import (
    DownloadResult,
    ImageFormat,
    StorageStats,
    StorageType,
    StoredFile,
    UploadResult,
)

__all__ = [
    # Protocol
    "StorageService",
    # Implementation
    "R2StorageService",
    "R2StorageSettings",
    # Schemas
    "DownloadResult",
    "ImageFormat",
    "StorageStats",
    "StorageType",
    "StoredFile",
    "UploadResult",
    # Exceptions
    "StorageConnectionError",
    "StorageDeleteError",
    "StorageDownloadError",
    "StorageError",
    "StorageNotFoundError",
    "StorageUploadError",
    "StorageValidationError",
]
