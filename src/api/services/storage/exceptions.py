"""Storage service exceptions."""

from __future__ import annotations


class StorageError(Exception):
    """Base exception for storage operations."""

    def __init__(self, message: str, cause: Exception | None = None) -> None:
        super().__init__(message)
        self.cause = cause


class StorageConnectionError(StorageError):
    """Raised when unable to connect to storage backend."""


class StorageUploadError(StorageError):
    """Raised when file upload fails."""


class StorageDownloadError(StorageError):
    """Raised when file download fails."""


class StorageDeleteError(StorageError):
    """Raised when file deletion fails."""


class StorageNotFoundError(StorageError):
    """Raised when requested object doesn't exist."""


class StorageValidationError(StorageError):
    """Raised when file validation fails (size, type, etc.)."""
