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


class StorageRangeNotSatisfiableError(StorageError):
    """Raised when a forwarded byte-range GetObject falls outside the object's bounds.

    Only raised when R2 itself rejects the range (e.g. the DB-recorded
    ``size_bytes`` used for pre-flight validation was stale) — the common
    "range is invalid" path is caught before any R2 call via
    ``parse_range``.
    """


class StorageValidationError(StorageError):
    """Raised when file validation fails (size, type, etc.)."""
