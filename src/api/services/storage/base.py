"""Storage service protocol definition."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable
from uuid import UUID

if TYPE_CHECKING:
    from .schemas import DownloadResult, ImageFormat, StorageType, StoredFile, UploadResult


@runtime_checkable
class StorageService(Protocol):
    """Protocol for storage backend implementations.

    Defines the contract for file storage operations. Implementations
    must handle all aspects of file lifecycle including upload, download,
    deletion, and URL generation.
    """

    async def upload(
        self,
        *,
        user_id: UUID,
        data: bytes,
        filename: str,
        content_type: str,
        storage_type: StorageType,
        job_id: UUID | None = None,
    ) -> UploadResult:
        """Upload a file to storage.

        Args:
            user_id: Owner of the file.
            data: Raw file bytes.
            filename: Original filename (for reference).
            content_type: MIME type of the file.
            storage_type: Whether this is an upload or output.
            job_id: Associated generation job (for outputs).

        Returns:
            Upload result with storage key and optional presigned URL.

        Raises:
            StorageValidationError: If file validation fails.
            StorageUploadError: If upload operation fails.
        """
        ...

    async def download(
        self,
        storage_key: str,
    ) -> bytes:
        """Download file content.

        Args:
            storage_key: Full storage key for the file.

        Returns:
            Raw file bytes.

        Raises:
            StorageNotFoundError: If file doesn't exist.
            StorageDownloadError: If download fails.
        """
        ...

    async def get_presigned_url(
        self,
        storage_key: str,
        *,
        expires_in: int = 3600,
    ) -> DownloadResult:
        """Generate a presigned URL for temporary access.

        Args:
            storage_key: Full storage key for the file.
            expires_in: URL validity duration in seconds.

        Returns:
            Download result with presigned URL and metadata.

        Raises:
            StorageNotFoundError: If file doesn't exist.
        """
        ...

    async def delete(
        self,
        storage_key: str,
    ) -> bool:
        """Delete a file from storage.

        Args:
            storage_key: Full storage key for the file.

        Returns:
            True if file was deleted, False if it didn't exist.

        Raises:
            StorageDeleteError: If deletion fails.
        """
        ...

    async def delete_many(
        self,
        storage_keys: list[str],
    ) -> int:
        """Delete multiple files from storage.

        Args:
            storage_keys: List of storage keys to delete.

        Returns:
            Number of files successfully deleted.

        Raises:
            StorageDeleteError: If deletion fails.
        """
        ...

    async def exists(
        self,
        storage_key: str,
    ) -> bool:
        """Check if a file exists in storage.

        Args:
            storage_key: Full storage key for the file.

        Returns:
            True if file exists, False otherwise.
        """
        ...

    async def list_user_files(
        self,
        user_id: UUID,
        *,
        storage_type: StorageType | None = None,
        limit: int = 100,
    ) -> list[StoredFile]:
        """List files for a user.

        Args:
            user_id: User to list files for.
            storage_type: Filter by upload or output (optional).
            limit: Maximum number of files to return.

        Returns:
            List of stored file metadata.
        """
        ...

    def build_storage_key(
        self,
        *,
        user_id: UUID,
        file_id: UUID,
        storage_type: StorageType,
        format: ImageFormat,
        job_id: UUID | None = None,
    ) -> str:
        """Build the storage key for a file.

        Key format:
        - Uploads: users/{user_id}/uploads/{file_id}.{ext}
        - Outputs: users/{user_id}/outputs/{job_id}/{file_id}.{ext}

        Args:
            user_id: Owner of the file.
            file_id: Unique file identifier.
            storage_type: Whether this is an upload or output.
            format: Image format determining extension.
            job_id: Job ID (required for outputs).

        Returns:
            Full storage key string.
        """
        ...

    async def health_check(self) -> bool:
        """Check if storage backend is accessible.

        Returns:
            True if storage is healthy, False otherwise.
        """
        ...

    async def close(self) -> None:
        """Close any open connections.

        Called during application shutdown.
        """
        ...
