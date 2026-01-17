"""User content service - orchestrates R2 storage and database operations.

This is the main service layer for handling user content (uploads and outputs).
It coordinates between R2 storage for actual file storage and PostgreSQL
for metadata tracking and efficient queries.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING
from uuid import UUID

import msgspec

from src.api.services.storage import (
    ImageFormat,
    R2StorageService,
    StorageNotFoundError,
    StorageType,
    StorageValidationError,
)
from src.db.repository import StorageRepository

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from src.db.models import GenerationOutput, UserImage

logger = logging.getLogger(__name__)


class UserContentError(Exception):
    """Base exception for user content operations."""


class UserContentNotFoundError(UserContentError):
    """Raised when requested content doesn't exist."""


class UserContentValidationError(UserContentError):
    """Raised when content validation fails."""


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


class UserContentService:
    """Service for managing user-uploaded and generated content.

    Coordinates between R2 storage (files) and PostgreSQL (metadata).
    Provides atomic operations that maintain consistency between both.
    """

    def __init__(
        self,
        storage: R2StorageService,
        session: AsyncSession,
        *,
        retention_days: int = 7,
    ) -> None:
        """Initialize user content service.

        Args:
            storage: R2 storage service for file operations.
            session: Database session for metadata operations.
            retention_days: Days to retain content before cleanup.
        """
        self._storage = storage
        self._repo = StorageRepository(session)
        self._retention_days = retention_days

    # -------------------------------------------------------------------------
    # Upload operations
    # -------------------------------------------------------------------------

    async def upload_image(
        self,
        *,
        user_id: UUID,
        data: bytes,
        filename: str,
        content_type: str,
    ) -> UploadedImage:
        """Upload an image for use in generation.

        Uploads to R2 and creates database record atomically.

        Args:
            user_id: Owner of the image.
            data: Raw image bytes.
            filename: Original filename.
            content_type: MIME type.

        Returns:
            UploadedImage with storage details.

        Raises:
            UserContentValidationError: If validation fails.
        """
        try:
            # Upload to R2 (validates size/format internally)
            result = await self._storage.upload(
                user_id=user_id,
                data=data,
                filename=filename,
                content_type=content_type,
                storage_type=StorageType.UPLOAD,
            )

            # Determine format for DB
            image_format = ImageFormat.from_content_type(content_type)
            now = datetime.now(timezone.utc)
            expires_at = now + timedelta(days=self._retention_days)

            # Create database record
            db_image = await self._repo.create_user_image(
                id=result.id,
                user_id=user_id,
                storage_key=result.storage_key,
                original_filename=filename,
                content_type=content_type,
                size_bytes=len(data),
                format=image_format.value,
                expires_at=expires_at,
            )

            logger.info(
                f"Uploaded image {result.id} for user {user_id}: " f"{filename} ({len(data)} bytes)"
            )

            return UploadedImage(
                id=db_image.id,
                storage_key=db_image.storage_key,
                filename=db_image.original_filename,
                content_type=db_image.content_type,
                size_bytes=db_image.size_bytes,
                created_at=db_image.created_at,
                expires_at=db_image.expires_at,
            )

        except StorageValidationError as e:
            raise UserContentValidationError(str(e)) from e

    async def get_upload(self, image_id: UUID) -> UserImage | None:
        """Get upload metadata by ID.

        Args:
            image_id: Image ID to look up.

        Returns:
            UserImage if found, None otherwise.
        """
        return await self._repo.get_user_image(image_id)

    async def get_upload_by_key(self, storage_key: str) -> UserImage | None:
        """Get upload metadata by storage key.

        Args:
            storage_key: R2 storage key.

        Returns:
            UserImage if found, None otherwise.
        """
        return await self._repo.get_user_image_by_key(storage_key)

    async def get_upload_access(
        self,
        image_id: UUID,
        *,
        expires_in: int = 3600,
    ) -> ImageAccess:
        """Get presigned URL for accessing an upload.

        Args:
            image_id: Image ID to access.
            expires_in: URL validity in seconds.

        Returns:
            ImageAccess with presigned URL.

        Raises:
            UserContentNotFoundError: If image doesn't exist.
        """
        image = await self._repo.get_user_image(image_id)
        if image is None:
            raise UserContentNotFoundError(f"Image not found: {image_id}")

        result = await self._storage.get_presigned_url(
            image.storage_key,
            expires_in=expires_in,
        )

        return ImageAccess(
            storage_key=result.storage_key,
            presigned_url=result.presigned_url,
            content_type=result.content_type,
            size_bytes=result.size_bytes,
            expires_in_seconds=result.expires_in_seconds,
        )

    async def download_upload(self, image_id: UUID) -> bytes:
        """Download upload content.

        Args:
            image_id: Image ID to download.

        Returns:
            Raw image bytes.

        Raises:
            UserContentNotFoundError: If image doesn't exist.
        """
        image = await self._repo.get_user_image(image_id)
        if image is None:
            raise UserContentNotFoundError(f"Image not found: {image_id}")

        try:
            return await self._storage.download(image.storage_key)
        except StorageNotFoundError as e:
            # DB record exists but R2 file missing - data inconsistency
            logger.error(f"R2 file missing for image {image_id}: {image.storage_key}")
            raise UserContentNotFoundError(f"Image file not found: {image_id}") from e

    async def list_user_uploads(
        self,
        user_id: UUID,
        *,
        limit: int = 100,
        offset: int = 0,
    ) -> list[UserImage]:
        """List uploads for a user.

        Args:
            user_id: User to list uploads for.
            limit: Maximum results.
            offset: Results to skip.

        Returns:
            List of UserImage metadata.
        """
        images = await self._repo.list_user_images(
            user_id,
            limit=limit,
            offset=offset,
        )
        return list(images)

    async def delete_upload(self, image_id: UUID) -> bool:
        """Delete an uploaded image.

        Removes from both R2 and database.

        Args:
            image_id: Image ID to delete.

        Returns:
            True if deleted, False if not found.
        """
        image = await self._repo.get_user_image(image_id)
        if image is None:
            return False

        # Delete from R2 first
        await self._storage.delete(image.storage_key)

        # Then delete DB record
        await self._repo.delete_user_image(image_id)

        logger.info(f"Deleted upload {image_id}")
        return True

    # -------------------------------------------------------------------------
    # Output operations
    # -------------------------------------------------------------------------

    async def store_output(
        self,
        *,
        user_id: UUID,
        job_id: UUID,
        data: bytes,
        content_type: str,
        output_index: int,
        input_image_id: UUID | None = None,
    ) -> GeneratedImage:
        """Store a generated output image.

        Uploads to R2 and creates database record atomically.

        Args:
            user_id: Owner of the output.
            job_id: Associated generation job.
            data: Raw image bytes.
            content_type: MIME type.
            output_index: Index in batch (0-based).
            input_image_id: Associated input image (for i2i).

        Returns:
            GeneratedImage with storage details.
        """
        # Upload to R2
        result = await self._storage.upload(
            user_id=user_id,
            data=data,
            filename=f"output_{output_index}.png",
            content_type=content_type,
            storage_type=StorageType.OUTPUT,
            job_id=job_id,
        )

        # Determine format
        image_format = ImageFormat.from_content_type(content_type)
        now = datetime.now(timezone.utc)
        expires_at = now + timedelta(days=self._retention_days)

        # Create database record
        db_output = await self._repo.create_output(
            id=result.id,
            user_id=user_id,
            job_id=job_id,
            storage_key=result.storage_key,
            content_type=content_type,
            size_bytes=len(data),
            format=image_format.value,
            output_index=output_index,
            expires_at=expires_at,
            input_image_id=input_image_id,
        )

        logger.info(
            f"Stored output {result.id} for job {job_id}: "
            f"index={output_index} ({len(data)} bytes)"
        )

        return GeneratedImage(
            id=db_output.id,
            job_id=db_output.job_id,
            storage_key=db_output.storage_key,
            content_type=db_output.content_type,
            size_bytes=db_output.size_bytes,
            output_index=db_output.output_index,
            created_at=db_output.created_at,
            expires_at=db_output.expires_at,
        )

    async def get_output(self, output_id: UUID) -> GenerationOutput | None:
        """Get output metadata by ID.

        Args:
            output_id: Output ID to look up.

        Returns:
            GenerationOutput if found, None otherwise.
        """
        return await self._repo.get_output(output_id)

    async def get_output_access(
        self,
        output_id: UUID,
        *,
        expires_in: int = 3600,
    ) -> ImageAccess:
        """Get presigned URL for accessing an output.

        Args:
            output_id: Output ID to access.
            expires_in: URL validity in seconds.

        Returns:
            ImageAccess with presigned URL.

        Raises:
            UserContentNotFoundError: If output doesn't exist.
        """
        output = await self._repo.get_output(output_id)
        if output is None:
            raise UserContentNotFoundError(f"Output not found: {output_id}")

        result = await self._storage.get_presigned_url(
            output.storage_key,
            expires_in=expires_in,
        )

        return ImageAccess(
            storage_key=result.storage_key,
            presigned_url=result.presigned_url,
            content_type=result.content_type,
            size_bytes=result.size_bytes,
            expires_in_seconds=result.expires_in_seconds,
        )

    async def download_output(self, output_id: UUID) -> bytes:
        """Download output content.

        Args:
            output_id: Output ID to download.

        Returns:
            Raw image bytes.

        Raises:
            UserContentNotFoundError: If output doesn't exist.
        """
        output = await self._repo.get_output(output_id)
        if output is None:
            raise UserContentNotFoundError(f"Output not found: {output_id}")

        try:
            return await self._storage.download(output.storage_key)
        except StorageNotFoundError as e:
            logger.error(f"R2 file missing for output {output_id}: {output.storage_key}")
            raise UserContentNotFoundError(f"Output file not found: {output_id}") from e

    async def list_job_outputs(self, job_id: UUID) -> list[GenerationOutput]:
        """List outputs for a job.

        Args:
            job_id: Job to list outputs for.

        Returns:
            List of GenerationOutput metadata ordered by index.
        """
        outputs = await self._repo.list_job_outputs(job_id)
        return list(outputs)

    async def list_user_outputs(
        self,
        user_id: UUID,
        *,
        limit: int = 100,
        offset: int = 0,
    ) -> list[GenerationOutput]:
        """List outputs for a user.

        Args:
            user_id: User to list outputs for.
            limit: Maximum results.
            offset: Results to skip.

        Returns:
            List of GenerationOutput metadata.
        """
        outputs = await self._repo.list_user_outputs(
            user_id,
            limit=limit,
            offset=offset,
        )
        return list(outputs)

    # -------------------------------------------------------------------------
    # Storage key utilities (for ComfyUI integration)
    # -------------------------------------------------------------------------

    def get_upload_storage_key(self, image_id: UUID, user_id: UUID, format: ImageFormat) -> str:
        """Get the R2 storage key for an upload.

        Useful for passing to ComfyUI S3 nodes.

        Args:
            image_id: Image file ID.
            user_id: Owner of the image.
            format: Image format.

        Returns:
            Full R2 storage key.
        """
        return self._storage.build_storage_key(
            user_id=user_id,
            file_id=image_id,
            storage_type=StorageType.UPLOAD,
            format=format,
        )

    def get_output_storage_key(
        self,
        output_id: UUID,
        user_id: UUID,
        job_id: UUID,
        format: ImageFormat,
    ) -> str:
        """Get the R2 storage key for an output.

        Args:
            output_id: Output file ID.
            user_id: Owner of the output.
            job_id: Associated job.
            format: Image format.

        Returns:
            Full R2 storage key.
        """
        return self._storage.build_storage_key(
            user_id=user_id,
            file_id=output_id,
            storage_type=StorageType.OUTPUT,
            format=format,
            job_id=job_id,
        )

    # -------------------------------------------------------------------------
    # Statistics
    # -------------------------------------------------------------------------

    async def get_user_stats(self, user_id: UUID) -> dict[str, int]:
        """Get storage statistics for a user.

        Args:
            user_id: User to get stats for.

        Returns:
            Dict with upload_count, output_count, total_bytes.
        """
        return await self._repo.get_user_storage_stats(user_id)
