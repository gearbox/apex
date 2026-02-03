"""Repository for storage-related database operations."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING
from uuid import UUID

from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.models import GenerationJob, GenerationOutput, UserImage

if TYPE_CHECKING:
    from collections.abc import Sequence


class StorageRepository:
    """Repository for storage-related database operations.

    Provides data access methods for UserImage, GenerationJob, and
    GenerationOutput models. All methods are async and use the
    provided session for transaction management.
    """

    def __init__(self, session: AsyncSession) -> None:
        """Initialize repository with database session.

        Args:
            session: Async SQLAlchemy session.
        """
        self._session = session

    # -------------------------------------------------------------------------
    # UserImage operations
    # -------------------------------------------------------------------------

    async def create_user_image(
        self,
        *,
        id: UUID,
        user_id: UUID,
        storage_key: str,
        original_filename: str,
        content_type: str,
        size_bytes: int,
        format: str,
        expires_at: datetime,
    ) -> UserImage:
        """Create a new user image record.

        Args:
            id: Unique image ID (matches R2 file ID).
            user_id: Owner of the image.
            storage_key: Full R2 storage key.
            original_filename: Original uploaded filename.
            content_type: MIME type.
            size_bytes: File size.
            format: Image format (png, jpeg, webp).
            expires_at: When the image should be cleaned up.

        Returns:
            Created UserImage instance.
        """
        image = UserImage(
            id=id,
            user_id=user_id,
            storage_key=storage_key,
            original_filename=original_filename,
            content_type=content_type,
            size_bytes=size_bytes,
            format=format,
            expires_at=expires_at,
        )
        self._session.add(image)
        await self._session.flush()
        return image

    async def get_user_image(self, image_id: UUID) -> UserImage | None:
        """Get a user image by ID.

        Args:
            image_id: Image ID to look up.

        Returns:
            UserImage if found, None otherwise.
        """
        return await self._session.get(UserImage, image_id)

    async def get_user_image_by_key(self, storage_key: str) -> UserImage | None:
        """Get a user image by storage key.

        Args:
            storage_key: R2 storage key.

        Returns:
            UserImage if found, None otherwise.
        """
        result = await self._session.execute(
            select(UserImage).where(UserImage.storage_key == storage_key)
        )
        return result.scalar_one_or_none()

    async def list_user_images(
        self,
        user_id: UUID,
        *,
        limit: int = 100,
        offset: int = 0,
    ) -> Sequence[UserImage]:
        """List images for a user.

        Args:
            user_id: User to list images for.
            limit: Maximum results to return.
            offset: Number of results to skip.

        Returns:
            List of UserImage instances.
        """
        result = await self._session.execute(
            select(UserImage)
            .where(UserImage.user_id == user_id)
            .order_by(UserImage.created_at.desc())
            .limit(limit)
            .offset(offset)
        )
        return result.scalars().all()

    async def delete_user_image(self, image_id: UUID) -> bool:
        """Delete a user image record.

        Args:
            image_id: Image ID to delete.

        Returns:
            True if deleted, False if not found.
        """
        image = await self._session.get(UserImage, image_id)
        if image is None:
            return False

        await self._session.delete(image)
        # Flush to ensure the delete is issued; commit is handled by caller.
        await self._session.flush()
        return True

    async def get_expired_images(
        self,
        before: datetime | None = None,
        limit: int = 1000,
    ) -> Sequence[UserImage]:
        """Get images past their expiration date.

        Args:
            before: Consider expired if expires_at < before (default: now).
            limit: Maximum results to return.

        Returns:
            List of expired UserImage instances.
        """
        if before is None:
            before = datetime.now(timezone.utc)

        result = await self._session.execute(
            select(UserImage)
            .where(UserImage.expires_at < before)
            .order_by(UserImage.expires_at)
            .limit(limit)
        )
        return result.scalars().all()

    # -------------------------------------------------------------------------
    # GenerationJob operations
    # -------------------------------------------------------------------------
    # FIXME: Use generation_type and JobStatus enums instead of str where applicable.
    async def create_job(
        self,
        *,
        id: UUID,
        user_id: UUID,
        name: str,
        prompt: str,
        generation_type: str = "i2i",
        status: str = "pending",
    ) -> GenerationJob:
        """Create a new generation job.

        Args:
            id: Unique job ID.
            user_id: Owner of the job.
            name: Job name.
            prompt: Generation prompt.
            status: Initial status.

        Returns:
            Created GenerationJob instance.
        """
        job = GenerationJob(
            id=id,
            user_id=user_id,
            name=name,
            prompt=prompt,
            status=status,
            generation_type=generation_type,
        )
        self._session.add(job)
        await self._session.flush()
        return job

    async def get_job(self, job_id: UUID) -> GenerationJob | None:
        """Get a job by ID.

        Args:
            job_id: Job ID to look up.

        Returns:
            GenerationJob if found, None otherwise.
        """
        return await self._session.get(GenerationJob, job_id)

    async def update_job_status(
        self,
        job_id: UUID,
        status: str,
        *,
        comfyui_prompt_id: str | None = None,
        started_at: datetime | None = None,
        completed_at: datetime | None = None,
    ) -> GenerationJob | None:
        """Update job status and timestamps.

        Args:
            job_id: Job ID to update.
            status: New status.
            comfyui_prompt_id: ComfyUI prompt ID (optional).
            started_at: Job start time (optional).
            completed_at: Job completion time (optional).

        Returns:
            Updated GenerationJob if found, None otherwise.
        """
        from src.api.schemas import JobStatus

        job = await self.get_job(job_id)
        if job is None:
            return None

        job.status = JobStatus(status)
        if comfyui_prompt_id is not None:
            job.comfyui_prompt_id = comfyui_prompt_id
        if started_at is not None:
            job.started_at = started_at
        if completed_at is not None:
            job.completed_at = completed_at

        await self._session.flush()
        return job

    async def list_user_jobs(
        self,
        user_id: UUID,
        *,
        status: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> Sequence[GenerationJob]:
        """List jobs for a user.

        Args:
            user_id: User to list jobs for.
            status: Filter by status (optional).
            limit: Maximum results to return.
            offset: Number of results to skip.

        Returns:
            List of GenerationJob instances.
        """
        query = select(GenerationJob).where(GenerationJob.user_id == user_id)

        if status is not None:
            query = query.where(GenerationJob.status == status)

        result = await self._session.execute(
            query.order_by(GenerationJob.created_at.desc()).limit(limit).offset(offset)
        )
        return result.scalars().all()

    # -------------------------------------------------------------------------
    # GenerationOutput operations
    # -------------------------------------------------------------------------

    async def create_output(
        self,
        *,
        id: UUID,
        user_id: UUID,
        job_id: UUID,
        storage_key: str,
        content_type: str,
        size_bytes: int,
        format: str,
        output_index: int,
        expires_at: datetime,
        input_image_id: UUID | None = None,
    ) -> GenerationOutput:
        """Create a new generation output record.

        Args:
            id: Unique output ID (matches R2 file ID).
            user_id: Owner of the output.
            job_id: Associated generation job.
            storage_key: Full R2 storage key.
            content_type: MIME type.
            size_bytes: File size.
            format: Image format.
            output_index: Index in batch (0-based).
            expires_at: When the output should be cleaned up.
            input_image_id: Associated input image (for i2i).

        Returns:
            Created GenerationOutput instance.
        """
        output = GenerationOutput(
            id=id,
            user_id=user_id,
            job_id=job_id,
            storage_key=storage_key,
            content_type=content_type,
            size_bytes=size_bytes,
            format=format,
            output_index=output_index,
            expires_at=expires_at,
            input_image_id=input_image_id,
        )
        self._session.add(output)
        await self._session.flush()
        return output

    async def get_output(self, output_id: UUID) -> GenerationOutput | None:
        """Get an output by ID.

        Args:
            output_id: Output ID to look up.

        Returns:
            GenerationOutput if found, None otherwise.
        """
        return await self._session.get(GenerationOutput, output_id)

    async def list_job_outputs(self, job_id: UUID) -> Sequence[GenerationOutput]:
        """List outputs for a job.

        Args:
            job_id: Job to list outputs for.

        Returns:
            List of GenerationOutput instances ordered by index.
        """
        result = await self._session.execute(
            select(GenerationOutput)
            .where(GenerationOutput.job_id == job_id)
            .order_by(GenerationOutput.output_index)
        )
        return result.scalars().all()

    async def list_user_outputs(
        self,
        user_id: UUID,
        *,
        limit: int = 100,
        offset: int = 0,
    ) -> Sequence[GenerationOutput]:
        """List outputs for a user.

        Args:
            user_id: User to list outputs for.
            limit: Maximum results to return.
            offset: Number of results to skip.

        Returns:
            List of GenerationOutput instances.
        """
        result = await self._session.execute(
            select(GenerationOutput)
            .where(GenerationOutput.user_id == user_id)
            .order_by(GenerationOutput.created_at.desc())
            .limit(limit)
            .offset(offset)
        )
        return result.scalars().all()

    async def get_expired_outputs(
        self,
        before: datetime | None = None,
        limit: int = 1000,
    ) -> Sequence[GenerationOutput]:
        """Get outputs past their expiration date.

        Args:
            before: Consider expired if expires_at < before (default: now).
            limit: Maximum results to return.

        Returns:
            List of expired GenerationOutput instances.
        """
        if before is None:
            before = datetime.now(timezone.utc)

        result = await self._session.execute(
            select(GenerationOutput)
            .where(GenerationOutput.expires_at < before)
            .order_by(GenerationOutput.expires_at)
            .limit(limit)
        )
        return result.scalars().all()

    async def delete_outputs_batch(self, output_ids: list[UUID]) -> int:
        """Delete multiple output records.

        Args:
            output_ids: List of output IDs to delete.

        Returns:
            Number of records deleted.
        """
        if not output_ids:
            return 0

        # First count how many rows match, using typed SQLAlchemy APIs
        count_result = await self._session.execute(
            select(func.count(GenerationOutput.id)).where(GenerationOutput.id.in_(output_ids))
        )
        (count,) = count_result.one()

        # Then perform the actual delete
        await self._session.execute(
            delete(GenerationOutput).where(GenerationOutput.id.in_(output_ids))
        )

        return int(count)

    # -------------------------------------------------------------------------
    # Statistics
    # -------------------------------------------------------------------------

    async def get_user_storage_stats(
        self,
        user_id: UUID,
    ) -> dict[str, int]:
        """Get storage statistics for a user.

        Args:
            user_id: User to get stats for.

        Returns:
            Dict with upload_count, output_count, total_bytes.
        """
        # Count and sum uploads
        upload_result = await self._session.execute(
            select(
                func.count(UserImage.id),
                func.coalesce(func.sum(UserImage.size_bytes), 0),
            ).where(UserImage.user_id == user_id)
        )
        upload_count, upload_bytes = upload_result.one()

        # Count and sum outputs
        output_result = await self._session.execute(
            select(
                func.count(GenerationOutput.id),
                func.coalesce(func.sum(GenerationOutput.size_bytes), 0),
            ).where(GenerationOutput.user_id == user_id)
        )
        output_count, output_bytes = output_result.one()

        return {
            "upload_count": upload_count,
            "output_count": output_count,
            "total_bytes": upload_bytes + output_bytes,
        }
