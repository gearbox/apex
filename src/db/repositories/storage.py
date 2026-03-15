"""Repository for storage-related database operations."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, cast
from uuid import UUID

from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.enums import GenerationType, JobStatus, Provider
from src.db.models import GenerationJob, GenerationOutput, UserImage

if TYPE_CHECKING:
    from collections.abc import Sequence


class StorageRepository:
    """Repository for storage-related database operations.

    Provides data access methods for UserImage, GenerationJob, and
    GenerationOutput models. All methods are async and use the
    provided session for transaction management.

    Single-resource lookup methods accept an optional ``user_id``
    keyword argument. When provided, the query includes a compound
    WHERE clause so that ownership is enforced at the database level.
    When omitted (``None``), these methods fall back to a plain
    primary-key lookup — suitable for internal / system operations
    such as cleanup jobs and background polling.
    """

    def __init__(self, session: AsyncSession) -> None:
        """Initialize repository with database session.

        Args:
            session: Async SQLAlchemy session.
        """
        self._session = session

    # -------------------------------------------------------------------------
    # Private helpers
    # -------------------------------------------------------------------------

    async def _get_by_id_with_optional_owner(
        self,
        model: type[UserImage] | type[GenerationOutput] | type[GenerationJob],
        pk: UUID,
        *,
        user_id: UUID | None = None,
    ) -> UserImage | GenerationOutput | GenerationJob | None:
        """Fetch a single record by PK, optionally scoped to a user.

        When ``user_id`` is ``None`` this delegates to ``session.get()``
        which benefits from SQLAlchemy's identity map cache.  When
        ``user_id`` is provided a compound ``WHERE`` is issued instead
        so that ownership is checked by the database in a single round
        trip.

        Args:
            model: SQLAlchemy model class.
            pk: Primary key value.
            user_id: Optional owner filter.

        Returns:
            Model instance or ``None``.
        """
        if user_id is None:
            return cast(
                UserImage | GenerationOutput | GenerationJob | None,
                await self._session.get(model, pk),
            )

        result = await self._session.execute(
            select(model).where(model.id == pk, model.user_id == user_id)
        )
        return cast(
            UserImage | GenerationOutput | GenerationJob | None, result.scalar_one_or_none()
        )

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
        """Create a new user upload record.

        Args:
            id: Unique upload ID (matches R2 file ID).
            user_id: Owner of the upload.
            storage_key: Full R2 storage key.
            original_filename: Original uploaded filename.
            content_type: MIME type.
            size_bytes: File size.
            format: Image format (png, jpeg, webp).
            expires_at: When the upload should be cleaned up.

        Returns:
            Created UserImage instance.
        """
        upload = UserImage(
            id=id,
            user_id=user_id,
            storage_key=storage_key,
            original_filename=original_filename,
            content_type=content_type,
            size_bytes=size_bytes,
            format=format,
            expires_at=expires_at,
        )
        self._session.add(upload)
        await self._session.flush()
        return upload

    async def get_user_image(
        self,
        image_id: UUID,
        *,
        user_id: UUID | None = None,
    ) -> UserImage | None:
        """Get a user image by ID.

        Args:
            image_id: Image ID to look up.
            user_id: When provided, only returns the image if owned
                by this user. ``None`` skips the ownership check
                (for internal / system use).

        Returns:
            UserImage if found, None otherwise.
        """
        return await self._get_by_id_with_optional_owner(  # type: ignore[return-value]
            UserImage, image_id, user_id=user_id
        )

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

    async def count_user_images(self, user_id: UUID) -> int:
        """Count total uploaded images for a user.

        Args:
            user_id: User ID.

        Returns:
            Total image count.
        """
        result = await self._session.execute(
            select(func.count()).select_from(UserImage).where(UserImage.user_id == user_id)
        )
        return int(result.scalar_one())

    async def list_user_images(
        self,
        user_id: UUID,
        *,
        limit: int = 100,
        offset: int = 0,
        cursor_ts: datetime | None = None,
        cursor_id: UUID | None = None,
    ) -> Sequence[UserImage]:
        """List images for a user.

        Supports both offset-based and cursor-based (keyset) pagination.
        When ``cursor_ts`` and ``cursor_id`` are supplied, offset is ignored
        and keyset filtering is applied instead.

        Args:
            user_id: User to list images for.
            limit: Maximum results to return.
            offset: Number of results to skip (ignored when cursor supplied).
            cursor_ts: ``created_at`` of the last item on the previous page.
            cursor_id: ``id`` of the last item on the previous page.

        Returns:
            List of UserImage instances ordered by ``created_at DESC``.
        """
        from sqlalchemy import and_, or_

        query = select(UserImage).where(UserImage.user_id == user_id)

        if cursor_ts is not None and cursor_id is not None:
            query = query.where(
                or_(
                    UserImage.created_at < cursor_ts,
                    and_(UserImage.created_at == cursor_ts, UserImage.id < cursor_id),
                )
            )
            result = await self._session.execute(
                query.order_by(UserImage.created_at.desc(), UserImage.id.desc()).limit(limit)
            )
        else:
            result = await self._session.execute(
                query.order_by(UserImage.created_at.desc()).limit(limit).offset(offset)
            )
        return result.scalars().all()

    async def delete_user_image(
        self,
        image_id: UUID,
        *,
        user_id: UUID | None = None,
    ) -> bool:
        """Delete a user image record.

        Args:
            image_id: Image ID to delete.
            user_id: When provided, only deletes if owned by this user.

        Returns:
            True if deleted, False if not found.
        """
        image = await self.get_user_image(image_id, user_id=user_id)
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
            before = datetime.now(UTC)

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

    async def create_job(
        self,
        *,
        id: UUID,
        user_id: UUID,
        name: str,
        prompt: str,
        generation_type: GenerationType = GenerationType.I2I,
        status: JobStatus = JobStatus.PENDING,
        provider: Provider = Provider.AISHA,
        model: str | None = None,
        aspect_ratio: str | None = None,
    ) -> GenerationJob:
        """Create a new generation job.

        Args:
            id: Unique job ID.
            user_id: Owner of the job.
            name: Job name.
            prompt: Generation prompt.
            generation_type: Type of generation (t2i, i2i, t2v, i2v).
            status: Initial status.
            provider: Generation provider (aisha, grok).
            model: Model identifier.
            aspect_ratio: Aspect ratio string, e.g. ``16:9``.

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
            provider=provider,
            model=model,
            aspect_ratio=aspect_ratio,
        )
        self._session.add(job)
        await self._session.flush()
        return job

    async def get_job(
        self,
        job_id: UUID,
        *,
        user_id: UUID | None = None,
    ) -> GenerationJob | None:
        """Get a job by ID.

        Args:
            job_id: Job ID to look up.
            user_id: When provided, only returns if owned by this user.
                by this user. ``None`` skips the ownership check
                (for internal use, e.g. background polling).

        Returns:
            GenerationJob if found, None otherwise.
        """
        return await self._get_by_id_with_optional_owner(  # type: ignore[return-value]
            GenerationJob,
            job_id,
            user_id=user_id,
        )

    async def update_job_status(
        self,
        job_id: UUID,
        status: JobStatus,
        *,
        comfyui_prompt_id: str | None = None,
        external_request_id: str | None = None,
        started_at: datetime | None = None,
        completed_at: datetime | None = None,
    ) -> GenerationJob | None:
        """Update job status and timestamps.

        Args:
            job_id: Job ID to update.
            status: New status.
            comfyui_prompt_id: ComfyUI prompt ID (optional).
            external_request_id: External provider request ID.
            started_at: Job start time (optional).
            completed_at: Job completion time (optional).

        Returns:
            Updated GenerationJob if found, None otherwise.
        """
        job = await self.get_job(job_id)
        if job is None:
            return None

        job.status = status

        # TODO: deprecate comfyui_prompt_id
        # Handle both legacy and new parameter names
        request_id = external_request_id or comfyui_prompt_id
        if request_id is not None:
            job.external_request_id = request_id
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
        status: JobStatus | None = None,
        provider: Provider | None = None,
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

        if provider is not None:
            query = query.where(GenerationJob.provider == provider)

        result = await self._session.execute(
            query.order_by(GenerationJob.created_at.desc()).limit(limit).offset(offset)
        )
        return result.scalars().all()

    async def list_pending_video_jobs(
        self,
        provider: str = Provider.GROK.value,
    ) -> Sequence[GenerationJob]:
        """List pending video generation jobs for polling.

        Args:
            provider: Provider to filter by.

        Returns:
            List of GenerationJob instances needing polling.
        """
        video_types = ["t2v", "i2v"]
        pending_statuses = [JobStatus.QUEUED.value, JobStatus.RUNNING.value]

        result = await self._session.execute(
            select(GenerationJob)
            .where(GenerationJob.provider == provider)
            .where(GenerationJob.status.in_(pending_statuses))
            .where(GenerationJob.generation_type.in_(video_types))
            .where(GenerationJob.external_request_id.isnot(None))
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
        is_thumbnail: bool = False,
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
            is_thumbnail: Whether this output is a thumbnail (e.g. video poster frame).

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
            is_thumbnail=is_thumbnail,
        )
        self._session.add(output)
        await self._session.flush()
        return output

    async def get_output(
        self,
        output_id: UUID,
        *,
        user_id: UUID | None = None,
    ) -> GenerationOutput | None:
        """Get an output by ID.

        Args:
            output_id: Output ID to look up.
            user_id: When provided, only returns if owned by this user.

        Returns:
            GenerationOutput if found, None otherwise.
        """
        return await self._get_by_id_with_optional_owner(  # type: ignore[return-value]
            GenerationOutput,
            output_id,
            user_id=user_id,
        )

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

    async def count_user_outputs(self, user_id: UUID) -> int:
        """Count total generated outputs for a user.

        Args:
            user_id: User ID.

        Returns:
            Total output count.
        """
        result = await self._session.execute(
            select(func.count())
            .select_from(GenerationOutput)
            .where(GenerationOutput.user_id == user_id)
        )
        return int(result.scalar_one())

    async def list_user_outputs(
        self,
        user_id: UUID,
        *,
        limit: int = 100,
        offset: int = 0,
        cursor_ts: datetime | None = None,
        cursor_id: UUID | None = None,
    ) -> Sequence[GenerationOutput]:
        """List outputs for a user.

        Supports both offset-based and cursor-based (keyset) pagination.
        When ``cursor_ts`` and ``cursor_id`` are supplied, offset is ignored
        and keyset filtering is applied instead.

        Args:
            user_id: User to list outputs for.
            limit: Maximum results to return.
            offset: Number of results to skip (ignored when cursor supplied).
            cursor_ts: ``created_at`` of the last item on the previous page.
            cursor_id: ``id`` of the last item on the previous page.

        Returns:
            List of GenerationOutput instances ordered by ``created_at DESC``.
        """
        from sqlalchemy import and_, or_

        query = select(GenerationOutput).where(GenerationOutput.user_id == user_id)

        if cursor_ts is not None and cursor_id is not None:
            query = query.where(
                or_(
                    GenerationOutput.created_at < cursor_ts,
                    and_(
                        GenerationOutput.created_at == cursor_ts,
                        GenerationOutput.id < cursor_id,
                    ),
                )
            )
            result = await self._session.execute(
                query.order_by(
                    GenerationOutput.created_at.desc(), GenerationOutput.id.desc()
                ).limit(limit)
            )
        else:
            result = await self._session.execute(
                query.order_by(GenerationOutput.created_at.desc()).limit(limit).offset(offset)
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
            before = datetime.now(UTC)

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

    async def delete_output(
        self,
        output_id: UUID,
        *,
        user_id: UUID | None = None,
    ) -> bool:
        """Delete an output record.

        Args:
            output_id: Output ID to delete.
            user_id: When provided, only deletes if owned by this user.

        Returns:
            True if deleted, False if not found.
        """
        output = await self.get_output(output_id, user_id=user_id)
        if output is None:
            return False
        await self._session.delete(output)
        await self._session.flush()
        return True

    # -------------------------------------------------------------------------
    # Statistics
    # -------------------------------------------------------------------------

    async def get_user_storage_stats(self, user_id: UUID) -> dict[str, int]:
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
