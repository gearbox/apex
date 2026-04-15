"""Repository for generation job database operations."""

from __future__ import annotations

from typing import TYPE_CHECKING

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from src.core.enums import GenerationType, JobStatus, Provider
from src.db.models.storage import GenerationJob
from src.db.repositories.base import BaseRepository

if TYPE_CHECKING:
    from collections.abc import Sequence
    from datetime import datetime
    from uuid import UUID


class JobRepository(BaseRepository[GenerationJob]):
    """Data access layer for GenerationJob records.

    Single-resource lookups accept an optional ``user_id`` kwarg.
    When provided the query includes a compound WHERE so ownership
    is enforced at the database level. When omitted (``None``),
    a plain primary-key lookup is used — suitable for internal /
    system operations such as background polling.
    """

    _model = GenerationJob

    def __init__(self, session: AsyncSession) -> None:
        super().__init__(session)

    async def create(
        self,
        *,
        id: UUID,
        user_id: UUID,
        name: str,
        prompt: str,
        product_id: str,
        generation_type: GenerationType = GenerationType.I2I,
        status: JobStatus = JobStatus.PENDING,
        provider: Provider = Provider.AISHA,
        model: str | None = None,
        aspect_ratio: str | None = None,
        source_job_id: UUID | None = None,
        source_output_id: UUID | None = None,
        input_image_id: UUID | None = None,
    ) -> GenerationJob:
        """Create a new generation job.

        Args:
            id: Unique job ID.
            user_id: Owner of the job.
            name: Job name.
            prompt: Generation prompt.
            product_id: Product this job belongs to.
            generation_type: Type of generation (t2i, i2i, t2v, i2v).
            status: Initial status.
            provider: Generation provider (aisha, grok).
            model: Model identifier.
            aspect_ratio: Aspect ratio string, e.g. ``16:9``.
            source_job_id: ID of the source job for lineage tracking.
            source_output_id: ID of the source output used as input.
            input_image_id: ID of the uploaded image used as input.

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
            product_id=product_id,
            source_job_id=source_job_id,
            source_output_id=source_output_id,
            input_image_id=input_image_id,
        )
        self._session.add(job)
        await self._session.flush()
        return job

    async def get(
        self,
        job_id: UUID,
        *,
        user_id: UUID | None = None,
    ) -> GenerationJob | None:
        """Get a job by ID, optionally scoped to a user.

        When ``user_id`` is provided, soft-deleted jobs are excluded
        (user-facing). When ``user_id`` is ``None``, all jobs are
        returned including soft-deleted (internal/system use).

        Args:
            job_id: Job ID to look up.
            user_id: When provided, only returns if owned by this user
                and not soft-deleted. ``None`` skips both checks
                (for internal use).

        Returns:
            GenerationJob if found, None otherwise.
        """
        if user_id is None:
            return await self._session.get(GenerationJob, job_id)

        result = await self._session.execute(
            select(GenerationJob).where(
                GenerationJob.id == job_id,
                GenerationJob.user_id == user_id,
                GenerationJob.is_deleted.is_(False),
            )
        )
        return result.scalar_one_or_none()

    async def update_status(
        self,
        job_id: UUID,
        status: JobStatus,
        *,
        external_request_id: str | None = None,
        started_at: datetime | None = None,
        completed_at: datetime | None = None,
    ) -> GenerationJob | None:
        """Update job status and timestamps.

        Args:
            job_id: Job ID to update.
            status: New status.
            external_request_id: External provider request ID.
            started_at: Job start time (optional).
            completed_at: Job completion time (optional).

        Returns:
            Updated GenerationJob if found, None otherwise.
        """
        job = await self.get(job_id)
        if job is None:
            return None

        job.status = status
        if external_request_id is not None:
            job.external_request_id = external_request_id
        if started_at is not None:
            job.started_at = started_at
        if completed_at is not None:
            job.completed_at = completed_at

        await self._session.flush()
        return job

    async def list_by_user(
        self,
        user_id: UUID,
        *,
        status: JobStatus | None = None,
        provider: Provider | None = None,
        generation_type: GenerationType | None = None,
        limit: int = 20,
        cursor_ts: datetime | None = None,
        cursor_id: UUID | None = None,
        eager_load_outputs: bool = False,
    ) -> Sequence[GenerationJob]:
        """List jobs for a user with cursor-based pagination and optional filters.

        Uses limit+1 fetch pattern. Caller checks ``len(result) > limit``
        to determine ``has_more``.

        Args:
            user_id: User to list jobs for.
            status: Filter by job status (optional).
            provider: Filter by provider (optional).
            generation_type: Filter by generation type (optional).
            limit: Page size (fetches limit+1 for has_more check).
            cursor_ts: ``created_at`` of the last item on the previous page.
            cursor_id: ``id`` of the last item on the previous page.
            eager_load_outputs: When True, eagerly loads the ``outputs``
                relationship via ``selectinload`` to avoid N+1 queries
                when building response DTOs.

        Returns:
            List of GenerationJob instances ordered by
            ``(created_at DESC, id DESC)``.
        """
        from sqlalchemy import literal, tuple_

        query = select(GenerationJob).where(GenerationJob.user_id == user_id)

        # Exclude soft-deleted jobs from user-facing listings
        query = query.where(GenerationJob.is_deleted.is_(False))

        if status is not None:
            query = query.where(GenerationJob.status == status)
        if provider is not None:
            query = query.where(GenerationJob.provider == provider)
        if generation_type is not None:
            query = query.where(GenerationJob.generation_type == generation_type)

        if eager_load_outputs:
            query = query.options(selectinload(GenerationJob.outputs))

        if cursor_ts is not None and cursor_id is not None:
            query = query.where(
                tuple_(GenerationJob.created_at, GenerationJob.id)
                < tuple_(literal(cursor_ts), literal(cursor_id))
            )

        result = await self._session.execute(
            query.order_by(
                GenerationJob.created_at.desc(),
                GenerationJob.id.desc(),
            ).limit(limit + 1)
        )
        return result.scalars().all()

    async def soft_delete(
        self,
        job_id: UUID,
        *,
        user_id: UUID,
    ) -> GenerationJob | None:
        """Soft-delete a job — marks it as deleted without removing the record.

        The job record and R2 outputs are retained until the retention
        policy cleans them up. The job stops appearing in user-facing
        list/get results.

        Uses a direct query that bypasses the is_deleted filter so that
        soft-deleting an already-deleted job is idempotent.

        Args:
            job_id: Job to soft-delete.
            user_id: Owner check — only the owner can delete their own job.

        Returns:
            Updated GenerationJob if found and owned, None otherwise.
        """
        result = await self._session.execute(
            select(GenerationJob).where(
                GenerationJob.id == job_id,
                GenerationJob.user_id == user_id,
            )
        )
        job = result.scalar_one_or_none()
        if job is None:
            return None

        job.is_deleted = True
        await self._session.flush()
        return job

    async def list_pending_video_jobs(
        self,
        provider: Provider = Provider.GROK,
    ) -> Sequence[GenerationJob]:
        """List pending video generation jobs for polling.

        Returns jobs that are queued or running, have a video generation
        type, and have an ``external_request_id`` set (indicating the
        provider accepted the request).

        Args:
            provider: Provider enum to filter by.

        Returns:
            List of GenerationJob instances needing polling.
        """
        video_types = [GenerationType.T2V, GenerationType.I2V]
        pending_statuses = [JobStatus.QUEUED, JobStatus.RUNNING]

        result = await self._session.execute(
            select(GenerationJob)
            .where(GenerationJob.provider == provider)
            .where(GenerationJob.status.in_(pending_statuses))
            .where(GenerationJob.generation_type.in_(video_types))
            .where(GenerationJob.external_request_id.isnot(None))
        )
        return result.scalars().all()
