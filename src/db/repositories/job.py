"""Repository for generation job database operations."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.enums import GenerationType, JobStatus, Provider
from src.db.models import GenerationJob

if TYPE_CHECKING:
    from collections.abc import Sequence
    from datetime import datetime


class JobRepository:
    """Data access layer for GenerationJob records.

    Single-resource lookups accept an optional ``user_id`` kwarg.
    When provided the query includes a compound WHERE so ownership
    is enforced at the database level. When omitted (``None``),
    a plain primary-key lookup is used — suitable for internal /
    system operations such as background polling.
    """

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

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

        Args:
            job_id: Job ID to look up.
            user_id: When provided, only returns if owned by this user.
                ``None`` skips the ownership check (for internal use).

        Returns:
            GenerationJob if found, None otherwise.
        """
        if user_id is None:
            return cast(
                GenerationJob | None,
                await self._session.get(GenerationJob, job_id),
            )
        result = await self._session.execute(
            select(GenerationJob).where(
                GenerationJob.id == job_id,
                GenerationJob.user_id == user_id,
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
        limit: int = 50,
        offset: int = 0,
    ) -> Sequence[GenerationJob]:
        """List jobs for a user with optional filters.

        Args:
            user_id: User to list jobs for.
            status: Filter by status (optional).
            provider: Filter by provider (optional).
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
