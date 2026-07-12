"""Repository for FrameExtractionJob database operations."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from sqlalchemy import select

from src.core.enums import FrameExtractionStatus
from src.db.models.frame_extraction import FrameExtractionJob
from src.db.repositories.base import BaseRepository

if TYPE_CHECKING:
    from uuid import UUID

    from sqlalchemy.ext.asyncio import AsyncSession


class FrameExtractionJobRepository(BaseRepository[FrameExtractionJob]):
    """Data access layer for FrameExtractionJob records."""

    _model = FrameExtractionJob

    def __init__(self, session: AsyncSession) -> None:
        super().__init__(session)

    async def create(
        self,
        *,
        id: UUID,
        user_id: UUID,
        product_id: str,
        kind: str,
        params: dict[str, Any],
        source_output_id: UUID | None = None,
        source_upload_id: UUID | None = None,
    ) -> FrameExtractionJob:
        """Create a new queued frame extraction job.

        Exactly one of ``source_output_id`` / ``source_upload_id`` must be
        set — enforced by ``ck_frame_extraction_jobs_exactly_one_source``.
        """
        job = FrameExtractionJob(
            id=id,
            user_id=user_id,
            product_id=product_id,
            kind=kind,
            status=FrameExtractionStatus.QUEUED.value,
            source_output_id=source_output_id,
            source_upload_id=source_upload_id,
            params=params,
        )
        self._session.add(job)
        await self._session.flush()
        return job

    async def get(
        self,
        job_id: UUID,
        *,
        user_id: UUID | None = None,
    ) -> FrameExtractionJob | None:
        """Get a job by ID, optionally scoped to a user."""
        return await self._get_with_optional_owner(job_id, user_id=user_id)

    async def claim_next(self) -> FrameExtractionJob | None:
        """Claim the oldest queued job for processing.

        Uses ``SELECT ... FOR UPDATE SKIP LOCKED`` so concurrent worker
        instances never claim the same row. Caller must commit the
        transaction promptly (claim-then-commit, before doing any actual
        ffmpeg work) — see FrameExtractionWorker.

        Returns:
            The claimed job (status transitioned to RUNNING, in-memory and
            flushed but not committed), or None if no job is queued.
        """
        result = await self._session.execute(
            select(FrameExtractionJob)
            .where(FrameExtractionJob.status == FrameExtractionStatus.QUEUED.value)
            .order_by(FrameExtractionJob.created_at)
            .limit(1)
            .with_for_update(skip_locked=True)
        )
        job = result.scalar_one_or_none()
        if job is None:
            return None

        job.status = FrameExtractionStatus.RUNNING.value
        job.started_at = datetime.now(UTC)
        await self._session.flush()
        return job

    async def mark_completed(self, job_id: UUID, *, result: dict[str, Any]) -> None:
        """Mark a job completed with its result payload."""
        job = await self._session.get(FrameExtractionJob, job_id)
        if job is None:
            return
        job.status = FrameExtractionStatus.COMPLETED.value
        job.result = result
        job.finished_at = datetime.now(UTC)
        await self._session.flush()

    async def mark_failed(self, job_id: UUID, *, error: str) -> None:
        """Mark a job failed with an error message."""
        job = await self._session.get(FrameExtractionJob, job_id)
        if job is None:
            return
        job.status = FrameExtractionStatus.FAILED.value
        job.error = error
        job.finished_at = datetime.now(UTC)
        await self._session.flush()
