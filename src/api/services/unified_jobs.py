"""Unified job service — cross-provider job history and detail.

Builds ``UnifiedJobResponse`` objects from ``GenerationJob`` DB records
by fetching outputs and generating presigned R2 URLs.

Replaces the in-memory ``JobManager`` for the user-facing jobs API.
``JobManager`` / ``GrokJobService`` remain as *execution* services;
this service is read-oriented (history, gallery, status polling).
"""

from __future__ import annotations

from uuid import UUID

import structlog
from sqlalchemy import literal, select, tuple_
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from src.api.schemas.jobs import JobOutputItem, UnifiedJobResponse
from src.api.schemas.pagination import CursorPage, decode_cursor, encode_cursor
from src.api.services.aisha_job_service import AishaJobService
from src.api.services.grok.job_service import GrokJobService
from src.api.services.storage import R2StorageService
from src.core.enums import GenerationType, JobStatus, Provider
from src.db.models.storage import GenerationJob
from src.db.repositories.storage import StorageRepository

logger = structlog.get_logger(__name__)

# Presigned URL TTL in seconds (1 hour)
_URL_TTL = 3600


class UnifiedJobService:
    """Read-side service for the cross-provider jobs API.

    Args:
        storage: R2 storage service for presigned URL generation.
        grok_job_service: Grok execution service used to poll async video jobs.
            Pass ``None`` when Grok is not configured.
    """

    def __init__(
        self,
        *,
        storage: R2StorageService | None,
        grok_job_service: GrokJobService | None,
        aisha_job_service: AishaJobService | None = None,
    ) -> None:
        self._storage = storage
        self._grok = grok_job_service
        self._aisha = aisha_job_service

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    async def get_job(
        self,
        job_id: UUID,
        user_id: UUID,
        *,
        session: AsyncSession,
    ) -> UnifiedJobResponse | None:
        """Get a single job by ID, scoped to the authenticated user.

        For queued/running Grok video jobs, this triggers a poll to xAI so
        the returned status is always fresh (poll-on-read pattern).

        Args:
            job_id: Job to fetch.
            user_id: Owner filter (enforced at DB level).
            session: DB session.

        Returns:
            ``UnifiedJobResponse`` or ``None`` if not found / not owned.
        """
        repo = StorageRepository(session)
        job = await repo.get_job(job_id, user_id=user_id)
        if job is None:
            return None

        # Poll Grok async video jobs on read — keeps status fresh without a
        # background worker being required for MVP.
        if (
            self._grok is not None
            and job.provider == Provider.GROK.value
            and job.generation_type in ("t2v", "i2v", "v2v")
            and job.status in (JobStatus.QUEUED.value, JobStatus.RUNNING.value)
        ):
            try:
                updated = await self._grok.poll_video_job(session, job_id)
                if updated is not None:
                    job = updated
            except Exception:
                logger.exception("unified_jobs.poll_on_read_failed", job_id=str(job_id))

        # Poll Aisha image jobs on read — downloads outputs and transitions status.
        if (
            self._aisha is not None
            and job.provider == Provider.AISHA.value
            and job.status in (JobStatus.QUEUED.value, JobStatus.RUNNING.value)
        ):
            try:
                updated = await self._aisha.poll_image_job(session, job_id)
                if updated is not None:
                    job = updated
            except Exception:
                logger.exception("unified_jobs.aisha_poll_on_read_failed", job_id=str(job_id))

        return await self._build_response(job, session=session)

    async def list_jobs(
        self,
        user_id: UUID,
        *,
        session: AsyncSession,
        status: JobStatus | None = None,
        provider: str | None = None,
        generation_type: GenerationType | None = None,
        limit: int = 20,
        cursor: str | None = None,
    ) -> CursorPage[UnifiedJobResponse]:
        """List jobs for a user with optional filters and cursor pagination.

        Args:
            user_id: Owner.
            session: DB session.
            status: Optional status filter.
            provider: Optional provider filter (``grok``, ``aisha``).
            generation_type: Optional type filter.
            limit: Page size (max 100).
            cursor: Opaque cursor token from a previous response's
                ``next_cursor`` field.

        Returns:
            Cursor-paginated job list.
        """
        limit = min(limit, 100)

        cursor_ts = None
        cursor_id = None
        if cursor is not None:
            cursor_ts, cursor_id = decode_cursor(cursor)

        data_q = (
            select(GenerationJob)
            .where(GenerationJob.user_id == user_id)
            .options(selectinload(GenerationJob.outputs))
        )

        if status is not None:
            data_q = data_q.where(GenerationJob.status == status.value)
        if provider is not None:
            data_q = data_q.where(GenerationJob.provider == provider)
        if generation_type is not None:
            data_q = data_q.where(GenerationJob.generation_type == generation_type.value)

        if cursor_ts is not None and cursor_id is not None:
            data_q = data_q.where(
                tuple_(GenerationJob.created_at, GenerationJob.id)
                < tuple_(literal(cursor_ts), literal(cursor_id))
            )

        jobs_result = await session.execute(
            data_q.order_by(GenerationJob.created_at.desc(), GenerationJob.id.desc()).limit(
                limit + 1
            )
        )
        jobs = list(jobs_result.scalars().all())

        has_more = len(jobs) > limit
        if has_more:
            jobs = jobs[:limit]

        items = []
        for job in jobs:
            items.append(await self._build_response(job, session=session))

        next_cursor: str | None = None
        if has_more and jobs:
            last = jobs[-1]
            next_cursor = encode_cursor(last.created_at, last.id)

        return CursorPage(
            items=items,
            limit=limit,
            has_more=has_more,
            next_cursor=next_cursor,
        )

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------

    async def _build_response(
        self,
        job: GenerationJob,
        *,
        session: AsyncSession,
    ) -> UnifiedJobResponse:
        """Build a full ``UnifiedJobResponse`` for a DB job record.

        Fetches outputs and generates presigned URLs.
        """
        repo = StorageRepository(session)
        db_outputs = await repo.list_job_outputs(job.id)

        output_items: list[JobOutputItem] = []
        thumbnail_url: str | None = None

        for out in db_outputs:
            try:
                if self._storage is None:
                    logger.warning("unified_jobs.presigned_url_skipped", output_id=str(out.id))
                    continue
                url_result = await self._storage.get_presigned_url(
                    out.storage_key, expires_in=_URL_TTL
                )
                presigned = url_result.presigned_url
            except Exception:
                logger.warning("unified_jobs.presigned_url_failed", output_id=str(out.id))
                continue

            item = JobOutputItem(
                id=out.id,
                url=presigned,
                content_type=out.content_type,
                format=out.format,
                size_bytes=out.size_bytes,
                output_index=out.output_index,
                is_thumbnail=getattr(out, "is_thumbnail", False),
            )
            output_items.append(item)

            if item.is_thumbnail and thumbnail_url is None:
                thumbnail_url = presigned

        # For image jobs without an explicit thumbnail flag, use first output
        if thumbnail_url is None and output_items:
            first = output_items[0]
            if "image" in first.content_type:
                thumbnail_url = first.url

        return UnifiedJobResponse(
            id=job.id,
            name=job.name,
            status=JobStatus(job.status),
            provider=job.provider,
            model=job.model,
            generation_type=GenerationType(job.generation_type),
            prompt=job.prompt,
            negative_prompt=job.negative_prompt,
            aspect_ratio=job.aspect_ratio,
            token_cost=job.token_cost,
            created_at=job.created_at,
            started_at=job.started_at,
            completed_at=job.completed_at,
            outputs=output_items,
            thumbnail_url=thumbnail_url,
            error=job.error_message,
        )
