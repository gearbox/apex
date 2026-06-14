"""Unified job service — cross-provider job history and detail.

Builds ``UnifiedJobResponse`` objects from ``GenerationJob`` DB records
by fetching outputs and generating presigned R2 URLs.

GET /v1/jobs/{id} is DB-only: no ComfyUI calls in the request path.
Aisha job status is updated by AishaJobPoller (background worker).
"""

from __future__ import annotations

from uuid import UUID

import structlog
from sqlalchemy import inspect
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.schemas.jobs import JobOutputItem, UnifiedJobResponse
from src.api.schemas.pagination import CursorPage, decode_cursor, encode_cursor
from src.api.services.grok.job_service import GrokJobService
from src.api.services.storage import R2StorageService
from src.core.enums import GenerationType, JobStatus, Provider
from src.db.models.storage import GenerationJob, GenerationOutput
from src.db.repositories.job import JobRepository
from src.db.repositories.output import OutputRepository

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
    ) -> None:
        self._storage = storage
        self._grok = grok_job_service

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

        Aisha jobs are updated by the background AishaJobPoller; this method
        reads from the DB only — no ComfyUI calls in the request path.

        Args:
            job_id: Job to fetch.
            user_id: Owner filter (enforced at DB level).
            session: DB session.

        Returns:
            ``UnifiedJobResponse`` or ``None`` if not found / not owned.
        """
        job_repo = JobRepository(session)
        job = await job_repo.get(job_id, user_id=user_id)
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

        return await self._build_response(job, session=session)

    async def list_jobs(
        self,
        user_id: UUID,
        *,
        session: AsyncSession,
        status: JobStatus | None = None,
        provider: Provider | None = None,
        generation_type: GenerationType | None = None,
        limit: int = 20,
        cursor: str | None = None,
    ) -> CursorPage[UnifiedJobResponse]:
        """List jobs for a user with optional filters and cursor pagination.

        Delegates filtering and cursor pagination to ``JobRepository.list_by_user``,
        then assembles ``UnifiedJobResponse`` DTOs with presigned output URLs.

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

        job_repo = JobRepository(session)
        jobs = list(
            await job_repo.list_by_user(
                user_id,
                status=status,
                provider=provider,
                generation_type=generation_type,
                limit=limit,
                cursor_ts=cursor_ts,
                cursor_id=cursor_id,
                eager_load_outputs=True,
            )
        )

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

    async def _get_job_outputs(
        self,
        job: GenerationJob,
        session: AsyncSession,
    ) -> list[GenerationOutput]:
        """Resolve outputs for a job, preferring the eagerly-loaded relationship.

        When the ``outputs`` relationship was populated by ``selectinload``
        (list path), returns them sorted by ``output_index`` with no extra query.
        Otherwise falls back to a repository query (single-job path).

        Args:
            job: GenerationJob instance.
            session: DB session for the fallback query.

        Returns:
            Outputs sorted by ``output_index``.
        """
        if "outputs" in inspect(job).dict:
            return sorted(job.outputs, key=lambda o: o.output_index)
        output_repo = OutputRepository(session)
        return list(await output_repo.list_by_job(job.id))

    async def _build_response(
        self,
        job: GenerationJob,
        *,
        session: AsyncSession,
    ) -> UnifiedJobResponse:
        """Build a full ``UnifiedJobResponse`` for a DB job record.

        Uses eagerly-loaded ``job.outputs`` when available, falls back
        to a repo query when outputs aren't preloaded.
        """
        db_outputs = await self._get_job_outputs(job, session)

        # Separate primary outputs from thumbnails; build presigned URL map.
        primary_outputs = [o for o in db_outputs if not getattr(o, "is_thumbnail", False)]
        thumbnail_outputs = [o for o in db_outputs if getattr(o, "is_thumbnail", False)]
        # thumbnail keyed by parent_output_id for O(1) lookup
        thumb_by_parent: dict[UUID, GenerationOutput] = {}
        for _t in thumbnail_outputs:
            _pid = getattr(_t, "parent_output_id", None)
            if _pid is not None:
                thumb_by_parent[_pid] = _t

        output_items: list[JobOutputItem] = []
        thumbnail_url: str | None = None

        # Pre-sign thumbnail URLs so we can attach them per-output
        thumb_presigned: dict[UUID, str] = {}
        for thumb in thumbnail_outputs:
            try:
                if self._storage is None:
                    continue
                r = await self._storage.get_presigned_url(thumb.storage_key, expires_in=_URL_TTL)
                thumb_presigned[thumb.id] = r.presigned_url
            except Exception:
                logger.warning("unified_jobs.presigned_url_failed", output_id=str(thumb.id))

        for out in primary_outputs:
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

            # Attach this output's thumbnail URL if available
            thumb_obj = thumb_by_parent.get(out.id)
            out_thumb_url = thumb_presigned.get(thumb_obj.id) if thumb_obj else None

            item = JobOutputItem(
                id=out.id,
                url=presigned,
                content_type=out.content_type,
                format=out.format,
                size_bytes=out.size_bytes,
                output_index=out.output_index,
                thumbnail_url=out_thumb_url,
                is_thumbnail=False,
            )
            output_items.append(item)

            # Top-level thumbnail_url = cover output's thumbnail (first one found)
            if thumbnail_url is None and out_thumb_url is not None:
                thumbnail_url = out_thumb_url

        # Legacy fallback: jobs without thumbnails use first image URL
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
