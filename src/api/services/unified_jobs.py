"""Unified job service — cross-provider job history and detail.

Builds ``UnifiedJobResponse`` objects from ``GenerationJob`` DB records.
Output URLs are stable content-proxy paths — no presigned URL generation.

GET /v1/jobs/{id} is DB-only: no ComfyUI calls in the request path.
Aisha job status is updated by AishaJobPoller (background worker).
"""

from __future__ import annotations

from collections import defaultdict
from uuid import UUID

import structlog
from sqlalchemy import inspect
from sqlalchemy.exc import NoInspectionAvailable
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.schemas.jobs import JobOutputItem, UnifiedJobResponse
from src.api.schemas.pagination import CursorPage, decode_cursor, encode_cursor
from src.api.services.generation.base import GenerationProvider
from src.api.services.media import build_output_media
from src.core.enums import GenerationType, JobStatus, Provider
from src.db.models.storage import GenerationJob, GenerationOutput
from src.db.repositories.job import JobRepository
from src.db.repositories.output import OutputRepository

logger = structlog.get_logger(__name__)


class UnifiedJobService:
    """Read-side service for the cross-provider jobs API.

    Args:
        providers: Provider registry (same mapping wired into
            ``GenerationService``), used to poll-on-read fresh status for
            providers with an async backend (e.g. Grok video).
    """

    def __init__(
        self,
        *,
        providers: dict[Provider, GenerationProvider],
    ) -> None:
        self._providers = providers

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

        Delegates to the resolved provider's ``refresh_job`` hook — a no-op
        for providers without an async backend (Aisha; updated instead by the
        background AishaJobPoller), a poll to xAI for queued/running Grok
        video jobs. This keeps status fresh without a background worker
        being required for MVP, while staying DB-only for providers that
        don't need it.

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

        try:
            provider_key = Provider(job.provider)
        except ValueError:
            logger.warning(
                "unified_jobs.unknown_provider", job_id=str(job_id), provider=job.provider
            )
            provider_key = None
        provider = self._providers.get(provider_key) if provider_key is not None else None
        if provider is not None:
            try:
                updated = await provider.refresh_job(session, job)
                if updated is not None:
                    job = updated
            except Exception:
                # Serve the stale row on poll failure — same isolation for
                # every provider's refresh_job, not just Grok's.
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
        then assembles ``UnifiedJobResponse`` DTOs with stable proxy-path output URLs.

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

    async def _build_response(
        self,
        job: GenerationJob,
        *,
        session: AsyncSession,
    ) -> UnifiedJobResponse:
        """Build a full ``UnifiedJobResponse`` for a DB job record.

        Uses eagerly-loaded ``job.outputs`` when available (list path); falls
        back to two repository queries (single-job path): one for full outputs
        and one batch for their derivatives.
        """
        output_repo = OutputRepository(session)
        derivatives_map: dict[UUID, list[GenerationOutput]] = defaultdict(list)

        try:
            _outputs_loaded = "outputs" in inspect(job).dict
        except NoInspectionAvailable:
            _outputs_loaded = False

        if _outputs_loaded:
            # Eagerly loaded — separate full vs. thumbnails in Python
            all_outputs = list(job.outputs)
            full_outputs = sorted(
                [o for o in all_outputs if not o.is_thumbnail],
                key=lambda o: o.output_index,
            )
            for o in all_outputs:
                if o.is_thumbnail and o.parent_output_id is not None:
                    derivatives_map[o.parent_output_id].append(o)
        else:
            # Single-job path — list_by_job returns full outputs only
            full_outputs = list(await output_repo.list_by_job(job.id))
            full_ids = [o.id for o in full_outputs]
            raw_map = await output_repo.batch_derivatives(full_ids)
            for k, v in raw_map.items():
                derivatives_map[k] = v

        output_items: list[JobOutputItem] = [
            JobOutputItem(
                id=out.id,
                output_index=out.output_index,
                media=build_output_media(out, list(derivatives_map.get(out.id, []))),
            )
            for out in full_outputs
        ]

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
            error=job.error_message,
        )
