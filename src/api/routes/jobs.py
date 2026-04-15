"""Unified jobs API — cross-provider job history, detail, and deletion.

Replaces the legacy in-memory ``JobController`` in ``generation.py``.
All providers (Grok, Aisha) surface through this single controller.

Endpoints:
  GET  /v1/jobs           — paginated job list (filterable)
  GET  /v1/jobs/{job_id}  — single job detail with outputs + thumbnails
  DELETE /v1/jobs/{job_id} — soft-hide job from history
"""

from __future__ import annotations

from collections.abc import Sequence
from uuid import UUID

import structlog
from litestar import Controller, Response, delete, get
from litestar.di import Provide
from litestar.status_codes import (
    HTTP_200_OK,
    HTTP_204_NO_CONTENT,
    HTTP_404_NOT_FOUND,
)
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.dependencies.auth import get_current_user_id
from src.api.schemas.jobs import UnifiedJobResponse
from src.api.schemas.pagination import CursorPage
from src.api.security import auth_guard
from src.api.services.unified_jobs import UnifiedJobService
from src.core.enums import GenerationType, JobStatus, Provider

logger = structlog.get_logger(__name__)


class UnifiedJobController(Controller):
    """Cross-provider job history and status endpoints.

    All responses include full generation parameters and presigned output URLs
    so the frontend can render the gallery without extra API calls.
    """

    path = "/v1/jobs"
    tags: Sequence[str] | None = ["Jobs"]
    guards = [auth_guard]
    dependencies = {"current_user_id": Provide(get_current_user_id)}

    # -------------------------------------------------------------------------
    # GET /v1/jobs
    # -------------------------------------------------------------------------

    @get("/")
    async def list_jobs(
        self,
        current_user_id: UUID,
        session: AsyncSession,
        unified_job_service: UnifiedJobService,
        status: JobStatus | None = None,
        provider: Provider | None = None,
        generation_type: GenerationType | None = None,
        limit: int = 20,
        cursor: str | None = None,
    ) -> CursorPage[UnifiedJobResponse]:
        """List generation jobs for the authenticated user.

        Supports filtering by status, provider, and generation type.
        Results are ordered by creation date (newest first).

        Query parameters:
          - ``status``: Filter by job status (pending, queued, running, completed, failed)
          - ``provider``: Filter by provider (grok, aisha)
          - ``generation_type``: Filter by type (t2i, i2i, t2v, i2v, v2v)
          - ``limit``: Page size (default 20, max 100)
          - ``cursor``: Opaque cursor from a previous response's ``next_cursor``
            field.  Pass to fetch the next page.
        """
        return await unified_job_service.list_jobs(
            current_user_id,
            session=session,
            status=status,
            provider=provider,
            generation_type=generation_type,
            limit=limit,
            cursor=cursor,
        )

    # -------------------------------------------------------------------------
    # GET /v1/jobs/{job_id}
    # -------------------------------------------------------------------------

    @get("/{job_id:uuid}")
    async def get_job(
        self,
        current_user_id: UUID,
        job_id: UUID,
        session: AsyncSession,
        unified_job_service: UnifiedJobService,
    ) -> Response[UnifiedJobResponse]:
        """Get a single job by ID.

        For in-progress Grok video jobs, this polls xAI for the latest status
        before responding (poll-on-read pattern — no separate background worker
        required for MVP).

        Returns 404 if the job does not exist or is not owned by the caller.
        """
        job = await unified_job_service.get_job(job_id, current_user_id, session=session)

        if job is None:
            return Response(
                content=None,  # type: ignore[arg-type]
                status_code=HTTP_404_NOT_FOUND,
            )

        return Response(content=job, status_code=HTTP_200_OK)

    # -------------------------------------------------------------------------
    # DELETE /v1/jobs/{job_id}
    # -------------------------------------------------------------------------

    @delete("/{job_id:uuid}", status_code=HTTP_204_NO_CONTENT)
    async def delete_job(
        self,
        current_user_id: UUID,
        job_id: UUID,
        session: AsyncSession,
    ) -> None:
        """Soft-hide a job from the user's history.

        The job record and R2 outputs are retained until the normal retention
        policy cleans them up. The job simply stops appearing in list results.

        Returns 404 if the job does not exist or is not owned by the caller.

        NOTE: Full hard-delete with R2 cleanup is a future enhancement.
              For MVP, the retention-based cleanup is sufficient.
        """
        from src.db.repositories.job import JobRepository

        job = await JobRepository(session).get(job_id, user_id=current_user_id)

        if job is None:
            # Litestar 204 handler — raise NotFoundException for proper 404
            from litestar.exceptions import NotFoundException

            raise NotFoundException(detail=f"Job {job_id} not found")

        # Soft-hide: mark as hidden by setting a sentinel status
        # For MVP, we repurpose the error_message field as a hidden flag
        # and filter it in list queries.  Migration 007 should add a proper
        # ``is_hidden`` bool column.
        job.error_message = "__hidden__"
        await session.commit()

        logger.info("job.hidden", job_id=str(job_id), user_id=str(current_user_id))
