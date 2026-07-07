"""Background worker for polling Grok video generation jobs.

Session-per-job + semaphore fan-out (mirrors AishaJobPoller): one short
session lists candidate jobs, then each job is polled in its own session.
All state transitions go through JobStateTransitionService — this worker
never mutates job.status directly and never calls BillingService itself;
refunds are settled inside the transition.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import structlog
from sqlalchemy import select

from src.api.schemas.events import EventType, JobProgressPayload
from src.api.services.job_state_transition import JobStateTransitionService
from src.core.enums import GenerationType, JobStatus, VideoPollStatus
from src.db.models import GenerationJob
from src.workers.base import PeriodicWorker

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from src.api.services.billing import BillingService
    from src.api.services.event_bus import EventBus
    from src.api.services.grok.job_service import GrokJobService
    from src.core.config import Settings
    from src.db import DatabaseManager

logger = structlog.get_logger(__name__)

_VIDEO_GENERATION_TYPES = (
    GenerationType.T2V,
    GenerationType.I2V,
    GenerationType.V2V,
    GenerationType.FLF2V,
)
_PENDING_STATUSES = (JobStatus.QUEUED, JobStatus.RUNNING)


class GrokVideoWorker(PeriodicWorker):
    """Polls xAI for in-progress Grok video jobs and drives transitions.

    Periodically checks for jobs in QUEUED or RUNNING status with video
    generation types and polls the xAI API for results.
    """

    def __init__(
        self,
        db_manager: DatabaseManager,
        job_service: GrokJobService,
        billing_service: BillingService,
        settings: Settings,
        event_bus: EventBus | None = None,
        *,
        redis_enabled: bool = False,
    ) -> None:
        """Initialize the video worker.

        Args:
            db_manager: Database manager for sessions.
            job_service: Grok job service for polling.
            billing_service: Billing service, needed by JobStateTransitionService
                for timeout/failure refunds.
            settings: Application settings.
            event_bus: Optional event bus for publishing job events.
            redis_enabled: Whether Redis is configured (enables the leader lease).
        """
        super().__init__(
            name="grok_video_poller",
            interval_seconds=settings.grok_video_poll_interval,
            redis_enabled=redis_enabled,
        )
        self._db_manager = db_manager
        self._job_service = job_service
        self._billing = billing_service
        self._max_poll_time = settings.grok_video_max_poll_time
        self._event_bus = event_bus
        self._max_concurrent_polls = settings.grok_video_max_concurrent_polls

    def _make_transition_service(self, session: AsyncSession) -> JobStateTransitionService:
        return JobStateTransitionService(
            session=session,
            event_bus=self._event_bus,
            billing_service=self._billing,
        )

    async def _emit_progress(self, job: GenerationJob, progress_pct: int) -> None:
        if not self._event_bus:
            return
        await self._event_bus.publish(
            user_id=job.user_id,
            event_type=EventType.JOB_PROGRESS,
            payload=JobProgressPayload(
                job_id=job.id,
                progress_pct=progress_pct,
                generation_type=job.generation_type,
            ),
        )

    async def run_once(self) -> None:
        """Fetch candidate jobs in one session, then poll each in its own session."""
        async with self._db_manager.session() as session:
            result = await session.execute(
                select(GenerationJob)
                .where(GenerationJob.status.in_(_PENDING_STATUSES))
                .where(GenerationJob.generation_type.in_(_VIDEO_GENERATION_TYPES))
                .where(GenerationJob.external_request_id.isnot(None))
            )
            jobs = list(result.scalars().all())

        if not jobs:
            return

        logger.debug("grok.video_jobs_found", count=len(jobs))
        sem = asyncio.Semaphore(self._max_concurrent_polls)

        async def guarded(job: GenerationJob) -> None:
            async with sem:
                try:
                    async with self._db_manager.session() as job_session:
                        await self._poll_one(job, job_session)
                except Exception:
                    logger.exception("grok.video_worker.poll_one_failed", job_id=str(job.id))

        await asyncio.gather(*(guarded(j) for j in jobs))

    async def _poll_one(self, job: GenerationJob, session: AsyncSession) -> None:
        """Poll one job and drive its state transition through JobStateTransitionService."""
        ts = self._make_transition_service(session)
        product_id = str(job.product_id)

        # Timeout check first — a job past the ceiling is failed regardless
        # of what xAI currently reports.
        if job.started_at is not None:
            elapsed = (datetime.now(UTC) - job.started_at).total_seconds()
            if elapsed > self._max_poll_time:
                logger.warning("grok.video_job_timeout", job_id=str(job.id), elapsed_s=int(elapsed))
                # Refund policy: the timeout path had NO refund before this
                # rewrite (F2) — refund=True here adds the missing one.
                await ts.transition_to_failed(
                    job.id,
                    error_message=f"Video generation timed out after {elapsed:.0f} seconds",
                    refund=True,
                    product_id=product_id,
                )
                return

        outcome = await self._job_service.poll_video_job_for_worker(session, job.id)

        if outcome.status == VideoPollStatus.STILL_RUNNING:
            if job.status == JobStatus.QUEUED:
                await ts.transition_to_running(job.id)
            else:
                # xAI does not expose granular progress.
                await self._emit_progress(job, progress_pct=50)
            return

        if outcome.status == VideoPollStatus.COMPLETED:
            # Outputs were already persisted (flushed, not committed) inside
            # poll_video_job_for_worker's session — passing outputs=[] here
            # means the transition only performs the status update, and its
            # commit() covers both atomically.
            await ts.transition_to_completed(job.id, outputs=[], product_id=product_id)
            return

        # FAILED. poll_video_job_for_worker's only failure source is
        # GrokAPIError from get_video_result — unlike the synchronous
        # submission paths in job_service (create_image_job/start_video_job),
        # which already refund via billing_service.refund on GrokAPIError,
        # this polling-failure path has never refunded. refund=True is
        # correct here: there is no double-charge risk.
        await ts.transition_to_failed(
            job.id,
            error_message=outcome.error_message or "Video polling failed",
            refund=True,
            product_id=product_id,
        )
