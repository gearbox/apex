"""Background worker for polling Grok video generation jobs.

Session-per-job + semaphore fan-out (mirrors AishaJobPoller): one short
session lists candidate jobs, then each job is polled in its own session.
All state transitions go through JobStateTransitionService — this worker
never mutates job.status directly and never calls BillingService itself;
refunds are settled inside the transition.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import structlog

from src.api.schemas.events import EventType, JobProgressPayload
from src.api.services.grok import GrokRateLimitError, GrokTimeoutError

# Retained as a module symbol for downstream test/extension imports. Terminal
# transitions themselves are now owned by GrokJobService.
from src.api.services.job_state_transition import JobStateTransitionService  # noqa: F401
from src.core.enums import JobStatus, Provider, VideoPollStatus
from src.db.repositories.job import JobRepository
from src.workers.base import PeriodicWorker

if TYPE_CHECKING:
    from collections.abc import Callable

    from redis.asyncio import Redis
    from sqlalchemy.ext.asyncio import AsyncSession

    from src.api.services.billing import BillingService
    from src.api.services.event_bus import EventBus
    from src.api.services.grok.job_service import GrokJobService
    from src.api.services.ops_event_bus import OpsEventBus
    from src.core.config import Settings
    from src.db import DatabaseManager
    from src.db.models import GenerationJob

logger = structlog.get_logger(__name__)


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
        ops_event_bus: OpsEventBus | None = None,
        *,
        redis_enabled: bool = False,
        redis_client_factory: Callable[[], Redis],
    ) -> None:
        """Initialize the video worker.

        Args:
            db_manager: Database manager for sessions.
            job_service: Grok job service for polling.
            billing_service: Billing service, needed by JobStateTransitionService
                for timeout/failure refunds.
            settings: Application settings.
            event_bus: Optional event bus for publishing job events.
            ops_event_bus: Optional ops event bus for admin notifications.
            redis_enabled: Whether Redis is configured (enables the leader lease).
        """
        super().__init__(
            name="grok_video_poller",
            interval_seconds=settings.grok_video_poll_interval,
            redis_enabled=redis_enabled,
            redis_client_factory=redis_client_factory,
        )
        self._db_manager = db_manager
        self._job_service = job_service
        self._billing = billing_service
        self._event_bus = event_bus
        self._ops_event_bus = ops_event_bus
        self._max_concurrent_polls = settings.grok_video_max_concurrent_polls

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
            jobs = list(
                await JobRepository(session).list_pending_video_jobs(provider=Provider.GROK)
            )

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
        """Poll one job; GrokJobService owns every terminal settlement."""
        product_id = str(job.product_id)

        try:
            outcome = await self._job_service.poll_video_job_for_worker(session, job.id)
        except (GrokRateLimitError, GrokTimeoutError) as exc:
            failure = exc.failure
            logger.warning(
                "grok.video_poll_transient",
                job_id=str(job.id),
                user_id=str(job.user_id),
                product_id=product_id,
                provider_job_id=failure.provider_request_id or job.external_request_id,
                failure_kind=failure.kind.value,
                provider_status=failure.provider_status_code,
                retryable=failure.retryable,
            )
            return  # no transition, no progress emit; retried next tick (D2)

        if outcome.status == VideoPollStatus.STILL_RUNNING:
            if job.status == JobStatus.RUNNING:
                # xAI does not expose granular progress.
                await self._emit_progress(job, progress_pct=50)
            else:
                await self._job_service.settle_video_poll_outcome(
                    session,
                    job_id=job.id,
                    outcome=outcome,
                    product_id=product_id,
                )
            return

        if outcome.status == VideoPollStatus.COMPLETED:
            await self._job_service.settle_video_poll_outcome(
                session,
                job_id=job.id,
                outcome=outcome,
                product_id=product_id,
            )
            return

        # FAILED is settled by the same GrokJobService entry point used by
        # poll-on-read. It owns the normalized public failure, billing policy,
        # conditional terminal state update, commit, and post-commit events.
        await self._job_service.settle_video_poll_outcome(
            session,
            job_id=job.id,
            outcome=outcome,
            product_id=product_id,
        )
