"""Background worker for polling one provider's async video generation jobs.

Session-per-job + semaphore fan-out (mirrors AishaJobPoller): one short
session lists candidate jobs, then each job is polled in its own session.
All state transitions go through the provider's ``settle_video_poll_outcome``
— this worker never mutates job.status directly and never calls
BillingService itself; refunds are settled inside the transition.

One instance exists per provider, each with its own lease name, interval and
semaphore, so providers poll and finalize independently and concurrently.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import structlog

from src.api.schemas.events import EventType, JobProgressPayload
from src.core.enums import JobStatus, VideoPollStatus
from src.db.repositories.job import JobRepository
from src.workers.base import PeriodicWorker

if TYPE_CHECKING:
    from collections.abc import Callable

    from redis.asyncio import Redis
    from sqlalchemy.ext.asyncio import AsyncSession

    from src.api.services.event_bus import EventBus
    from src.api.services.video_jobs.protocol import AsyncVideoJobService
    from src.db import DatabaseManager
    from src.db.models import GenerationJob

logger = structlog.get_logger(__name__)


class AsyncVideoJobWorker(PeriodicWorker):
    """Polls one provider for in-progress video jobs and drives transitions."""

    def __init__(
        self,
        *,
        name: str | None = None,
        job_service: AsyncVideoJobService,
        interval_seconds: int,
        max_concurrent_polls: int,
        db_manager: DatabaseManager,
        event_bus: EventBus | None = None,
        redis_enabled: bool = False,
        redis_client_factory: Callable[[], Redis],
    ) -> None:
        """Initialize the video worker.

        Args:
            name: Worker name; feeds the leader-lease key, so it must be stable
                across releases (and unique per provider). Defaults to the job
                service's ``VIDEO_POLLER_NAME`` — the one definition per provider.
            job_service: The provider's polling/settlement service.
            interval_seconds: Seconds between ticks.
            max_concurrent_polls: Semaphore bound on concurrent job polls.
            db_manager: Database manager for sessions.
            event_bus: Optional event bus for publishing job progress events.
            redis_enabled: Whether Redis is configured (enables the leader lease).
            redis_client_factory: Factory for the operational Redis client.
        """
        super().__init__(
            name=name if name is not None else job_service.VIDEO_POLLER_NAME,
            interval_seconds=interval_seconds,
            redis_enabled=redis_enabled,
            redis_client_factory=redis_client_factory,
        )
        self._db_manager = db_manager
        self._job_service = job_service
        self._event_bus = event_bus
        self._max_concurrent_polls = max_concurrent_polls

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
        provider = self._job_service.provider
        async with self._db_manager.session() as session:
            jobs = list(await JobRepository(session).list_pending_video_jobs(provider=provider))

        if not jobs:
            return

        logger.debug("video_job.jobs_found", provider=provider.value, count=len(jobs))
        sem = asyncio.Semaphore(self._max_concurrent_polls)

        async def guarded(job: GenerationJob) -> None:
            async with sem:
                try:
                    async with self._db_manager.session() as job_session:
                        await self._poll_one(job, job_session)
                except Exception:
                    logger.exception(
                        "video_job.worker.poll_one_failed",
                        provider=provider.value,
                        job_id=str(job.id),
                    )

        await asyncio.gather(*(guarded(j) for j in jobs))

    async def _poll_one(self, job: GenerationJob, session: AsyncSession) -> None:
        """Poll one job; the provider service owns every terminal settlement."""
        product_id = str(job.product_id)

        try:
            outcome = await self._job_service.poll_video_job_for_worker(session, job.id)
        except self._job_service.transient_poll_errors as exc:
            failure = exc.failure  # type: ignore[attr-defined]
            logger.warning(
                "video_job.poll_transient",
                provider=self._job_service.provider.value,
                job_id=str(job.id),
                user_id=str(job.user_id),
                product_id=product_id,
                provider_job_id=failure.provider_request_id or job.external_request_id,
                failure_kind=failure.kind.value,
                provider_status=failure.provider_status_code,
                retryable=failure.retryable,
            )
            return  # no transition, no progress emit; retried next tick (D2)

        if outcome.status == VideoPollStatus.STILL_RUNNING and job.status == JobStatus.RUNNING:
            # Nothing to settle for an already-running job. Only a provider that
            # declares a progress value emits an event (no fabricated percentages).
            progress_pct = self._job_service.running_progress_pct
            if progress_pct is not None:
                await self._emit_progress(job, progress_pct=progress_pct)
            return

        # QUEUED -> RUNNING, COMPLETED and FAILED all settle through the same
        # provider entry point used by poll-on-read. It owns the normalized
        # public failure, billing policy, conditional terminal state update,
        # commit, and post-commit events.
        await self._job_service.settle_video_poll_outcome(
            session,
            job_id=job.id,
            outcome=outcome,
            product_id=product_id,
        )
