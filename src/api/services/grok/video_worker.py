"""Background worker for polling Grok video generation jobs.

This worker runs in the background and periodically polls for
video generation jobs that are waiting for completion.
"""

from __future__ import annotations

import asyncio
import contextlib
from datetime import UTC, datetime
from typing import TYPE_CHECKING
from uuid import UUID

import structlog
from sqlalchemy import select

from src.api.schemas.events import EventType, JobProgressPayload, JobStatusPayload
from src.core.enums import GenerationType, JobStatus
from src.db.models import GenerationJob

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from src.api.services.event_bus import EventBus
    from src.api.services.grok.job_service import GrokJobService
    from src.core.config import Settings
    from src.db import DatabaseManager

logger = structlog.get_logger(__name__)


class GrokVideoWorker:
    """Background worker for polling Grok video generation jobs.

    Periodically checks for jobs in QUEUED or RUNNING status
    with video generation types and polls xAI API for results.
    """

    def __init__(
        self,
        db_manager: DatabaseManager,
        job_service: GrokJobService,
        settings: Settings,
        event_bus: EventBus | None = None,
    ) -> None:
        """Initialize the video worker.

        Args:
            db_manager: Database manager for sessions.
            job_service: Grok job service for polling.
            settings: Application settings.
            event_bus: Optional event bus for publishing job events.
        """
        self._db = db_manager
        self._job_service = job_service
        self._poll_interval = settings.grok_video_poll_interval
        self._max_poll_time = settings.grok_video_max_poll_time
        self._event_bus = event_bus
        self._running = False
        self._task: asyncio.Task[None] | None = None

    @staticmethod
    def _normalize_status(status: JobStatus | str) -> str:
        return status if isinstance(status, str) else status.value

    async def _emit_status_changed(
        self,
        job: GenerationJob,
        previous_status: str,
        new_status: str,
    ) -> None:
        if not self._event_bus:
            return
        await self._event_bus.publish(
            user_id=job.user_id,
            event_type=EventType.JOB_STATUS_CHANGED,
            payload=JobStatusPayload(
                job_id=job.id,
                status=new_status,
                previous_status=previous_status,
                generation_type=job.generation_type,
                provider=job.provider,
            ),
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

    async def start(self) -> None:
        """Start the background worker."""
        if self._running:
            logger.warning("grok.video_worker_already_running")
            return

        self._running = True
        self._task = asyncio.create_task(self._run_loop())
        logger.info(
            "grok.video_worker_started",
            poll_interval=self._poll_interval,
            max_poll_time=self._max_poll_time,
        )

    async def stop(self) -> None:
        """Stop the background worker."""
        self._running = False
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None
        logger.info("grok.video_worker_stopped")

    async def _run_loop(self) -> None:
        """Main worker loop."""
        while self._running:
            try:
                await self._poll_pending_jobs()
            except Exception:
                logger.exception("grok.video_worker_loop_error")

            # Wait for next poll interval
            await asyncio.sleep(self._poll_interval)

    async def _poll_pending_jobs(self) -> None:
        """Find and poll all pending video jobs."""
        async with self._db.session() as session:
            # Find jobs that need polling
            # - Status is QUEUED or RUNNING
            # - Generation type is video (T2V, I2V, V2V)
            # - Has xAI request ID (stored in external_request_id)
            # - Not timed out
            video_types = [
                GenerationType.T2V.value,
                GenerationType.I2V.value,
                GenerationType.V2V.value,
                GenerationType.FLF2V.value,
            ]
            pending_statuses = [JobStatus.QUEUED.value, JobStatus.RUNNING.value]

            result = await session.execute(
                select(GenerationJob)
                .where(GenerationJob.status.in_(pending_statuses))
                .where(GenerationJob.generation_type.in_(video_types))
                .where(GenerationJob.external_request_id.isnot(None))
            )
            jobs = result.scalars().all()

            if not jobs:
                return

            logger.debug("grok.video_jobs_found", count=len(jobs))

            for job in jobs:
                await self._poll_single_job(session, job)

    async def _poll_single_job(
        self,
        session: AsyncSession,
        job: GenerationJob,
    ) -> None:
        """Poll a single video job for completion."""
        job_id: UUID = job.id

        # Check for timeout
        if job.started_at:
            elapsed = (datetime.now(UTC) - job.started_at).total_seconds()
            if elapsed > self._max_poll_time:
                logger.warning("grok.video_job_timeout", job_id=str(job_id), elapsed_s=int(elapsed))
                prev_status = self._normalize_status(job.status)
                job.status = JobStatus.FAILED
                job.error_message = f"Video generation timed out after {elapsed:.0f} seconds"
                job.completed_at = datetime.now(UTC)
                await session.commit()
                await self._emit_status_changed(job, prev_status, JobStatus.FAILED.value)
                return

        try:
            # Capture status before polling so we can emit the delta
            previous_status = self._normalize_status(job.status)
            # Poll the job
            updated_job = await self._job_service.poll_video_job(session, job_id)
            await session.commit()

            if updated_job is None:
                logger.warning("grok.video_job_not_found", job_id=str(job_id))
                return

            if updated_job.status == JobStatus.COMPLETED.value:
                logger.info("grok.video_job_completed", job_id=str(job_id))
                await self._emit_status_changed(job, previous_status, updated_job.status)
            elif updated_job.status == JobStatus.FAILED.value:
                logger.warning(
                    "grok.video_job_failed", job_id=str(job_id), error=updated_job.error_message
                )
                await self._emit_status_changed(job, previous_status, updated_job.status)
            elif updated_job.status == JobStatus.RUNNING.value:
                await self._emit_progress(
                    job, progress_pct=50
                )  # Grok API does not expose granular progress

        except Exception as e:
            logger.error("grok.video_job_poll_failed", job_id=str(job_id), error=str(e))
            # Don't fail the job on transient errors, let it retry
            await session.rollback()


class GrokVideoWorkerManager:
    """Manager for the video worker lifecycle.

    Provides a clean interface for starting/stopping the worker
    during application lifecycle.
    """

    _worker: GrokVideoWorker | None = None

    @classmethod
    async def start(
        cls,
        db_manager: DatabaseManager,
        job_service: GrokJobService,
        settings: Settings,
        event_bus: EventBus | None = None,
    ) -> None:
        """Start the video worker.

        Args:
            db_manager: Database manager.
            job_service: Grok job service.
            settings: Application settings.
            event_bus: Optional event bus for publishing job events.
        """
        if cls._worker is not None:
            logger.warning("grok.video_worker_already_initialized")
            return

        cls._worker = GrokVideoWorker(db_manager, job_service, settings, event_bus=event_bus)
        await cls._worker.start()

    @classmethod
    async def stop(cls) -> None:
        """Stop the video worker."""
        if cls._worker is not None:
            await cls._worker.stop()
            cls._worker = None
