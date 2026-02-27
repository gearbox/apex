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

from src.core.enums import GenerationType, JobStatus
from src.db.models import GenerationJob

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

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
    ) -> None:
        """Initialize the video worker.

        Args:
            db_manager: Database manager for sessions.
            job_service: Grok job service for polling.
            settings: Application settings.
        """
        self._db = db_manager
        self._job_service = job_service
        self._poll_interval = settings.grok_video_poll_interval
        self._max_poll_time = settings.grok_video_max_poll_time
        self._running = False
        self._task: asyncio.Task[None] | None = None

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
                job.status = JobStatus.FAILED
                job.error_message = f"Video generation timed out after {elapsed:.0f} seconds"
                job.completed_at = datetime.now(UTC)
                await session.commit()
                return

        try:
            # Poll the job
            updated_job = await self._job_service.poll_video_job(session, job_id)
            await session.commit()

            if updated_job is None:
                logger.warning("grok.video_job_not_found", job_id=str(job_id))
                return

            if updated_job.status == JobStatus.COMPLETED.value:
                logger.info("grok.video_job_completed", job_id=str(job_id))
            elif updated_job.status == JobStatus.FAILED.value:
                logger.warning("grok.video_job_failed", job_id=str(job_id), error=updated_job.error_message)

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
    ) -> None:
        """Start the video worker.

        Args:
            db_manager: Database manager.
            job_service: Grok job service.
            settings: Application settings.
        """
        if cls._worker is not None:
            logger.warning("grok.video_worker_already_initialized")
            return

        cls._worker = GrokVideoWorker(db_manager, job_service, settings)
        await cls._worker.start()

    @classmethod
    async def stop(cls) -> None:
        """Stop the video worker."""
        if cls._worker is not None:
            await cls._worker.stop()
            cls._worker = None
