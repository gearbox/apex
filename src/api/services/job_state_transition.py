"""Single chokepoint for all GenerationJob state transitions.

Wraps the DB update, commits, and publishes a JobStatusChanged event on the
EventBus after a successful DB commit. All Aisha job state changes flow through
here so future SSE or audit hooks have exactly one place to hook into.

Design invariants:
- Every transition method is idempotent: a second call for a job that is already
  in the target (or a terminal) state is a no-op — returns the current row without
  re-publishing or re-settling billing.
- Event publish happens AFTER the DB commit. A publish failure is logged but does
  NOT roll back — the DB is the source of truth; a missed event is a degraded SSE
  experience, not a data-loss bug.
- The service owns one DB session per instance. Do not reuse instances across
  transactional scopes.
"""

from __future__ import annotations

import dataclasses
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any, cast
from uuid import UUID

import structlog
from sqlalchemy import CursorResult, func
from sqlalchemy import update as sa_update
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.schemas.events import EventType, JobStatusPayload
from src.core.enums import JobStatus
from src.db.models.storage import GenerationJob
from src.db.repositories.job import JobRepository
from src.db.repositories.output import OutputRepository

if TYPE_CHECKING:
    from src.api.services.billing import BillingService
    from src.api.services.event_bus import EventBus

logger = structlog.get_logger(__name__)


@dataclasses.dataclass
class GenerationOutputData:
    """Data needed to persist one ComfyUI output image.

    Fields come from the R2 upload result. ``user_id``, ``job_id``, and
    ``product_id`` are taken from the job row inside the transition.
    """

    id: UUID
    storage_key: str
    content_type: str
    size_bytes: int
    format: str
    output_index: int
    expires_at: datetime


class JobStateTransitionService:
    """Atomically transitions GenerationJob status and publishes events.

    Intended for use inside a single transactional scope (one DB session per
    instance). The worker creates a fresh instance per job poll via the
    ``transition_service_factory`` pattern.
    """

    def __init__(
        self,
        session: AsyncSession,
        event_bus: EventBus | None,
        billing_service: BillingService,
    ) -> None:
        self._session = session
        self._event_bus = event_bus
        self._billing = billing_service
        self._job_repo = JobRepository(session)
        self._output_repo = OutputRepository(session)

    # -------------------------------------------------------------------------
    # Public transition methods
    # -------------------------------------------------------------------------

    async def transition_to_running(
        self,
        job_id: UUID,
        *,
        comfyui_prompt_id: str | None = None,
    ) -> GenerationJob:
        """QUEUED → RUNNING.

        Idempotent: if the job is already RUNNING or terminal, returns the
        current row without publishing or committing again.
        """
        job = await self._session.get(GenerationJob, job_id)
        if job is None:
            raise ValueError(f"Job {job_id} not found")

        old_status = str(job.status)
        if old_status != JobStatus.QUEUED.value:
            return job

        update_values: dict[str, object] = {
            "status": JobStatus.RUNNING.value,
            "started_at": func.now(),
        }
        if comfyui_prompt_id is not None:
            update_values["external_request_id"] = comfyui_prompt_id

        result = cast(
            CursorResult[Any],
            await self._session.execute(
                sa_update(GenerationJob)
                .where(
                    GenerationJob.id == job_id,
                    GenerationJob.status == JobStatus.QUEUED.value,
                )
                .values(**update_values)
                .execution_options(synchronize_session=False)
            ),
        )

        if result.rowcount == 0:
            # Concurrent update won the race — fetch current state without publish
            await self._session.refresh(job)
            return job

        await self._session.commit()
        await self._session.refresh(job)
        await self._publish(job, old_status=old_status)
        logger.info("job.transition.running", job_id=str(job_id))
        return job

    async def transition_to_completed(
        self,
        job_id: UUID,
        *,
        outputs: list[GenerationOutputData],
        product_id: str,
    ) -> GenerationJob:
        """QUEUED|RUNNING → COMPLETED. Persists outputs in the same transaction.

        Idempotent: no-op if already COMPLETED or terminal.
        """
        job = await self._session.get(GenerationJob, job_id)
        if job is None:
            raise ValueError(f"Job {job_id} not found")

        old_status = str(job.status)
        if old_status not in (JobStatus.QUEUED.value, JobStatus.RUNNING.value):
            return job

        result = cast(
            CursorResult[Any],
            await self._session.execute(
                sa_update(GenerationJob)
                .where(
                    GenerationJob.id == job_id,
                    GenerationJob.status.in_([JobStatus.QUEUED.value, JobStatus.RUNNING.value]),
                )
                .values(
                    status=JobStatus.COMPLETED.value,
                    completed_at=func.now(),
                )
                .execution_options(synchronize_session=False)
            ),
        )

        if result.rowcount == 0:
            await self._session.refresh(job)
            return job

        # Persist outputs in same transaction
        for out in outputs:
            await self._output_repo.create(
                id=out.id,
                user_id=job.user_id,
                job_id=job_id,
                storage_key=out.storage_key,
                content_type=out.content_type,
                size_bytes=out.size_bytes,
                format=out.format,
                output_index=out.output_index,
                expires_at=out.expires_at,
                product_id=product_id,
            )

        await self._session.commit()
        await self._session.refresh(job)
        await self._publish(job, old_status=old_status)
        logger.info(
            "job.transition.completed",
            job_id=str(job_id),
            output_count=len(outputs),
        )
        return job

    async def transition_to_failed(
        self,
        job_id: UUID,
        *,
        error_message: str,
        refund: bool = True,
        product_id: str,
    ) -> GenerationJob:
        """QUEUED|RUNNING → FAILED.

        Issues a billing refund when ``refund=True`` and a debit reservation
        exists. Refund failures are logged but do not roll back the status
        transition — the DB reflects FAILED even if the refund can't be processed.
        """
        job = await self._session.get(GenerationJob, job_id)
        if job is None:
            raise ValueError(f"Job {job_id} not found")

        old_status = str(job.status)
        if old_status not in (JobStatus.QUEUED.value, JobStatus.RUNNING.value):
            return job

        result = cast(
            CursorResult[Any],
            await self._session.execute(
                sa_update(GenerationJob)
                .where(
                    GenerationJob.id == job_id,
                    GenerationJob.status.in_([JobStatus.QUEUED.value, JobStatus.RUNNING.value]),
                )
                .values(
                    status=JobStatus.FAILED.value,
                    completed_at=func.now(),
                    error_message=error_message[:2000],
                )
                .execution_options(synchronize_session=False)
            ),
        )

        if result.rowcount == 0:
            await self._session.refresh(job)
            return job

        await self._session.commit()
        await self._session.refresh(job)
        await self._publish(job, old_status=old_status)
        logger.warning(
            "job.transition.failed",
            job_id=str(job_id),
            error=error_message[:200],
        )

        if refund:
            await self._try_refund(job_id, product_id=product_id)

        return job

    # -------------------------------------------------------------------------
    # Internals
    # -------------------------------------------------------------------------

    async def _publish(self, job: GenerationJob, *, old_status: str) -> None:
        if self._event_bus is None:
            return
        try:
            payload = JobStatusPayload(
                job_id=job.id,
                status=str(job.status),
                previous_status=old_status,
                generation_type=str(job.generation_type),
                provider=str(job.provider),
            )
            await self._event_bus.publish(
                user_id=job.user_id,
                event_type=EventType.JOB_STATUS_CHANGED,
                payload=payload,
            )
        except Exception:
            logger.exception(
                "job.state_transition.publish_failed",
                job_id=str(job.id),
            )

    async def _try_refund(self, job_id: UUID, *, product_id: str) -> None:
        try:
            await self._billing.refund(
                job_id=job_id,
                description="Job failed — tokens refunded",
                session=self._session,
                product_id=product_id,
            )
            await self._session.commit()
        except Exception:
            await self._session.rollback()
            logger.exception(
                "job.state_transition.refund_failed",
                job_id=str(job_id),
            )

    @staticmethod
    def make_output_expires_at(retention_days: int = 7) -> datetime:
        return datetime.now(UTC) + timedelta(days=retention_days)
