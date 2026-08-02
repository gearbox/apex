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

import structlog
from sqlalchemy import CursorResult, func, or_
from sqlalchemy import update as sa_update

from src.api.schemas.events import EventType, JobStatusPayload
from src.api.schemas.ops_events import GenerationFailedOpsPayload, OpsEventType
from src.api.services.generation.public_errors import (
    LEGACY_FAILED_JOB_MESSAGE,
    public_error_for_job,
)
from src.api.services.ops_event_bus import OpsEventBus
from src.core.enums import JobStatus
from src.db.models.storage import GenerationJob, GenerationMaterializationAttempt
from src.db.repositories.job import JobRepository
from src.db.repositories.output import OutputRepository

if TYPE_CHECKING:
    from uuid import UUID

    from sqlalchemy.ext.asyncio import AsyncSession
    from sqlalchemy.sql.elements import ColumnElement

    from src.api.services.billing import BalanceEvent, BillingService
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
    is_thumbnail: bool = False
    parent_output_id: UUID | None = None
    thumbnail_max_edge: int | None = None
    width: int | None = None
    height: int | None = None


@dataclasses.dataclass(frozen=True)
class _DeferredFailurePublication:
    """All notifications that become eligible only after a UoW commit."""

    job_id: UUID
    user_id: UUID
    old_status: str
    status: str
    generation_type: str
    provider: str
    failure_code: str | None
    public_error_message: str | None
    product_id: str
    reserve_event: BalanceEvent | None
    refund_event: BalanceEvent | None


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
        ops_event_bus: OpsEventBus | None = None,
    ) -> None:
        self._session = session
        self._event_bus = event_bus
        self._billing = billing_service
        self._ops_event_bus = (
            ops_event_bus if ops_event_bus is not None else OpsEventBus(enabled=False)
        )
        self._job_repo = JobRepository(session)
        self._output_repo = OutputRepository(session)

    # -------------------------------------------------------------------------
    # Public transition methods
    # -------------------------------------------------------------------------

    async def transition_to_running(
        self,
        job_id: UUID,
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

        result = cast(
            "CursorResult[Any]",
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
        job, _ = await self._transition_to_completed(
            job_id,
            outputs=outputs,
            product_id=product_id,
        )
        return job

    async def transition_claimed_video_to_completed(
        self,
        job_id: UUID,
        *,
        claim_token: str,
        outputs: list[GenerationOutputData],
        product_id: str,
        materialization_attempt_id: UUID | None = None,
    ) -> tuple[GenerationJob, bool]:
        """Complete a Grok video only while holding its durable DB lease.

        ``claim_token`` is acquired and committed before any R2 side effect.
        The CAS below prevents an expired/replaced claimant from attaching its
        materialized outputs to a job settled by another caller.
        """
        return await self._transition_to_completed(
            job_id,
            outputs=outputs,
            product_id=product_id,
            claim_token=claim_token,
            materialization_attempt_id=materialization_attempt_id,
        )

    async def _transition_to_completed(
        self,
        job_id: UUID,
        *,
        outputs: list[GenerationOutputData],
        product_id: str,
        claim_token: str | None = None,
        materialization_attempt_id: UUID | None = None,
    ) -> tuple[GenerationJob, bool]:
        job = await self._session.get(GenerationJob, job_id)
        if job is None:
            raise ValueError(f"Job {job_id} not found")

        old_status = str(job.status)
        if old_status not in (JobStatus.QUEUED.value, JobStatus.RUNNING.value):
            return job, False

        conditions: list[ColumnElement[bool]] = [
            GenerationJob.id == job_id,
            GenerationJob.status.in_([JobStatus.QUEUED.value, JobStatus.RUNNING.value]),
        ]
        values: dict[str, object] = {
            "status": JobStatus.COMPLETED.value,
            "completed_at": func.now(),
            # Terminal rows never retain a claim pair, including ordinary
            # non-video completions and compatibility paths.
            "finalization_claim_token": None,
            "finalization_lease_expires_at": None,
        }
        if claim_token is not None:
            conditions.extend(
                [
                    GenerationJob.finalization_claim_token == claim_token,
                ]
            )

        result = cast(
            "CursorResult[Any]",
            await self._session.execute(
                sa_update(GenerationJob)
                .where(*conditions)
                .values(**values)
                .execution_options(synchronize_session=False)
            ),
        )

        if result.rowcount == 0:
            await self._session.refresh(job)
            return job, False

        # Persist outputs in same transaction — parents before children so the
        # self-FK on parent_output_id is satisfiable within a single transaction.
        sorted_outputs = sorted(outputs, key=lambda o: 1 if o.is_thumbnail else 0)
        for out in sorted_outputs:
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
                is_thumbnail=out.is_thumbnail,
                parent_output_id=out.parent_output_id,
                thumbnail_max_edge=out.thumbnail_max_edge,
                width=out.width,
                height=out.height,
            )

        if materialization_attempt_id is not None:
            # This update shares the output/job commit.  A completed job can
            # never look like an abandoned R2 attempt after a restart.
            await self._session.execute(
                sa_update(GenerationMaterializationAttempt)
                .where(
                    GenerationMaterializationAttempt.id == materialization_attempt_id,
                    GenerationMaterializationAttempt.product_id == product_id,
                )
                .values(state="committed", completed_at=func.now(), updated_at=func.now())
            )

        await self._session.commit()
        await self._session.refresh(job)
        await self._publish(job, old_status=old_status)
        logger.info(
            "job.transition.completed",
            job_id=str(job_id),
            output_count=len(outputs),
        )
        return job, True

    async def transition_to_failed(
        self,
        job_id: UUID,
        *,
        error_message: str | None = None,
        public_error_message: str | None = None,
        failure_code: str | None = None,
        refund: bool = True,
        product_id: str,
        reserve_event: BalanceEvent | None = None,
        commit: bool = True,
    ) -> tuple[GenerationJob, bool]:
        """PENDING|QUEUED|RUNNING → FAILED.

        For non-billable failures the conditional FAILED transition and its
        refund ledger row are one database transaction.  A refund failure
        rolls back the status transition too, leaving the job retryable.  Set
        ``commit=False`` only for a request-level unit of work that will add
        its idempotency result and commit before calling
        :meth:`publish_deferred_failure`.

        Returns:
            (job, did_transition) where did_transition is True if THIS call moved
            the job from QUEUED|RUNNING to FAILED. False if the job was already
            terminal (idempotent no-op — no publish, no refund).
        """
        job = await self._session.get(GenerationJob, job_id)
        if job is None:
            raise ValueError(f"Job {job_id} not found")

        old_status = str(job.status)
        in_flight_statuses = (
            JobStatus.PENDING.value,
            JobStatus.QUEUED.value,
            JobStatus.RUNNING.value,
        )
        if old_status not in in_flight_statuses:
            return job, False

        # ``error_message`` remains internal diagnostics for legacy call
        # sites. It is deliberately kept separate from the public column and
        # event payload; new paths pass ``public_error_message`` instead.
        safe_message = public_error_message or LEGACY_FAILED_JOB_MESSAGE

        result = cast(
            "CursorResult[Any]",
            await self._session.execute(
                sa_update(GenerationJob)
                .where(
                    GenerationJob.id == job_id,
                    GenerationJob.status.in_(in_flight_statuses),
                    # A live finalizer owns this job until its database-time
                    # lease expires.  Once expired, failure and completion
                    # race on the same conditional update; the winner is
                    # authoritative.  The claim pair is cleared below for
                    # every terminal result.
                    or_(
                        GenerationJob.finalization_claim_token.is_(None),
                        GenerationJob.finalization_lease_expires_at <= func.now(),
                    ),
                )
                .values(
                    status=JobStatus.FAILED.value,
                    completed_at=func.now(),
                    error_message=error_message[:2000] if error_message is not None else None,
                    public_error_message=safe_message[:2000],
                    failure_code=failure_code[:100] if failure_code is not None else None,
                    finalization_claim_token=None,
                    finalization_lease_expires_at=None,
                )
                .execution_options(synchronize_session=False)
            ),
        )

        if result.rowcount == 0:
            # Concurrent update won the race — already terminal.
            await self._session.refresh(job)
            return job, False

        refund_event: BalanceEvent | None = None
        if refund and job.debit_transaction_id is not None:
            try:
                refund_result = await self._billing.refund(
                    job_id=job_id,
                    description="Job failed — tokens refunded",
                    session=self._session,
                    product_id=product_id,
                )
                refund_event = refund_result.event
            except Exception:
                # A terminal, non-billable job must never retain its debit
                # merely because the compensating ledger write failed.
                await self._session.rollback()
                logger.exception(
                    "job.transition.refund_failed_rolled_back",
                    job_id=str(job_id),
                )
                raise
        elif refund:
            # A pre-reservation/billing failure can still have a job row in
            # the unit of work.  Never turn that into RefundNotEligibleError:
            # there is no confirmed debit to compensate.
            logger.info("job.transition.refund_skipped_no_debit", job_id=str(job_id))

        if not commit:
            await self._session.refresh(job)
            self._deferred_failure = self._build_deferred_failure_publication(
                job=job,
                old_status=old_status,
                product_id=product_id,
                reserve_event=reserve_event,
                refund_event=refund_event,
                failure_code=failure_code,
                public_error_message=safe_message,
            )
            return job, True

        await self._session.commit()
        await self._session.refresh(job)
        await self.publish_deferred_failure(
            self._build_deferred_failure_publication(
                job=job,
                old_status=old_status,
                product_id=product_id,
                reserve_event=reserve_event,
                refund_event=refund_event,
                failure_code=failure_code,
                public_error_message=safe_message,
            )
        )

        return job, True

    @property
    def deferred_failure(self) -> _DeferredFailurePublication | None:
        """Post-commit events captured by ``transition_to_failed(commit=False)``."""
        return getattr(self, "_deferred_failure", None)

    async def publish_deferred_failure(self, publication: _DeferredFailurePublication) -> None:
        """Publish an already-committed failure settlement exactly once."""
        await self._publish_status_payload(
            job_id=publication.job_id,
            user_id=publication.user_id,
            status=publication.status,
            old_status=publication.old_status,
            generation_type=publication.generation_type,
            provider=publication.provider,
            failure_code=publication.failure_code,
            public_error_message=publication.public_error_message,
        )
        logger.warning(
            "job.transition.failed",
            job_id=str(publication.job_id),
            failure_code=publication.failure_code,
            refund=publication.refund_event is not None,
        )
        await self._ops_event_bus.publish(
            event_type=OpsEventType.GENERATION_FAILED,
            product_id=publication.product_id,
            payload=GenerationFailedOpsPayload(
                job_id=publication.job_id,
                user_id=publication.user_id,
                provider=publication.provider,
                generation_type=publication.generation_type,
            ),
        )
        if self._event_bus is not None:
            # Do not call the transport with empty placeholders.  Besides
            # avoiding needless work, this preserves the useful invariant
            # that every publish_balance invocation represents a committed
            # ledger event.
            if publication.reserve_event is not None:
                await self._event_bus.publish_balance(publication.reserve_event)
            if publication.refund_event is not None:
                await self._event_bus.publish_balance(publication.refund_event)

    # -------------------------------------------------------------------------
    # Internals
    # -------------------------------------------------------------------------

    @staticmethod
    def _build_deferred_failure_publication(
        *,
        job: GenerationJob,
        old_status: str,
        product_id: str,
        reserve_event: BalanceEvent | None,
        refund_event: BalanceEvent | None,
        failure_code: str | None,
        public_error_message: str,
    ) -> _DeferredFailurePublication:
        """Capture immutable data before the request session can be released."""
        return _DeferredFailurePublication(
            job_id=job.id,
            user_id=job.user_id,
            old_status=old_status,
            status=JobStatus.FAILED.value,
            generation_type=str(job.generation_type),
            provider=str(job.provider),
            failure_code=failure_code[:100] if failure_code is not None else None,
            public_error_message=public_error_message[:2000],
            product_id=product_id,
            reserve_event=reserve_event,
            refund_event=refund_event,
        )

    async def _publish(self, job: GenerationJob, *, old_status: str) -> None:
        await self._publish_status_payload(
            job_id=job.id,
            user_id=job.user_id,
            status=str(job.status),
            old_status=old_status,
            generation_type=str(job.generation_type),
            provider=str(job.provider),
            failure_code=job.failure_code,
            public_error_message=job.public_error_message,
        )

    async def _publish_status_payload(
        self,
        *,
        job_id: UUID,
        user_id: UUID,
        status: str,
        old_status: str,
        generation_type: str,
        provider: str,
        failure_code: str | None,
        public_error_message: str | None,
    ) -> None:
        if self._event_bus is None:
            return
        try:
            payload = JobStatusPayload(
                job_id=job_id,
                status=status,
                previous_status=old_status,
                generation_type=generation_type,
                provider=provider,
                failure_code=failure_code,
                error_message=public_error_for_job(
                    status=status,
                    public_error_message=public_error_message,
                ),
            )
            await self._event_bus.publish(
                user_id=user_id,
                event_type=EventType.JOB_STATUS_CHANGED,
                payload=payload,
            )
        except Exception:
            logger.exception(
                "job.state_transition.publish_failed",
                job_id=str(job_id),
            )

    @staticmethod
    def make_output_expires_at(retention_days: int = 7) -> datetime:
        return datetime.now(UTC) + timedelta(days=retention_days)
