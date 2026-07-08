"""Sweep in-flight Aisha jobs to FAILED when their GPU session terminates."""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING
from uuid import UUID

import structlog

from src.api.services.job_state_transition import JobStateTransitionService
from src.db.repositories.job import JobRepository

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

    from src.api.services.billing import BillingService
    from src.api.services.event_bus import EventBus

logger = structlog.get_logger(__name__)


@dataclasses.dataclass(frozen=True)
class JobSweepResult:
    swept_count: int
    error_count: int
    skipped_count: int


class JobSweepService:
    """Sweeps in-flight Aisha jobs to FAILED when their GPU session terminates.

    Called from session-lifecycle code paths that drive a session into a
    terminal-or-stopping state (stop, provisioning failure). Never called
    from the request path of pause — pause is gated on zero in-flight
    jobs by precondition, not by reactive sweep.

    Idempotent: re-running on the same session is safe — already-terminal
    jobs are recognized via the explicit did_transition flag from
    transition_to_failed, not by inferring from error_message.

    Best-effort: per-job failures are logged and isolated. The sweep
    completes for every other job. The caller continues with its own
    cleanup regardless of sweep outcome.
    """

    def __init__(
        self,
        *,
        session_factory: async_sessionmaker[AsyncSession],
        event_bus: EventBus | None,
        billing_service: BillingService,
    ) -> None:
        self._session_factory = session_factory
        self._event_bus = event_bus
        self._billing = billing_service

    async def sweep_session(
        self,
        gpu_session_id: UUID,
        *,
        product_id: str,
        reason: str,
    ) -> JobSweepResult:
        """Mark all in-flight Aisha jobs for the given session as FAILED.

        Args:
            gpu_session_id: Session whose jobs to fail.
            product_id: Required by JobStateTransitionService for
                refund scoping.
            reason: Human-readable error_message stamped on each job.
                Truncated to 500 chars.

        Returns:
            JobSweepResult with counts. Never raises.
        """
        truncated_reason = reason[:500]

        async with self._session_factory() as session:
            repo = JobRepository(session)
            jobs = list(await repo.list_in_flight_for_session(gpu_session_id))

            if not jobs:
                return JobSweepResult(swept_count=0, error_count=0, skipped_count=0)

            ts = JobStateTransitionService(
                session=session,
                event_bus=self._event_bus,
                billing_service=self._billing,
            )

            swept = 0
            errored = 0
            skipped = 0

            for job in jobs:
                try:
                    _, did_transition = await ts.transition_to_failed(
                        job.id,
                        error_message=truncated_reason,
                        refund=True,
                        product_id=product_id,
                    )
                    if did_transition:
                        swept += 1
                    else:
                        skipped += 1
                except Exception:
                    errored += 1
                    logger.exception(
                        "job_sweep.transition_failed",
                        gpu_session_id=str(gpu_session_id),
                        job_id=str(job.id),
                    )

        logger.info(
            "job_sweep.completed",
            gpu_session_id=str(gpu_session_id),
            swept=swept,
            errored=errored,
            skipped=skipped,
            reason=reason[:200],
        )
        return JobSweepResult(swept_count=swept, error_count=errored, skipped_count=skipped)

    async def sweep_session_best_effort(
        self,
        *,
        session_id: UUID,
        product_id: str,
        reason: str,
        log_event: str,
    ) -> None:
        """Best-effort wrapper around sweep_session.

        Catches and logs exceptions; never raises. Logs the result on success
        via log_event with the JobSweepResult fields as kwargs.

        Use this from lifecycle code (session stop, provisioning failure)
        where a sweep failure must NOT block the rest of the cleanup. The
        underlying age-based timeout in AishaJobPoller is the safety net.

        Args:
            session_id: Session whose in-flight jobs to fail.
            product_id: Pass the session's product_id (for refund scoping).
            reason: Human-readable error_message; truncated internally.
            log_event: Structlog event name on success
                (e.g. "gpu_session.stop.job_sweep" or
                "gpu_session.provision.job_sweep").
        """
        try:
            result = await self.sweep_session(
                session_id,
                product_id=product_id,
                reason=reason,
            )
            logger.info(
                log_event,
                session_id=str(session_id),
                **dataclasses.asdict(result),
            )
        except Exception:
            logger.exception(
                "%s_error",
                log_event,
                session_id=str(session_id),
            )
