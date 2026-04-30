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
    terminal-or-stopping state (stop, provisioning failure). Never called from
    the request path of pause — pause is gated on zero in-flight jobs by
    precondition, not by reactive sweep.

    Idempotent: re-running on the same session is safe — already-terminal jobs
    are no-ops in JobStateTransitionService.transition_to_failed.

    Best-effort: per-job failures are logged and isolated. The sweep completes
    for every other job. The caller continues with its own cleanup regardless of
    sweep outcome.
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
            product_id: Required by JobStateTransitionService for refund
                scoping. Pass the session's product_id.
            reason: Human-readable error_message stamped on each job.

        Returns:
            JobSweepResult with counts of failed and errored jobs.
            Never raises — failures are logged and counted.
        """
        async with self._session_factory() as session:
            repo = JobRepository(session)
            jobs = list(await repo.list_in_flight_for_session(gpu_session_id))

        if not jobs:
            return JobSweepResult(swept_count=0, error_count=0, skipped_count=0)

        swept = 0
        errored = 0
        skipped = 0
        truncated_reason = reason[:500]

        for job in jobs:
            try:
                async with self._session_factory() as ts_session:
                    ts = JobStateTransitionService(
                        session=ts_session,
                        event_bus=self._event_bus,
                        billing_service=self._billing,
                    )
                    result = await ts.transition_to_failed(
                        job.id,
                        error_message=truncated_reason,
                        refund=True,
                        product_id=product_id,
                    )
                    # transition_to_failed is a no-op (returns current row) when
                    # the job is already terminal. Detect this by checking whether
                    # the completed_at was just written vs. already present.
                    # Using the rowcount-based path inside the transition service
                    # is not exposed; we instead treat a mismatch in error_message
                    # as "already terminal / not our transition". This is advisory
                    # only (counts go to logs, not to the user).
                    from src.core.enums import JobStatus

                    if str(result.status) == JobStatus.FAILED.value and (
                        result.error_message is not None
                        and result.error_message[:500] == truncated_reason[:500]
                    ):
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
