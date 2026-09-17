"""Background worker: reconciles sessions with NULL billing_finalized_at, plus
'failed' pre-active sessions whose base reservation was never refunded (S3).

When GpuSessionService._finalize_billing fails both of its in-line retry
attempts, the session is left in status='stopped' with
billing_finalized_at NULL and 'gpu_session.billing.finalization_deferred'
logged at ERROR. This worker periodically picks up those sessions and
re-invokes _finalize_billing.

Separately (S3, round-2 remediation): GpuSessionService.fail_pre_active_session
and GpuProvisioningWorker._mark_failed both swallow a refund failure and
continue, leaving a 'failed' session's base reservation debited with nothing
to retry it — billing_finalized_at doesn't apply to that status, so the
finalization sweep above never sees these. This worker's second pass picks up
GpuSessionRepository.list_pending_refund_reconciliation candidates and
re-invokes GpuSessionService.reconcile_pending_refund, which treats an
already-completed refund (RefundNotEligibleError) as success.

Runs once every ``settings.billing_reconciler_interval_minutes`` (default 10).

**Backoff, not exclusion (X1, round-5 remediation).** A session that keeps
failing reconciliation is never dropped from either sweep query — it stays
retryable forever, per the documented contract that billing_finalized_at
stays NULL so the worker keeps retrying once the underlying issue is fixed.
Round-3's T7 instead excluded any row past ``quarantine_threshold`` attempts,
which silently stopped reconciling it forever (the only externally visible
signal, an ERROR log, also stopped — indistinguishable from recovery). This
version paces retries with per-session exponential backoff
(``compute_next_attempt_at``) written to
``GpuSession.billing_reconciliation_next_attempt_at`` instead: a persistently
failing row costs one attempt per backoff period rather than one per sweep,
which is what keeps it from flooding ops alerts or head-of-line-blocking
healthy candidates — the two properties the threshold predicate was actually
protecting — without ever making the row unreachable.
"""

from __future__ import annotations

import time
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

import structlog

from src.db.repositories.gpu_session import GpuSessionRepository
from src.workers.base import PeriodicWorker

if TYPE_CHECKING:
    from collections.abc import Callable

    from redis.asyncio import Redis
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

    from src.api.services.gpu_session.service import GpuSessionService
    from src.core.config import Settings
    from src.db.models.gpu_session import GpuSession

logger = structlog.get_logger(__name__)


def compute_next_attempt_at(
    attempts: int, *, now: datetime, base_minutes: int, cap_hours: int
) -> datetime:
    """Exponential backoff for the next reconciliation attempt: ``base * 2**(n-1)``, capped.

    ``attempts`` is the *new* (post-increment) attempt count, so the first
    failure (attempts=1) backs off by exactly ``base_minutes``. Pure function
    so the schedule (and its cap) can be unit-tested without touching a DB.

    The cap is applied to the plain-int minute count *before* a ``timedelta``
    is constructed: ``2 ** (attempts - 1)`` is unbounded (a session can fail
    indefinitely), and ``timedelta`` raises ``OverflowError`` once a delay
    this large is built directly — Python ints have no such limit, so the
    ``min()`` must happen first.
    """
    delay_minutes = min(base_minutes * (2 ** (attempts - 1)), cap_hours * 60)
    return now + timedelta(minutes=delay_minutes)


class BillingReconcilerWorker(PeriodicWorker):
    """Reconciles stopped GPU sessions whose billing finalization is pending.

    See module docstring. Lifecycle and loop shape mirror
    OrphanedTunnelCleanupWorker.
    """

    def __init__(
        self,
        *,
        session_factory: async_sessionmaker[AsyncSession],
        gpu_session_service: GpuSessionService,
        settings: Settings,
        redis_enabled: bool = False,
        redis_client_factory: Callable[[], Redis],
    ) -> None:
        super().__init__(
            name="billing_reconciler",
            interval_seconds=settings.billing_reconciler_interval_minutes * 60,
            initial_delay_seconds=30.0,
            jitter_seconds=10.0,
            redis_enabled=redis_enabled,
            redis_client_factory=redis_client_factory,
        )
        self._session_factory = session_factory
        self._service = gpu_session_service
        self._settings = settings

    async def run_once(self) -> None:
        """One reconciliation sweep: billing finalization, then refund reconciliation.

        The two candidate sets are disjoint by construction — finalization only
        ever matches status='stopped', refund reconciliation only ever matches
        status='failed' — so a session is never double-processed in one sweep.
        """
        started = time.monotonic()
        grace_cutoff = datetime.now(UTC) - timedelta(
            minutes=self._settings.billing_reconciler_grace_period_minutes
        )

        await self._sweep_finalization(grace_cutoff)
        await self._sweep_refund_reconciliation(grace_cutoff)

        elapsed_ms = int((time.monotonic() - started) * 1000)
        logger.debug("billing_reconciler.sweep.done", duration_ms=elapsed_ms)

    async def _sweep_finalization(self, grace_cutoff: datetime) -> None:
        """Retry billing_finalized_at IS NULL / status='stopped' sessions."""
        # 1. Pull candidates in one short transaction.
        async with self._session_factory() as db:
            repo = GpuSessionRepository(db)
            candidates = await repo.list_pending_billing_finalization(
                grace_cutoff=grace_cutoff,
                limit=self._settings.billing_reconciler_max_per_sweep,
                now=datetime.now(UTC),
            )

        if not candidates:
            logger.debug("billing_reconciler.finalization_sweep.no_candidates")
            return

        reconciled = 0
        still_failing = 0
        quarantined = 0

        # 2. Classify each candidate. _process_session encapsulates the
        #    finalize-or-bump decision so this loop stays linear.
        for session_row in candidates:
            outcome = await self._process_session(session_row)
            if outcome == "reconciled":
                reconciled += 1
            elif outcome == "quarantined":
                quarantined += 1
            else:  # still_failing
                still_failing += 1

        logger.info(
            "billing_reconciler.finalization_sweep.done",
            candidates=len(candidates),
            reconciled=reconciled,
            still_failing=still_failing,
            quarantined=quarantined,
        )

    async def _sweep_refund_reconciliation(self, grace_cutoff: datetime) -> None:
        """Retry the base-reservation refund for 'failed' pre-active sessions (S3)."""
        async with self._session_factory() as db:
            repo = GpuSessionRepository(db)
            candidates = await repo.list_pending_refund_reconciliation(
                grace_cutoff=grace_cutoff,
                limit=self._settings.billing_reconciler_max_per_sweep,
                now=datetime.now(UTC),
            )

        if not candidates:
            logger.debug("billing_reconciler.refund_sweep.no_candidates")
            return

        reconciled = 0
        still_failing = 0
        quarantined = 0

        for session_row in candidates:
            outcome = await self._process_refund_session(session_row)
            if outcome == "reconciled":
                reconciled += 1
            elif outcome == "quarantined":
                quarantined += 1
            else:  # still_failing
                still_failing += 1

        logger.info(
            "billing_reconciler.refund_sweep.done",
            candidates=len(candidates),
            reconciled=reconciled,
            still_failing=still_failing,
            quarantined=quarantined,
        )

    async def _process_refund_session(self, session_row: GpuSession) -> str:
        """Run refund reconciliation on one session and classify the outcome.

        Returns one of: ``"reconciled"``, ``"still_failing"``, ``"quarantined"``.
        Mirrors ``_process_session``'s shape, reusing the same
        ``billing_finalization_attempts`` counter/backoff schedule — a
        session is only ever a candidate for one of the two sweeps (see
        ``run_once``), so the shared counter can't conflate the two failure kinds.
        """
        try:
            success = await self._service.reconcile_pending_refund(session_row)
        except Exception:
            logger.exception(
                "billing_reconciler.refund_session_error",
                session_id=str(session_row.id),
            )
            success = False

        if success:
            logger.info(
                "billing_reconciler.refund_session_reconciled",
                session_id=str(session_row.id),
                attempts_before=session_row.billing_finalization_attempts,
            )
            return "reconciled"

        previous_count = session_row.billing_finalization_attempts
        try:
            new_count = await self._bump_and_check_quarantine(session_row)
        except Exception:
            logger.exception(
                "billing_reconciler.refund_bump_error",
                session_id=str(session_row.id),
            )
            return "still_failing"

        threshold = self._settings.billing_reconciler_quarantine_threshold
        if new_count >= threshold:
            log = logger.error if previous_count < threshold else logger.warning
            log(
                "billing_reconciler.refund_session_quarantined",
                session_id=str(session_row.id),
                attempts=new_count,
                quarantine=True,
            )
            return "quarantined"

        return "still_failing"

    async def _process_session(self, session_row: GpuSession) -> str:
        """Run finalize on one session and classify the outcome.

        Returns one of: ``"reconciled"``, ``"still_failing"``, ``"quarantined"``.

        Calls the service's public ``finalize_billing_for_session`` method,
        which returns True on success. On failure (or worker-level exception),
        bumps the attempt counter and emits the quarantine log if the
        threshold has been crossed.
        """
        try:
            success = await self._service.finalize_billing_for_session(session_row)
        except Exception:
            # A worker-level exception (DB connection, etc.) shouldn't poison
            # the sweep. Log and treat as still-failing so attempts get bumped.
            logger.exception(
                "billing_reconciler.session_error",
                session_id=str(session_row.id),
            )
            success = False

        if success:
            logger.info(
                "billing_reconciler.session_reconciled",
                session_id=str(session_row.id),
                attempts_before=session_row.billing_finalization_attempts,
            )
            return "reconciled"

        # Failed: bump the attempt counter; emit quarantine log if threshold hit.
        previous_count = session_row.billing_finalization_attempts
        try:
            new_count = await self._bump_and_check_quarantine(session_row)
        except Exception:
            logger.exception(
                "billing_reconciler.bump_error",
                session_id=str(session_row.id),
            )
            return "still_failing"

        threshold = self._settings.billing_reconciler_quarantine_threshold
        if new_count >= threshold:
            # ERROR exactly once, on crossing the threshold — past that, WARNING,
            # so a chronically failing session doesn't re-trigger the ops alert
            # every sweep forever (X1, round-5 remediation).
            log = logger.error if previous_count < threshold else logger.warning
            log(
                "billing_reconciler.session_quarantined",
                session_id=str(session_row.id),
                attempts=new_count,
                stopped_at=session_row.stopped_at.isoformat() if session_row.stopped_at else None,
                quarantine=True,
            )
            return "quarantined"

        return "still_failing"

    async def _bump_and_check_quarantine(self, session_row: GpuSession) -> int:
        """Bump, then calculate/store backoff from the returned DB count.

        The counter increment is atomic in SQL. Its returned value, not the
        candidate object's possibly stale attempt count, selects the backoff
        exponent; both writes commit together in this short transaction (Y2,
        round-6 remediation).
        """
        now = datetime.now(UTC)
        async with self._session_factory() as db:
            repo = GpuSessionRepository(db)
            new_count = await repo.increment_billing_finalization_attempts(session_row.id)
            next_attempt_at = compute_next_attempt_at(
                new_count,
                now=now,
                base_minutes=self._settings.billing_reconciler_backoff_base_minutes,
                cap_hours=self._settings.billing_reconciler_backoff_cap_hours,
            )
            await repo.set_billing_reconciliation_next_attempt_at(
                session_row.id, next_attempt_at=next_attempt_at
            )
            await db.commit()
        return new_count
