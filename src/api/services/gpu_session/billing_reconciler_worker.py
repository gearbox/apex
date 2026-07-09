"""Background worker: reconciles sessions with NULL billing_finalized_at.

When GpuSessionService._finalize_billing fails both of its in-line retry
attempts, the session is left in status='stopped' with
billing_finalized_at NULL and 'gpu_session.billing.finalization_deferred'
logged at ERROR. This worker periodically picks up those sessions and
re-invokes _finalize_billing.

Runs once every ``settings.billing_reconciler_interval_minutes`` (default 10).
"""

from __future__ import annotations

import time
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

import structlog

from src.db.repositories.gpu_session import GpuSessionRepository
from src.workers.base import PeriodicWorker

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

    from src.api.services.gpu_session.service import GpuSessionService
    from src.core.config import Settings
    from src.db.models.gpu_session import GpuSession

logger = structlog.get_logger(__name__)


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
    ) -> None:
        super().__init__(
            name="billing_reconciler",
            interval_seconds=settings.billing_reconciler_interval_minutes * 60,
            initial_delay_seconds=30.0,
            jitter_seconds=10.0,
            redis_enabled=redis_enabled,
        )
        self._session_factory = session_factory
        self._service = gpu_session_service
        self._settings = settings

    async def run_once(self) -> None:
        """One reconciliation sweep."""
        started = time.monotonic()
        grace_cutoff = datetime.now(UTC) - timedelta(
            minutes=self._settings.billing_reconciler_grace_period_minutes
        )

        # 1. Pull candidates in one short transaction.
        async with self._session_factory() as db:
            repo = GpuSessionRepository(db)
            candidates = await repo.list_pending_billing_finalization(
                grace_cutoff=grace_cutoff,
                limit=self._settings.billing_reconciler_max_per_sweep,
            )

        if not candidates:
            logger.debug("billing_reconciler.sweep.no_candidates")
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

        elapsed_ms = int((time.monotonic() - started) * 1000)
        logger.info(
            "billing_reconciler.sweep.done",
            candidates=len(candidates),
            reconciled=reconciled,
            still_failing=still_failing,
            quarantined=quarantined,
            duration_ms=elapsed_ms,
        )

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
        try:
            new_count = await self._bump_and_check_quarantine(session_row)
        except Exception:
            logger.exception(
                "billing_reconciler.bump_error",
                session_id=str(session_row.id),
            )
            return "still_failing"

        if new_count >= self._settings.billing_reconciler_quarantine_threshold:
            logger.error(
                "billing_reconciler.session_quarantined",
                session_id=str(session_row.id),
                attempts=new_count,
                stopped_at=session_row.stopped_at.isoformat() if session_row.stopped_at else None,
                quarantine=True,
            )
            return "quarantined"

        return "still_failing"

    async def _bump_and_check_quarantine(self, session_row: GpuSession) -> int:
        """Increment attempt counter; return the new value."""
        async with self._session_factory() as db:
            repo = GpuSessionRepository(db)
            new_count = await repo.increment_billing_finalization_attempts(session_row.id)
            await db.commit()
        return new_count
