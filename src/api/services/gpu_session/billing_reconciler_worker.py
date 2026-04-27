"""Background worker: reconciles sessions with NULL billing_finalized_at.

When GpuSessionService._finalize_billing fails both of its in-line retry
attempts, the session is left in status='stopped' with
billing_finalized_at NULL and 'gpu_session.billing.finalization_deferred'
logged at ERROR. This worker periodically picks up those sessions and
re-invokes _finalize_billing.

Runs once every ``settings.billing_reconciler_interval_minutes`` (default 10).
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

import structlog

from src.db.models.gpu_session import GpuSession
from src.db.repositories.gpu_session import GpuSessionRepository

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

    from src.api.services.gpu_session.service import GpuSessionService
    from src.core.config import Settings

logger = structlog.get_logger(__name__)


class BillingReconcilerWorker:
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
    ) -> None:
        self._session_factory = session_factory
        self._service = gpu_session_service
        self._settings = settings
        self._running = False
        self._task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        """Start the reconciler worker."""
        if self._running or self._task is not None:
            return
        self._running = True
        self._task = asyncio.create_task(self._run_loop())
        logger.info(
            "billing_reconciler_worker.started",
            interval_minutes=self._settings.billing_reconciler_interval_minutes,
            grace_period_minutes=self._settings.billing_reconciler_grace_period_minutes,
            quarantine_threshold=self._settings.billing_reconciler_quarantine_threshold,
        )

    async def stop(self) -> None:
        """Stop the reconciler worker."""
        if not self._running:
            return
        self._running = False
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None
        logger.info("billing_reconciler_worker.stopped")

    async def _run_loop(self) -> None:
        """Main periodic loop."""
        while self._running:
            try:
                await self._sweep_once()
            except Exception:
                logger.exception("billing_reconciler_worker.sweep_error")
            await asyncio.sleep(self._settings.billing_reconciler_interval_minutes * 60)

    async def _sweep_once(self) -> None:
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

        # 2. Process each. _finalize_billing handles its own transaction
        #    boundaries via the service's session_factory; we don't pass
        #    our session in.
        for session_row in candidates:
            finalized = False
            try:
                await self._service._finalize_billing(session_row)

                # Re-check the column to determine whether the inner method
                # succeeded (billing_finalized_at stamped) or failed silently
                # (column still NULL).
                async with self._session_factory() as db:
                    repo = GpuSessionRepository(db)
                    refreshed = await repo.get_by_id(session_row.id)

                if refreshed is not None and refreshed.billing_finalized_at is not None:
                    finalized = True
                    reconciled += 1
                    logger.info(
                        "billing_reconciler.session_reconciled",
                        session_id=str(session_row.id),
                        attempts_before=session_row.billing_finalization_attempts,
                    )
            except Exception:
                # A worker-level exception (DB connection, etc.) shouldn't
                # poison the sweep. Log and move on; next sweep tries again.
                logger.exception(
                    "billing_reconciler.session_error",
                    session_id=str(session_row.id),
                )

            if not finalized:
                still_failing += 1
                try:
                    new_count = await self._bump_and_check_quarantine(session_row)
                    if new_count >= self._settings.billing_reconciler_quarantine_threshold:
                        quarantined += 1
                        logger.error(
                            "billing_reconciler.session_quarantined",
                            session_id=str(session_row.id),
                            attempts=new_count,
                            stopped_at=session_row.stopped_at.isoformat()
                            if session_row.stopped_at
                            else None,
                            quarantine=True,
                        )
                except Exception:
                    logger.exception(
                        "billing_reconciler.bump_error",
                        session_id=str(session_row.id),
                    )

        elapsed_ms = int((time.monotonic() - started) * 1000)
        logger.info(
            "billing_reconciler.sweep.done",
            candidates=len(candidates),
            reconciled=reconciled,
            still_failing=still_failing,
            quarantined=quarantined,
            duration_ms=elapsed_ms,
        )

    async def _bump_and_check_quarantine(self, session_row: GpuSession) -> int:
        """Increment attempt counter; return the new value."""
        async with self._session_factory() as db:
            repo = GpuSessionRepository(db)
            new_count = await repo.increment_billing_finalization_attempts(session_row.id)
            await db.commit()
        return new_count
