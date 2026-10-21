"""Compensation of orphaned video tasks (``accepted_task_receipts``).

An orphan is an upstream task a provider accepted whose request later rolled
back: the user got a 500 and no ledger entry survived. The reconciler therefore
**cancels or absorbs, and never charges** — it imports nothing from billing and
writes no ledger row.

Per receipt it observes first, then acts, outside any transaction:

* upstream ``CANCELLED``                        → ``cancelled``
* ``PENDING`` → cancel → cancelled              → ``cancelled``
* ``PENDING`` → refused / provider has no cancel → ``absorbed``
* anything else (running or finished), or gone  → ``absorbed`` with the status

One worker per provider, each with its own lease, so providers never block each
other and polling is not slowed by reconciliation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import structlog

from src.api.services.video_jobs.protocol import OrphanCancelOutcome, OrphanTaskState
from src.api.services.video_jobs.receipts import publish_orphan_resolved
from src.core.enums import AcceptedTaskReceiptStatus
from src.db.repositories.accepted_task_receipt import AcceptedTaskReceiptRepository
from src.workers.base import PeriodicWorker

if TYPE_CHECKING:
    from collections.abc import Callable

    from redis.asyncio import Redis

    from src.api.services.ops_event_bus import OpsEventBus
    from src.api.services.video_jobs.protocol import OrphanTaskCompensator
    from src.core.config import Settings
    from src.db.models.accepted_task_receipt import AcceptedTaskReceipt
    from src.db.session import DatabaseManager

logger = structlog.get_logger(__name__)

_UNKNOWN_STATUS = "unknown"


class AcceptedTaskReconciler:
    """Resolves one provider's orphaned receipts."""

    def __init__(
        self,
        receipt_db: DatabaseManager,
        compensator: OrphanTaskCompensator,
        ops_event_bus: OpsEventBus,
        *,
        grace_seconds: int,
        batch_size: int = 50,
    ) -> None:
        """Wire the reconciler.

        Args:
            receipt_db: Database manager the receipts live on.
            compensator: The provider's cancel/observe adapter.
            ops_event_bus: Ops event bus.
            grace_seconds: Minimum receipt age. It only has to cover the gap between
                the receipt commit and the request's ``DELETE``; the row lock, not
                this value, protects a slow request.
            batch_size: Receipts handled per tick.
        """
        self._db = receipt_db
        self._compensator = compensator
        self._ops = ops_event_bus
        self._grace_seconds = grace_seconds
        self._batch_size = batch_size

    async def run_once(self) -> int:
        """Reconcile one batch.

        Returns:
            The number of receipts resolved this tick.
        """
        provider = self._compensator.provider.value
        # Short transaction: the rows are only *selected* here (SKIP LOCKED skips
        # every in-flight request, which holds the delete lock). The lock is
        # released before any network call.
        async with self._db.session() as session:
            receipts = list(
                await AcceptedTaskReceiptRepository(session).lock_due_pending(
                    provider=provider,
                    grace_seconds=self._grace_seconds,
                    limit=self._batch_size,
                )
            )
        resolved = 0
        for receipt in receipts:
            try:
                if await self._reconcile_one(receipt):
                    resolved += 1
            except Exception:
                logger.exception(
                    "video_receipt.reconcile_failed",
                    provider=provider,
                    job_id=str(receipt.job_id),
                    external_task_id=receipt.external_task_id,
                )
        return resolved

    async def _reconcile_one(self, receipt: AcceptedTaskReceipt) -> bool:
        try:
            status, upstream = await self._decide(receipt)
        except self._compensator.transient_errors as exc:
            logger.warning(
                "video_receipt.reconcile_transient",
                **self._log_fields(receipt),
                error_type=type(exc).__name__,
            )
            async with self._db.session() as session:
                await AcceptedTaskReceiptRepository(session).mark_checked(
                    provider=receipt.provider, external_task_id=receipt.external_task_id
                )
            return False
        except Exception:
            # Permanent: retrying forever would only repeat it. The ops alert
            # carries the task id for a manual audit.
            logger.exception(
                "video_receipt.reconcile_permanent_failure", **self._log_fields(receipt)
            )
            status, upstream = AcceptedTaskReceiptStatus.ABSORBED, _UNKNOWN_STATUS
        return await self._resolve(receipt, status, upstream)

    async def _decide(
        self, receipt: AcceptedTaskReceipt
    ) -> tuple[AcceptedTaskReceiptStatus, str | None]:
        task_id = receipt.external_task_id
        observed = await self._compensator.observe(task_id)
        if observed.state is OrphanTaskState.CANCELLED:
            return AcceptedTaskReceiptStatus.CANCELLED, observed.upstream_status
        if observed.state is OrphanTaskState.PENDING:
            outcome = await self._compensator.cancel(task_id)
            if outcome is OrphanCancelOutcome.CANCELLED:
                return AcceptedTaskReceiptStatus.CANCELLED, observed.upstream_status
            return AcceptedTaskReceiptStatus.ABSORBED, observed.upstream_status
        if observed.state is OrphanTaskState.GONE:
            return AcceptedTaskReceiptStatus.ABSORBED, _UNKNOWN_STATUS
        return AcceptedTaskReceiptStatus.ABSORBED, observed.upstream_status

    async def _resolve(
        self,
        receipt: AcceptedTaskReceipt,
        status: AcceptedTaskReceiptStatus,
        upstream_status: str | None,
    ) -> bool:
        async with self._db.session() as session:
            # ``resolve`` re-checks ``status = 'pending'``: a concurrent resolution is a no-op.
            updated = await AcceptedTaskReceiptRepository(session).resolve(
                provider=receipt.provider,
                external_task_id=receipt.external_task_id,
                status=status,
                upstream_status=upstream_status,
            )
        if updated is None:
            return False
        event = (
            "video_receipt.orphan_cancelled"
            if status is AcceptedTaskReceiptStatus.CANCELLED
            else "video_receipt.orphan_absorbed"
        )
        logger.warning(event, **self._log_fields(receipt), upstream_status=upstream_status)
        await publish_orphan_resolved(self._ops, updated, resolution=status.value)
        return True

    @staticmethod
    def _log_fields(receipt: AcceptedTaskReceipt) -> dict[str, str]:
        return {
            "provider": receipt.provider,
            "job_id": str(receipt.job_id),
            "external_task_id": receipt.external_task_id,
            "user_id": str(receipt.user_id),
            "product_id": receipt.product_id,
        }


class AcceptedTaskReconcilerWorker(PeriodicWorker):
    """Periodic driver for one provider's ``AcceptedTaskReconciler``."""

    def __init__(
        self,
        *,
        name: str,
        reconciler: AcceptedTaskReconciler,
        interval_seconds: int,
        redis_enabled: bool,
        redis_client_factory: Callable[[], Redis],
    ) -> None:
        """Args:
        name: Worker (and leader-lease) name — the provider's ``RECEIPT_RECONCILER_NAME``.
        reconciler: The provider's reconciler.
        interval_seconds: Seconds between ticks.
        redis_enabled: Whether Redis is configured (enables the leader lease).
        redis_client_factory: Factory for the operational Redis client.
        """
        super().__init__(
            name=name,
            interval_seconds=interval_seconds,
            redis_enabled=redis_enabled,
            redis_client_factory=redis_client_factory,
        )
        self._reconciler = reconciler

    async def run_once(self) -> None:
        """Reconcile one batch."""
        await self._reconciler.run_once()


def build_receipt_reconciler_worker(
    *,
    name: str,
    receipt_db: DatabaseManager,
    compensator: OrphanTaskCompensator,
    ops_event_bus: OpsEventBus,
    settings: Settings,
    redis_enabled: bool,
    redis_client_factory: Callable[[], Redis],
) -> AcceptedTaskReconcilerWorker:
    """Build one provider's receipt reconciler worker.

    The single construction site shared by the in-process wiring and the standalone
    worker processes, so both agree on the lease name, grace and interval.

    Args:
        name: Worker (and leader-lease) name — the provider's ``RECEIPT_RECONCILER_NAME``.
        receipt_db: Database manager the receipts live on.
        compensator: The provider's cancel/observe adapter.
        ops_event_bus: Ops event bus.
        settings: Application settings (grace and interval).
        redis_enabled: Whether Redis is configured (enables the leader lease).
        redis_client_factory: Factory for the operational Redis client.

    Returns:
        A worker that has not been started.
    """
    return AcceptedTaskReconcilerWorker(
        name=name,
        reconciler=AcceptedTaskReconciler(
            receipt_db,
            compensator,
            ops_event_bus,
            grace_seconds=settings.video_task_receipt_grace_seconds,
        ),
        interval_seconds=settings.video_task_receipt_reconcile_interval_seconds,
        redis_enabled=redis_enabled,
        redis_client_factory=redis_client_factory,
    )
