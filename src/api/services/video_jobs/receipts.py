"""Accepted-task receipts: never lose track of a task a provider accepted.

A provider accepts an async video task *inside* the still-open request
transaction, and nothing is durable until the route commits. Any failure in
between rolls back the job row and the reservation while the task keeps running
upstream. The receipt closes that window without changing the transaction model:

1. Right after acceptance the task id is inserted and **committed on a dedicated
   small connection pool** (``receipt_db``), never the request pool — N
   concurrent submits each holding a request connection would otherwise starve
   waiting for a second one.
2. The request transaction then ``DELETE``s that row. The delete holds the row
   lock until the request ends: a commit removes the receipt atomically with the
   job; a rollback brings it back, unlocked — an orphan by construction.
3. The reconciler reads with ``FOR UPDATE SKIP LOCKED``, so it can never touch
   an in-flight request's receipt. The table holds only orphans.

An orphan is cancelled or absorbed, **never charged**: the user received a 500
and no ledger entry survived.
"""

from __future__ import annotations

import asyncio
import contextlib
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

import structlog
from sqlalchemy.exc import SQLAlchemyError

from src.api.schemas.ops_events import (
    OpsEventType,
    VideoTaskOrphanedOpsPayload,
    VideoTaskReceiptUnrecordedOpsPayload,
)
from src.api.services.video_jobs.protocol import OrphanCancelOutcome
from src.core.enums import AcceptedTaskReceiptStatus
from src.db.repositories.accepted_task_receipt import AcceptedTaskReceiptRepository
from src.db.session import DatabaseManager

if TYPE_CHECKING:
    from uuid import UUID

    from sqlalchemy.ext.asyncio import AsyncSession

    from src.api.services.ops_event_bus import OpsEventBus
    from src.api.services.video_jobs.protocol import OrphanTaskCompensator
    from src.core.config import Settings
    from src.core.enums import Provider
    from src.db.models.accepted_task_receipt import AcceptedTaskReceipt

logger = structlog.get_logger(__name__)

_BACKOFF_SECONDS: Final[tuple[float, ...]] = (0.05, 0.2, 0.5)
_RETRYABLE: Final[tuple[type[Exception], ...]] = (SQLAlchemyError, OSError, TimeoutError)
_IMMEDIATE_CANCEL_TIMEOUT_SECONDS: Final[float] = 5.0


@dataclass(frozen=True, slots=True)
class AcceptedTask:
    """What the job service knows the instant a provider accepts a task."""

    provider: Provider
    external_task_id: str
    job_id: UUID
    user_id: UUID
    product_id: str
    model: str
    token_cost: int


async def publish_orphan_resolved(
    ops_event_bus: OpsEventBus, receipt: AcceptedTaskReceipt, *, resolution: str
) -> None:
    """Publish ``VIDEO_TASK_ORPHANED`` for a receipt's first resolution (never raises)."""
    await ops_event_bus.publish(
        event_type=OpsEventType.VIDEO_TASK_ORPHANED,
        product_id=receipt.product_id,
        payload=VideoTaskOrphanedOpsPayload(
            provider=receipt.provider,
            external_task_id=receipt.external_task_id,
            job_id=receipt.job_id,
            user_id=receipt.user_id,
            model=receipt.model,
            token_cost=receipt.token_cost,
            resolution=resolution,
            upstream_status=receipt.upstream_status,
        ),
    )


class AcceptedTaskReceipts:
    """Writes, claims and resolves accepted-task receipts."""

    def __init__(
        self,
        receipt_db: DatabaseManager,
        ops_event_bus: OpsEventBus,
        *,
        record_attempts: int,
    ) -> None:
        """Wire the store.

        Args:
            receipt_db: A **second** ``DatabaseManager`` with its own small pool:
                the receipt write is autonomous while the request holds a
                connection from the main pool.
            ops_event_bus: Ops event bus.
            record_attempts: Tries per ``record`` call before giving up.
        """
        self._db = receipt_db
        self._ops = ops_event_bus
        self._attempts = max(1, record_attempts)

    async def record(self, task: AcceptedTask) -> bool:
        """Insert and commit the receipt on ``receipt_db``.

        Idempotent (``ON CONFLICT DO NOTHING``) and shielded: a cancelled caller
        still gets the write finished. Never raises, except ``CancelledError``
        once the shielded write has completed.

        Returns:
            ``True`` once the receipt is durable; ``False`` after the final
            failure, which is logged at ERROR and published to ops with the task
            id so an operator can cancel it by hand.
        """
        inner = asyncio.ensure_future(self._record_with_retries(task))
        try:
            return await asyncio.shield(inner)
        except asyncio.CancelledError:
            with contextlib.suppress(Exception, asyncio.CancelledError):
                await asyncio.shield(inner)
            raise

    async def _record_with_retries(self, task: AcceptedTask) -> bool:
        last_error: BaseException | None = None
        for attempt in range(self._attempts):
            try:
                async with self._db.session() as session:
                    await AcceptedTaskReceiptRepository(session).insert(
                        provider=task.provider.value,
                        external_task_id=task.external_task_id,
                        job_id=task.job_id,
                        user_id=task.user_id,
                        product_id=task.product_id,
                        model=task.model,
                        token_cost=task.token_cost,
                    )
            except _RETRYABLE as exc:
                last_error = exc
                logger.warning(
                    "video_receipt.record_retry",
                    provider=task.provider.value,
                    job_id=str(task.job_id),
                    attempt=attempt + 1,
                    error_type=type(exc).__name__,
                )
                if attempt + 1 < self._attempts:
                    await asyncio.sleep(_BACKOFF_SECONDS[min(attempt, len(_BACKOFF_SECONDS) - 1)])
            else:
                logger.info(
                    "video_receipt.recorded",
                    provider=task.provider.value,
                    job_id=str(task.job_id),
                    external_task_id=task.external_task_id,
                )
                return True

        logger.error(
            "video_receipt.unrecorded",
            provider=task.provider.value,
            job_id=str(task.job_id),
            external_task_id=task.external_task_id,
            user_id=str(task.user_id),
            product_id=task.product_id,
            error_type=type(last_error).__name__ if last_error is not None else None,
        )
        await self._ops.publish(
            event_type=OpsEventType.VIDEO_TASK_RECEIPT_UNRECORDED,
            product_id=task.product_id,
            payload=VideoTaskReceiptUnrecordedOpsPayload(
                provider=task.provider.value,
                external_task_id=task.external_task_id,
                job_id=task.job_id,
                user_id=task.user_id,
                model=task.model,
            ),
        )
        return False

    @staticmethod
    async def claim_for_request(
        session: AsyncSession, *, provider: Provider, external_task_id: str
    ) -> None:
        """Delete the receipt on the *request* session, locking it until the request ends.

        Requires ``READ COMMITTED`` (the PostgreSQL default and ours) so this
        statement sees the receipt committed a moment earlier on another
        connection; under ``REPEATABLE READ`` the snapshot predates it and the
        delete would match nothing.
        """
        deleted = await AcceptedTaskReceiptRepository(session).delete_pending(
            provider=provider.value, external_task_id=external_task_id
        )
        if deleted == 0:
            logger.error(
                "video_receipt.claim_missing",
                provider=provider.value,
                external_task_id=external_task_id,
            )

    async def resolve(
        self,
        *,
        provider: Provider,
        external_task_id: str,
        status: AcceptedTaskReceiptStatus,
        upstream_status: str | None,
    ) -> None:
        """Resolve a pending receipt and publish ``VIDEO_TASK_ORPHANED`` once. Never raises."""
        try:
            async with self._db.session() as session:
                receipt = await AcceptedTaskReceiptRepository(session).resolve(
                    provider=provider.value,
                    external_task_id=external_task_id,
                    status=status,
                    upstream_status=upstream_status,
                )
            if receipt is not None:
                await publish_orphan_resolved(self._ops, receipt, resolution=status.value)
        except Exception:
            logger.exception(
                "video_receipt.resolve_failed",
                provider=provider.value,
                external_task_id=external_task_id,
            )


async def record_acceptance(
    receipts: AcceptedTaskReceipts, session: AsyncSession, task: AcceptedTask
) -> bool:
    """Record the receipt, then claim it under the request transaction's lock.

    The single entry point both job services call right after a provider accepts
    a task. When the receipt could not be recorded nothing is claimed and the
    saga continues: failing the request would not help, since the main
    transaction might still commit with the task id stored on the job.
    """
    recorded = await receipts.record(task)
    if recorded:
        await receipts.claim_for_request(
            session, provider=task.provider, external_task_id=task.external_task_id
        )
    return recorded


async def cancel_orphan_immediately(
    *,
    receipts: AcceptedTaskReceipts,
    compensator: OrphanTaskCompensator,
    session: AsyncSession,
    task: AcceptedTask,
) -> None:
    """Best-effort cancel of a task whose post-acceptance persistence just failed.

    Any exception after acceptance inside a job service propagates and the
    generation saga rolls back, so the task is certainly orphaned and its first
    seconds are the best chance it is still cancellable. Every failure here is
    logged and swallowed: the caller re-raises the original error.

    The request transaction is rolled back *before* the receipt is resolved: it
    still holds the receipt's row lock, and resolving on another connection would
    wait on the very rollback this call is blocking. The saga rolls back anyway.
    """
    outcome: OrphanCancelOutcome | None = None
    try:
        outcome = await asyncio.shield(
            asyncio.wait_for(
                compensator.cancel(task.external_task_id),
                timeout=_IMMEDIATE_CANCEL_TIMEOUT_SECONDS,
            )
        )
    except Exception as exc:
        logger.warning(
            "video_receipt.immediate_cancel_failed",
            provider=task.provider.value,
            job_id=str(task.job_id),
            external_task_id=task.external_task_id,
            error_type=type(exc).__name__,
        )
    with contextlib.suppress(Exception):
        await session.rollback()
    if outcome is OrphanCancelOutcome.CANCELLED:
        await receipts.resolve(
            provider=task.provider,
            external_task_id=task.external_task_id,
            status=AcceptedTaskReceiptStatus.CANCELLED,
            upstream_status="CANCELED",
        )
    logger.info(
        "video_receipt.immediate_cancel",
        provider=task.provider.value,
        job_id=str(task.job_id),
        external_task_id=task.external_task_id,
        outcome=outcome.value if outcome is not None else None,
    )


def build_accepted_task_receipts(
    settings: Settings, ops_event_bus: OpsEventBus
) -> tuple[DatabaseManager, AcceptedTaskReceipts]:
    """Build the dedicated receipt pool and the store on top of it.

    The caller owns the returned ``DatabaseManager`` and must ``close()`` it on
    shutdown next to the main manager. The pool is deliberately small and
    separate: the receipt write is autonomous while the request holds a
    connection from the main pool, and sharing it would starve under a burst.
    """
    receipt_db = DatabaseManager(
        settings.database_url,
        pool_size=settings.video_task_receipt_db_pool_size,
        max_overflow=settings.video_task_receipt_db_max_overflow,
        pool_timeout=settings.video_task_receipt_db_pool_timeout_seconds,
        echo=settings.db_echo,
    )
    receipts = AcceptedTaskReceipts(
        receipt_db,
        ops_event_bus,
        record_attempts=settings.video_task_receipt_record_attempts,
    )
    return receipt_db, receipts
