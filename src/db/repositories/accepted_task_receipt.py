"""Data access for ``accepted_task_receipts`` (see ``AcceptedTaskReceipt``).

Plain-argument methods only: the receipt store and the reconciler own the
policy, this class owns the SQL (SRP). The session passed in decides which
connection a statement runs on — the store uses a dedicated pool for writes
and the *request* session for ``delete_pending``.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any, cast

from sqlalchemy import delete, select, text, update
from sqlalchemy.dialects.postgresql import insert as pg_insert

from src.core.enums import AcceptedTaskReceiptStatus
from src.core.uid import new_id
from src.db.models.accepted_task_receipt import AcceptedTaskReceipt

if TYPE_CHECKING:
    from collections.abc import Sequence
    from uuid import UUID

    from sqlalchemy.engine import CursorResult
    from sqlalchemy.ext.asyncio import AsyncSession


class AcceptedTaskReceiptRepository:
    """SQL for accepted-task receipts."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def insert(
        self,
        *,
        provider: str,
        external_task_id: str,
        job_id: UUID,
        user_id: UUID,
        product_id: str,
        model: str,
        token_cost: int,
    ) -> None:
        """Insert a pending receipt; a repeat for the same task is a no-op."""
        stmt = (
            pg_insert(AcceptedTaskReceipt)
            .values(
                id=new_id(),
                provider=provider,
                external_task_id=external_task_id,
                job_id=job_id,
                user_id=user_id,
                product_id=product_id,
                model=model,
                token_cost=token_cost,
                status=AcceptedTaskReceiptStatus.PENDING.value,
            )
            .on_conflict_do_nothing(index_elements=["provider", "external_task_id"])
        )
        await self._session.execute(stmt)

    async def delete_pending(self, *, provider: str, external_task_id: str) -> int:
        """Delete the pending receipt; holds its row lock until the session ends.

        Returns:
            Rows deleted (0 or 1).
        """
        result = cast(
            "CursorResult[Any]",
            await self._session.execute(
                delete(AcceptedTaskReceipt).where(
                    AcceptedTaskReceipt.provider == provider,
                    AcceptedTaskReceipt.external_task_id == external_task_id,
                    AcceptedTaskReceipt.status == AcceptedTaskReceiptStatus.PENDING.value,
                )
            ),
        )
        return int(result.rowcount)

    async def lock_due_pending(
        self, *, provider: str, grace_seconds: int, limit: int
    ) -> Sequence[AcceptedTaskReceipt]:
        """Select due pending receipts, skipping rows an in-flight request has locked."""
        cutoff = datetime.now(UTC) - timedelta(seconds=grace_seconds)
        stmt = (
            select(AcceptedTaskReceipt)
            .where(
                AcceptedTaskReceipt.provider == provider,
                AcceptedTaskReceipt.status == AcceptedTaskReceiptStatus.PENDING.value,
                AcceptedTaskReceipt.accepted_at < cutoff,
            )
            .order_by(AcceptedTaskReceipt.accepted_at)
            .limit(limit)
            .with_for_update(skip_locked=True)
        )
        return (await self._session.execute(stmt)).scalars().all()

    async def mark_checked(self, *, provider: str, external_task_id: str) -> None:
        """Record a transient reconcile attempt; the receipt stays pending."""
        await self._session.execute(
            update(AcceptedTaskReceipt)
            .where(
                AcceptedTaskReceipt.provider == provider,
                AcceptedTaskReceipt.external_task_id == external_task_id,
                AcceptedTaskReceipt.status == AcceptedTaskReceiptStatus.PENDING.value,
            )
            .values(last_checked_at=datetime.now(UTC))
        )

    async def resolve(
        self,
        *,
        provider: str,
        external_task_id: str,
        status: AcceptedTaskReceiptStatus,
        upstream_status: str | None,
    ) -> AcceptedTaskReceipt | None:
        """Resolve a pending receipt exactly once.

        Returns:
            The updated row on the first resolution, ``None`` when it was not
            pending (already resolved, or deleted by a committed request).
        """
        # Bounded wait for a row an in-flight request may still hold (never indefinite).
        await self._session.execute(text("SET LOCAL lock_timeout = '2s'"))
        now = datetime.now(UTC)
        result = await self._session.execute(
            update(AcceptedTaskReceipt)
            .where(
                AcceptedTaskReceipt.provider == provider,
                AcceptedTaskReceipt.external_task_id == external_task_id,
                AcceptedTaskReceipt.status == AcceptedTaskReceiptStatus.PENDING.value,
            )
            .values(
                status=status.value,
                upstream_status=upstream_status,
                last_checked_at=now,
                resolved_at=now,
            )
            .returning(AcceptedTaskReceipt)
        )
        return result.scalar_one_or_none()
