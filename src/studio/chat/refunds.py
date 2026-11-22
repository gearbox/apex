"""Refunding a failed chat job's reservation - the one place a chat job's money comes back.

Used by the task supervisor's fail paths and by the sweep, always **in the same transaction as the
guarded update to ``failed``** and only when that update hit the row: that is what makes a refund
happen exactly once, whichever of the task and the sweep got there first.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from uuid import UUID

    from sqlalchemy.ext.asyncio import AsyncSession

    from src.api.services.billing import BalanceEvent, BillingService

__all__ = ["refund_reservation"]


async def refund_reservation(
    billing: BillingService,
    session: AsyncSession,
    *,
    job_id: UUID,
    price: int,
    product_id: str,
    user_id: UUID,
    reason: str,
    description: str,
) -> BalanceEvent | None:
    """Refund a job's reservation in full; nothing is written for a free job (``price == 0``).

    Args:
        billing: The ledger.
        session: The transaction that marked the job ``failed``.
        job_id: The ledger's ``job_id``: the artist message's or the draft's id.
        price: What the job reserved.
        product_id: Product scope.
        user_id: Owner scope.
        reason: The failure code, recorded in the ledger metadata.
        description: The ledger line.

    Returns:
        The balance event to publish **after** the transaction commits, or ``None``.
    """
    if price <= 0:
        return None
    refunded = await billing.refund(
        job_id,
        description=description,
        metadata={"studio_chat_job_id": str(job_id), "reason": reason},
        session=session,
        product_id=product_id,
        user_id=user_id,
    )
    return refunded.event
