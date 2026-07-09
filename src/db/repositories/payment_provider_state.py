"""Persistence for per-product payment provider runtime state."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert

from src.core.uid import new_id
from src.db.models.billing import PaymentProviderState

if TYPE_CHECKING:
    from collections.abc import Sequence
    from uuid import UUID

    from sqlalchemy.ext.asyncio import AsyncSession

    from src.core.product import PaymentProvider


class PaymentProviderStateRepository:
    """Atomic provider-state queries and upserts."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def get_states(self, product_id: str) -> Sequence[PaymentProviderState]:
        result = await self._session.execute(
            select(PaymentProviderState).where(PaymentProviderState.product_id == product_id)
        )
        return result.scalars().all()

    async def upsert_state(
        self,
        product_id: str,
        provider: PaymentProvider,
        *,
        is_enabled: bool | None = None,
        display_order: int | None = None,
        updated_by: UUID | None,
    ) -> PaymentProviderState:
        values = {
            "id": new_id(),
            "product_id": product_id,
            "provider": provider.value,
            "is_enabled": True if is_enabled is None else is_enabled,
            "display_order": 0 if display_order is None else display_order,
            "updated_by": updated_by,
            "updated_at": datetime.now(UTC),
        }
        update_values: dict[str, object] = {
            "updated_by": updated_by,
            "updated_at": values["updated_at"],
        }
        if is_enabled is not None:
            update_values["is_enabled"] = is_enabled
        if display_order is not None:
            update_values["display_order"] = display_order

        statement = (
            insert(PaymentProviderState)
            .values(**values)
            .on_conflict_do_update(
                constraint="uq_payment_provider_state",
                set_=update_values,
            )
            .returning(PaymentProviderState)
        )
        result = await self._session.execute(statement)
        state = result.scalar_one()
        # ON CONFLICT may return an identity already present in the session;
        # refresh so callers see the values written by the UPDATE branch.
        await self._session.refresh(state)
        return state
