"""PostgreSQL provider-state upsert behavior."""

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.product import PaymentProvider
from src.db.models.billing import PaymentProviderState
from src.db.repositories.payment_provider_state import PaymentProviderStateRepository

pytestmark = pytest.mark.asyncio


async def test_upsert_updates_one_row(db_session: AsyncSession, make_user) -> None:  # type: ignore[no-untyped-def]
    actor = await make_user()
    repository = PaymentProviderStateRepository(db_session)
    first = await repository.upsert_state(
        "vex",
        PaymentProvider.STRIPE,
        is_enabled=False,
        updated_by=actor.id,
    )
    second = await repository.upsert_state(
        "vex",
        PaymentProvider.STRIPE,
        is_enabled=True,
        display_order=4,
        updated_by=actor.id,
    )
    rows = (
        (
            await db_session.execute(
                select(PaymentProviderState).where(
                    PaymentProviderState.product_id == "vex",
                    PaymentProviderState.provider == PaymentProvider.STRIPE.value,
                )
            )
        )
        .scalars()
        .all()
    )
    assert first.id == second.id
    assert len(rows) == 1
    assert rows[0].is_enabled is True
    assert rows[0].display_order == 4
