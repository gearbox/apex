"""Integration tests for Phase 1F billing paths against a real PostgreSQL schema.

These tests exercise the specific invariants that Phase 1F introduces:

1. ``check_and_reserve(..., job_id=<gpu_session.id>, ...)`` — the base reservation.
   Before dropping the FK on ``token_transactions.job_id``, this insert would
   raise ``ForeignKeyViolationError`` because ``gpu_session.id`` is not a valid
   ``generation_jobs.id``. This is the regression test that would have caught
   the bug discovered in Phase 1F round 2.

2. ``check_and_reserve(..., job_id=None, ...)`` — the overage debit. Must be
   allowed to carry a NULL ``job_id`` with the parent session link in metadata.

3. ``partial_refund`` against a debit inserted without a generation job must
   succeed and follow the cumulative-refund invariant.

Kept deliberately narrow and DB-facing: no mocks of billing or repo internals.
"""

from __future__ import annotations

from uuid import uuid4

import pytest
import pytest_asyncio
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.services.billing import BillingService, RefundNotEligibleError
from src.db.models.billing import TokenTransaction
from src.db.repositories.billing import BillingRepository

pytestmark = pytest.mark.asyncio


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture
async def billing_service() -> BillingService:
    """BillingService instance without EventBus (SSE is out of scope for DB integration)."""
    return BillingService(event_bus=None)


async def _seed_balance(session: AsyncSession, account_id, amount: int) -> None:
    """Directly insert a credit transaction so the account has balance to debit from."""
    txn = TokenTransaction(
        id=uuid4(),
        account_id=account_id,
        transaction_type="credit",
        amount=amount,
        balance_after=amount,
        product_id="vex",
    )
    session.add(txn)
    await session.flush()


# ---------------------------------------------------------------------------
# The critical regression test — GPU session reservation with non-job job_id
# ---------------------------------------------------------------------------


async def test_reservation_accepts_non_job_uuid_as_job_id(
    billing_service: BillingService,
    billing_repo: BillingRepository,
    db_session: AsyncSession,
    make_user,
    make_token_account,
) -> None:
    """Phase 1F regression: ``job_id`` may be a gpu_sessions.id, not a generation_jobs.id.

    Before dropping the FK constraint ``token_transactions_job_id_fkey`` (which
    referenced ``generation_jobs.id``), this insert would have raised
    ``ForeignKeyViolationError`` on the first production start_session call.
    """
    user = await make_user(email=f"gpu-resv-{uuid4().hex[:6]}@example.com")
    account = await make_token_account(account_type="personal", user=user)
    await _seed_balance(db_session, account.id, 1000)

    # A UUID that is explicitly NOT in generation_jobs — mimics GpuSession.id.
    fake_session_id = uuid4()

    txn = await billing_service.check_and_reserve(
        account.id,
        500,
        fake_session_id,
        metadata={
            "type": "gpu_session_base",
            "model_type": "aisha-image",
            "bundle_name": "wan_2.2_i2v",
        },
        session=db_session,
        product_id="vex",
        user_id=user.id,
    )

    # Reservation must have succeeded and be retrievable via the job lookup.
    assert txn.amount == -500
    assert txn.job_id == fake_session_id
    fetched = await billing_repo.get_debit_for_job(fake_session_id)
    assert fetched is not None
    assert fetched.id == txn.id


async def test_overage_debit_with_null_job_id_succeeds(
    billing_service: BillingService,
    db_session: AsyncSession,
    make_user,
    make_token_account,
) -> None:
    """GPU session overage debits carry ``job_id=None`` with a metadata link.

    This keeps the one-debit-per-job invariant on the base reservation while
    still creating the overage charge. The DB must accept a NULL ``job_id``
    (which was already true — this test just locks it down against schema regressions).
    """
    user = await make_user(email=f"gpu-overage-{uuid4().hex[:6]}@example.com")
    account = await make_token_account(account_type="personal", user=user)
    await _seed_balance(db_session, account.id, 1000)
    parent_session_id = uuid4()

    txn = await billing_service.check_and_reserve(
        account.id,
        100,
        None,  # job_id=None for non-job billing events
        metadata={
            "type": "gpu_session_overage",
            "session_id": str(parent_session_id),
            "billable_minutes": 6,
        },
        session=db_session,
        product_id="vex",
        user_id=user.id,
    )

    assert txn.amount == -100
    assert txn.job_id is None
    assert txn.metadata_["session_id"] == str(parent_session_id)


async def test_partial_refund_against_non_job_reservation(
    billing_service: BillingService,
    billing_repo: BillingRepository,
    db_session: AsyncSession,
    make_user,
    make_token_account,
) -> None:
    """End-to-end: reserve tokens against a fake session id, then partial-refund.

    Exercises the full Phase 1F underage path: check_and_reserve creates the
    base debit with a non-job job_id; partial_refund looks it up via
    get_debit_for_job(job_id) and creates a REFUND txn for a portion of the amount.
    """
    user = await make_user(email=f"gpu-refund-{uuid4().hex[:6]}@example.com")
    account = await make_token_account(account_type="personal", user=user)
    await _seed_balance(db_session, account.id, 1000)
    session_id = uuid4()

    # Reserve 500
    await billing_service.check_and_reserve(
        account.id,
        500,
        session_id,
        metadata={"type": "gpu_session_base"},
        session=db_session,
        product_id="vex",
        user_id=user.id,
    )

    # Refund 100 of it (session used only 4 min out of 5-min reservation, say)
    refund_txn = await billing_service.partial_refund(
        session_id,
        100,
        description="GPU session partial refund: used 4min, reserved 5min minimum",
        session=db_session,
        product_id="vex",
        user_id=user.id,
    )

    assert refund_txn.amount == 100
    assert refund_txn.job_id == session_id
    # Balance = 1000 - 500 + 100 = 600
    balance = await billing_repo.get_balance(account.id)
    assert balance == 600


async def test_partial_refund_cumulative_invariant_enforced_on_real_db(
    billing_service: BillingService,
    db_session: AsyncSession,
    make_user,
    make_token_account,
) -> None:
    """Round-1 bug regression: two partial_refund calls on the same job must
    respect the cumulative cap.
    """
    user = await make_user(email=f"gpu-cumul-{uuid4().hex[:6]}@example.com")
    account = await make_token_account(account_type="personal", user=user)
    await _seed_balance(db_session, account.id, 1000)
    session_id = uuid4()

    await billing_service.check_and_reserve(
        account.id,
        500,
        session_id,
        metadata={"type": "gpu_session_base"},
        session=db_session,
        product_id="vex",
        user_id=user.id,
    )

    # First partial refund: 300 — OK
    await billing_service.partial_refund(
        session_id,
        300,
        description="refund 1",
        session=db_session,
        product_id="vex",
        user_id=user.id,
    )

    # Second partial refund: 250 would bring cumulative to 550 > 500 original — reject
    with pytest.raises(RefundNotEligibleError, match="would exceed original debit"):
        await billing_service.partial_refund(
            session_id,
            250,
            description="refund 2 (should fail)",
            session=db_session,
            product_id="vex",
            user_id=user.id,
        )

    # Third call at the boundary — 200 brings cumulative to exactly 500, allowed
    await billing_service.partial_refund(
        session_id,
        200,
        description="refund 3 at boundary",
        session=db_session,
        product_id="vex",
        user_id=user.id,
    )

    # Verify the ledger: one DEBIT of -500, two REFUNDs totaling +500 → balance back to seed
    result = await db_session.execute(
        select(TokenTransaction)
        .where(TokenTransaction.account_id == account.id)
        .order_by(TokenTransaction.created_at)
    )
    txns = result.scalars().all()
    # seed credit + 1 debit + 2 refunds = 4
    assert len(txns) == 4
    refund_amounts = sorted(t.amount for t in txns if t.transaction_type == "refund")
    assert refund_amounts == [200, 300]
