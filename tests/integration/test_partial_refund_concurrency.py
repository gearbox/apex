"""Integration test for ``BillingService.partial_refund`` row-lock concurrency.

Round 4 / Issue 3: two concurrent ``partial_refund`` calls on the same
``job_id`` would, before the fix, both pass the cumulative-invariant check
(both read the same pre-insert ``sum_refunds_for_job`` snapshot) and both
commit, producing a cumulative over-refund.

The fix: ``BillingRepository.get_debit_for_job(for_update=True)`` acquires a
row-level lock on the debit, so concurrent partial-refund callers serialize.
This test verifies the serialization works against a real PostgreSQL.

Self-contained — does not use the standard ``db_session`` SAVEPOINT fixture
because that fixture only allocates one connection and we need two truly
independent connections to demonstrate the row lock. We commit setup data
directly and clean up via DELETE in ``finally``.
"""

from __future__ import annotations

import asyncio
from uuid import UUID, uuid4

import pytest
import pytest_asyncio
from sqlalchemy import delete, text
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession

from src.api.services.billing import BillingService, RefundNotEligibleError
from src.core.uid import new_id
from src.db.models.billing import TokenAccount, TokenTransaction
from src.db.models.user import User

pytestmark = pytest.mark.asyncio


@pytest_asyncio.fixture
async def billing_service() -> BillingService:
    return BillingService()


async def _seed_user_account_and_debit(
    engine: AsyncEngine, *, debit_amount: int, balance: int
) -> tuple[User, TokenAccount, TokenTransaction]:
    """Seed a user, account, and a single base-reservation debit. Commits."""
    user = User(
        id=new_id(),
        email=f"concurrency-{uuid4().hex[:8]}@example.com",
        password_hash="x" * 64,
        product_id="vex",
        is_active=True,
    )
    account = TokenAccount(
        id=new_id(),
        user_id=user.id,
        account_type="personal",
        product_id="vex",
    )
    # Seed the balance via a CREDIT, then post the DEBIT.
    credit = TokenTransaction(
        id=new_id(),
        account_id=account.id,
        transaction_type="credit",
        amount=balance,
        balance_after=balance,
        product_id="vex",
    )
    debit_job_id = new_id()
    debit = TokenTransaction(
        id=new_id(),
        account_id=account.id,
        transaction_type="debit",
        amount=-debit_amount,
        balance_after=balance - debit_amount,
        job_id=debit_job_id,
        product_id="vex",
        metadata_={"type": "gpu_session_base"},
    )

    async with AsyncSession(bind=engine, expire_on_commit=False) as session:
        session.add_all([user, account, credit, debit])
        await session.commit()
        # Re-fetch so caller has detached objects with stable values.
        await session.refresh(user)
        await session.refresh(account)
        await session.refresh(debit)
    return user, account, debit


async def _cleanup(engine: AsyncEngine, account_id, user_id) -> None:
    """Delete all rows seeded for the test (transactions, account, user).

    Uses ``session_replication_role = replica`` in the cleanup transaction so
    the ``token_transactions`` immutability trigger is bypassed for test
    teardown. The trigger correctly forbids mutation in normal app code.
    """
    async with AsyncSession(bind=engine, expire_on_commit=False) as session:
        # Disable user-defined triggers for this transaction only.
        await session.execute(text("SET LOCAL session_replication_role = 'replica'"))
        await session.execute(
            delete(TokenTransaction).where(TokenTransaction.account_id == account_id)
        )
        await session.execute(delete(TokenAccount).where(TokenAccount.id == account_id))
        await session.execute(delete(User).where(User.id == user_id))
        await session.commit()


async def test_partial_refund_concurrent_serializes_via_row_lock(
    billing_service: BillingService,
    db_engine: AsyncEngine,
) -> None:
    """Two concurrent partial_refund calls of 300 each against an original
    debit of 500 must not both succeed.

    Without the row lock, both callers read ``sum_refunds_for_job=0``, both
    pass ``0 + 300 <= 500``, both insert REFUND of 300 → cumulative 600 > 500.
    With the row lock, the second caller blocks on the first's lock, then
    sees ``sum_refunds_for_job=300`` after the first commits, and raises
    ``RefundNotEligibleError`` because ``300 + 300 > 500``.
    """
    user, account, debit = await _seed_user_account_and_debit(
        db_engine, debit_amount=500, balance=1000
    )
    assert debit.job_id is not None
    job_id: UUID = debit.job_id

    try:
        # Two independent sessions on two separate connections.
        async def attempt_refund(amount: int) -> str:
            async with (
                AsyncSession(bind=db_engine, expire_on_commit=False) as session,
                session.begin(),
            ):
                try:
                    await billing_service.partial_refund(
                        job_id,
                        amount,
                        description=f"concurrent attempt {amount}",
                        session=session,
                        product_id="vex",
                        user_id=user.id,
                    )
                except RefundNotEligibleError as exc:
                    return f"rejected: {exc}"
                else:
                    return "committed"

        # Run concurrently. The row lock serializes them: one wins, one fails
        # the cumulative invariant check.
        results = await asyncio.gather(
            attempt_refund(300),
            attempt_refund(300),
            return_exceptions=True,
        )

        committed = [r for r in results if r == "committed"]
        rejected = [r for r in results if isinstance(r, str) and r.startswith("rejected")]

        assert len(committed) == 1, f"Expected exactly 1 commit, got {results}"
        assert len(rejected) == 1, f"Expected exactly 1 rejection, got {results}"

        # Verify the ledger state matches the serialized outcome: exactly one
        # REFUND of 300, balance = 1000 - 500 + 300 = 800.
        async with AsyncSession(bind=db_engine, expire_on_commit=False) as session:
            result = await session.execute(
                text(
                    "SELECT COALESCE(SUM(amount), 0) FROM token_transactions "
                    "WHERE job_id = :jid AND transaction_type = 'refund'"
                ),
                {"jid": job_id},
            )
            total_refunded = result.scalar_one()
            assert total_refunded == 300, (
                f"Expected exactly 300 refunded (one of two concurrent 300s), "
                f"got {total_refunded}. The row lock did NOT prevent over-refund."
            )
    finally:
        await _cleanup(db_engine, account.id, user.id)


async def test_partial_refund_compatible_concurrent_amounts_both_succeed(
    billing_service: BillingService,
    db_engine: AsyncEngine,
) -> None:
    """When two concurrent refunds together fit under the cap, both succeed.

    Original debit 500. Two concurrent refunds of 200 each → 400 total, under
    the cap. The row lock serializes them, both pass the invariant check,
    both commit. Verifies the lock doesn't reject legitimate concurrent work.
    """
    user, account, debit = await _seed_user_account_and_debit(
        db_engine, debit_amount=500, balance=1000
    )
    assert debit.job_id is not None
    job_id: UUID = debit.job_id

    try:

        async def attempt_refund(amount: int) -> str:
            async with (
                AsyncSession(bind=db_engine, expire_on_commit=False) as session,
                session.begin(),
            ):
                try:
                    await billing_service.partial_refund(
                        job_id,
                        amount,
                        description=f"compat attempt {amount}",
                        session=session,
                        product_id="vex",
                        user_id=user.id,
                    )
                except RefundNotEligibleError as exc:
                    return f"rejected: {exc}"
                else:
                    return "committed"

        results = await asyncio.gather(
            attempt_refund(200),
            attempt_refund(200),
        )
        assert results == ["committed", "committed"], f"Both should succeed, got {results}"

        async with AsyncSession(bind=db_engine, expire_on_commit=False) as session:
            result = await session.execute(
                text(
                    "SELECT COALESCE(SUM(amount), 0) FROM token_transactions "
                    "WHERE job_id = :jid AND transaction_type = 'refund'"
                ),
                {"jid": job_id},
            )
            assert result.scalar_one() == 400
    finally:
        await _cleanup(db_engine, account.id, user.id)
