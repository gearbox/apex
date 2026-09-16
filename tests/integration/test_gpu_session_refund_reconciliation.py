"""Integration coverage for S3 (round-2 remediation): refund reconciliation for
'failed' pre-active GPU sessions whose base reservation was never refunded.

Both GpuSessionService.fail_pre_active_session and GpuProvisioningWorker._mark_failed
swallow a refund failure and continue; list_pending_billing_finalization (the only
reconciler before this fix) only ever matches status='stopped', so a 'failed' session
with an unrefunded reservation was lost forever. These tests exercise the new
GpuSessionRepository.list_pending_refund_reconciliation query and
GpuSessionService.reconcile_pending_refund against real Postgres.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID, uuid4

import pytest
import pytest_asyncio
from sqlalchemy import delete, select, text
from sqlalchemy.ext.asyncio import async_sessionmaker

from src.api.services.billing import BillingService
from src.api.services.billing_errors import RefundNotEligibleError, RefundNotEligibleReason
from src.api.services.gpu_session import GpuSessionService, NullNodeCooldownStore
from src.core.enums import GpuSessionStatus
from src.core.uid import new_id
from src.db.models.billing import TokenAccount, TokenTransaction
from src.db.models.gpu_session import GpuSession
from src.db.models.user import User
from src.db.repositories.billing import BillingRepository
from src.db.repositories.gpu_session import GpuSessionRepository

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession

pytestmark = pytest.mark.asyncio


@pytest.fixture
def session_factory(db_engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(bind=db_engine, expire_on_commit=False)


@pytest_asyncio.fixture(autouse=True)
async def _cleanup(
    session_factory: async_sessionmaker[AsyncSession],
) -> AsyncGenerator[None]:
    yield
    async with session_factory() as db:
        user_ids = (
            (await db.execute(select(User.id).where(User.email.like("refund-reconcile-%"))))
            .scalars()
            .all()
        )
        if not user_ids:
            return
        account_ids = (
            (await db.execute(select(TokenAccount.id).where(TokenAccount.user_id.in_(user_ids))))
            .scalars()
            .all()
        )
        # token_transactions carries a BEFORE UPDATE OR DELETE trigger
        # (enforce_token_transactions_immutable) blocking all mutation, by name
        # per project_token_transactions_immutable — never DISABLE TRIGGER ALL.
        # FK order otherwise: token_transactions/gpu_sessions (RESTRICT on
        # account_id) before token_accounts, then users.
        await db.execute(
            text(
                "ALTER TABLE token_transactions DISABLE TRIGGER enforce_token_transactions_immutable"
            )
        )
        await db.execute(
            delete(TokenTransaction).where(TokenTransaction.account_id.in_(account_ids))
        )
        await db.execute(
            text(
                "ALTER TABLE token_transactions ENABLE TRIGGER enforce_token_transactions_immutable"
            )
        )
        await db.execute(delete(GpuSession).where(GpuSession.user_id.in_(user_ids)))
        await db.execute(delete(TokenAccount).where(TokenAccount.id.in_(account_ids)))
        await db.execute(delete(User).where(User.id.in_(user_ids)))
        await db.commit()


async def _seed_balance(
    session_factory: async_sessionmaker[AsyncSession], account_id: UUID, amount: int
) -> None:
    """Directly insert a credit transaction so the account has balance to debit from."""
    async with session_factory() as db, db.begin():
        db.add(
            TokenTransaction(
                id=new_id(),
                account_id=account_id,
                transaction_type="credit",
                amount=amount,
                balance_after=amount,
                product_id="vex",
            )
        )


async def _seed_user_and_account(
    session_factory: async_sessionmaker[AsyncSession],
) -> tuple[User, TokenAccount]:
    user = User(
        id=new_id(),
        email=f"refund-reconcile-{uuid4().hex}@example.com",
        password_hash="hash",
        product_id="vex",
    )
    async with session_factory() as db, db.begin():
        db.add(user)
    account = TokenAccount(id=new_id(), account_type="personal", user_id=user.id, product_id="vex")
    async with session_factory() as db, db.begin():
        db.add(account)
    await _seed_balance(session_factory, account.id, 10_000)
    return user, account


async def _seed_failed_session_with_debit(
    session_factory: async_sessionmaker[AsyncSession],
    *,
    user: User,
    account: TokenAccount,
    started_at: datetime | None,
    created_at: datetime | None = None,
    stopped_at: datetime | None = None,
    with_debit: bool = True,
    billing_finalization_attempts: int = 0,
) -> GpuSession:
    """A terminal 'failed' pre-active session, optionally with a real
    base-reservation debit but (deliberately) no refund — the exact state a
    swallowed refund failure leaves behind. ``stopped_at`` is the terminal
    timestamp the T5 grace-window check now uses (mirroring what
    ``fail_pre_active_session``/``_mark_failed`` stamp on the real path) —
    ``created_at`` no longer controls eligibility and is only here for realism.
    ``with_debit=False`` reproduces the T4 anomaly: ``account_id`` set but the
    reservation debit was never written.
    """
    gpu_session = GpuSession(
        id=new_id(),
        user_id=user.id,
        product_id="vex",
        status=GpuSessionStatus.failed,
        bundle_name="retry-bundle",
        model_type="aisha-image",
        account_id=account.id,
        started_at=started_at,
        stopped_at=stopped_at,
        error_message="node_provision_script_failed",
        billing_finalization_attempts=billing_finalization_attempts,
    )
    if created_at is not None:
        gpu_session.created_at = created_at
    async with session_factory() as db, db.begin():
        db.add(gpu_session)
        if with_debit:
            billing_service = BillingService()
            await billing_service.check_and_reserve(
                account.id,
                500,
                gpu_session.id,
                metadata={"type": "gpu_session_reservation"},
                description="GPU session base reservation",
                session=db,
                product_id="vex",
                user_id=user.id,
            )
    return gpu_session


def _make_service(
    session_factory: async_sessionmaker[AsyncSession], *, billing_service: object
) -> GpuSessionService:
    return GpuSessionService(
        vastai_client=AsyncMock(),
        cf_client=AsyncMock(),
        bundle_index=MagicMock(),
        session_factory=session_factory,
        settings=MagicMock(),
        billing_service=billing_service,  # type: ignore[arg-type]
        cooldown_store=NullNodeCooldownStore(),
        provisioning_script_service=AsyncMock(),
    )


class TestListPendingRefundReconciliation:
    async def test_failed_pre_active_session_with_no_refund_is_selected(
        self, session_factory: async_sessionmaker[AsyncSession]
    ) -> None:
        user, account = await _seed_user_and_account(session_factory)
        session = await _seed_failed_session_with_debit(
            session_factory,
            user=user,
            account=account,
            started_at=None,
            stopped_at=datetime.now(UTC) - timedelta(hours=1),
        )

        async with session_factory() as db:
            candidates = await GpuSessionRepository(db).list_pending_refund_reconciliation(
                grace_cutoff=datetime.now(UTC), limit=50
            )

        assert session.id in {row.id for row in candidates}

    async def test_a_session_that_reached_active_is_not_selected(
        self, session_factory: async_sessionmaker[AsyncSession]
    ) -> None:
        user, account = await _seed_user_and_account(session_factory)
        session = await _seed_failed_session_with_debit(
            session_factory,
            user=user,
            account=account,
            started_at=datetime.now(UTC) - timedelta(hours=2),
            stopped_at=datetime.now(UTC) - timedelta(hours=1),
        )

        async with session_factory() as db:
            candidates = await GpuSessionRepository(db).list_pending_refund_reconciliation(
                grace_cutoff=datetime.now(UTC), limit=50
            )

        assert session.id not in {row.id for row in candidates}

    async def test_a_session_within_the_grace_window_is_not_yet_selected(
        self, session_factory: async_sessionmaker[AsyncSession]
    ) -> None:
        user, account = await _seed_user_and_account(session_factory)
        session = await _seed_failed_session_with_debit(
            session_factory,
            user=user,
            account=account,
            started_at=None,
            stopped_at=datetime.now(UTC),
        )

        async with session_factory() as db:
            candidates = await GpuSessionRepository(db).list_pending_refund_reconciliation(
                grace_cutoff=datetime.now(UTC) - timedelta(minutes=2), limit=50
            )

        assert session.id not in {row.id for row in candidates}

    async def test_grace_window_is_measured_from_stopped_at_not_created_at(
        self, session_factory: async_sessionmaker[AsyncSession]
    ) -> None:
        """T5: a session can legitimately run for a long provisioning window
        before failing — created_at alone must not decide eligibility. A
        session created long ago but that only just failed (stopped_at now)
        is still inside the grace period; one created recently but that failed
        a while ago (stopped_at old) is already eligible."""
        user, account = await _seed_user_and_account(session_factory)
        old_creation_recent_failure = await _seed_failed_session_with_debit(
            session_factory,
            user=user,
            account=account,
            started_at=None,
            created_at=datetime.now(UTC) - timedelta(hours=2),
            stopped_at=datetime.now(UTC),
        )
        recent_creation_old_failure = await _seed_failed_session_with_debit(
            session_factory,
            user=user,
            account=account,
            started_at=None,
            created_at=datetime.now(UTC),
            stopped_at=datetime.now(UTC) - timedelta(hours=1),
        )

        async with session_factory() as db:
            candidates = await GpuSessionRepository(db).list_pending_refund_reconciliation(
                grace_cutoff=datetime.now(UTC) - timedelta(minutes=2), limit=50
            )

        candidate_ids = {row.id for row in candidates}
        assert old_creation_recent_failure.id not in candidate_ids
        assert recent_creation_old_failure.id in candidate_ids

    async def test_an_already_refunded_session_is_never_selected(
        self, session_factory: async_sessionmaker[AsyncSession]
    ) -> None:
        user, account = await _seed_user_and_account(session_factory)
        session = await _seed_failed_session_with_debit(
            session_factory,
            user=user,
            account=account,
            started_at=None,
            stopped_at=datetime.now(UTC) - timedelta(hours=1),
        )
        async with session_factory() as db, db.begin():
            await BillingService().refund(
                session.id,
                description="already refunded",
                session=db,
                product_id="vex",
                user_id=user.id,
            )

        async with session_factory() as db:
            candidates = await GpuSessionRepository(db).list_pending_refund_reconciliation(
                grace_cutoff=datetime.now(UTC), limit=50
            )

        assert session.id not in {row.id for row in candidates}

    async def test_no_debit_ever_written_is_never_selected(
        self, session_factory: async_sessionmaker[AsyncSession]
    ) -> None:
        """T4: account_id set but the reservation debit was never written (the
        session failed before check_and_reserve ran) — without the has_debit
        requirement this row matched forever, since refund() raises
        RefundNotEligibleError every time and reconcile_pending_refund treated
        that as success without ever removing the row from the candidate set."""
        user, account = await _seed_user_and_account(session_factory)
        session = await _seed_failed_session_with_debit(
            session_factory,
            user=user,
            account=account,
            started_at=None,
            stopped_at=datetime.now(UTC) - timedelta(hours=1),
            with_debit=False,
        )

        async with session_factory() as db:
            candidates = await GpuSessionRepository(db).list_pending_refund_reconciliation(
                grace_cutoff=datetime.now(UTC), limit=50
            )

        assert session.id not in {row.id for row in candidates}

    async def test_debit_present_no_refund_is_selected_then_excluded_after_reconcile(
        self, session_factory: async_sessionmaker[AsyncSession]
    ) -> None:
        user, account = await _seed_user_and_account(session_factory)
        session = await _seed_failed_session_with_debit(
            session_factory,
            user=user,
            account=account,
            started_at=None,
            stopped_at=datetime.now(UTC) - timedelta(hours=1),
        )

        async with session_factory() as db:
            before = await GpuSessionRepository(db).list_pending_refund_reconciliation(
                grace_cutoff=datetime.now(UTC), limit=50
            )
        assert session.id in {row.id for row in before}

        service = _make_service(session_factory, billing_service=BillingService())
        assert await service.reconcile_pending_refund(session) is True

        async with session_factory() as db:
            after = await GpuSessionRepository(db).list_pending_refund_reconciliation(
                grace_cutoff=datetime.now(UTC), limit=50
            )
        assert session.id not in {row.id for row in after}

    async def test_debit_and_refund_both_present_is_not_selected(
        self, session_factory: async_sessionmaker[AsyncSession]
    ) -> None:
        user, account = await _seed_user_and_account(session_factory)
        session = await _seed_failed_session_with_debit(
            session_factory,
            user=user,
            account=account,
            started_at=None,
            stopped_at=datetime.now(UTC) - timedelta(hours=1),
        )
        async with session_factory() as db, db.begin():
            await BillingService().refund(
                session.id,
                description="already refunded",
                session=db,
                product_id="vex",
                user_id=user.id,
            )

        async with session_factory() as db:
            candidates = await GpuSessionRepository(db).list_pending_refund_reconciliation(
                grace_cutoff=datetime.now(UTC), limit=50
            )

        assert session.id not in {row.id for row in candidates}

    async def test_no_debit_rows_do_not_head_of_line_block_genuinely_stuck_rows(
        self, session_factory: async_sessionmaker[AsyncSession]
    ) -> None:
        """T4: without the has_debit filter, the two permanent no-DEBIT
        candidates — being the oldest — would fill a limit=2 sweep and starve
        the genuinely stuck (has-debit, no-refund) row below them."""
        user, account = await _seed_user_and_account(session_factory)
        now = datetime.now(UTC)
        no_debit_1 = await _seed_failed_session_with_debit(
            session_factory,
            user=user,
            account=account,
            started_at=None,
            stopped_at=now - timedelta(hours=3),
            with_debit=False,
        )
        no_debit_2 = await _seed_failed_session_with_debit(
            session_factory,
            user=user,
            account=account,
            started_at=None,
            stopped_at=now - timedelta(hours=2),
            with_debit=False,
        )
        genuinely_stuck = await _seed_failed_session_with_debit(
            session_factory,
            user=user,
            account=account,
            started_at=None,
            stopped_at=now - timedelta(hours=1),
        )

        async with session_factory() as db:
            candidates = await GpuSessionRepository(db).list_pending_refund_reconciliation(
                grace_cutoff=now, limit=2
            )

        candidate_ids = {row.id for row in candidates}
        assert no_debit_1.id not in candidate_ids
        assert no_debit_2.id not in candidate_ids
        assert genuinely_stuck.id in candidate_ids

    async def test_quarantined_session_is_excluded_from_selection(
        self, session_factory: async_sessionmaker[AsyncSession]
    ) -> None:
        """T7: past the quarantine threshold, a candidate must stop being
        re-selected — quarantine means "stop trying", not "keep trying and
        keep shouting" via an ERROR log every sweep forever."""
        user, account = await _seed_user_and_account(session_factory)
        session = await _seed_failed_session_with_debit(
            session_factory,
            user=user,
            account=account,
            started_at=None,
            stopped_at=datetime.now(UTC) - timedelta(hours=1),
            billing_finalization_attempts=5,
        )

        async with session_factory() as db:
            candidates = await GpuSessionRepository(db).list_pending_refund_reconciliation(
                grace_cutoff=datetime.now(UTC), limit=50, quarantine_threshold=5
            )

        assert session.id not in {row.id for row in candidates}

    async def test_quarantine_threshold_none_disables_the_filter(
        self, session_factory: async_sessionmaker[AsyncSession]
    ) -> None:
        user, account = await _seed_user_and_account(session_factory)
        session = await _seed_failed_session_with_debit(
            session_factory,
            user=user,
            account=account,
            started_at=None,
            stopped_at=datetime.now(UTC) - timedelta(hours=1),
            billing_finalization_attempts=999,
        )

        async with session_factory() as db:
            candidates = await GpuSessionRepository(db).list_pending_refund_reconciliation(
                grace_cutoff=datetime.now(UTC), limit=50
            )

        assert session.id in {row.id for row in candidates}


class TestReconcilePendingRefund:
    async def test_reconciler_picks_it_up_and_it_no_longer_appears(
        self, session_factory: async_sessionmaker[AsyncSession]
    ) -> None:
        user, account = await _seed_user_and_account(session_factory)
        session = await _seed_failed_session_with_debit(
            session_factory,
            user=user,
            account=account,
            started_at=None,
            created_at=datetime.now(UTC) - timedelta(hours=1),
        )

        service = _make_service(session_factory, billing_service=BillingService())
        success = await service.reconcile_pending_refund(session)
        assert success is True

        async with session_factory() as db:
            txns = await BillingRepository(db).get_debit_for_job(session.id)
            assert txns is not None
            has_refund = await BillingRepository(db).has_refund_for_job(session.id)
            assert has_refund is True

            candidates = await GpuSessionRepository(db).list_pending_refund_reconciliation(
                grace_cutoff=datetime.now(UTC), limit=50
            )
        assert session.id not in {row.id for row in candidates}

    async def test_already_refunded_is_treated_as_success_no_duplicate_transaction(
        self, session_factory: async_sessionmaker[AsyncSession]
    ) -> None:
        user, account = await _seed_user_and_account(session_factory)
        session = await _seed_failed_session_with_debit(
            session_factory,
            user=user,
            account=account,
            started_at=None,
            created_at=datetime.now(UTC) - timedelta(hours=1),
        )
        billing_service = BillingService()
        async with session_factory() as db, db.begin():
            await billing_service.refund(
                session.id,
                description="first refund",
                session=db,
                product_id="vex",
                user_id=user.id,
            )

        service = _make_service(session_factory, billing_service=billing_service)
        success = await service.reconcile_pending_refund(session)
        assert success is True

        async with session_factory() as db:
            result = await db.execute(
                select(TokenTransaction).where(
                    TokenTransaction.job_id == session.id,
                    TokenTransaction.transaction_type == "refund",
                )
            )
            refund_txns = result.scalars().all()
        assert len(refund_txns) == 1

    async def test_missing_debit_is_a_genuine_anomaly_not_treated_as_success(
        self, session_factory: async_sessionmaker[AsyncSession]
    ) -> None:
        """T4: reconcile_pending_refund must distinguish 'someone already
        refunded this' (success) from 'the debit that should exist doesn't'
        (a real anomaly — the row got here via a direct call, not the query,
        which now filters this shape out) by returning False so the worker
        bumps its attempt counter instead of logging false success."""
        user, account = await _seed_user_and_account(session_factory)
        session = await _seed_failed_session_with_debit(
            session_factory,
            user=user,
            account=account,
            started_at=None,
            stopped_at=datetime.now(UTC) - timedelta(hours=1),
            with_debit=False,
        )

        service = _make_service(session_factory, billing_service=BillingService())
        success = await service.reconcile_pending_refund(session)

        assert success is False

    async def test_account_not_found_is_not_treated_as_success(
        self, session_factory: async_sessionmaker[AsyncSession]
    ) -> None:
        """U6: a structured `reason` discriminator, not string-matching on
        str(exc), decides the branch — ACCOUNT_NOT_FOUND is a genuine anomaly
        like NO_DEBIT_FOUND, not the ordinary 'someone already refunded this'
        race, and must not be silently reported as success."""
        user, account = await _seed_user_and_account(session_factory)
        session = await _seed_failed_session_with_debit(
            session_factory,
            user=user,
            account=account,
            started_at=None,
            stopped_at=datetime.now(UTC) - timedelta(hours=1),
        )
        broken_billing = AsyncMock()
        broken_billing.refund.side_effect = RefundNotEligibleError(
            "Account not found for refund", reason=RefundNotEligibleReason.ACCOUNT_NOT_FOUND
        )
        service = _make_service(session_factory, billing_service=broken_billing)

        assert await service.reconcile_pending_refund(session) is False

    async def test_no_account_id_is_a_trivial_success(
        self, session_factory: async_sessionmaker[AsyncSession]
    ) -> None:
        user, _account = await _seed_user_and_account(session_factory)
        session = GpuSession(
            id=new_id(),
            user_id=user.id,
            product_id="vex",
            status=GpuSessionStatus.failed,
            bundle_name="retry-bundle",
            model_type="aisha-image",
            account_id=None,
            started_at=None,
        )
        async with session_factory() as db, db.begin():
            db.add(session)

        service = _make_service(session_factory, billing_service=BillingService())
        assert await service.reconcile_pending_refund(session) is True


class TestFailPreActiveSessionSwallowedRefundIsReconcilable:
    async def test_refund_failure_leaves_the_session_terminal_and_reconcilable(
        self, session_factory: async_sessionmaker[AsyncSession]
    ) -> None:
        """fail_pre_active_session must still transition to 'failed' (and the
        caller — the webhook — must still see success) even when the refund
        itself blows up; the new reconciliation query is what recovers it."""
        user, account = await _seed_user_and_account(session_factory)
        gpu_session = GpuSession(
            id=new_id(),
            user_id=user.id,
            product_id="vex",
            status=GpuSessionStatus.provisioning,
            bundle_name="retry-bundle",
            model_type="aisha-image",
            account_id=account.id,
            callback_token_hash="irrelevant-for-this-test",
        )
        async with session_factory() as db, db.begin():
            db.add(gpu_session)
            billing_service = BillingService()
            await billing_service.check_and_reserve(
                account.id,
                500,
                gpu_session.id,
                metadata={"type": "gpu_session_reservation"},
                description="GPU session base reservation",
                session=db,
                product_id="vex",
                user_id=user.id,
            )

        broken_billing = AsyncMock()
        broken_billing.refund.side_effect = RuntimeError("simulated transient failure")
        service = _make_service(session_factory, billing_service=broken_billing)

        result = await service.fail_pre_active_session(gpu_session.id, reason="node said so")

        assert result is not None
        assert result.status == GpuSessionStatus.failed
        broken_billing.refund.assert_awaited_once()

        async with session_factory() as db:
            has_refund = await BillingRepository(db).has_refund_for_job(gpu_session.id)
            assert has_refund is False

            candidates = await GpuSessionRepository(db).list_pending_refund_reconciliation(
                grace_cutoff=datetime.now(UTC), limit=50
            )
        assert gpu_session.id in {row.id for row in candidates}

        # A subsequent reconciliation attempt (real billing this time) succeeds.
        recovery_service = _make_service(session_factory, billing_service=BillingService())
        async with session_factory() as db:
            refreshed = await GpuSessionRepository(db).get_by_id(gpu_session.id)
        assert refreshed is not None
        assert await recovery_service.reconcile_pending_refund(refreshed) is True

        async with session_factory() as db:
            has_refund_after = await BillingRepository(db).has_refund_for_job(gpu_session.id)
        assert has_refund_after is True
