"""Tests for BillingService's event-builder contract (C6/D3).

BillingService is a stateless, pure event-builder w.r.t. real-time events:
every ledger-mutating method returns a ``BalanceEvent`` (via ``BillingResult``
/ ``SettleUsageResult``) instead of publishing it. Publishing is exclusively
the caller's job, via ``EventBus.publish_balance``, strictly after the
caller's own commit — see ``tests/integration/test_billing_event_ordering.py``
for the commit-ordering invariant itself.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from src.api.services.billing import (
    BalanceEvent,
    BillingResult,
    BillingService,
    SettleUsageResult,
)
from src.api.services.event_bus import EventBus
from src.core.enums import TransactionType

pytestmark = pytest.mark.unit


def _make_account(
    account_id: object = None,
    *,
    is_active: bool = True,
    account_type: str = "personal",
    user_id: object = None,
    organization_id: object = None,
) -> MagicMock:
    account = MagicMock()
    account.id = account_id or uuid4()
    account.is_active = is_active
    account.account_type = account_type
    account.user_id = user_id
    account.organization_id = organization_id
    account.organization = None
    return account


def _make_txn(**overrides: object) -> MagicMock:
    txn = MagicMock()
    txn.id = uuid4()
    for k, v in overrides.items():
        setattr(txn, k, v)
    return txn


class TestMutatingMethodsReturnEventAndDoNotPublish:
    """Every ledger-mutating method returns its event; none ever publishes."""

    async def test_check_and_reserve_returns_event_without_publishing(self) -> None:
        await self._assert_no_publish_and_event(
            method="check_and_reserve",
            call=lambda service, session, account, user_id: service.check_and_reserve(
                account.id,
                10,
                uuid4(),
                metadata={},
                session=session,
                product_id="vex",
                user_id=user_id,
            ),
            expected_type=BillingResult,
            expected_transaction_type=TransactionType.DEBIT.value,
        )

    async def test_credit_returns_event_without_publishing(self) -> None:
        await self._assert_no_publish_and_event(
            method="credit",
            call=lambda service, session, account, user_id: service.credit(
                account.id,
                100,
                uuid4(),
                description="test credit",
                session=session,
                product_id="vex",
                user_id=user_id,
            ),
            expected_type=BillingResult,
            expected_transaction_type=TransactionType.CREDIT.value,
        )

    async def test_admin_adjust_returns_event_without_publishing(self) -> None:
        await self._assert_no_publish_and_event(
            method="admin_adjust",
            call=lambda service, session, account, _user_id: service.admin_adjust(
                account.id,
                50,
                uuid4(),
                description="bonus",
                session=session,
                product_id="vex",
            ),
            expected_type=BillingResult,
            expected_transaction_type=TransactionType.ADMIN_ADJUSTMENT.value,
            account_user_id_for_target=True,
        )

    async def test_settle_session_usage_returns_event_without_publishing(self) -> None:
        service = BillingService()
        session = AsyncMock()
        user_id = uuid4()
        account = _make_account()
        txn = _make_txn(amount=-50)

        with (
            patch("src.api.services.billing.BillingRepository") as MockRepo,
            patch.object(EventBus, "publish_balance", new=AsyncMock()) as mock_publish,
            patch.object(EventBus, "publish", new=AsyncMock()) as mock_publish_legacy,
        ):
            repo = MockRepo.return_value
            repo.get_account_for_update = AsyncMock(return_value=account)
            repo.get_balance = AsyncMock(return_value=100)
            repo.create_transaction = AsyncMock(return_value=txn)

            result = await service.settle_session_usage(
                account.id,
                50,
                session_id=uuid4(),
                model_type="aisha-image",
                session=session,
                product_id="vex",
                user_id=user_id,
            )

        assert isinstance(result, SettleUsageResult)
        assert isinstance(result.event, BalanceEvent)
        assert result.event.user_ids == [user_id]
        assert result.event.transaction_type == TransactionType.DEBIT.value
        mock_publish.assert_not_awaited()
        mock_publish_legacy.assert_not_awaited()

    async def _assert_no_publish_and_event(
        self,
        *,
        method: str,
        call,
        expected_type: type,
        expected_transaction_type: str,
        account_user_id_for_target: bool = False,
    ) -> None:
        service = BillingService()
        session = AsyncMock()
        user_id = uuid4()
        account = _make_account(user_id=user_id) if account_user_id_for_target else _make_account()
        txn = _make_txn()

        with (
            patch("src.api.services.billing.BillingRepository") as MockRepo,
            patch.object(EventBus, "publish_balance", new=AsyncMock()) as mock_publish,
            patch.object(EventBus, "publish", new=AsyncMock()) as mock_publish_legacy,
        ):
            repo = MockRepo.return_value
            repo.get_account_for_update = AsyncMock(return_value=account)
            repo.get_debit_for_job = AsyncMock(return_value=_make_txn(amount=-100))
            repo.has_refund_for_job = AsyncMock(return_value=False)
            repo.sum_refunds_for_job = AsyncMock(return_value=0)
            repo.get_balance = AsyncMock(return_value=100)
            repo.create_transaction = AsyncMock(return_value=txn)
            repo.get_member_user_ids = AsyncMock(return_value=[])

            result = await call(
                service, session, account, None if account_user_id_for_target else user_id
            )

        assert isinstance(result, expected_type), f"{method} did not return {expected_type}"
        assert isinstance(result.event, BalanceEvent), f"{method} did not build a BalanceEvent"
        assert result.event.user_ids == [user_id]
        assert result.event.transaction_type == expected_transaction_type
        mock_publish.assert_not_awaited()
        mock_publish_legacy.assert_not_awaited()


class TestEventNoneWhenNoTarget:
    """``event`` is None whenever there is no SSE target — no user_id resolved."""

    async def test_event_none_when_no_event_bus_or_no_user(self) -> None:
        """No ``user_id`` passed → BalanceEvent is None (nothing to target).

        BillingService holds no event_bus reference at all any more (it is a
        pure, stateless event-builder — C8), so "no event bus" is structurally
        guaranteed; this test pins the remaining "no user" case.
        """
        service = BillingService()
        session = AsyncMock()
        account = _make_account()
        txn = _make_txn(amount=-10)

        with patch("src.api.services.billing.BillingRepository") as MockRepo:
            repo = MockRepo.return_value
            repo.get_account_for_update = AsyncMock(return_value=account)
            repo.get_balance = AsyncMock(return_value=100)
            repo.create_transaction = AsyncMock(return_value=txn)

            result = await service.check_and_reserve(
                account.id,
                10,
                uuid4(),
                metadata={},
                session=session,
                product_id="vex",
                user_id=None,
            )

        assert result.event is None

    async def test_admin_adjust_event_none_for_enterprise_account_with_no_members(self) -> None:
        service = BillingService()
        session = AsyncMock()
        account = _make_account(account_type="enterprise", organization_id=uuid4())
        txn = _make_txn(amount=100)

        with patch("src.api.services.billing.BillingRepository") as MockRepo:
            repo = MockRepo.return_value
            repo.get_account_for_update = AsyncMock(return_value=account)
            repo.get_balance = AsyncMock(return_value=0)
            repo.create_transaction = AsyncMock(return_value=txn)
            repo.get_member_user_ids = AsyncMock(return_value=[])

            result = await service.admin_adjust(
                account.id,
                100,
                uuid4(),
                description="empty org",
                session=session,
                product_id="synthara",
            )

        assert result.event is None
