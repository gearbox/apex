"""Tests for billing services."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID, uuid4

import pytest

from src.api.services.billing import BillingService
from src.api.services.billing_errors import (
    AccountInactiveError,
    AccountNotFoundError,
    InsufficientBalanceError,
    PriceNotFoundError,
    RefundNotEligibleError,
)
from src.api.services.moderation import (
    ComfyUIModerationDetector,
    GrokModerationDetector,
)
from src.api.services.pricing import PricingService
from src.core.enums import TransactionType

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def billing_service() -> BillingService:
    return BillingService()


@pytest.fixture
def pricing_service() -> PricingService:
    return PricingService()


@pytest.fixture
def mock_session() -> AsyncMock:
    session = AsyncMock()
    session.flush = AsyncMock()
    session.commit = AsyncMock()
    return session


def _make_account(
    account_id: UUID | None = None,
    is_active: bool = True,
    account_type: str = "personal",
) -> MagicMock:
    account = MagicMock()
    account.id = account_id or uuid4()
    account.is_active = is_active
    account.account_type = account_type
    account.organization = None
    return account


def _make_transaction(
    txn_id: UUID | None = None,
    account_id: UUID | None = None,
    amount: int = -10,
) -> MagicMock:
    txn = MagicMock()
    txn.id = txn_id or uuid4()
    txn.account_id = account_id or uuid4()
    txn.amount = amount
    return txn


# ---------------------------------------------------------------------------
# BillingService.check_and_reserve
# ---------------------------------------------------------------------------


class TestCheckAndReserve:
    async def test_successful_reserve(
        self, billing_service: BillingService, mock_session: AsyncMock
    ) -> None:
        account_id = uuid4()
        job_id = uuid4()
        account = _make_account(account_id=account_id)
        txn = _make_transaction(account_id=account_id, amount=-10)

        with patch("src.api.services.billing.BillingRepository") as MockRepo:
            repo = MockRepo.return_value
            repo.get_account_for_update = AsyncMock(return_value=account)
            repo.get_balance = AsyncMock(return_value=100)
            repo.create_transaction = AsyncMock(return_value=txn)

            result = await billing_service.check_and_reserve(
                account_id,
                10,
                job_id,
                metadata={"provider": "grok"},
                session=mock_session,
                product_id="vex",
            )

        assert result.txn.id == txn.id
        repo.create_transaction.assert_awaited_once()
        call_kwargs = repo.create_transaction.call_args.kwargs
        assert call_kwargs["amount"] == -10
        assert call_kwargs["balance_after"] == 90
        assert call_kwargs["transaction_type"] == TransactionType.DEBIT.value

    async def test_insufficient_balance(
        self, billing_service: BillingService, mock_session: AsyncMock
    ) -> None:
        account_id = uuid4()
        account = _make_account(account_id=account_id)

        with patch("src.api.services.billing.BillingRepository") as MockRepo:
            repo = MockRepo.return_value
            repo.get_account_for_update = AsyncMock(return_value=account)
            repo.get_balance = AsyncMock(return_value=5)

            with pytest.raises(InsufficientBalanceError) as exc_info:
                await billing_service.check_and_reserve(
                    account_id,
                    10,
                    uuid4(),
                    metadata={},
                    session=mock_session,
                    product_id="vex",
                )

            assert exc_info.value.balance == 5
            assert exc_info.value.required == 10

    async def test_account_not_found(
        self, billing_service: BillingService, mock_session: AsyncMock
    ) -> None:
        with patch("src.api.services.billing.BillingRepository") as MockRepo:
            repo = MockRepo.return_value
            repo.get_account_for_update = AsyncMock(return_value=None)

            with pytest.raises(AccountNotFoundError):
                await billing_service.check_and_reserve(
                    uuid4(), 10, uuid4(), metadata={}, session=mock_session, product_id="vex"
                )

    async def test_inactive_account(
        self, billing_service: BillingService, mock_session: AsyncMock
    ) -> None:
        account = _make_account(is_active=False)

        with patch("src.api.services.billing.BillingRepository") as MockRepo:
            repo = MockRepo.return_value
            repo.get_account_for_update = AsyncMock(return_value=account)

            with pytest.raises(AccountInactiveError):
                await billing_service.check_and_reserve(
                    account.id, 10, uuid4(), metadata={}, session=mock_session, product_id="vex"
                )


# ---------------------------------------------------------------------------
# BillingService.refund
# ---------------------------------------------------------------------------


class TestRefund:
    async def test_successful_refund(
        self, billing_service: BillingService, mock_session: AsyncMock
    ) -> None:
        job_id = uuid4()
        account_id = uuid4()
        debit = _make_transaction(account_id=account_id, amount=-25)
        refund_txn = _make_transaction(account_id=account_id, amount=25)
        account = _make_account(account_id=account_id)

        with patch("src.api.services.billing.BillingRepository") as MockRepo:
            repo = MockRepo.return_value
            repo.get_debit_for_job = AsyncMock(return_value=debit)
            repo.has_refund_for_job = AsyncMock(return_value=False)
            repo.get_account_for_update = AsyncMock(return_value=account)
            repo.get_balance = AsyncMock(return_value=75)
            repo.create_transaction = AsyncMock(return_value=refund_txn)

            result = await billing_service.refund(
                job_id, description="test refund", session=mock_session, product_id="vex"
            )

        assert result.txn.amount == 25
        call_kwargs = repo.create_transaction.call_args.kwargs
        assert call_kwargs["amount"] == 25
        assert call_kwargs["balance_after"] == 100
        assert call_kwargs["transaction_type"] == TransactionType.REFUND.value

    async def test_no_debit_found(
        self, billing_service: BillingService, mock_session: AsyncMock
    ) -> None:
        with patch("src.api.services.billing.BillingRepository") as MockRepo:
            repo = MockRepo.return_value
            repo.get_debit_for_job = AsyncMock(return_value=None)

            with pytest.raises(RefundNotEligibleError):
                await billing_service.refund(
                    uuid4(), description="test", session=mock_session, product_id="vex"
                )

    async def test_already_refunded(
        self, billing_service: BillingService, mock_session: AsyncMock
    ) -> None:
        debit = _make_transaction(amount=-10)

        with patch("src.api.services.billing.BillingRepository") as MockRepo:
            repo = MockRepo.return_value
            repo.get_debit_for_job = AsyncMock(return_value=debit)
            repo.has_refund_for_job = AsyncMock(return_value=True)

            with pytest.raises(RefundNotEligibleError):
                await billing_service.refund(
                    uuid4(), description="test", session=mock_session, product_id="vex"
                )


class TestPartialRefund:
    async def test_successful_partial_refund(
        self, billing_service: BillingService, mock_session: AsyncMock
    ) -> None:
        """Partial refund under original debit succeeds and creates a REFUND txn."""
        job_id = uuid4()
        account_id = uuid4()
        debit = _make_transaction(account_id=account_id, amount=-500)
        refund_txn = _make_transaction(account_id=account_id, amount=200)
        account = _make_account(account_id=account_id)

        with patch("src.api.services.billing.BillingRepository") as MockRepo:
            repo = MockRepo.return_value
            repo.get_debit_for_job = AsyncMock(return_value=debit)
            repo.sum_refunds_for_job = AsyncMock(return_value=0)
            repo.get_account_for_update = AsyncMock(return_value=account)
            repo.get_balance = AsyncMock(return_value=75)
            repo.create_transaction = AsyncMock(return_value=refund_txn)

            result = await billing_service.partial_refund(
                job_id,
                amount=200,
                description="GPU session partial refund: used 3min, reserved 5min minimum",
                session=mock_session,
                product_id="vex",
                user_id=uuid4(),
            )

        assert result.txn.amount == 200
        call_kwargs = repo.create_transaction.call_args.kwargs
        assert call_kwargs["amount"] == 200
        assert call_kwargs["balance_after"] == 275  # 75 + 200
        assert call_kwargs["transaction_type"] == TransactionType.REFUND.value
        assert call_kwargs["job_id"] == job_id

    async def test_rejects_zero_or_negative_amount(
        self, billing_service: BillingService, mock_session: AsyncMock
    ) -> None:
        """Amount must be > 0; zero or negative is a caller bug."""
        for bad_amount in (0, -1, -100):
            with pytest.raises(RefundNotEligibleError, match="must be positive"):
                await billing_service.partial_refund(
                    uuid4(),
                    amount=bad_amount,
                    description="test",
                    session=mock_session,
                    product_id="vex",
                )

    async def test_rejects_amount_exceeding_debit_on_first_call(
        self, billing_service: BillingService, mock_session: AsyncMock
    ) -> None:
        """With no prior refunds, requesting > original debit fails."""
        debit = _make_transaction(amount=-500)

        with patch("src.api.services.billing.BillingRepository") as MockRepo:
            repo = MockRepo.return_value
            repo.get_debit_for_job = AsyncMock(return_value=debit)
            repo.sum_refunds_for_job = AsyncMock(return_value=0)

            with pytest.raises(RefundNotEligibleError, match="would exceed original debit"):
                await billing_service.partial_refund(
                    uuid4(),
                    amount=501,
                    description="test",
                    session=mock_session,
                    product_id="vex",
                )

    async def test_rejects_cumulative_refund_exceeding_original_debit(
        self, billing_service: BillingService, mock_session: AsyncMock
    ) -> None:
        """Key contract: already_refunded + amount MUST NOT exceed original debit.

        Simulates: original debit = 500, prior refund = 300, request = 250.
        300 + 250 = 550 > 500 → reject.
        """
        debit = _make_transaction(amount=-500)

        with patch("src.api.services.billing.BillingRepository") as MockRepo:
            repo = MockRepo.return_value
            repo.get_debit_for_job = AsyncMock(return_value=debit)
            repo.sum_refunds_for_job = AsyncMock(return_value=300)  # prior refunds

            with pytest.raises(RefundNotEligibleError) as exc_info:
                await billing_service.partial_refund(
                    uuid4(),
                    amount=250,
                    description="test",
                    session=mock_session,
                    product_id="vex",
                )
            # Verify the error message includes the relevant numbers
            msg = str(exc_info.value)
            assert "already_refunded=300" in msg
            assert "requested=250" in msg
            assert "original=500" in msg

    async def test_allows_cumulative_refund_up_to_original_debit(
        self, billing_service: BillingService, mock_session: AsyncMock
    ) -> None:
        """Boundary: already_refunded + amount == original_debit must succeed."""
        account_id = uuid4()
        debit = _make_transaction(account_id=account_id, amount=-500)
        refund_txn = _make_transaction(account_id=account_id, amount=200)
        account = _make_account(account_id=account_id)

        with patch("src.api.services.billing.BillingRepository") as MockRepo:
            repo = MockRepo.return_value
            repo.get_debit_for_job = AsyncMock(return_value=debit)
            repo.sum_refunds_for_job = AsyncMock(return_value=300)  # prior refunds
            repo.get_account_for_update = AsyncMock(return_value=account)
            repo.get_balance = AsyncMock(return_value=100)
            repo.create_transaction = AsyncMock(return_value=refund_txn)

            # 300 + 200 == 500, exactly at the boundary → allowed
            await billing_service.partial_refund(
                uuid4(),
                amount=200,
                description="final partial refund",
                session=mock_session,
                product_id="vex",
            )

        repo.create_transaction.assert_awaited_once()

    async def test_raises_when_no_debit_exists(
        self, billing_service: BillingService, mock_session: AsyncMock
    ) -> None:
        with patch("src.api.services.billing.BillingRepository") as MockRepo:
            repo = MockRepo.return_value
            repo.get_debit_for_job = AsyncMock(return_value=None)

            with pytest.raises(RefundNotEligibleError, match="No debit transaction found"):
                await billing_service.partial_refund(
                    uuid4(),
                    amount=100,
                    description="test",
                    session=mock_session,
                    product_id="vex",
                )


# ---------------------------------------------------------------------------
# PricingService.get_price
# ---------------------------------------------------------------------------


class TestPricingService:
    async def test_get_price_found(
        self, pricing_service: PricingService, mock_session: AsyncMock
    ) -> None:
        rule = MagicMock()
        rule.token_cost = 50

        with patch("src.api.services.pricing.BillingRepository") as MockRepo:
            repo = MockRepo.return_value
            repo.get_active_price = AsyncMock(return_value=rule)

            price = await pricing_service.get_price(
                "grok", "t2i", "grok-imagine-image", session=mock_session
            )

        assert price == 50

    async def test_get_price_not_found(
        self, pricing_service: PricingService, mock_session: AsyncMock
    ) -> None:
        with patch("src.api.services.pricing.BillingRepository") as MockRepo:
            repo = MockRepo.return_value
            repo.get_active_price = AsyncMock(return_value=None)

            with pytest.raises(PriceNotFoundError):
                await pricing_service.get_price("grok", "t2i", "nonexistent", session=mock_session)


# ---------------------------------------------------------------------------
# ModerationDetector tests
# ---------------------------------------------------------------------------


class TestGrokModerationDetector:
    def test_no_moderation(self) -> None:
        detector = GrokModerationDetector()
        result = detector.classify({"respect_moderation": True}, None)
        assert not result.is_moderated
        assert not result.is_provider_error

    def test_moderated(self) -> None:
        detector = GrokModerationDetector()
        result = detector.classify({"respect_moderation": False}, None)
        assert result.is_moderated
        assert not result.is_provider_error
        assert result.reason == "content_moderated"

    def test_provider_error(self) -> None:
        detector = GrokModerationDetector()
        result = detector.classify(None, RuntimeError("API down"))
        assert not result.is_moderated
        assert result.is_provider_error

    def test_no_response_no_exception(self) -> None:
        detector = GrokModerationDetector()
        result = detector.classify(None, None)
        assert not result.is_moderated
        assert not result.is_provider_error

    def test_missing_key_defaults_to_not_moderated(self) -> None:
        detector = GrokModerationDetector()
        result = detector.classify({}, None)
        assert not result.is_moderated


class TestComfyUIModerationDetector:
    def test_no_moderation(self) -> None:
        detector = ComfyUIModerationDetector()
        result = detector.classify(None, None)
        assert not result.is_moderated
        assert not result.is_provider_error

    def test_provider_error(self) -> None:
        detector = ComfyUIModerationDetector()
        result = detector.classify(None, RuntimeError("connection error"))
        assert not result.is_moderated
        assert result.is_provider_error


# ---------------------------------------------------------------------------
# BillingService.admin_adjust
# ---------------------------------------------------------------------------


class TestAdminAdjust:
    async def test_positive_adjustment(
        self, billing_service: BillingService, mock_session: AsyncMock
    ) -> None:
        account_id = uuid4()
        admin_id = uuid4()
        account = _make_account(account_id=account_id)
        txn = _make_transaction(account_id=account_id, amount=100)

        with patch("src.api.services.billing.BillingRepository") as MockRepo:
            repo = MockRepo.return_value
            repo.get_account_for_update = AsyncMock(return_value=account)
            repo.get_balance = AsyncMock(return_value=50)
            repo.create_transaction = AsyncMock(return_value=txn)
            await billing_service.admin_adjust(
                account_id,
                100,
                admin_id,
                description="bonus",
                session=mock_session,
                product_id="vex",
            )

        call_kwargs = repo.create_transaction.call_args.kwargs
        assert call_kwargs["amount"] == 100
        assert call_kwargs["balance_after"] == 150

    async def test_negative_adjustment_exceeds_balance(
        self, billing_service: BillingService, mock_session: AsyncMock
    ) -> None:
        account_id = uuid4()
        account = _make_account(account_id=account_id)

        with patch("src.api.services.billing.BillingRepository") as MockRepo:
            repo = MockRepo.return_value
            repo.get_account_for_update = AsyncMock(return_value=account)
            repo.get_balance = AsyncMock(return_value=30)

            with pytest.raises(InsufficientBalanceError):
                await billing_service.admin_adjust(
                    account_id,
                    -50,
                    uuid4(),
                    description="deduction",
                    session=mock_session,
                    product_id="vex",
                )

    async def test_positive_adjustment_builds_event_personal(
        self, billing_service: BillingService, mock_session: AsyncMock
    ) -> None:
        """Personal account adjustment builds a BalanceEvent for the account owner.

        BillingService no longer publishes (C6) — it returns a BalanceEvent
        via BillingResult, and the caller publishes post-commit through
        EventBus.publish_balance.
        """
        account_id = uuid4()
        owner_user_id = uuid4()
        admin_id = uuid4()
        account = _make_account(account_id=account_id, account_type="personal")
        account.user_id = owner_user_id
        account.organization_id = None
        txn = _make_transaction(account_id=account_id, amount=100)

        with patch("src.api.services.billing.BillingRepository") as MockRepo:
            repo = MockRepo.return_value
            repo.get_account_for_update = AsyncMock(return_value=account)
            repo.get_balance = AsyncMock(return_value=50)
            repo.create_transaction = AsyncMock(return_value=txn)

            result = await billing_service.admin_adjust(
                account_id,
                100,
                admin_id,
                description="bonus",
                session=mock_session,
                product_id="vex",
            )

        assert result.event is not None
        assert result.event.user_ids == [owner_user_id]
        assert result.event.account_id == account_id
        assert result.event.balance == 150
        assert result.event.delta == 100
        assert result.event.transaction_type == "admin_adjustment"

    async def test_positive_adjustment_event_targets_all_org_members(
        self, billing_service: BillingService, mock_session: AsyncMock
    ) -> None:
        """Enterprise account adjustment's event targets every org member."""
        account_id = uuid4()
        org_id = uuid4()
        member_ids = [uuid4(), uuid4(), uuid4()]
        admin_id = uuid4()
        account = _make_account(account_id=account_id, account_type="enterprise")
        account.user_id = None
        account.organization_id = org_id
        txn = _make_transaction(account_id=account_id, amount=200)

        with patch("src.api.services.billing.BillingRepository") as MockRepo:
            repo = MockRepo.return_value
            repo.get_account_for_update = AsyncMock(return_value=account)
            repo.get_balance = AsyncMock(return_value=300)
            repo.create_transaction = AsyncMock(return_value=txn)
            repo.get_member_user_ids = AsyncMock(return_value=member_ids)

            result = await billing_service.admin_adjust(
                account_id,
                200,
                admin_id,
                description="enterprise bonus",
                session=mock_session,
                product_id="synthara",
            )

        assert result.event is not None
        assert set(result.event.user_ids) == set(member_ids)

    async def test_adjustment_result_returns_txn(
        self, billing_service: BillingService, mock_session: AsyncMock
    ) -> None:
        """admin_adjust always returns the created transaction via BillingResult.txn."""
        account_id = uuid4()
        account = _make_account(account_id=account_id, account_type="personal")
        account.user_id = uuid4()
        account.organization_id = None
        txn = _make_transaction(account_id=account_id, amount=50)

        with patch("src.api.services.billing.BillingRepository") as MockRepo:
            repo = MockRepo.return_value
            repo.get_account_for_update = AsyncMock(return_value=account)
            repo.get_balance = AsyncMock(return_value=100)
            repo.create_transaction = AsyncMock(return_value=txn)

            result = await billing_service.admin_adjust(
                account_id,
                50,
                uuid4(),
                description="no-bus test",
                session=mock_session,
                product_id="vex",
            )

        assert result.txn == txn

    async def test_enterprise_adjustment_no_members_no_event(
        self, billing_service: BillingService, mock_session: AsyncMock
    ) -> None:
        """Enterprise account with no members still adjusts correctly, just no BalanceEvent."""
        account_id = uuid4()
        org_id = uuid4()
        account = _make_account(account_id=account_id, account_type="enterprise")
        account.user_id = None
        account.organization_id = org_id
        txn = _make_transaction(account_id=account_id, amount=100)

        with patch("src.api.services.billing.BillingRepository") as MockRepo:
            repo = MockRepo.return_value
            repo.get_account_for_update = AsyncMock(return_value=account)
            repo.get_balance = AsyncMock(return_value=0)
            repo.create_transaction = AsyncMock(return_value=txn)
            repo.get_member_user_ids = AsyncMock(return_value=[])

            result = await billing_service.admin_adjust(
                account_id,
                100,
                uuid4(),
                description="empty org",
                session=mock_session,
                product_id="synthara",
            )

        assert result.event is None
