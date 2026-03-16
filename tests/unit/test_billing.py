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

        assert result.id == txn.id
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

        assert result.amount == 25
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
