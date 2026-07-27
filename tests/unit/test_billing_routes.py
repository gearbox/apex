"""Tests for BillingController routes not covered by the topup provider guard
tests — the top-up options endpoint (C5) and the idempotency-hash/amount
interaction (C5 note)."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import msgspec
import pytest

from src.api.routes.billing import BillingController
from src.api.schemas.billing import TopUpOptionsResponse, TopUpStripeRequest
from src.api.services.idempotency import IdempotencyConflictError, IdempotencyService
from src.core.config import Settings
from src.core.product_registry import VEX_CONFIG

pytestmark = pytest.mark.unit

_JWT_SECRET = "test-secret-key-that-is-definitely-long-enough-32bytes"


class TestGetTransactions:
    """Issue A regression: the transaction history response must carry the
    neutral payment_method discriminator and never the gateway name."""

    async def test_credit_row_exposes_payment_method_not_gateway_name(self) -> None:
        txn = MagicMock(
            id=uuid4(),
            transaction_type="credit",
            amount=1000,
            balance_after=1000,
            description="Token purchase via crypto payment",
            metadata_={"payment_method": "crypto"},
            job_id=None,
            payment_id=uuid4(),
            created_at=datetime.now(UTC),
            created_by=None,
        )
        billing_service = AsyncMock()
        billing_service.resolve_account_for_user = AsyncMock(return_value=MagicMock(id=uuid4()))
        billing_service.get_transaction_history = AsyncMock(return_value=[txn])

        result = await BillingController.get_transactions.fn(
            MagicMock(),
            current_user_id=uuid4(),
            session=AsyncMock(),
            billing_service=billing_service,
        )

        assert result.items[0].payment_method == "crypto"

        body = msgspec.json.encode(result)
        assert b"nowpayments" not in body
        assert b"stripe" not in body


class TestGetTopupOptions:
    async def test_topup_options_reflects_settings(self) -> None:
        settings = Settings(
            jwt_secret_key=_JWT_SECRET,
            billing_pricing_tiers={5: 0, 20: 5, 90: 15},
            billing_tokens_per_usd=42,
            billing_min_topup_usd=5,
            billing_max_topup_usd=500,
        )

        result = await BillingController.get_topup_options.fn(
            MagicMock(), settings=settings, product_id="vex"
        )

        assert isinstance(result, TopUpOptionsResponse)
        assert result.min_amount_usd == 5
        assert result.max_amount_usd == 500
        assert result.tokens_per_usd == 42
        assert [(t.threshold_usd, t.discount_pct) for t in result.tiers] == [
            (5, 0),
            (20, 5),
            (90, 15),
        ]

    async def test_topup_options_uses_default_settings_tiers(self) -> None:
        settings = Settings(jwt_secret_key=_JWT_SECRET)

        result = await BillingController.get_topup_options.fn(
            MagicMock(), settings=settings, product_id="vex"
        )

        assert [(t.threshold_usd, t.discount_pct) for t in result.tiers] == [
            (10, 0),
            (50, 0),
            (100, 5),
            (250, 10),
        ]


class TestGetPricing:
    async def test_pricing_response_includes_input_token_cost(self) -> None:
        rule = MagicMock()
        rule.id = uuid4()
        rule.provider = "grok"
        rule.generation_type = "i2i"
        rule.model = "grok-imagine-image"
        rule.token_cost = 20
        rule.input_token_cost = 3
        rule.is_active = True
        rule.effective_from = datetime.now(UTC)
        rule.effective_until = None
        rule.notes = None

        pricing_service = AsyncMock()
        pricing_service.list_catalog = AsyncMock(return_value=[rule])

        result = await BillingController.get_pricing.fn(
            MagicMock(),
            session=AsyncMock(),
            pricing_service=pricing_service,
            product_config=VEX_CONFIG,
        )

        assert result[0].token_cost == 20
        assert result[0].input_token_cost == 3


class TestIdempotencyKeyAmountConflict:
    async def test_same_idempotency_key_different_amount_conflicts(self) -> None:
        """The idempotency request hash covers the full request body, so a
        retried key with a different amount_usd must be rejected as a
        conflict rather than silently reusing the cached response."""
        low_hash = IdempotencyService.hash_request(
            msgspec.json.encode(TopUpStripeRequest(amount_usd=10))
        )
        high_hash = IdempotencyService.hash_request(
            msgspec.json.encode(TopUpStripeRequest(amount_usd=100))
        )
        assert low_hash != high_hash

    async def test_check_raises_conflict_when_hash_differs_for_same_key(self) -> None:
        """End-to-end through the real IdempotencyService: acquiring a key
        with one amount, then reusing it with a different amount, raises
        IdempotencyConflictError instead of replaying or silently proceeding."""
        service = IdempotencyService()
        user_id = uuid4()
        shared_key = "shared-key"

        first_hash = IdempotencyService.hash_request(
            msgspec.json.encode(TopUpStripeRequest(amount_usd=10))
        )
        second_hash = IdempotencyService.hash_request(
            msgspec.json.encode(TopUpStripeRequest(amount_usd=100))
        )

        existing_record = MagicMock()
        existing_record.request_hash = first_hash

        mock_repo = MagicMock()
        mock_repo.try_acquire = AsyncMock(return_value=None)  # key already exists
        mock_repo.get_existing = AsyncMock(return_value=existing_record)

        with (
            patch("src.api.services.idempotency.IdempotencyRepository", return_value=mock_repo),
            pytest.raises(IdempotencyConflictError, match="different request body"),
        ):
            await service.check(
                user_id=user_id,
                product_id="vex",
                idempotency_key=shared_key,
                operation="payment",
                request_hash=second_hash,
                session=AsyncMock(),
            )
