"""Tests for BillingController routes not covered by the topup provider guard
tests — the top-up options endpoint (C5) and the idempotency-hash/amount
interaction (C5 note)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import msgspec
import pytest

from src.api.routes.billing import BillingController
from src.api.schemas.billing import TopUpOptionsResponse, TopUpStripeRequest
from src.api.services.idempotency import IdempotencyConflictError, IdempotencyService
from src.core.config import Settings

pytestmark = pytest.mark.unit

_JWT_SECRET = "test-secret-key-that-is-definitely-long-enough-32bytes"


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
