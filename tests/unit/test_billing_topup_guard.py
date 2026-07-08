"""Tests for the per-product payment-provider guard on top-up routes (F1/D1).

ProductConfig.supports_payment_provider() previously had zero call sites —
both top-up endpoints accepted any provider for any product. These tests
exercise the route handlers directly (via the Litestar handler's ``.fn``,
bypassing the full DI/app stack) to verify the guard rejects unsupported
providers with 403 *before* the idempotency key is consumed.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest
from litestar.exceptions import HTTPException

from src.api.routes.billing import BillingController
from src.api.schemas.billing import (
    NowPaymentsInvoiceResponse,
    StripeCheckoutResponse,
    TopUpNowPaymentsRequest,
    TopUpStripeRequest,
)
from src.core.product_registry import SYNTHARA_CONFIG, VEX_CONFIG

pytestmark = pytest.mark.unit


class TestTopupStripeProviderGuard:
    async def test_topup_stripe_403_when_provider_unsupported(self) -> None:
        """Neither real product currently lacks Stripe, so exercise the guard
        against a bare product_config double reporting STRIPE unsupported —
        this pins the guard's behavior (403, no idempotency check) independent
        of which real products happen to support Stripe today."""
        idempotency_service = AsyncMock()
        product_config = MagicMock()
        product_config.supports_payment_provider = MagicMock(return_value=False)

        with pytest.raises(HTTPException) as exc_info:
            await BillingController.topup_stripe.fn(
                MagicMock(),
                current_user_id=uuid4(),
                data=TopUpStripeRequest(amount_usd=100),
                session=AsyncMock(),
                billing_service=AsyncMock(),
                payment_service=AsyncMock(),
                product_id="vex",
                product_config=product_config,
                idempotency_service=idempotency_service,
                idempotency_key_header="key-1",
            )

        assert exc_info.value.status_code == 403
        idempotency_service.check.assert_not_awaited()

    async def test_topup_stripe_succeeds_when_provider_supported(self) -> None:
        billing_service = AsyncMock()
        billing_service.resolve_account_for_user = AsyncMock(return_value=MagicMock(id=uuid4()))
        payment_service = AsyncMock()
        payment_service.create_stripe_checkout = AsyncMock(
            return_value=MagicMock(
                checkout_url="https://checkout.stripe.com/x",
                session_id="cs_test_1",
                payment_id=uuid4(),
            )
        )
        idempotency_service = AsyncMock()
        idempotency_service.check = AsyncMock(return_value=uuid4())

        response = await BillingController.topup_stripe.fn(
            MagicMock(),
            current_user_id=uuid4(),
            data=TopUpStripeRequest(amount_usd=100),
            session=AsyncMock(),
            billing_service=billing_service,
            payment_service=payment_service,
            product_id="vex",
            product_config=VEX_CONFIG,
            idempotency_service=idempotency_service,
            idempotency_key_header="key-1",
        )

        idempotency_service.check.assert_awaited_once()
        assert response.status_code == 201
        assert isinstance(response.content, StripeCheckoutResponse)


class TestTopupNowPaymentsProviderGuard:
    async def test_topup_nowpayments_403_when_provider_unsupported(self) -> None:
        """Synthara's ProductConfig does not include NOWPAYMENTS in
        payment_providers — the real, current configuration for this guard."""
        idempotency_service = AsyncMock()

        with pytest.raises(HTTPException) as exc_info:
            await BillingController.topup_nowpayments.fn(
                MagicMock(),
                current_user_id=uuid4(),
                data=TopUpNowPaymentsRequest(amount_usd=100, pay_currency="btc"),
                session=AsyncMock(),
                billing_service=AsyncMock(),
                payment_service=AsyncMock(),
                product_id="synthara",
                product_config=SYNTHARA_CONFIG,
                idempotency_service=idempotency_service,
                idempotency_key_header="key-1",
            )

        assert exc_info.value.status_code == 403
        idempotency_service.check.assert_not_awaited()

    async def test_topup_nowpayments_succeeds_when_provider_supported(self) -> None:
        billing_service = AsyncMock()
        billing_service.resolve_account_for_user = AsyncMock(return_value=MagicMock(id=uuid4()))
        payment_service = AsyncMock()
        payment_service.create_nowpayments_invoice = AsyncMock(
            return_value=MagicMock(
                invoice_url="https://nowpayments.io/pay/xyz",
                payment_id=uuid4(),
            )
        )
        idempotency_service = AsyncMock()
        idempotency_service.check = AsyncMock(return_value=uuid4())

        response = await BillingController.topup_nowpayments.fn(
            MagicMock(),
            current_user_id=uuid4(),
            data=TopUpNowPaymentsRequest(amount_usd=100, pay_currency="btc"),
            session=AsyncMock(),
            billing_service=billing_service,
            payment_service=payment_service,
            product_id="vex",
            product_config=VEX_CONFIG,
            idempotency_service=idempotency_service,
            idempotency_key_header="key-1",
        )

        idempotency_service.check.assert_awaited_once()
        assert response.status_code == 201
        assert isinstance(response.content, NowPaymentsInvoiceResponse)


class TestRejectedTopupIdempotencyKey:
    async def test_rejected_topup_does_not_consume_idempotency_key(self) -> None:
        """A 403-rejected top-up must not call idempotency_service.check() at
        all — the guard runs strictly before it (D1) — so a later call with
        the *same* key against a supported provider succeeds fresh, as if the
        rejected attempt never happened."""
        idempotency_service = AsyncMock()
        shared_key = "shared-idempotency-key"

        # First: rejected (NowPayments unsupported on Synthara).
        with pytest.raises(HTTPException) as exc_info:
            await BillingController.topup_nowpayments.fn(
                MagicMock(),
                current_user_id=uuid4(),
                data=TopUpNowPaymentsRequest(amount_usd=100, pay_currency="btc"),
                session=AsyncMock(),
                billing_service=AsyncMock(),
                payment_service=AsyncMock(),
                product_id="synthara",
                product_config=SYNTHARA_CONFIG,
                idempotency_service=idempotency_service,
                idempotency_key_header=shared_key,
            )
        assert exc_info.value.status_code == 403
        idempotency_service.check.assert_not_awaited()

        # Second: same key, supported product/provider — must proceed as a
        # brand-new request (idempotency_service.check is finally invoked).
        idempotency_service.check = AsyncMock(return_value=uuid4())
        billing_service = AsyncMock()
        billing_service.resolve_account_for_user = AsyncMock(return_value=MagicMock(id=uuid4()))
        payment_service = AsyncMock()
        payment_service.create_nowpayments_invoice = AsyncMock(
            return_value=MagicMock(
                invoice_url="https://nowpayments.io/pay/xyz",
                payment_id=uuid4(),
            )
        )

        response = await BillingController.topup_nowpayments.fn(
            MagicMock(),
            current_user_id=uuid4(),
            data=TopUpNowPaymentsRequest(amount_usd=100, pay_currency="btc"),
            session=AsyncMock(),
            billing_service=billing_service,
            payment_service=payment_service,
            product_id="vex",
            product_config=VEX_CONFIG,
            idempotency_service=idempotency_service,
            idempotency_key_header=shared_key,
        )

        idempotency_service.check.assert_awaited_once()
        check_kwargs = idempotency_service.check.await_args.kwargs
        assert check_kwargs["idempotency_key"] == shared_key
        assert check_kwargs["product_id"] == "vex"
        assert response.status_code == 201
