"""Top-up route runtime provider-state, amount-bounds, and idempotency behavior.

The provider-neutral refactor moved per-provider guard checks (support/enabled)
and amount-bounds checks inside ``PaymentService.create_charge`` — they used
to live as separate ``ProductConfig.supports_payment_provider()`` calls in the
route handler. These tests exercise the route handlers directly (via the
Litestar handler's ``.fn``, bypassing the full DI/app stack) to verify the
route-level idempotency-key envelope around whatever ``create_charge`` raises:
403 (disabled provider) and 400 (bad amount) both release the idempotency key
via ``idempotency_service.fail()``, and a generic exception is not misrouted
to a 400 response.
"""

from unittest.mock import ANY, AsyncMock, MagicMock
from uuid import uuid4

import msgspec
import pytest
from litestar.exceptions import HTTPException

from src.api.routes.billing import BillingController
from src.api.schemas.billing import (
    NowPaymentsInvoiceResponse,
    StripeCheckoutResponse,
    TopUpNowPaymentsRequest,
    TopUpStripeRequest,
)
from src.api.services.billing_errors import PaymentProviderDisabledError, TopUpAmountError
from src.api.services.payments import CreatedCharge
from src.core.product import PaymentProvider
from src.core.product_registry import VEX_CONFIG

pytestmark = pytest.mark.unit


async def test_disabled_provider_fails_idempotency_record() -> None:
    record_id = uuid4()
    idempotency = AsyncMock()
    idempotency.check = AsyncMock(return_value=record_id)
    billing = AsyncMock()
    billing.resolve_account_for_user = AsyncMock(return_value=MagicMock(id=uuid4()))
    payment = AsyncMock()
    payment.create_charge = AsyncMock(
        side_effect=PaymentProviderDisabledError(PaymentProvider.STRIPE)
    )
    with pytest.raises(PaymentProviderDisabledError):
        await BillingController.topup_stripe.fn(
            MagicMock(),
            current_user_id=uuid4(),
            data=TopUpStripeRequest(amount_usd=100),
            session=AsyncMock(),
            billing_service=billing,
            payment_service=payment,
            product_id="vex",
            product_config=VEX_CONFIG,
            idempotency_service=idempotency,
            idempotency_key_header="key-1",
        )
    idempotency.fail.assert_awaited_once_with(record_id, session=ANY)


async def test_success_maps_normalized_result_to_existing_schema() -> None:
    idempotency = AsyncMock()
    idempotency.check = AsyncMock(return_value=uuid4())
    billing = AsyncMock()
    billing.resolve_account_for_user = AsyncMock(return_value=MagicMock(id=uuid4()))
    payment_id = uuid4()
    payment = AsyncMock()
    payment.create_charge = AsyncMock(
        return_value=CreatedCharge(
            redirect_url="https://checkout.stripe/1",
            external_id="cs_1",
            payment_id=payment_id,
        )
    )
    response = await BillingController.topup_stripe.fn(
        MagicMock(),
        current_user_id=uuid4(),
        data=TopUpStripeRequest(amount_usd=100),
        session=AsyncMock(),
        billing_service=billing,
        payment_service=payment,
        product_id="vex",
        product_config=VEX_CONFIG,
        idempotency_service=idempotency,
        idempotency_key_header="key-1",
    )
    assert response.content.checkout_url == "https://checkout.stripe/1"
    assert response.content.session_id == "cs_1"


class TestTopupAmountValidation:
    """Ported from the pre-refactor TestTopupAmountValidation: amount bounds
    are now enforced inside PaymentService.create_charge (TopUpAmountError),
    common to every provider — the route-level 400 mapping is unchanged."""

    async def test_topup_stripe_amount_error_returns_400_and_releases_key(self) -> None:
        record_id = uuid4()
        billing_service = AsyncMock()
        billing_service.resolve_account_for_user = AsyncMock(return_value=MagicMock(id=uuid4()))
        payment_service = AsyncMock()
        payment_service.create_charge = AsyncMock(
            side_effect=TopUpAmountError("amount_usd must be between 5 and 10000 (got 1)")
        )
        idempotency_service = AsyncMock()
        idempotency_service.check = AsyncMock(return_value=record_id)

        with pytest.raises(HTTPException) as exc_info:
            await BillingController.topup_stripe.fn(
                MagicMock(),
                current_user_id=uuid4(),
                data=TopUpStripeRequest(amount_usd=1),
                session=AsyncMock(),
                billing_service=billing_service,
                payment_service=payment_service,
                product_id="vex",
                product_config=VEX_CONFIG,
                idempotency_service=idempotency_service,
                idempotency_key_header="key-amount-low",
            )

        assert exc_info.value.status_code == 400
        assert "amount_usd must be between" in str(exc_info.value.detail)
        idempotency_service.fail.assert_awaited_once_with(record_id, session=ANY)

    async def test_topup_nowpayments_amount_error_returns_400_and_releases_key(self) -> None:
        record_id = uuid4()
        billing_service = AsyncMock()
        billing_service.resolve_account_for_user = AsyncMock(return_value=MagicMock(id=uuid4()))
        payment_service = AsyncMock()
        payment_service.create_charge = AsyncMock(
            side_effect=TopUpAmountError("amount_usd must be between 5 and 10000 (got 10001)")
        )
        idempotency_service = AsyncMock()
        idempotency_service.check = AsyncMock(return_value=record_id)

        with pytest.raises(HTTPException) as exc_info:
            await BillingController.topup_nowpayments.fn(
                MagicMock(),
                current_user_id=uuid4(),
                data=TopUpNowPaymentsRequest(amount_usd=10001, pay_currency="btc"),
                session=AsyncMock(),
                billing_service=billing_service,
                payment_service=payment_service,
                product_id="vex",
                product_config=VEX_CONFIG,
                idempotency_service=idempotency_service,
                idempotency_key_header="key-amount-high",
            )

        assert exc_info.value.status_code == 400
        assert "amount_usd must be between" in str(exc_info.value.detail)
        idempotency_service.fail.assert_awaited_once_with(record_id, session=ANY)

    async def test_topup_stripe_generic_exception_is_not_mapped_to_400(self) -> None:
        """A configuration/runtime error (e.g. unconfigured provider key) must
        propagate as-is — only TopUpAmountError and PaymentProviderDisabledError
        get their own handling; everything else is a 500 upstream."""
        record_id = uuid4()
        billing_service = AsyncMock()
        billing_service.resolve_account_for_user = AsyncMock(return_value=MagicMock(id=uuid4()))
        payment_service = AsyncMock()
        payment_service.create_charge = AsyncMock(
            side_effect=RuntimeError("Stripe secret key not configured for product vex")
        )
        idempotency_service = AsyncMock()
        idempotency_service.check = AsyncMock(return_value=record_id)

        with pytest.raises(RuntimeError, match="Stripe secret key"):
            await BillingController.topup_stripe.fn(
                MagicMock(),
                current_user_id=uuid4(),
                data=TopUpStripeRequest(amount_usd=100),
                session=AsyncMock(),
                billing_service=billing_service,
                payment_service=payment_service,
                product_id="vex",
                product_config=VEX_CONFIG,
                idempotency_service=idempotency_service,
                idempotency_key_header="key-config-error",
            )

        idempotency_service.fail.assert_awaited_once_with(record_id, session=ANY)


class TestRejectedTopupIdempotencyKeyIsReusable:
    """Ported from the pre-refactor TestRejectedTopupIdempotencyKey (D1): a
    request rejected via PaymentProviderDisabledError must release its
    idempotency key (via fail()) so a retry with the *same* key against a
    supported provider proceeds as a brand-new request."""

    async def test_disabled_provider_then_retry_with_same_key_succeeds(self) -> None:
        shared_key = "shared-idempotency-key"
        record_id = uuid4()
        idempotency_service = AsyncMock()
        idempotency_service.check = AsyncMock(return_value=record_id)
        billing_service = AsyncMock()
        billing_service.resolve_account_for_user = AsyncMock(return_value=MagicMock(id=uuid4()))
        payment_service = AsyncMock()
        payment_service.create_charge = AsyncMock(
            side_effect=PaymentProviderDisabledError(PaymentProvider.NOWPAYMENTS)
        )

        with pytest.raises(PaymentProviderDisabledError):
            await BillingController.topup_nowpayments.fn(
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
        idempotency_service.fail.assert_awaited_once_with(record_id, session=ANY)

        # Retry with the same key: check() is invoked fresh (fail() unlocked
        # it) and this time the charge succeeds.
        idempotency_service.check = AsyncMock(return_value=uuid4())
        payment_service.create_charge = AsyncMock(
            return_value=CreatedCharge(
                redirect_url="https://nowpayments.io/pay/xyz",
                external_id="np_1",
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
        assert isinstance(response.content, NowPaymentsInvoiceResponse)


async def test_topup_stripe_success_response_shape() -> None:
    idempotency_service = AsyncMock()
    idempotency_service.check = AsyncMock(return_value=uuid4())
    billing_service = AsyncMock()
    billing_service.resolve_account_for_user = AsyncMock(return_value=MagicMock(id=uuid4()))
    payment_service = AsyncMock()
    payment_service.create_charge = AsyncMock(
        return_value=CreatedCharge(
            redirect_url="https://checkout.stripe.com/x",
            external_id="cs_test_1",
            payment_id=uuid4(),
        )
    )

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


class TestOptionalPayCurrency:
    """D1: an unset pay_currency deserializes fine, and the route forwards
    ``extra`` only when a non-blank ticker was pinned — the gateway then owns
    the omission-from-payload decision (see nowpayments_gateway tests)."""

    def test_request_without_pay_currency_deserializes(self) -> None:
        request = msgspec.json.decode(b'{"amount_usd": 100}', type=TopUpNowPaymentsRequest)
        assert request.pay_currency is None

    def test_request_with_pay_currency_deserializes(self) -> None:
        request = msgspec.json.decode(
            b'{"amount_usd": 100, "pay_currency": "btc"}', type=TopUpNowPaymentsRequest
        )
        assert request.pay_currency == "btc"

    async def test_unset_pay_currency_forwards_empty_extra(self) -> None:
        idempotency_service = AsyncMock()
        idempotency_service.check = AsyncMock(return_value=uuid4())
        billing_service = AsyncMock()
        billing_service.resolve_account_for_user = AsyncMock(return_value=MagicMock(id=uuid4()))
        payment_service = AsyncMock()
        payment_service.create_charge = AsyncMock(
            return_value=CreatedCharge(
                redirect_url="https://nowpayments.io/pay/xyz",
                external_id="np_1",
                payment_id=uuid4(),
            )
        )

        await BillingController.topup_nowpayments.fn(
            MagicMock(),
            current_user_id=uuid4(),
            data=TopUpNowPaymentsRequest(amount_usd=100),
            session=AsyncMock(),
            billing_service=billing_service,
            payment_service=payment_service,
            product_id="vex",
            product_config=VEX_CONFIG,
            idempotency_service=idempotency_service,
            idempotency_key_header="key-unset-currency",
        )

        assert payment_service.create_charge.await_args.kwargs["extra"] == {}

    async def test_whitespace_only_pay_currency_forwards_empty_extra(self) -> None:
        """Normalize with .strip() before the truthiness check so a single
        space doesn't pin a bogus ticker."""
        idempotency_service = AsyncMock()
        idempotency_service.check = AsyncMock(return_value=uuid4())
        billing_service = AsyncMock()
        billing_service.resolve_account_for_user = AsyncMock(return_value=MagicMock(id=uuid4()))
        payment_service = AsyncMock()
        payment_service.create_charge = AsyncMock(
            return_value=CreatedCharge(
                redirect_url="https://nowpayments.io/pay/xyz",
                external_id="np_1",
                payment_id=uuid4(),
            )
        )

        await BillingController.topup_nowpayments.fn(
            MagicMock(),
            current_user_id=uuid4(),
            data=TopUpNowPaymentsRequest(amount_usd=100, pay_currency=" "),
            session=AsyncMock(),
            billing_service=billing_service,
            payment_service=payment_service,
            product_id="vex",
            product_config=VEX_CONFIG,
            idempotency_service=idempotency_service,
            idempotency_key_header="key-whitespace-currency",
        )

        assert payment_service.create_charge.await_args.kwargs["extra"] == {}

    async def test_pinned_pay_currency_is_stripped_and_forwarded(self) -> None:
        idempotency_service = AsyncMock()
        idempotency_service.check = AsyncMock(return_value=uuid4())
        billing_service = AsyncMock()
        billing_service.resolve_account_for_user = AsyncMock(return_value=MagicMock(id=uuid4()))
        payment_service = AsyncMock()
        payment_service.create_charge = AsyncMock(
            return_value=CreatedCharge(
                redirect_url="https://nowpayments.io/pay/xyz",
                external_id="np_1",
                payment_id=uuid4(),
            )
        )

        await BillingController.topup_nowpayments.fn(
            MagicMock(),
            current_user_id=uuid4(),
            data=TopUpNowPaymentsRequest(amount_usd=100, pay_currency="  usdcmatic  "),
            session=AsyncMock(),
            billing_service=billing_service,
            payment_service=payment_service,
            product_id="vex",
            product_config=VEX_CONFIG,
            idempotency_service=idempotency_service,
            idempotency_key_header="key-pinned-currency",
        )

        assert payment_service.create_charge.await_args.kwargs["extra"] == {
            "pay_currency": "usdcmatic"
        }
