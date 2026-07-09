"""Top-up route runtime provider-state and idempotency behavior."""

from unittest.mock import ANY, AsyncMock, MagicMock
from uuid import uuid4

import pytest

from src.api.routes.billing import BillingController
from src.api.schemas.billing import TopUpStripeRequest
from src.api.services.billing_errors import PaymentProviderDisabledError
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
