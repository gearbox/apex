"""Stripe gateway translation tests."""

from unittest.mock import MagicMock, patch

import pytest

from src.api.services.payments.contracts import WebhookEnvelope
from src.api.services.payments.stripe_gateway import StripeGateway
from src.core.config import Settings
from src.core.enums import PaymentStatus

pytestmark = pytest.mark.unit


def _settings() -> Settings:
    return Settings(
        jwt_secret_key="test-secret-key-that-is-definitely-long-enough-32bytes",
        stripe_webhook_secret_vex="whsec_test",
    )


async def test_completed_checkout_maps_to_normalized_outcome() -> None:
    event = MagicMock()
    event.id = "evt_1"
    event.type = "checkout.session.completed"
    event.data.object = {"id": "cs_1"}
    with patch("stripe.Webhook.construct_event", return_value=event):
        outcome = await StripeGateway(_settings()).verify_webhook(
            WebhookEnvelope(
                raw_body=b"{}",
                headers={"stripe-signature": "sig"},
                product_id="vex",
            )
        )
    assert outcome.lookup.by == "external_id"
    assert outcome.lookup.value == "cs_1"
    assert outcome.status is PaymentStatus.COMPLETED
    assert outcome.metadata_patch == {"webhook_event_id": "evt_1"}


async def test_non_checkout_event_is_ignored() -> None:
    event = MagicMock(id="evt_2", type="customer.created")
    with patch("stripe.Webhook.construct_event", return_value=event):
        outcome = await StripeGateway(_settings()).verify_webhook(
            WebhookEnvelope(
                raw_body=b"{}",
                headers={"stripe-signature": "sig"},
                product_id="vex",
            )
        )
    assert outcome.status is None
