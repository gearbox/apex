"""Stripe gateway translation tests."""

from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
import stripe

from src.api.services.billing_errors import PaymentVerificationError, PaymentVerificationReason
from src.api.services.payments.contracts import ChargeContext, WebhookEnvelope
from src.api.services.payments.stripe_gateway import StripeGateway
from src.core.config import Settings
from src.core.enums import PaymentStatus
from src.core.product_registry import VEX_CONFIG
from src.core.topup_pricing import build_quote, topup_tiers_for

pytestmark = pytest.mark.unit


def _settings() -> Settings:
    return Settings(
        jwt_secret_key="test-secret-key-that-is-definitely-long-enough-32bytes",
        stripe_secret_key_vex="sk_test_vex_123",
        stripe_webhook_secret_vex="whsec_test",
    )


async def test_create_charge_uses_per_product_key_and_frontend_origin() -> None:
    """C6/C7: per-product Stripe key and success/cancel URLs built from
    ProductConfig.frontend_origin, not a hardcoded placeholder."""
    settings = _settings()
    quote = build_quote(
        100, tiers=topup_tiers_for("vex", settings), tokens_per_usd=settings.billing_tokens_per_usd
    )

    fake_session = MagicMock(
        id="cs_test_abc123", url="https://checkout.stripe.com/pay/cs_test_abc123"
    )
    create_async_mock = AsyncMock(return_value=fake_session)
    captured: dict[str, object] = {}

    def _fake_stripe_client(*, api_key: str) -> MagicMock:
        captured["api_key"] = api_key
        client = MagicMock()
        client.v1.checkout.sessions.create_async = create_async_mock
        return client

    with patch("stripe.StripeClient", side_effect=_fake_stripe_client):
        result = await StripeGateway(settings).create_charge(
            ChargeContext(
                payment_id=uuid4(),
                account_id=uuid4(),
                user_id=uuid4(),
                product_config=VEX_CONFIG,
                quote=quote,
            )
        )

    assert captured["api_key"] == "sk_test_vex_123"
    assert result.external_id == "cs_test_abc123"
    assert result.redirect_url == fake_session.url

    params = create_async_mock.call_args.kwargs["params"]
    assert params["success_url"] == (
        f"{VEX_CONFIG.frontend_origin}{settings.stripe_checkout_success_path}"
    )
    assert params["cancel_url"] == (
        f"{VEX_CONFIG.frontend_origin}{settings.stripe_checkout_cancel_path}"
    )
    assert params["line_items"][0]["price_data"]["unit_amount"] == int(quote.total_due * 100)


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


async def test_missing_signature_header_raises() -> None:
    with pytest.raises(PaymentVerificationError, match="signature header missing") as exc_info:
        await StripeGateway(_settings()).verify_webhook(
            WebhookEnvelope(raw_body=b"{}", headers={}, product_id="vex")
        )
    assert exc_info.value.reason is PaymentVerificationReason.MISSING_SIGNATURE_HEADER


async def test_bad_signature_raises_signature_mismatch() -> None:
    with (
        patch(
            "stripe.Webhook.construct_event",
            side_effect=stripe.SignatureVerificationError(  # type: ignore[no-untyped-call]
                "Bad sig", "sig-header", b"{}"
            ),
        ),
        pytest.raises(PaymentVerificationError, match="signature invalid") as exc_info,
    ):
        await StripeGateway(_settings()).verify_webhook(
            WebhookEnvelope(
                raw_body=b"{}",
                headers={"stripe-signature": "sig"},
                product_id="vex",
            )
        )
    exc = exc_info.value
    assert exc.reason is PaymentVerificationReason.SIGNATURE_MISMATCH
    assert exc.context["error"] == "Bad sig"
    assert "{}" not in str(exc.context)


async def test_malformed_payload_raises_malformed_json() -> None:
    with (
        patch("stripe.Webhook.construct_event", side_effect=ValueError("bad payload")),
        pytest.raises(PaymentVerificationError, match="payload malformed") as exc_info,
    ):
        await StripeGateway(_settings()).verify_webhook(
            WebhookEnvelope(
                raw_body=b"not json",
                headers={"stripe-signature": "sig"},
                product_id="vex",
            )
        )
    exc = exc_info.value
    assert exc.reason is PaymentVerificationReason.MALFORMED_JSON
    assert exc.context["error"] == "bad payload"
