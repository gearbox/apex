"""Stripe Checkout payment gateway adapter."""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING, Any

import stripe

from src.api.services.billing_errors import PaymentVerificationError, PaymentVerificationReason
from src.api.services.payments.contracts import (
    ChargeContext,
    ChargeResult,
    PaymentLookup,
    WebhookEnvelope,
    WebhookOutcome,
)
from src.core.enums import PaymentStatus
from src.core.product import PaymentProvider

if TYPE_CHECKING:
    from src.core.config import Settings


class StripeGateway:
    """Translate Stripe Checkout API calls and webhook events."""

    provider = PaymentProvider.STRIPE

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    async def create_charge(self, ctx: ChargeContext) -> ChargeResult:
        product_id = ctx.product_config.slug
        quote = ctx.quote
        client = stripe.StripeClient(api_key=self._settings.stripe_secret_key_for(product_id))
        checkout_session = await client.v1.checkout.sessions.create_async(
            params={
                "mode": "payment",
                "line_items": [
                    {
                        "price_data": {
                            "currency": "usd",
                            "product_data": {
                                "name": "Usage credits",
                                "description": (
                                    f"{quote.tokens_granted} tokens "
                                    f"(${quote.credits_usd} credit, {quote.discount_pct}% off)"
                                ),
                            },
                            "unit_amount": int(quote.total_due * 100),
                        },
                        "quantity": 1,
                    }
                ],
                "metadata": {
                    "account_id": str(ctx.account_id),
                    "credits_usd": str(quote.credits_usd),
                },
                "success_url": (
                    f"{ctx.product_config.frontend_origin}"
                    f"{self._settings.stripe_checkout_success_path}"
                ),
                "cancel_url": (
                    f"{ctx.product_config.frontend_origin}"
                    f"{self._settings.stripe_checkout_cancel_path}"
                ),
            }
        )
        return ChargeResult(
            external_id=checkout_session.id,
            redirect_url=checkout_session.url or "",
            currency="USD",
            provider_metadata={
                "checkout_session_id": checkout_session.id,
                "credits_usd": quote.credits_usd,
                "discount_pct": quote.discount_pct,
            },
        )

    async def verify_webhook(self, envelope: WebhookEnvelope) -> WebhookOutcome:
        sig_header = envelope.headers.get("stripe-signature", "")
        if not sig_header:
            raise PaymentVerificationError(
                "Stripe signature header missing",
                reason=PaymentVerificationReason.MISSING_SIGNATURE_HEADER,
                context={"body_len": str(len(envelope.raw_body))},
            )

        # SignatureVerificationError.__str__ returns only its `message` arg
        # (StripeError.__str__ never surfaces http_body/sig_header), so str(exc)
        # is safe to log — it never leaks the payload or secret (D2).
        try:
            event = stripe.Webhook.construct_event(
                envelope.raw_body,
                sig_header,
                self._settings.stripe_webhook_secret_for(envelope.product_id),
            )
        except stripe.SignatureVerificationError as exc:
            raise PaymentVerificationError(
                "Stripe signature invalid",
                reason=PaymentVerificationReason.SIGNATURE_MISMATCH,
                context={
                    "error": str(exc),
                    "body_len": str(len(envelope.raw_body)),
                    "body_sha256_prefix": hashlib.sha256(envelope.raw_body).hexdigest()[:16],
                },
            ) from exc
        except ValueError as exc:
            raise PaymentVerificationError(
                "Stripe payload malformed",
                reason=PaymentVerificationReason.MALFORMED_JSON,
                context={"error": str(exc), "body_len": str(len(envelope.raw_body))},
            ) from exc

        event_id = str(event.id)
        if event.type != "checkout.session.completed":
            return WebhookOutcome(
                lookup=PaymentLookup(by="external_id", value=event_id),
                status=None,
            )

        checkout_session: Any = event.data.object
        return WebhookOutcome(
            lookup=PaymentLookup(by="external_id", value=str(checkout_session["id"])),
            status=PaymentStatus.COMPLETED,
            metadata_patch={"webhook_event_id": event_id},
        )
