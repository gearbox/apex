"""NowPayments invoice and IPN gateway adapter."""

from __future__ import annotations

import hashlib
import hmac
import json
from decimal import Decimal, InvalidOperation
from typing import TYPE_CHECKING, Any
from uuid import UUID

import httpx
import structlog

from src.api.services.billing_errors import PaymentVerificationError
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
    from collections.abc import Callable

    from src.core.config import Settings

logger = structlog.get_logger(__name__)

_STATUS_MAP: dict[str, PaymentStatus] = {
    "waiting": PaymentStatus.PENDING,
    "confirming": PaymentStatus.PENDING,
    "sending": PaymentStatus.PENDING,
    "partially_paid": PaymentStatus.PARTIALLY_PAID,
    "failed": PaymentStatus.FAILED,
    "expired": PaymentStatus.FAILED,
    "refunded": PaymentStatus.REFUNDED,
    "finished": PaymentStatus.COMPLETED,
}


class NowPaymentsGateway:
    """Translate NowPayments invoices and signed IPN payloads."""

    provider = PaymentProvider.NOWPAYMENTS

    def __init__(
        self,
        settings: Settings,
        client_factory: Callable[[], httpx.AsyncClient] | None = None,
    ) -> None:
        self._settings = settings
        self._client_factory = client_factory or httpx.AsyncClient

    async def create_charge(self, ctx: ChargeContext) -> ChargeResult:
        quote = ctx.quote
        product_id = ctx.product_config.slug
        pay_currency = ctx.extra.get("pay_currency", "")
        order_id = json.dumps(
            {
                "account_id": str(ctx.account_id),
                "payment_id": str(ctx.payment_id),
                "credits_usd": quote.credits_usd,
            }
        )

        async with self._client_factory() as client:
            response = await client.post(
                f"{self._settings.nowpayments_api_base.rstrip('/')}/v1/invoice",
                headers={
                    "x-api-key": self._settings.nowpayments_api_key_for(product_id),
                    "Content-Type": "application/json",
                },
                json={
                    "price_amount": float(quote.total_due),
                    "price_currency": "usd",
                    "pay_currency": pay_currency,
                    "order_id": order_id,
                    "order_description": (
                        f"{quote.tokens_granted} tokens "
                        f"(${quote.credits_usd} credit, {quote.discount_pct}% off)"
                    ),
                },
            )
            try:
                response.raise_for_status()
            except httpx.HTTPStatusError:
                logger.warning(
                    "payment.invoice_creation_failed",
                    provider=self.provider.value,
                    status_code=response.status_code,
                    body=response.text,
                    product_id=product_id,
                )
                raise
            data: dict[str, Any] = response.json()

        return ChargeResult(
            external_id=str(data.get("id", "")),
            redirect_url=str(data.get("invoice_url", "")),
            currency=pay_currency.upper(),
            provider_metadata={
                **data,
                "credits_usd": quote.credits_usd,
                "discount_pct": quote.discount_pct,
            },
        )

    async def verify_webhook(self, envelope: WebhookEnvelope) -> WebhookOutcome:
        try:
            parsed = json.loads(envelope.raw_body, parse_float=str, parse_int=str)
        except json.JSONDecodeError as exc:
            raise PaymentVerificationError("NowPayments IPN payload is not valid JSON") from exc
        if not isinstance(parsed, dict):
            raise PaymentVerificationError("NowPayments IPN payload must be a JSON object")

        canonical = json.dumps(parsed, sort_keys=True, separators=(",", ":")).encode()
        expected = hmac.new(
            self._settings.nowpayments_ipn_secret_for(envelope.product_id).encode(),
            canonical,
            hashlib.sha512,
        ).hexdigest()
        if not hmac.compare_digest(expected, envelope.headers.get("x-nowpayments-sig", "")):
            raise PaymentVerificationError("NowPayments HMAC signature invalid")

        try:
            order = json.loads(str(parsed.get("order_id", "")))
            internal_payment_id = UUID(str(order["payment_id"]))
        except (json.JSONDecodeError, KeyError, ValueError, TypeError) as exc:
            raise PaymentVerificationError("NowPayments IPN order_id is malformed") from exc

        raw_status = str(parsed.get("payment_status", ""))
        status = _STATUS_MAP.get(raw_status)
        if status is None:
            logger.warning(
                "payment.ipn_unknown_status",
                internal_payment_id=str(internal_payment_id),
                raw_status=raw_status,
            )

        amount_paid: Decimal | None = None
        if status in {PaymentStatus.COMPLETED, PaymentStatus.PARTIALLY_PAID}:
            paid_raw = parsed.get("actually_paid") or parsed.get("price_amount") or "0"
            try:
                amount_paid = Decimal(str(paid_raw))
            except InvalidOperation as exc:
                raise PaymentVerificationError(
                    "NowPayments IPN actually_paid is malformed"
                ) from exc

        metadata: dict[str, Any] = {
            "ipn_payment_id": str(parsed.get("payment_id", "")),
            "ipn_payload": parsed,
        }
        if amount_paid is not None:
            metadata["ipn_actually_paid"] = str(amount_paid)

        return WebhookOutcome(
            lookup=PaymentLookup(by="payment_id", value=str(internal_payment_id)),
            status=status,
            metadata_patch=metadata,
            amount_paid=amount_paid,
        )
