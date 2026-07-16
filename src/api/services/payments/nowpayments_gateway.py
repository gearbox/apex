"""NowPayments invoice, IPN, and currency-catalog gateway adapter."""

from __future__ import annotations

import hashlib
import hmac
import json
from decimal import Decimal, InvalidOperation
from typing import TYPE_CHECKING, Any
from urllib.parse import urljoin, urlparse
from uuid import UUID

import httpx
import structlog

from src.api.services.billing_errors import PaymentCatalogError, PaymentVerificationError
from src.api.services.payments.catalog import CurrencyDetails
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

# NowPayments' currency logo_url values are frequently site-relative
# (e.g. "/images/coins/btc.svg") and resolve against their marketing site,
# not the api.nowpayments.io API base. Absolute logo_url values are only
# accepted from these observed nowpayments-owned hosts (D11) — anything
# else fails loud rather than silently fetching from an arbitrary host.
_NOWPAYMENTS_SITE_BASE = "https://nowpayments.io"
_ALLOWED_LOGO_HOSTS = frozenset({"nowpayments.io", "www.nowpayments.io"})

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
        pay_currency = ctx.extra.get("pay_currency", "").strip()
        order_id = json.dumps(
            {
                "account_id": str(ctx.account_id),
                "payment_id": str(ctx.payment_id),
                "credits_usd": quote.credits_usd,
            }
        )

        payload: dict[str, Any] = {
            "price_amount": float(quote.total_due),
            "price_currency": "usd",
            "order_id": order_id,
            "order_description": (
                f"{quote.tokens_granted} tokens "
                f"(${quote.credits_usd} credit, {quote.discount_pct}% off)"
            ),
        }
        if pay_currency:
            payload["pay_currency"] = pay_currency

        async with self._client_factory() as client:
            response = await client.post(
                f"{self._settings.nowpayments_api_base.rstrip('/')}/v1/invoice",
                headers={
                    "x-api-key": self._settings.nowpayments_api_key_for(product_id),
                    "Content-Type": "application/json",
                },
                json=payload,
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
            currency=pay_currency.upper() if pay_currency else "USD",
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

        # HMAC compatibility contract: the NowPayments dashboard IPN format MUST be
        # "All-Strings". parse_float=str/parse_int=str + canonical json.dumps
        # round-trips All-Strings bodies byte-identically; the "Classic" format
        # (raw JSON numbers) re-serializes differently and fails every signature.
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
        amount_due: Decimal | None = None
        if status in {PaymentStatus.COMPLETED, PaymentStatus.PARTIALLY_PAID}:
            paid_raw = parsed.get("actually_paid")
            due_raw = parsed.get("pay_amount")
            if paid_raw is None or due_raw is None:
                raise PaymentVerificationError(
                    "NowPayments IPN missing actually_paid/pay_amount for settled status"
                )
            try:
                amount_paid = Decimal(str(paid_raw))
                amount_due = Decimal(str(due_raw))
            except InvalidOperation as exc:
                raise PaymentVerificationError(
                    "NowPayments IPN actually_paid/pay_amount is malformed"
                ) from exc
            if amount_due <= 0:
                raise PaymentVerificationError("NowPayments IPN pay_amount must be positive")

        metadata: dict[str, Any] = {
            "ipn_payment_id": str(parsed.get("payment_id", "")),
            "ipn_payload": parsed,
        }
        if amount_paid is not None:
            metadata["ipn_actually_paid"] = str(amount_paid)
            metadata["ipn_pay_amount"] = str(amount_due)

        settled_currency: str | None = None
        if raw_pay_currency := parsed.get("pay_currency"):
            settled_currency = str(raw_pay_currency).upper()
            metadata["ipn_pay_currency"] = settled_currency

        return WebhookOutcome(
            lookup=PaymentLookup(by="payment_id", value=str(internal_payment_id)),
            status=status,
            metadata_patch=metadata,
            amount_paid=amount_paid,
            amount_due=amount_due,
            settled_currency=settled_currency,
        )

    async def list_merchant_currencies(self, product_id: str) -> list[str]:
        """Return the dashboard-checked, uppercased, deduplicated ticker list.

        This is the sole availability authority (D2) — full-currencies only
        decorates. Any non-2xx response propagates via ``raise_for_status``
        (an HTTP-level catalog sync failure); a 2xx response with the wrong
        shape raises ``PaymentCatalogError``.
        """
        async with self._client_factory() as client:
            response = await client.get(
                f"{self._settings.nowpayments_api_base.rstrip('/')}/v1/merchant/coins",
                headers={"x-api-key": self._settings.nowpayments_api_key_for(product_id)},
            )
            response.raise_for_status()
            data: Any = response.json()

        if not isinstance(data, dict):
            raise PaymentCatalogError("NowPayments merchant/coins payload must be a JSON object")
        raw_currencies = data.get("selectedCurrencies")
        if not isinstance(raw_currencies, list) or not all(
            isinstance(item, str) for item in raw_currencies
        ):
            raise PaymentCatalogError(
                "NowPayments merchant/coins 'selectedCurrencies' must be a list of strings"
            )

        seen: dict[str, None] = {}
        for raw in raw_currencies:
            if ticker := raw.strip().upper():
                seen[ticker] = None
        return list(seen)

    async def list_full_currencies(self, product_id: str) -> dict[str, CurrencyDetails]:
        """Return the provider's full currency universe, keyed by uppercased ticker.

        Fetched once as a full table (their universe is large) rather than
        per-ticker. Shape violations at the top level fail loud; missing
        per-item fields degrade to ``None`` rather than failing the whole sync.
        """
        async with self._client_factory() as client:
            response = await client.get(
                f"{self._settings.nowpayments_api_base.rstrip('/')}/v1/full-currencies",
                headers={"x-api-key": self._settings.nowpayments_api_key_for(product_id)},
            )
            response.raise_for_status()
            data: Any = response.json()

        if not isinstance(data, dict):
            raise PaymentCatalogError("NowPayments full-currencies payload must be a JSON object")
        raw_entries = data.get("currencies")
        if not isinstance(raw_entries, list):
            raise PaymentCatalogError("NowPayments full-currencies 'currencies' must be a list")

        details: dict[str, CurrencyDetails] = {}
        for entry in raw_entries:
            if not isinstance(entry, dict):
                continue
            raw_code = entry.get("code")
            if not isinstance(raw_code, str) or not raw_code.strip():
                continue
            ticker = raw_code.strip().upper()

            raw_name = entry.get("name")
            name = raw_name.strip() if isinstance(raw_name, str) and raw_name.strip() else None

            raw_network = entry.get("network")
            network = (
                raw_network.strip().upper()
                if isinstance(raw_network, str) and raw_network.strip()
                else None
            )

            raw_logo = entry.get("logo_url")
            logo_url = (
                self._resolve_logo_url(raw_logo)
                if isinstance(raw_logo, str) and raw_logo.strip()
                else None
            )

            details[ticker] = CurrencyDetails(
                ticker=ticker,
                name=name,
                network=network,
                logo_url=logo_url,
            )
        return details

    @staticmethod
    def _resolve_logo_url(raw_logo_url: str) -> str:
        """Resolve a possibly-relative provider logo URL against nowpayments.io (D11)."""
        resolved = urljoin(_NOWPAYMENTS_SITE_BASE, raw_logo_url)
        host = urlparse(resolved).hostname
        if host not in _ALLOWED_LOGO_HOSTS:
            raise PaymentCatalogError(f"Disallowed NowPayments logo host: {host}")
        return resolved
