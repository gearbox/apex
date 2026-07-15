"""Normalized contracts shared by payment gateways and the orchestrator."""

from __future__ import annotations

from collections.abc import Mapping
from decimal import Decimal
from typing import Any, Literal
from uuid import UUID

import msgspec

from src.core.enums import PaymentStatus
from src.core.product import ProductConfig
from src.core.topup_pricing import TopUpQuote


class ChargeContext(msgspec.Struct, frozen=True, kw_only=True):
    """Provider-neutral inputs required to create a payment charge."""

    payment_id: UUID
    account_id: UUID
    user_id: UUID
    product_config: ProductConfig
    quote: TopUpQuote
    extra: dict[str, str] = {}


class ChargeResult(msgspec.Struct, frozen=True, kw_only=True):
    """Provider-neutral result returned after creating a charge."""

    external_id: str
    redirect_url: str
    currency: str
    provider_metadata: dict[str, Any]


class CreatedCharge(msgspec.Struct, frozen=True, kw_only=True):
    """Persisted charge details returned to billing routes."""

    redirect_url: str
    external_id: str
    payment_id: UUID


class WebhookEnvelope(msgspec.Struct, frozen=True, kw_only=True):
    """Raw webhook request data needed for provider verification."""

    raw_body: bytes
    headers: Mapping[str, str]
    product_id: str


class PaymentLookup(msgspec.Struct, frozen=True, kw_only=True):
    """How the orchestrator should locate the payment row."""

    by: Literal["external_id", "payment_id"]
    value: str


class WebhookOutcome(msgspec.Struct, frozen=True, kw_only=True):
    """Verified and normalized provider webhook outcome.

    ``amount_paid`` and ``amount_due`` are the settled amount and the
    expected amount, both denominated in the same provider-defined unit
    system (e.g. NowPayments' pay currency) — never mixed with fiat
    ``payment.amount_usd``. Set together when proportional crediting is
    required; left ``None`` (Stripe path) for a plain full credit on
    ``COMPLETED``.
    """

    lookup: PaymentLookup
    status: PaymentStatus | None
    metadata_patch: dict[str, Any] = {}
    amount_paid: Decimal | None = None
    amount_due: Decimal | None = None
    settled_currency: str | None = None
    """Uppercased provider ticker of the currency the customer actually paid
    in, when the provider reports it; the orchestrator syncs
    ``Payment.currency`` to this value."""
