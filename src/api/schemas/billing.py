"""Billing API schemas using msgspec."""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

import msgspec

from src.core.enums import AccountType

# --- Response structs ---


class BalanceResponse(msgspec.Struct, kw_only=True):
    """Token balance for an account."""

    account_id: UUID
    account_type: str
    balance: int
    organization_name: str | None = None


class TransactionResponse(msgspec.Struct, kw_only=True):
    """Single token transaction."""

    id: UUID
    transaction_type: str
    amount: int
    balance_after: int
    description: str | None
    metadata: dict[str, Any]
    job_id: UUID | None
    payment_id: UUID | None
    created_at: datetime
    created_by: UUID | None


class PricingRuleResponse(msgspec.Struct, kw_only=True):
    """Pricing catalog entry."""

    id: UUID
    provider: str
    generation_type: str
    model: str | None
    token_cost: int
    is_active: bool
    effective_from: datetime
    effective_until: datetime | None
    notes: str | None


class TokenPackageResponse(msgspec.Struct, kw_only=True):
    """Available token purchase package."""

    id: str
    name: str
    tokens: int
    bonus_tokens: int
    total_tokens: int
    price_usd: str  # string to avoid float precision issues in JSON


class PaymentResponse(msgspec.Struct, kw_only=True):
    """Payment record."""

    id: UUID
    payment_provider: str
    status: str
    amount_usd: str
    tokens_granted: int
    currency: str
    created_at: datetime
    completed_at: datetime | None


class AdminAdjustResponse(msgspec.Struct, kw_only=True):
    """Response for admin balance adjustment."""

    transaction: TransactionResponse
    new_balance: int


class StripeCheckoutResponse(msgspec.Struct, kw_only=True):
    """Response for Stripe checkout session creation."""

    checkout_url: str
    session_id: str
    payment_id: UUID


class NowPaymentsInvoiceResponse(msgspec.Struct, kw_only=True):
    """Response for NowPayments invoice creation."""

    invoice_url: str
    payment_id: UUID


# --- Account preference structs ---


class SetBillingAccountRequest(msgspec.Struct, forbid_unknown_fields=True, kw_only=True):
    """Request to set the preferred billing account."""

    account: AccountType


class BillingAccountResponse(msgspec.Struct, kw_only=True):
    """Response for billing account preference queries."""

    preferred_account: str | None
    message: str


# --- Request structs ---


class TopUpStripeRequest(msgspec.Struct, forbid_unknown_fields=True, kw_only=True):
    """Request to create a Stripe checkout session."""

    package_id: str


class TopUpNowPaymentsRequest(msgspec.Struct, forbid_unknown_fields=True, kw_only=True):
    """Request to create a NowPayments invoice."""

    package_id: str
    pay_currency: str


class AdminAdjustRequest(msgspec.Struct, forbid_unknown_fields=True, kw_only=True):
    """Request for admin balance adjustment."""

    amount: int  # positive = credit, negative = debit
    description: str


class CreatePricingRuleRequest(msgspec.Struct, forbid_unknown_fields=True, kw_only=True):
    """Request to create a pricing rule."""

    provider: str
    generation_type: str
    model: str | None = None
    token_cost: int
    notes: str | None = None


class PatchPricingRuleRequest(msgspec.Struct, forbid_unknown_fields=True, kw_only=True):
    """Request to update a pricing rule."""

    token_cost: int | None = None
    is_active: bool | None = None
    effective_until: datetime | None = None
    notes: str | None = None
