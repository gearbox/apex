"""User-facing wording for billing ledger entries.

Ledger descriptions are shown to end users verbatim by
``GET /v1/billing/transactions``. They must never name an internal payment
gateway — the crypto/card distinction is all a user needs. The gateway
remains resolvable server-side via ``token_transactions.payment_id ->
payments.payment_provider`` (admin-only surface).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.core.product import PaymentMethodKind

_PARTIAL_SUFFIX = " (partial)"


def topup_description(kind: PaymentMethodKind, *, partial: bool) -> str:
    """Build the ledger description for a token top-up credit."""
    base = f"Token purchase via {kind.value} payment"
    return f"{base}{_PARTIAL_SUFFIX}" if partial else base
