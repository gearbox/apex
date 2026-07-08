"""Pure top-up pricing math — tier resolution and quote construction.

No I/O, no Settings import: both invoice-creation paths (``PaymentService``)
and the read-only options endpoint call the functions here with plain values,
so the UI summary box and the actual charge can never disagree.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from decimal import ROUND_HALF_UP, Decimal
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.core.config import Settings


@dataclasses.dataclass(frozen=True, slots=True)
class TopUpTier:
    """A single discount tier: amounts at or above ``threshold_usd`` qualify."""

    threshold_usd: int
    discount_pct: int


@dataclasses.dataclass(frozen=True, slots=True)
class TopUpQuote:
    """Resolved price quote for a top-up amount."""

    credits_usd: int
    """Nominal credits amount the user chose (pre-discount)."""

    discount_pct: int
    total_due: Decimal
    """Cents-quantized amount actually charged."""

    tokens_granted: int
    """credits_usd * tokens_per_usd — full, pre-discount token value."""


def build_tiers(raw: dict[int, int]) -> tuple[TopUpTier, ...]:
    """Build an ascending-by-threshold tier tuple from a raw settings dict."""
    return tuple(
        TopUpTier(threshold_usd=threshold, discount_pct=discount)
        for threshold, discount in sorted(raw.items(), key=lambda item: item[0])
    )


def resolve_discount(amount_usd: int, tiers: Sequence[TopUpTier]) -> int:
    """Return the discount pct for the highest threshold <= amount_usd, else 0."""
    discount_pct = 0
    for tier in tiers:
        if tier.threshold_usd <= amount_usd:
            discount_pct = tier.discount_pct
        else:
            break
    return discount_pct


def build_quote(
    amount_usd: int,
    *,
    tiers: Sequence[TopUpTier],
    tokens_per_usd: int,
) -> TopUpQuote:
    """Build a full price quote for ``amount_usd`` credits.

    Range validation (min/max top-up amount) is the caller's responsibility
    (``PaymentService``) — this module stays pure math.
    """
    discount_pct = resolve_discount(amount_usd, tiers)
    total_due = (Decimal(amount_usd) * (100 - discount_pct) / 100).quantize(
        Decimal("0.01"), rounding=ROUND_HALF_UP
    )
    return TopUpQuote(
        credits_usd=amount_usd,
        discount_pct=discount_pct,
        total_due=total_due,
        tokens_granted=amount_usd * tokens_per_usd,
    )


def topup_tiers_for(product_id: str, settings: Settings) -> tuple[TopUpTier, ...]:
    """Resolve top-up tiers for a product.

    Today every product shares the same global tier table from ``Settings``.
    ``product_id`` is intentionally unused — it names the seam so per-product
    pricing later is a data change, not a refactor.
    """
    del product_id
    return build_tiers(settings.billing_pricing_tiers)
