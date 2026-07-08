"""Unit tests for the pure top-up pricing quote module (D1/D8)."""

from __future__ import annotations

from decimal import Decimal
from itertools import pairwise

import pytest

from src.core.topup_pricing import build_quote, build_tiers, resolve_discount

pytestmark = pytest.mark.unit

_DEFAULT_RAW_TIERS: dict[int, int] = {10: 0, 50: 0, 100: 5, 250: 10}
_DEFAULT_TIERS = build_tiers(_DEFAULT_RAW_TIERS)


class TestBuildTiers:
    def test_tiers_sorted_ascending(self) -> None:
        raw = {250: 10, 10: 0, 100: 5, 50: 0}
        tiers = build_tiers(raw)
        thresholds = [t.threshold_usd for t in tiers]
        assert thresholds == sorted(thresholds)
        assert thresholds == [10, 50, 100, 250]


class TestResolveDiscount:
    @pytest.mark.parametrize(
        "amount_usd,expected_pct",
        [
            (10, 0),
            (9, 0),  # below lowest tier
            (50, 0),
            (49, 0),
            (100, 5),
            (99, 0),
            (250, 10),
            (249, 5),
            (1000, 10),  # well above highest tier — sticks at highest discount
        ],
    )
    def test_resolve_discount_tier_boundaries_inclusive(
        self, amount_usd: int, expected_pct: int
    ) -> None:
        assert resolve_discount(amount_usd, _DEFAULT_TIERS) == expected_pct

    def test_below_lowest_tier_gets_zero_discount(self) -> None:
        assert resolve_discount(1, _DEFAULT_TIERS) == 0

    @pytest.mark.parametrize("threshold,discount", list(_DEFAULT_RAW_TIERS.items()))
    def test_every_configured_threshold_inclusive_boundary(
        self, threshold: int, discount: int
    ) -> None:
        """Exactly at a threshold gets that tier's discount; one dollar below does not
        (unless a lower tier already covers it)."""
        assert resolve_discount(threshold, _DEFAULT_TIERS) == discount
        below_pct = resolve_discount(threshold - 1, _DEFAULT_TIERS)
        assert below_pct <= discount


class TestBuildQuote:
    def test_quote_total_due_cents_exact(self) -> None:
        quote = build_quote(100, tiers=_DEFAULT_TIERS, tokens_per_usd=100)
        assert quote.total_due == Decimal("95.00")
        assert quote.credits_usd == 100
        assert quote.discount_pct == 5
        assert quote.tokens_granted == 10_000

    def test_quote_zero_discount_charges_full_amount(self) -> None:
        quote = build_quote(10, tiers=_DEFAULT_TIERS, tokens_per_usd=100)
        assert quote.total_due == Decimal("10.00")
        assert quote.discount_pct == 0

    def test_quote_tokens_granted_always_pre_discount(self) -> None:
        quote = build_quote(250, tiers=_DEFAULT_TIERS, tokens_per_usd=100)
        # tokens_granted uses credits_usd, not total_due.
        assert quote.tokens_granted == 250 * 100
        assert quote.total_due == Decimal("225.00")

    def test_effective_price_per_token_non_increasing(self) -> None:
        """For a fixed tokens_per_usd, effective $/token must never increase
        as amount_usd increases — guaranteed by non-decreasing discounts."""
        amounts = list(range(1, 500))
        prices_per_token = []
        for amount in amounts:
            quote = build_quote(amount, tiers=_DEFAULT_TIERS, tokens_per_usd=100)
            price_per_token = quote.total_due / quote.tokens_granted
            prices_per_token.append(price_per_token)

        for earlier, later in pairwise(prices_per_token):
            assert later <= earlier
