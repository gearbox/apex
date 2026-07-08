"""Tests for Settings validators — billing pricing tiers (D6)."""

from __future__ import annotations

import pytest

from src.core.config import Settings

pytestmark = pytest.mark.unit

_JWT_SECRET = "test-secret-key-that-is-definitely-long-enough-32bytes"


class TestPricingTiersValidRejected:
    def test_pricing_tiers_non_monotonic_discounts_rejected(self) -> None:
        with pytest.raises(ValueError, match="non-decreasing"):
            Settings(
                jwt_secret_key=_JWT_SECRET,
                billing_pricing_tiers={10: 10, 50: 5},  # discount drops at higher threshold
            )

    def test_pricing_tiers_discount_above_90_rejected(self) -> None:
        with pytest.raises(ValueError, match=r"\[0, 90\]"):
            Settings(
                jwt_secret_key=_JWT_SECRET,
                billing_pricing_tiers={10: 0, 100: 95},
            )

    def test_min_topup_above_lowest_tier_rejected(self) -> None:
        with pytest.raises(ValueError, match="lowest configured tier threshold"):
            Settings(
                jwt_secret_key=_JWT_SECRET,
                billing_pricing_tiers={10: 0, 50: 0},
                billing_min_topup_usd=15,
            )

    def test_negative_threshold_rejected(self) -> None:
        with pytest.raises(ValueError, match="must be positive"):
            Settings(
                jwt_secret_key=_JWT_SECRET,
                billing_pricing_tiers={-10: 0, 50: 5},
            )

    def test_empty_tiers_rejected(self) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            Settings(jwt_secret_key=_JWT_SECRET, billing_pricing_tiers={})

    def test_min_topup_not_less_than_max_rejected(self) -> None:
        with pytest.raises(ValueError, match="billing_min_topup_usd must be less than"):
            Settings(
                jwt_secret_key=_JWT_SECRET,
                billing_min_topup_usd=100,
                billing_max_topup_usd=100,
            )


class TestPricingTiersValidAccepted:
    def test_default_tiers_accepted(self) -> None:
        settings = Settings(jwt_secret_key=_JWT_SECRET)
        assert settings.billing_pricing_tiers == {10: 0, 50: 0, 100: 5, 250: 10}

    def test_custom_valid_tiers_accepted(self) -> None:
        settings = Settings(
            jwt_secret_key=_JWT_SECRET,
            billing_pricing_tiers={5: 0, 20: 5, 90: 90},
            billing_min_topup_usd=5,
        )
        assert settings.billing_pricing_tiers == {5: 0, 20: 5, 90: 90}

    def test_flat_zero_discount_tiers_accepted(self) -> None:
        """Non-decreasing allows equal discounts across every tier."""
        settings = Settings(
            jwt_secret_key=_JWT_SECRET,
            billing_pricing_tiers={10: 0, 50: 0, 100: 0},
        )
        assert settings.billing_pricing_tiers == {10: 0, 50: 0, 100: 0}
