"""Tests for Settings validators — billing pricing tiers (D6)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

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


class TestNodeCooldownThresholds:
    def test_node_cooldown_base_gt_max_rejected(self) -> None:
        with pytest.raises(ValueError, match="node_cooldown_base_minutes must be <="):
            Settings(
                jwt_secret_key=_JWT_SECRET,
                node_cooldown_base_minutes=400,
                node_cooldown_max_minutes=360,
            )


class TestFrameExtractStaleRunningThreshold:
    def test_default_matches_worst_case_runtime(self) -> None:
        settings = Settings(jwt_secret_key=_JWT_SECRET)
        assert settings.frame_extract_stale_running_seconds == 1800

    def test_below_worst_case_rejected(self) -> None:
        with pytest.raises(ValueError, match="frame_extract_stale_running_seconds must be >="):
            Settings(
                jwt_secret_key=_JWT_SECRET,
                frame_extract_stale_running_seconds=300,
                frame_extract_ffmpeg_timeout_seconds=30,
            )

    def test_boundary_equal_to_worst_case_accepted(self) -> None:
        settings = Settings(
            jwt_secret_key=_JWT_SECRET,
            frame_extract_stale_running_seconds=1800,
            frame_extract_ffmpeg_timeout_seconds=30,
        )
        assert settings.frame_extract_stale_running_seconds == 1800


class TestNowpaymentsIpnCallbackUrl:
    def test_unset_accepted(self) -> None:
        settings = Settings(jwt_secret_key=_JWT_SECRET)
        assert settings.nowpayments_ipn_callback_url is None

    def test_valid_https_url_accepted(self) -> None:
        url = "https://api.example.com/v1/billing/webhooks/nowpayments"
        settings = Settings(jwt_secret_key=_JWT_SECRET, nowpayments_ipn_callback_url=url)
        assert settings.nowpayments_ipn_callback_url == url

    def test_valid_http_url_accepted_for_local_dev(self) -> None:
        url = "http://localhost:8000/v1/billing/webhooks/nowpayments"
        settings = Settings(jwt_secret_key=_JWT_SECRET, nowpayments_ipn_callback_url=url)
        assert settings.nowpayments_ipn_callback_url == url

    def test_scheme_less_url_rejected(self) -> None:
        with pytest.raises(ValueError, match="nowpayments_ipn_callback_url"):
            Settings(
                jwt_secret_key=_JWT_SECRET,
                nowpayments_ipn_callback_url="api.example.com/v1/billing/webhooks/nowpayments",
            )

    def test_garbage_value_rejected(self) -> None:
        with pytest.raises(ValueError, match="nowpayments_ipn_callback_url"):
            Settings(jwt_secret_key=_JWT_SECRET, nowpayments_ipn_callback_url="not a url")

    def test_ftp_scheme_rejected(self) -> None:
        with pytest.raises(ValueError, match="nowpayments_ipn_callback_url"):
            Settings(
                jwt_secret_key=_JWT_SECRET,
                nowpayments_ipn_callback_url="ftp://example.com/callback",
            )


class TestMaxRevocationEpochTtlSeconds:
    """issue #142 F3 — TokenRevocationService's epoch TTL must derive from
    the *upper bound* Field constraints, not the currently configured
    values, so lowering either setting later can't strand epoch keys with a
    TTL shorter than tokens minted under the old, higher setting."""

    def test_at_least_max_permitted_content_cookie_lifetime(self) -> None:
        settings = Settings(jwt_secret_key=_JWT_SECRET)
        # content_cookie_ttl_hours: Field(le=168) -> 168h in seconds.
        assert settings.max_revocation_epoch_ttl_seconds >= 168 * 3600

    def test_ignores_current_values_uses_upper_bounds(self) -> None:
        """A deployment running low current values must still get the TTL
        derived from the field's maximum, not from what's configured now."""
        settings = Settings(
            jwt_secret_key=_JWT_SECRET,
            jwt_access_token_expire_minutes=1,
            content_cookie_ttl_hours=1,
        )
        assert settings.max_revocation_epoch_ttl_seconds == 168 * 3600

    def test_matches_hardcoded_upper_bound_expectation(self) -> None:
        settings = Settings(jwt_secret_key=_JWT_SECRET)
        assert settings.max_revocation_epoch_ttl_seconds == max(60 * 60, 168 * 3600)


class TestDeploymentEnvironment:
    def test_defaults_to_development(self) -> None:
        settings = Settings(jwt_secret_key=_JWT_SECRET)
        assert settings.environment == "development"

    def test_env_var_sets_environment(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ENVIRONMENT", "staging")
        settings = Settings(jwt_secret_key=_JWT_SECRET)
        assert settings.environment == "staging"

    def test_invalid_value_rejected(self) -> None:
        with pytest.raises(ValidationError):
            Settings(jwt_secret_key=_JWT_SECRET, environment="prod")  # pyright: ignore[reportArgumentType]


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
