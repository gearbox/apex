"""Unit tests for product registry and ProductConfig logic."""

from __future__ import annotations

import pytest

from src.core.enums import ModelType, Product
from src.core.product import PaymentProvider
from src.core.product_registry import (
    SYNTHARA_CONFIG,
    VEX_CONFIG,
    resolve_product_by_domain,
    resolve_product_by_slug,
)


class TestProductConfigIsModelAllowed:
    """Tests for ProductConfig.is_model_allowed()."""

    def test_vex_allows_all_models_when_no_allowlist(self) -> None:
        """VEX has allowed_models=None, meaning all non-blocked models are allowed."""
        assert VEX_CONFIG.allowed_models is None
        assert VEX_CONFIG.is_model_allowed(ModelType.GROK_IMAGINE_IMAGE)
        assert VEX_CONFIG.is_model_allowed(ModelType.GROK_IMAGINE_VIDEO)
        assert VEX_CONFIG.is_model_allowed(ModelType.AISHA_IMAGE)

    def test_synthara_uses_allowlist(self) -> None:
        """SYNTHARA has an explicit allowlist of SFW models."""
        assert SYNTHARA_CONFIG.allowed_models is not None
        assert SYNTHARA_CONFIG.is_model_allowed(ModelType.GROK_IMAGINE_IMAGE)
        assert SYNTHARA_CONFIG.is_model_allowed(ModelType.GROK_IMAGINE_VIDEO)

    def test_synthara_blocks_models_not_in_allowlist(self) -> None:
        """Models not in SYNTHARA's allowlist are rejected."""
        assert not SYNTHARA_CONFIG.is_model_allowed(ModelType.AISHA_IMAGE)
        assert not SYNTHARA_CONFIG.is_model_allowed(ModelType.AISHA_VIDEO)

    def test_blocked_model_overrides_allowlist(self) -> None:
        """blocked_models takes priority over allowed_models=None."""
        # VEX has no blocklist, so AISHA models are allowed
        assert VEX_CONFIG.is_model_allowed(ModelType.AISHA_IMAGE)

    def test_blocked_model_on_open_allowlist(self) -> None:
        """If allowed_models is None but model is in blocked_models, it is rejected."""
        from dataclasses import replace

        cfg_with_block = replace(VEX_CONFIG, blocked_models=frozenset({ModelType.AISHA_IMAGE}))
        assert not cfg_with_block.is_model_allowed(ModelType.AISHA_IMAGE)
        assert cfg_with_block.is_model_allowed(ModelType.GROK_IMAGINE_IMAGE)


class TestProductConfigHasFeature:
    """Tests for ProductConfig.has_feature()."""

    def test_synthara_has_organizations_feature(self) -> None:
        assert SYNTHARA_CONFIG.has_feature("organizations")

    def test_vex_does_not_have_organizations_feature(self) -> None:
        assert not VEX_CONFIG.has_feature("organizations")

    def test_unknown_feature_returns_false(self) -> None:
        assert not VEX_CONFIG.has_feature("nonexistent_feature_xyz")
        assert not SYNTHARA_CONFIG.has_feature("nonexistent_feature_xyz")


class TestProductConfigSupportsPaymentProvider:
    """Tests for ProductConfig.supports_payment_provider()."""

    def test_vex_supports_stripe(self) -> None:
        assert VEX_CONFIG.supports_payment_provider(PaymentProvider.STRIPE)

    def test_vex_supports_nowpayments(self) -> None:
        assert VEX_CONFIG.supports_payment_provider(PaymentProvider.NOWPAYMENTS)

    def test_synthara_supports_stripe(self) -> None:
        assert SYNTHARA_CONFIG.supports_payment_provider(PaymentProvider.STRIPE)

    def test_synthara_does_not_support_nowpayments(self) -> None:
        assert not SYNTHARA_CONFIG.supports_payment_provider(PaymentProvider.NOWPAYMENTS)


class TestResolveProductByDomain:
    """Tests for resolve_product_by_domain()."""

    @pytest.mark.parametrize(
        "domain",
        ["vex.pics", "www.vex.pics", "app.vex.pics"],
    )
    def test_vex_domains(self, domain: str) -> None:
        cfg = resolve_product_by_domain(domain)
        assert cfg is not None
        assert cfg.product == Product.VEX

    @pytest.mark.parametrize(
        "domain",
        ["synthara.app", "www.synthara.app", "app.synthara.app"],
    )
    def test_synthara_domains(self, domain: str) -> None:
        cfg = resolve_product_by_domain(domain)
        assert cfg is not None
        assert cfg.product == Product.SYNTHARA

    def test_unknown_domain_returns_none(self) -> None:
        assert resolve_product_by_domain("unknown.example.com") is None
        assert resolve_product_by_domain("") is None
        assert resolve_product_by_domain("evil.vex.pics.attacker.com") is None

    def test_domain_with_port_is_stripped(self) -> None:
        """Domains like 'vex.pics:443' should resolve correctly."""
        cfg = resolve_product_by_domain("vex.pics:443")
        assert cfg is not None
        assert cfg.product == Product.VEX

    def test_domain_case_insensitive(self) -> None:
        cfg = resolve_product_by_domain("VEX.PICS")
        assert cfg is not None
        assert cfg.product == Product.VEX


class TestResolveProductBySlug:
    """Tests for resolve_product_by_slug()."""

    def test_valid_slug_vex(self) -> None:
        cfg = resolve_product_by_slug("vex")
        assert cfg is not None
        assert cfg.product == Product.VEX

    def test_valid_slug_synthara(self) -> None:
        cfg = resolve_product_by_slug("synthara")
        assert cfg is not None
        assert cfg.product == Product.SYNTHARA

    def test_invalid_slug_returns_none(self) -> None:
        assert resolve_product_by_slug("unknown") is None
        assert resolve_product_by_slug("") is None
        assert resolve_product_by_slug("VEX") is not None  # case-insensitive

    def test_slug_case_insensitive(self) -> None:
        cfg = resolve_product_by_slug("VEX")
        assert cfg is not None
        assert cfg.product == Product.VEX

        cfg2 = resolve_product_by_slug("SYNTHARA")
        assert cfg2 is not None
        assert cfg2.product == Product.SYNTHARA
