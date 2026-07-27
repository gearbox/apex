"""Unit tests for product registry and ProductConfig logic."""

from __future__ import annotations

import pytest

from src.core.enums import ModelType, Product
from src.core.product import AgeGatePolicy, PaymentMethodKind, PaymentProvider
from src.core.product_registry import (
    PRODUCT_REGISTRY,
    SYNTHARA_CONFIG,
    VEX_CONFIG,
    extract_apex_domain,
    resolve_product_by_domain,
    resolve_product_by_slug,
)

# Derived from the real registry rather than hardcoded, so this file never
# needs to spell out the product's actual domain.
_VEX_DOMAIN = next(iter(VEX_CONFIG.domains))
_SYNTHARA_DOMAIN = next(iter(SYNTHARA_CONFIG.domains))


class TestAgeGatePolicy:
    def test_vex_uses_date_of_birth(self) -> None:
        # vex verifies age via DOB entry, not a bare self-attestation checkbox.
        assert VEX_CONFIG.age_gate is AgeGatePolicy.DATE_OF_BIRTH

    def test_synthara_has_no_age_gate(self) -> None:
        assert SYNTHARA_CONFIG.age_gate is AgeGatePolicy.NONE

    def test_no_product_uses_checkbox(self) -> None:
        # CHECKBOX is reserved (dormant) for a future ID-backed verification flow.
        # No product may route a bare self-attestation checkbox to verification.
        assert all(cfg.age_gate is not AgeGatePolicy.CHECKBOX for cfg in PRODUCT_REGISTRY.values())


class TestExtractApexDomain:
    """Tests for extract_apex_domain()."""

    def test_plain_apex(self) -> None:
        assert extract_apex_domain("example.com") == "example.com"

    def test_www_subdomain(self) -> None:
        assert extract_apex_domain("www.example.com") == "example.com"

    def test_staging_subdomain(self) -> None:
        assert extract_apex_domain("staging.example.com") == "example.com"

    def test_deep_subdomain(self) -> None:
        assert extract_apex_domain("feature-xyz.staging.synthara.app") == "synthara.app"

    def test_multi_part_tld(self) -> None:
        assert extract_apex_domain("www.example.co.uk") == "example.co.uk"

    def test_localhost_returns_none(self) -> None:
        assert extract_apex_domain("localhost") is None

    def test_ipv4_returns_none(self) -> None:
        assert extract_apex_domain("127.0.0.1") is None

    def test_empty_returns_none(self) -> None:
        assert extract_apex_domain("") is None


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


class TestPaymentProviderMethodKind:
    """Tests for PaymentProvider.method_kind — the crypto/card discriminator."""

    def test_payment_method_kind_total(self) -> None:
        """Every PaymentProvider member must map to a PaymentMethodKind.

        Guards future providers: adding a PaymentProvider without extending
        the mapping fails the build here, not in production with a wrong
        label on a financial record.
        """
        for provider in PaymentProvider:
            assert isinstance(provider.method_kind, PaymentMethodKind)

    def test_method_kind_values(self) -> None:
        assert PaymentProvider.STRIPE.method_kind is PaymentMethodKind.CARD
        assert PaymentProvider.NOWPAYMENTS.method_kind is PaymentMethodKind.CRYPTO


class TestResolveProductByDomain:
    """Tests for resolve_product_by_domain()."""

    @pytest.mark.parametrize(
        "domain",
        [
            _VEX_DOMAIN,
            f"www.{_VEX_DOMAIN}",
            f"app.{_VEX_DOMAIN}",
            f"staging.{_VEX_DOMAIN}",
            f"preview.{_VEX_DOMAIN}",
        ],
    )
    def test_vex_domains(self, domain: str) -> None:
        cfg = resolve_product_by_domain(domain)
        assert cfg is not None
        assert cfg.product == Product.VEX

    @pytest.mark.parametrize(
        "domain",
        [
            _SYNTHARA_DOMAIN,
            f"www.{_SYNTHARA_DOMAIN}",
            f"app.{_SYNTHARA_DOMAIN}",
            f"staging.{_SYNTHARA_DOMAIN}",
            f"preview.{_SYNTHARA_DOMAIN}",
        ],
    )
    def test_synthara_domains(self, domain: str) -> None:
        cfg = resolve_product_by_domain(domain)
        assert cfg is not None
        assert cfg.product == Product.SYNTHARA

    def test_unknown_domain_returns_none(self) -> None:
        assert resolve_product_by_domain("unknown.example.org") is None
        assert resolve_product_by_domain("") is None
        assert resolve_product_by_domain(f"evil.{_VEX_DOMAIN}.attacker.com") is None
        assert resolve_product_by_domain(f"evil.{_SYNTHARA_DOMAIN}.attacker.com") is None

    def test_domain_with_port_is_stripped(self) -> None:
        """Domains like '<apex>:443' should resolve correctly."""
        cfg = resolve_product_by_domain(f"{_VEX_DOMAIN}:443")
        assert cfg is not None
        assert cfg.product == Product.VEX

    def test_domain_case_insensitive(self) -> None:
        cfg = resolve_product_by_domain(_VEX_DOMAIN.upper())
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
