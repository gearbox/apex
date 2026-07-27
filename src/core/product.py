"""Product configuration registry.

Each product defines its own policies for auth, models, content, billing,
and rate limits. Loaded once at startup, immutable at runtime.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Final

# Runtime import required: PEP 563 field annotations on ProductConfig are resolved
# by Litestar DI and msgspec at runtime.
from src.core.enums import ModelType, Product  # noqa: TC001

if TYPE_CHECKING:
    from collections.abc import Mapping


class AgeGatePolicy(StrEnum):
    """Age verification requirements."""

    NONE = "none"
    CHECKBOX = "checkbox"  # Simple "I am 18+" confirmation
    DATE_OF_BIRTH = "date_of_birth"  # DOB entry with validation
    # IDENTITY_VERIFICATION = "identity_verification"  # Future: Jumio/Veriff


class AuthMethod(StrEnum):
    """Supported authentication methods."""

    EMAIL_PASSWORD = "email_password"  # noqa: S105
    GOOGLE_OAUTH = "google_oauth"
    APPLE_OAUTH = "apple_oauth"
    # SSO_SAML = "sso_saml"  # Future: enterprise SSO


class PaymentProvider(StrEnum):
    """Supported payment providers."""

    STRIPE = "stripe"
    NOWPAYMENTS = "nowpayments"  # Crypto

    @property
    def method_kind(self) -> PaymentMethodKind:
        """User-facing payment method class for this gateway.

        Raises:
            ValueError: If the provider has no mapping — fail loud rather
                than defaulting to a wrong label on a financial record.
        """
        kind = _PROVIDER_METHOD_KIND.get(self)
        if kind is None:
            raise ValueError(f"No PaymentMethodKind mapped for provider {self.value!r}")
        return kind


class PaymentMethodKind(StrEnum):
    """User-facing payment method class — deliberately gateway-agnostic."""

    CARD = "card"
    CRYPTO = "crypto"


# Single source of truth for provider → user-facing method class. Every
# PaymentProvider member MUST appear here; test_payment_method_kind_total
# fails the build if a new provider is added without a mapping.
_PROVIDER_METHOD_KIND: Final[Mapping[PaymentProvider, PaymentMethodKind]] = {
    PaymentProvider.STRIPE: PaymentMethodKind.CARD,
    PaymentProvider.NOWPAYMENTS: PaymentMethodKind.CRYPTO,
}


class ContentRating(StrEnum):
    """Content rating levels."""

    SFW = "sfw"  # Safe for work only
    PERMISSIVE = "permissive"  # Adult content allowed with age gate


@dataclass(frozen=True)
class ContentPolicyConfig:
    """Content moderation policy per product."""

    rating: ContentRating
    allow_nsfw_generation: bool
    allow_nsfw_uploads: bool
    require_pre_moderation: bool  # Run content safety checks before generation
    max_prompt_length: int = 4096


@dataclass(frozen=True)
class ProductRateLimits:
    """Rate limit configuration per product.

    Format: "N/period" compatible with the `limits` library.
    """

    generation_per_user: str = "60/hour"
    generation_per_user_burst: str = "10/minute"
    upload_per_user: str = "100/hour"
    api_global: str = "1000/minute"


@dataclass(frozen=True)
class StripeConfig:
    """Per-product Stripe configuration.

    Actual keys come from Settings (env vars). This just holds
    the env var names to look up, plus product-specific Stripe metadata.
    """

    secret_key_env: str  # e.g. "stripe_secret_key_vex"
    webhook_secret_env: str  # e.g. "stripe_webhook_secret_vex"
    publishable_key_env: str  # e.g. "stripe_publishable_key_vex"


@dataclass(frozen=True)
class NowPaymentsConfig:
    """Per-product NowPayments configuration."""

    api_key_env: str  # e.g. "nowpayments_api_key_vex"
    ipn_secret_env: str  # e.g. "nowpayments_ipn_secret_vex"


@dataclass(frozen=True)
class ProductConfig:
    """Complete configuration for a product.

    Immutable at runtime. Loaded once at startup from the product registry.
    """

    product: Product
    slug: str
    display_name: str
    domains: frozenset[str]
    # Canonical frontend origin (scheme + host, no trailing slash), e.g.
    # "https://vex-domain.com". Used to build redirect URLs (e.g. Stripe Checkout
    # success/cancel) without hardcoding brand strings in service code.
    frontend_origin: str

    # Auth
    age_gate: AgeGatePolicy
    allowed_auth_methods: frozenset[AuthMethod]

    # Models — None means "all enabled models", otherwise explicit allowlist
    allowed_models: frozenset[ModelType] | None = None
    blocked_models: frozenset[ModelType] = field(default_factory=frozenset)

    # Content
    content_policy: ContentPolicyConfig = field(
        default_factory=lambda: ContentPolicyConfig(
            rating=ContentRating.SFW,
            allow_nsfw_generation=False,
            allow_nsfw_uploads=False,
            require_pre_moderation=True,
        )
    )

    # Billing
    payment_providers: frozenset[PaymentProvider] = field(
        default_factory=lambda: frozenset({PaymentProvider.STRIPE})
    )
    stripe_config: StripeConfig | None = None
    nowpayments_config: NowPaymentsConfig | None = None

    # Rate limits
    rate_limits: ProductRateLimits = field(default_factory=ProductRateLimits)

    # Feature flags — extensible set of feature slugs
    features: frozenset[str] = field(default_factory=frozenset)

    # Cookie domain — registrable domain for the content cookie (e.g. "vex-domain.com").
    # None for localhost/dev pseudo-products that don't set Domain on the cookie.
    cookie_domain: str | None = None

    def is_model_allowed(self, model: ModelType) -> bool:
        """Check if a model is accessible on this product."""
        if model in self.blocked_models:
            return False
        if self.allowed_models is not None:
            return model in self.allowed_models
        return True

    def has_feature(self, feature: str) -> bool:
        """Check if a feature flag is enabled for this product."""
        return feature in self.features

    def supports_payment_provider(self, provider: PaymentProvider) -> bool:
        """Check if a payment provider is enabled for this product."""
        return provider in self.payment_providers
