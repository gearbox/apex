"""Product registry — the single source of truth for product configurations.

Import PRODUCT_REGISTRY or use get_product_config() / resolve_product().
"""

from __future__ import annotations

from src.core.enums import ModelType, Product
from src.core.product import (
    AgeGatePolicy,
    AuthMethod,
    ContentPolicyConfig,
    ContentRating,
    NowPaymentsConfig,
    PaymentProvider,
    ProductConfig,
    ProductRateLimits,
    StripeConfig,
)

# ---------------------------------------------------------------------------
# Product definitions
# ---------------------------------------------------------------------------

VEX_CONFIG = ProductConfig(
    product=Product.VEX,
    slug="vex",
    display_name="vex.pics",
    domains=frozenset(
        {
            "vex.pics",
            "www.vex.pics",
            "app.vex.pics",
        }
    ),
    age_gate=AgeGatePolicy.CHECKBOX,
    allowed_auth_methods=frozenset(
        {
            AuthMethod.EMAIL_PASSWORD,
            AuthMethod.GOOGLE_OAUTH,
        }
    ),
    # vex.pics: all models allowed (permissive)
    allowed_models=None,
    blocked_models=frozenset(),
    content_policy=ContentPolicyConfig(
        rating=ContentRating.PERMISSIVE,
        allow_nsfw_generation=True,
        allow_nsfw_uploads=True,
        require_pre_moderation=True,  # Still moderate for illegal content
    ),
    payment_providers=frozenset(
        {
            PaymentProvider.NOWPAYMENTS,
            PaymentProvider.STRIPE,
        }
    ),
    stripe_config=StripeConfig(
        secret_key_env="stripe_secret_key_vex",
        webhook_secret_env="stripe_webhook_secret_vex",
        publishable_key_env="stripe_publishable_key_vex",
    ),
    nowpayments_config=NowPaymentsConfig(
        api_key_env="nowpayments_api_key_vex",
        ipn_secret_env="nowpayments_ipn_secret_vex",
    ),
    rate_limits=ProductRateLimits(
        generation_per_user="120/hour",
        generation_per_user_burst="15/minute",
        upload_per_user="200/hour",
        api_global="2000/minute",
    ),
    features=frozenset(),
)

SYNTHARA_CONFIG = ProductConfig(
    product=Product.SYNTHARA,
    slug="synthara",
    display_name="Synthara",
    domains=frozenset(
        {
            "synthara.app",
            "www.synthara.app",
            "app.synthara.app",
        }
    ),
    age_gate=AgeGatePolicy.NONE,
    allowed_auth_methods=frozenset(
        {
            AuthMethod.EMAIL_PASSWORD,
            AuthMethod.GOOGLE_OAUTH,
            AuthMethod.APPLE_OAUTH,
        }
    ),
    # synthara: only SFW models (explicit allowlist — add models here as needed)
    allowed_models=frozenset(
        {
            ModelType.GROK_IMAGINE_IMAGE,
            ModelType.GROK_IMAGINE_VIDEO,
            # Explicitly no NSFW models. Video models can be added when ready.
        }
    ),
    blocked_models=frozenset(),
    content_policy=ContentPolicyConfig(
        rating=ContentRating.SFW,
        allow_nsfw_generation=False,
        allow_nsfw_uploads=False,
        require_pre_moderation=True,
    ),
    payment_providers=frozenset(
        {
            PaymentProvider.STRIPE,
        }
    ),
    stripe_config=StripeConfig(
        secret_key_env="stripe_secret_key_synthara",
        webhook_secret_env="stripe_webhook_secret_synthara",
        publishable_key_env="stripe_publishable_key_synthara",
    ),
    nowpayments_config=None,
    rate_limits=ProductRateLimits(
        generation_per_user="300/hour",
        generation_per_user_burst="30/minute",
        upload_per_user="500/hour",
        api_global="5000/minute",
    ),
    features=frozenset({"organizations"}),
)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

PRODUCT_REGISTRY: dict[Product, ProductConfig] = {
    Product.VEX: VEX_CONFIG,
    Product.SYNTHARA: SYNTHARA_CONFIG,
}

# Domain → ProductConfig lookup (built once at import time)
_DOMAIN_MAP: dict[str, ProductConfig] = {}
for _cfg in PRODUCT_REGISTRY.values():
    for _domain in _cfg.domains:
        _DOMAIN_MAP[_domain.lower()] = _cfg


def get_product_config(product: Product) -> ProductConfig:
    """Get configuration for a product. Raises KeyError if unknown."""
    return PRODUCT_REGISTRY[product]


def resolve_product_by_domain(domain: str) -> ProductConfig | None:
    """Resolve product config from a domain string. Returns None if no match."""
    return _DOMAIN_MAP.get(domain.lower().split(":")[0])  # Strip port


def resolve_product_by_slug(slug: str) -> ProductConfig | None:
    """Resolve product config from a slug string (for X-Product-Id header)."""
    try:
        product = Product(slug.lower())
        return PRODUCT_REGISTRY.get(product)
    except ValueError:
        return None
