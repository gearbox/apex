"""Product registry — the single source of truth for product configurations.

Import PRODUCT_REGISTRY or use get_product_config() / resolve_product().
"""

from __future__ import annotations

import tldextract as _tldextract

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
        }
    ),
    frontend_origin="https://vex.pics",
    age_gate=AgeGatePolicy.DATE_OF_BIRTH,
    allowed_auth_methods=frozenset(
        {
            AuthMethod.EMAIL_PASSWORD,
            AuthMethod.GOOGLE_OAUTH,
        }
    ),
    # vex-domain.com: all models allowed (permissive)
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
        secret_key_env="stripe_secret_key_vex",  # noqa: S106
        webhook_secret_env="stripe_webhook_secret_vex",  # noqa: S106
        publishable_key_env="stripe_publishable_key_vex",
    ),
    nowpayments_config=NowPaymentsConfig(
        api_key_env="nowpayments_api_key_vex",
        ipn_secret_env="nowpayments_ipn_secret_vex",  # noqa: S106
    ),
    rate_limits=ProductRateLimits(
        generation_per_user="120/hour",
        generation_per_user_burst="15/minute",
        upload_per_user="200/hour",
        api_global="2000/minute",
    ),
    features=frozenset(),
    cookie_domain="vex.pics",
)

SYNTHARA_CONFIG = ProductConfig(
    product=Product.SYNTHARA,
    slug="synthara",
    display_name="Synthara",
    domains=frozenset(
        {
            "synthara.app",
        }
    ),
    frontend_origin="https://synthara.app",
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
        secret_key_env="stripe_secret_key_synthara",  # noqa: S106
        webhook_secret_env="stripe_webhook_secret_synthara",  # noqa: S106
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
    cookie_domain="synthara.app",
)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

PRODUCT_REGISTRY: dict[Product, ProductConfig] = {
    Product.VEX: VEX_CONFIG,
    Product.SYNTHARA: SYNTHARA_CONFIG,
}

_ALL_PRODUCTS: tuple[ProductConfig, ...] = tuple(PRODUCT_REGISTRY.values())


def extract_apex_domain(hostname: str) -> str | None:
    """Return the eTLD+1 for *hostname*, or ``None`` if the hostname is invalid/IP.

    Examples::

        extract_apex_domain("staging.vex-domain.com")   -> "vex-domain.com"
        extract_apex_domain("app.synthara.app")    -> "synthara.app"
        extract_apex_domain("vex-domain.com")            -> "vex-domain.com"
        extract_apex_domain("localhost")           -> None
        extract_apex_domain("127.0.0.1")           -> None

    ``tldextract`` uses the Mozilla Public Suffix List, so multi-part TLDs
    (e.g. ``.co.uk``) are handled correctly without bespoke string splitting.
    """
    ext = _tldextract.extract(hostname)
    return f"{ext.domain}.{ext.suffix}" if ext.domain and ext.suffix else None


def get_product_config(product: Product) -> ProductConfig:
    """Get configuration for a product. Raises KeyError if unknown."""
    return PRODUCT_REGISTRY[product]


def resolve_product_by_domain(hostname: str) -> ProductConfig | None:
    """Resolve product config from a raw hostname (may include subdomain).

    Matching is performed on the eTLD+1 so that ``staging.vex-domain.com``,
    ``app.vex-domain.com`` and ``vex-domain.com`` all resolve to the same product.
    """
    apex = extract_apex_domain(hostname.lower().split(":")[0])
    if apex is None:
        return None
    return next((config for config in _ALL_PRODUCTS if apex in config.domains), None)


def resolve_product_by_slug(slug: str) -> ProductConfig | None:
    """Resolve product config from a slug string (for X-Product-Id header)."""
    try:
        product = Product(slug.lower())
        return PRODUCT_REGISTRY.get(product)
    except ValueError:
        return None
