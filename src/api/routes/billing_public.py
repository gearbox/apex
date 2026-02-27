"""Public billing endpoints — no authentication required.

These endpoints power the purchase/pricing page on the frontend.

Endpoints:
  GET /api/v1/billing/packages   — available token packages with USD prices
  GET /api/v1/billing/pricing    — active per-operation prices from pricing_catalog
"""

from __future__ import annotations

from collections.abc import Sequence
from decimal import Decimal

import msgspec
import structlog
from litestar import Controller, get
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.config import TOKEN_PACKAGES

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class TokenPackageResponse(msgspec.Struct, kw_only=True):
    """A purchasable token package."""

    id: str
    """Unique package identifier (matches Stripe price lookup key)."""

    name: str
    """Display name, e.g. ``Starter``."""

    tokens: int
    """Number of tokens granted on purchase."""

    price_usd: str
    """Price in USD as a decimal string, e.g. ``"9.99"``."""

    popular: bool = False
    """True for the recommended / most popular package (UI badge)."""

    bonus_pct: int = 0
    """Bonus percentage above base rate, e.g. ``20`` for 20% extra tokens."""


class PricingRulePublicResponse(msgspec.Struct, kw_only=True):
    """Active price for a provider + generation_type + model combination."""

    provider: str
    generation_type: str
    model: str | None
    token_cost: int


class PublicPricingResponse(msgspec.Struct, kw_only=True):
    """Full public pricing catalog."""

    packages: list[TokenPackageResponse]
    prices: list[PricingRulePublicResponse]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_packages() -> list[TokenPackageResponse]:
    return sorted(
        [
            TokenPackageResponse(
                id=pkg.id,
                name=pkg.name,
                tokens=pkg.tokens,
                price_usd=str(pkg.price_usd),
                bonus_pct=round(pkg.bonus_tokens * 100 / pkg.tokens) if pkg.bonus_tokens else 0,
            )
            for pkg in TOKEN_PACKAGES.values()
        ],
        key=lambda p: Decimal(p.price_usd),
    )


# ---------------------------------------------------------------------------
# Controller
# ---------------------------------------------------------------------------


class PublicBillingController(Controller):
    """Public billing information — no auth required."""

    path = "/api/v1/billing"
    tags: Sequence[str] | None = ["Billing"]

    @get("/packages")
    async def list_packages(self) -> list[TokenPackageResponse]:
        """List available token packages for the purchase page.

        Returns all active packages ordered by price ascending.
        Prices and token amounts are sourced from application configuration
        (``TOKEN_PACKAGES`` in ``src/core/config.py``).
        """
        return _build_packages()

    @get("/pricing")
    async def get_pricing(
        self,
        session: AsyncSession,
    ) -> PublicPricingResponse:
        """Get active generation prices and available packages.

        Returns:
          - ``packages``: purchasable token packages
          - ``prices``: per-operation token costs from the pricing catalog

        This endpoint is used by the frontend to show cost estimates
        before a user submits a generation job.
        """
        from sqlalchemy import select

        from src.db.models.billing import PricingRule  # type: ignore[attr-defined]

        result = await session.execute(
            select(PricingRule).where(PricingRule.is_active == True)  # noqa: E712
        )
        rules = result.scalars().all()

        prices = [
            PricingRulePublicResponse(
                provider=rule.provider,
                generation_type=rule.generation_type,
                model=rule.model,
                token_cost=rule.token_cost,
            )
            for rule in rules
        ]

        return PublicPricingResponse(packages=_build_packages(), prices=prices)
