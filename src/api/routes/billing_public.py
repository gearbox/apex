"""Public, product-scoped billing discovery endpoints."""

from __future__ import annotations

from typing import TYPE_CHECKING

import msgspec
from litestar import Controller, get
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.services.payment_provider_state import PaymentProviderStateService
from src.core.config import Settings
from src.core.product import PaymentProvider, ProductConfig
from src.db.repositories.payment_currency import PaymentCurrencyRepository

if TYPE_CHECKING:
    from collections.abc import Sequence


class PublicPaymentProvider(msgspec.Struct, frozen=True, kw_only=True):
    """Provider information needed by the unauthenticated checkout UI."""

    provider: PaymentProvider
    display_order: int


class PublicCurrency(msgspec.Struct, frozen=True, kw_only=True):
    """Currency catalog entry for the unauthenticated checkout currency picker.

    Empty list ⇒ FE hides the picker and omits ``pay_currency``; the
    provider's own invoice-page picker takes over (D5) — checkout never
    depends on catalog state. ``logo_url: null`` ⇒ generic coin icon;
    ``name``/``network: null`` ⇒ ticker-only label.
    """

    ticker: str
    name: str | None
    network: str | None
    logo_url: str | None


class BillingPublicController(Controller):
    """Public billing capability discovery for the resolved product."""

    path = "/v1/billing"
    tags: Sequence[str] | None = ("Billing",)

    @get("/providers")
    async def list_payment_providers(
        self,
        session: AsyncSession,
        product_config: ProductConfig,
        payment_provider_state_service: PaymentProviderStateService,
    ) -> list[PublicPaymentProvider]:
        providers = await payment_provider_state_service.effective_providers(
            product_config, session=session
        )
        return [
            PublicPaymentProvider(
                provider=info.provider,
                display_order=info.display_order,
            )
            for info in providers
        ]

    @get("/currencies")
    async def list_currencies(
        self,
        session: AsyncSession,
        product_config: ProductConfig,
        settings: Settings,
    ) -> list[PublicCurrency]:
        """Available currencies for the product's catalog-capable providers.

        Cold/failed cache ⇒ ``[]`` (D5). Never gates on provider config or
        catalog freshness — a stale or empty cache degrades to "no picker",
        never a checkout error.
        """
        rows = await PaymentCurrencyRepository(session).list_currencies(
            product_config.slug, only_available=True
        )
        assets_base = (
            settings.r2_public_url_base.rstrip("/") if settings.r2_public_url_base else None
        )
        return [
            PublicCurrency(
                ticker=row.ticker,
                name=row.name,
                network=row.network,
                logo_url=f"{assets_base}/{row.logo_key}" if assets_base and row.logo_key else None,
            )
            for row in rows
        ]
