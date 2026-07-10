"""Public, product-scoped billing discovery endpoints."""

from __future__ import annotations

from typing import TYPE_CHECKING

import msgspec
from litestar import Controller, get
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.services.payment_provider_state import PaymentProviderStateService
from src.core.product import PaymentProvider, ProductConfig

if TYPE_CHECKING:
    from collections.abc import Sequence


class PublicPaymentProvider(msgspec.Struct, frozen=True, kw_only=True):
    """Provider information needed by the unauthenticated checkout UI."""

    provider: PaymentProvider
    display_order: int


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
