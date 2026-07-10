"""Superadmin runtime management of product payment providers."""

from __future__ import annotations

from typing import TYPE_CHECKING

from litestar import Controller, get, patch
from litestar.di import Provide
from litestar.exceptions import HTTPException, NotFoundException
from litestar.status_codes import HTTP_400_BAD_REQUEST
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.dependencies.auth import get_current_superadmin_user
from src.api.schemas.admin import PaymentProviderPatchRequest
from src.api.security import auth_guard
from src.api.services.billing_errors import UnknownProviderError
from src.api.services.payment_provider_state import (
    PaymentProviderStateService,
    ProviderInfo,
)
from src.core.product import PaymentProvider, ProductConfig
from src.db.models import User

if TYPE_CHECKING:
    from collections.abc import Sequence


class PaymentProviderAdminController(Controller):
    """Superadmin-only provider state endpoints."""

    path = "/v1/admin/payments/providers"
    tags: Sequence[str] | None = ("Payment Provider Management",)
    guards = [auth_guard]  # noqa: RUF012
    dependencies = {  # noqa: RUF012
        "superadmin": Provide(get_current_superadmin_user),
    }

    @get("/")
    async def list_providers(
        self,
        superadmin: User,
        session: AsyncSession,
        product_config: ProductConfig,
        payment_provider_state_service: PaymentProviderStateService,
    ) -> list[ProviderInfo]:
        del superadmin
        return await payment_provider_state_service.all_providers(product_config, session=session)

    @patch("/{provider:str}")
    async def update_provider(
        self,
        superadmin: User,
        provider: str,
        data: PaymentProviderPatchRequest,
        session: AsyncSession,
        product_config: ProductConfig,
        payment_provider_state_service: PaymentProviderStateService,
    ) -> ProviderInfo:
        if data.is_enabled is None and data.display_order is None:
            raise HTTPException(
                status_code=HTTP_400_BAD_REQUEST,
                detail="At least one of is_enabled or display_order is required",
            )
        try:
            provider_enum = PaymentProvider(provider)
        except ValueError as exc:
            raise NotFoundException(detail=f"Payment provider '{provider}' not found") from exc

        try:
            result = await payment_provider_state_service.set_state(
                product_config,
                provider_enum,
                is_enabled=data.is_enabled,
                display_order=data.display_order,
                actor_id=superadmin.id,
                session=session,
            )
        except UnknownProviderError as exc:
            raise NotFoundException(detail=str(exc)) from exc
        await session.commit()
        return result
