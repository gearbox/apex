"""Superadmin runtime management of product payment providers and currency catalog."""

from __future__ import annotations

import json
from datetime import datetime
from typing import TYPE_CHECKING

import msgspec
from litestar import Controller, get, patch, post
from litestar.di import Provide
from litestar.exceptions import HTTPException, NotFoundException
from litestar.status_codes import HTTP_400_BAD_REQUEST, HTTP_502_BAD_GATEWAY
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.dependencies.auth import get_current_superadmin_user
from src.api.schemas.admin import PaymentProviderPatchRequest
from src.api.security import auth_guard
from src.api.services.billing_errors import UnknownProviderError
from src.api.services.payment_currency_sync import PaymentCurrencySyncService, SyncResult
from src.api.services.payment_provider_state import (
    PaymentProviderStateService,
    ProviderInfo,
)
from src.core.product import PaymentProvider, ProductConfig
from src.core.uid import new_id
from src.db.models import AdminAuditLog, User
from src.db.repositories.admin import AdminRepository
from src.db.repositories.payment_currency import PaymentCurrencyRepository

if TYPE_CHECKING:
    from collections.abc import Sequence


class AdminCurrency(msgspec.Struct, frozen=True, kw_only=True):
    """Full catalog row (incl. unavailable) for superadmin visibility."""

    ticker: str
    provider: PaymentProvider
    is_available: bool
    name: str | None
    network: str | None
    logo_key: str | None
    logo_source_url: str | None
    logo_synced_at: datetime | None
    last_seen_at: datetime


class PaymentProviderAdminController(Controller):
    """Superadmin-only provider state and currency catalog endpoints."""

    path = "/v1/admin/payments"
    tags: Sequence[str] | None = ("Payment Provider Management",)
    guards = [auth_guard]  # noqa: RUF012
    dependencies = {  # noqa: RUF012
        "superadmin": Provide(get_current_superadmin_user),
    }

    @get("/providers")
    async def list_providers(
        self,
        superadmin: User,
        session: AsyncSession,
        product_config: ProductConfig,
        payment_provider_state_service: PaymentProviderStateService,
    ) -> list[ProviderInfo]:
        del superadmin
        return await payment_provider_state_service.all_providers(product_config, session=session)

    @patch("/providers/{provider:str}")
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

    @get("/currencies")
    async def list_currencies(
        self,
        superadmin: User,
        session: AsyncSession,
        product_config: ProductConfig,
    ) -> list[AdminCurrency]:
        """Full currency catalog including unavailable rows, for admin visibility."""
        del superadmin
        rows = await PaymentCurrencyRepository(session).list_currencies(
            product_config.slug, only_available=False
        )
        return [
            AdminCurrency(
                ticker=row.ticker,
                provider=PaymentProvider(row.provider),
                is_available=row.is_available,
                name=row.name,
                network=row.network,
                logo_key=row.logo_key,
                logo_source_url=row.logo_source_url,
                logo_synced_at=row.logo_synced_at,
                last_seen_at=row.last_seen_at,
            )
            for row in rows
        ]

    @post("/currencies/refresh")
    async def refresh_currencies(
        self,
        superadmin: User,
        session: AsyncSession,
        product_config: ProductConfig,
        payment_currency_sync_service: PaymentCurrencySyncService,
    ) -> list[SyncResult]:
        """Synchronously refresh the current product's currency catalog.

        A provider failure aborts the whole refresh (502) and leaves the
        previously synced catalog untouched — the route never commits a
        partial sync (D6).
        """
        try:
            results = await payment_currency_sync_service.refresh(product_config, session=session)
        except Exception as exc:
            raise HTTPException(status_code=HTTP_502_BAD_GATEWAY, detail=str(exc)) from exc

        detail = json.dumps(
            {
                "results": [
                    {
                        "provider": r.provider.value,
                        "upserted": r.upserted,
                        "deactivated": r.deactivated,
                    }
                    for r in results
                ]
            },
            sort_keys=True,
        )
        await AdminRepository(session).write_audit(
            AdminAuditLog(
                id=new_id(),
                actor_id=superadmin.id,
                target_user_id=None,
                product_id=product_config.slug,
                action="payment_currencies.refresh",
                detail=detail,
                source="api",
            )
        )
        await session.commit()
        return results
