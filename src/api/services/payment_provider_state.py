"""Capability-intersected runtime payment provider state."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import msgspec
import structlog

from src.api.services.billing_errors import UnknownProviderError
from src.core.product import PaymentProvider
from src.core.uid import new_id
from src.db.models.admin import AdminAuditLog
from src.db.repositories.admin import AdminRepository
from src.db.repositories.payment_provider_state import PaymentProviderStateRepository

if TYPE_CHECKING:
    from collections.abc import Callable
    from uuid import UUID

    from sqlalchemy.ext.asyncio import AsyncSession

    from src.core.config import Settings
    from src.core.product import ProductConfig
    from src.db.models.billing import PaymentProviderState

logger = structlog.get_logger(__name__)


class ProviderInfo(msgspec.Struct, frozen=True, kw_only=True):
    """Effective/admin-facing payment provider state."""

    provider: PaymentProvider
    is_enabled: bool
    display_order: int
    credentials_configured: bool


class PaymentProviderStateService:
    """Combine static product capability with runtime database overrides."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def _credentials_configured(self, product_id: str, provider: PaymentProvider) -> bool:
        checks: dict[PaymentProvider, tuple[Callable[[str], str], ...]] = {
            PaymentProvider.STRIPE: (
                self._settings.stripe_secret_key_for,
                self._settings.stripe_webhook_secret_for,
            ),
            PaymentProvider.NOWPAYMENTS: (
                self._settings.nowpayments_api_key_for,
                self._settings.nowpayments_ipn_secret_for,
            ),
        }
        try:
            return all(bool(resolve(product_id)) for resolve in checks[provider])
        except (KeyError, RuntimeError):
            return False

    async def _provider_infos(
        self,
        product_config: ProductConfig,
        *,
        session: AsyncSession,
        include_disabled: bool,
    ) -> list[ProviderInfo]:
        states = await PaymentProviderStateRepository(session).get_states(product_config.slug)
        by_provider = {state.provider: state for state in states}
        infos: list[ProviderInfo] = []
        for provider in product_config.payment_providers:
            state = by_provider.get(provider.value)
            is_enabled = state is None or state.is_enabled
            if is_enabled or include_disabled:
                infos.append(
                    ProviderInfo(
                        provider=provider,
                        is_enabled=is_enabled,
                        display_order=state.display_order if state is not None else 0,
                        credentials_configured=self._credentials_configured(
                            product_config.slug, provider
                        ),
                    )
                )
        return sorted(infos, key=lambda info: (info.display_order, info.provider.value))

    async def effective_providers(
        self, product_config: ProductConfig, *, session: AsyncSession
    ) -> list[ProviderInfo]:
        """Return enabled capability members in checkout display order."""

        return await self._provider_infos(product_config, session=session, include_disabled=False)

    async def all_providers(
        self, product_config: ProductConfig, *, session: AsyncSession
    ) -> list[ProviderInfo]:
        """Return all capability members, including runtime-disabled entries."""

        return await self._provider_infos(product_config, session=session, include_disabled=True)

    async def is_effective(
        self,
        product_config: ProductConfig,
        provider: PaymentProvider,
        *,
        session: AsyncSession,
    ) -> bool:
        if provider not in product_config.payment_providers:
            return False
        states = await PaymentProviderStateRepository(session).get_states(product_config.slug)
        state = next((row for row in states if row.provider == provider.value), None)
        return state is None or state.is_enabled

    async def set_state(
        self,
        product_config: ProductConfig,
        provider: PaymentProvider,
        *,
        is_enabled: bool | None,
        display_order: int | None,
        actor_id: UUID,
        session: AsyncSession,
    ) -> ProviderInfo:
        if provider not in product_config.payment_providers:
            raise UnknownProviderError(provider)

        repo = PaymentProviderStateRepository(session)
        existing = await repo.get_states(product_config.slug)
        previous: PaymentProviderState | None = next(
            (row for row in existing if row.provider == provider.value), None
        )
        previous_enabled = previous is None or previous.is_enabled
        previous_order = previous.display_order if previous is not None else 0
        state = await repo.upsert_state(
            product_config.slug,
            provider,
            is_enabled=is_enabled,
            display_order=display_order,
            updated_by=actor_id,
        )

        action: str | None = None
        if is_enabled is not None and is_enabled != previous_enabled:
            action = f"payment_provider.{'enable' if is_enabled else 'disable'}"
        elif display_order is not None and display_order != previous_order:
            action = "payment_provider.reorder"

        if action is not None:
            detail = json.dumps(
                {
                    "provider": provider.value,
                    "is_enabled": state.is_enabled,
                    "display_order": state.display_order,
                },
                sort_keys=True,
            )
            await AdminRepository(session).write_audit(
                AdminAuditLog(
                    id=new_id(),
                    actor_id=actor_id,
                    target_user_id=None,
                    product_id=product_config.slug,
                    action=action,
                    detail=detail,
                    source="api",
                )
            )
            logger.info(
                "payment_provider.state_changed",
                actor_id=str(actor_id),
                product_id=product_config.slug,
                provider=provider.value,
                is_enabled=state.is_enabled,
                display_order=state.display_order,
                action=action,
            )
        return ProviderInfo(
            provider=provider,
            is_enabled=state.is_enabled,
            display_order=state.display_order,
            credentials_configured=self._credentials_configured(product_config.slug, provider),
        )
