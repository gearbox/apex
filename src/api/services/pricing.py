"""Pricing service for token cost lookups."""

from __future__ import annotations

from typing import TYPE_CHECKING

import structlog

from src.api.services.billing_errors import PriceNotFoundError
from src.core.uid import new_id
from src.db.repositories.billing import (
    UNSET_OPTIONAL_UPDATE,
    BillingRepository,
    OptionalUpdate,
)

if TYPE_CHECKING:
    from collections.abc import Sequence
    from datetime import datetime
    from uuid import UUID

    from sqlalchemy.ext.asyncio import AsyncSession

    from src.db.models.billing import PricingRule

logger = structlog.get_logger(__name__)


class PricingService:
    """Service for pricing catalog operations."""

    async def _get_rule(
        self,
        provider: str,
        generation_type: str,
        model: str | None,
        *,
        session: AsyncSession,
    ) -> PricingRule:
        repo = BillingRepository(session)
        rule = await repo.get_active_price(provider, generation_type, model)
        if rule is None:
            raise PriceNotFoundError(
                f"No active pricing rule for {provider}/{generation_type}/{model}"
            )
        return rule

    async def get_price(
        self,
        provider: str,
        generation_type: str,
        model: str | None,
        *,
        session: AsyncSession,
    ) -> int:
        """Look up token cost for a generation.

        Priority: exact (provider, generation_type, model) first,
        then wildcard (provider, generation_type, NULL) fallback.

        Raises:
            PriceNotFoundError: If no active rule found.
        """
        rule = await self._get_rule(provider, generation_type, model, session=session)
        return rule.token_cost

    async def quote(
        self,
        provider: str,
        generation_type: str,
        model: str | None,
        *,
        n: int,
        input_image_count: int,
        session: AsyncSession,
    ) -> int:
        """Total token cost for a generation: (token_cost + input_token_cost * k) * n."""
        if n < 1:
            raise ValueError("n must be >= 1")
        if input_image_count < 0:
            raise ValueError("input_image_count must be >= 0")

        rule = await self._get_rule(provider, generation_type, model, session=session)
        return (rule.token_cost + rule.input_token_cost * input_image_count) * n

    async def list_catalog(
        self,
        *,
        active_only: bool = True,
        session: AsyncSession,
    ) -> Sequence[PricingRule]:
        repo = BillingRepository(session)
        return await repo.list_pricing_rules(active_only=active_only)

    async def create_rule(
        self,
        *,
        provider: str,
        generation_type: str,
        model: str | None,
        token_cost: int,
        input_token_cost: int = 0,
        notes: str | None,
        admin_id: UUID,
        session: AsyncSession,
    ) -> PricingRule:
        repo = BillingRepository(session)
        return await repo.create_pricing_rule(
            id=new_id(),
            provider=provider,
            generation_type=generation_type,
            model=model,
            token_cost=token_cost,
            input_token_cost=input_token_cost,
            notes=notes,
            created_by=admin_id,
        )

    async def deactivate_rule(
        self,
        rule_id: UUID,
        *,
        session: AsyncSession,
    ) -> None:
        """Soft deactivate — sets is_active=False. Never hard deletes."""
        repo = BillingRepository(session)
        rule = await repo.get_pricing_rule(rule_id)
        if rule is None:
            raise PriceNotFoundError(f"Pricing rule {rule_id} not found")
        rule.is_active = False
        await session.flush()

    async def update_rule(
        self,
        rule_id: UUID,
        *,
        token_cost: int | None = None,
        input_token_cost: int | None = None,
        is_active: bool | None = None,
        effective_until: OptionalUpdate[datetime] = UNSET_OPTIONAL_UPDATE,
        notes: OptionalUpdate[str] = UNSET_OPTIONAL_UPDATE,
        session: AsyncSession,
    ) -> PricingRule:
        repo = BillingRepository(session)
        rule = await repo.update_pricing_rule(
            rule_id,
            token_cost=token_cost,
            input_token_cost=input_token_cost,
            is_active=is_active,
            effective_until=effective_until,
            notes=notes,
        )
        if rule is None:
            raise PriceNotFoundError(f"Pricing rule {rule_id} not found")
        return rule
