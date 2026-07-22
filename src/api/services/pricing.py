"""Pricing service for token cost lookups."""

from __future__ import annotations

from typing import TYPE_CHECKING

import structlog

from src.api.services.billing_errors import PriceNotFoundError
from src.core.enums import ModelType
from src.core.uid import new_id
from src.db.repositories.billing import (
    UNSET_OPTIONAL_UPDATE,
    BillingRepository,
    OptionalUpdate,
)
from src.db.repositories.generation_model import GenerationModelRepository

if TYPE_CHECKING:
    from collections.abc import Sequence
    from datetime import datetime
    from uuid import UUID

    from sqlalchemy.ext.asyncio import AsyncSession

    from src.core.product import ProductConfig
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
        product_config: ProductConfig | None = None,
        session: AsyncSession,
    ) -> Sequence[PricingRule]:
        """List pricing rules.

        Args:
            active_only: Restrict to currently-active, in-effect rules.
            product_config: If given, also drop rules pinned to a specific model
                (`model` is not None) that is disabled or not allowed for this
                product. Rules with `model=None` (provider/generation_type-wide)
                are always kept since they aren't tied to one model. Pass None
                (admin management views) to see the unfiltered catalog.
        """
        repo = BillingRepository(session)
        rules = await repo.list_pricing_rules(active_only=active_only)
        if product_config is None:
            return rules

        model_repo = GenerationModelRepository(session)
        allowed_model_keys = {
            m.model_key for m in await model_repo.list_enabled_for_product(product_config)
        }

        def _rule_allowed(rule: PricingRule) -> bool:
            if rule.model is None:
                return True
            try:
                ModelType(rule.model)
            except ValueError:
                return True
            return rule.model in allowed_model_keys

        return [r for r in rules if _rule_allowed(r)]

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
