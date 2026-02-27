"""Pricing service for token cost lookups."""

from __future__ import annotations

from typing import TYPE_CHECKING
from uuid import UUID, uuid4

import structlog

from src.api.services.billing_errors import PriceNotFoundError
from src.db.repositories.billing import BillingRepository

if TYPE_CHECKING:
    from collections.abc import Sequence

    from sqlalchemy.ext.asyncio import AsyncSession

    from src.db.models.billing import PricingRule

logger = structlog.get_logger(__name__)


class PricingService:
    """Service for pricing catalog operations."""

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
        repo = BillingRepository(session)
        rule = await repo.get_active_price(provider, generation_type, model)
        if rule is None:
            raise PriceNotFoundError(
                f"No active pricing rule for {provider}/{generation_type}/{model}"
            )
        return rule.token_cost

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
        notes: str | None,
        admin_id: UUID,
        session: AsyncSession,
    ) -> PricingRule:
        repo = BillingRepository(session)
        return await repo.create_pricing_rule(
            id=uuid4(),
            provider=provider,
            generation_type=generation_type,
            model=model,
            token_cost=token_cost,
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
        is_active: bool | None = None,
        effective_until: object = None,  # datetime | None
        notes: str | None = None,
        session: AsyncSession,
    ) -> PricingRule:
        repo = BillingRepository(session)
        rule = await repo.get_pricing_rule(rule_id)
        if rule is None:
            raise PriceNotFoundError(f"Pricing rule {rule_id} not found")
        if token_cost is not None:
            rule.token_cost = token_cost
        if is_active is not None:
            rule.is_active = is_active
        if effective_until is not None:
            rule.effective_until = effective_until  # type: ignore[assignment]
        if notes is not None:
            rule.notes = notes
        await session.flush()
        return rule
