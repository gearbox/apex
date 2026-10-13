"""Pricing service for token cost lookups."""

from __future__ import annotations

from typing import TYPE_CHECKING

import structlog

from src.api.services.billing_errors import InvalidPricingRuleError, PriceNotFoundError
from src.core.enums import GenerationType, MediaKind, ModelType
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

    from src.core.enums import VideoResolution
    from src.core.product import ProductConfig
    from src.db.models.billing import PricingRule

logger = structlog.get_logger(__name__)


def validate_rule_invariants(
    *,
    generation_type: str,
    token_cost: int,
    input_token_cost: int,
    per_second_token_cost: int,
    has_video_resolution: bool,
) -> None:
    """Check the catalog invariants for the *resulting* state of one rule.

    The database ``chk_pricing_*`` constraints stay as the backstop; this runs
    first so a violation is a 422 naming the problem, never a 500 — and so a
    rule that ``quote()`` would later reject for every end user cannot be stored.

    Raises:
        InvalidPricingRuleError: ``generation_type`` is unknown, a cost is negative, the
            rule costs nothing, or a per-second / resolution-specific rule targets a
            generation type without video output.
    """
    try:
        mode = GenerationType(generation_type)
    except ValueError:
        raise InvalidPricingRuleError(f"Unknown generation_type '{generation_type}'") from None

    if min(token_cost, input_token_cost, per_second_token_cost) < 0:
        raise InvalidPricingRuleError("Pricing costs must not be negative")
    if token_cost == 0 and per_second_token_cost == 0:
        raise InvalidPricingRuleError(
            "A pricing rule needs token_cost > 0 or per_second_token_cost > 0"
        )
    if (
        per_second_token_cost > 0 or has_video_resolution
    ) and mode.output_kind is not MediaKind.VIDEO:
        raise InvalidPricingRuleError(
            f"per_second_token_cost and video_resolution apply to video generation types only; "
            f"'{generation_type}' produces {mode.output_kind.value}"
        )


class PricingService:
    """Service for pricing catalog operations."""

    async def _get_rule(
        self,
        provider: str,
        generation_type: str,
        model: str | None,
        video_resolution: VideoResolution | None = None,
        *,
        session: AsyncSession,
    ) -> PricingRule:
        repo = BillingRepository(session)
        rule = await repo.get_active_price(provider, generation_type, model, video_resolution)
        if rule is None:
            resolution = f"/{video_resolution.value}" if video_resolution is not None else ""
            raise PriceNotFoundError(
                f"No active pricing rule for {provider}/{generation_type}/{model}{resolution}"
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

        Priority: exact (provider, generation_type, model) first, then wildcard
        (provider, generation_type, NULL); see ``BillingRepository.get_active_price``
        for the video-resolution tiers.

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
        source_media_count: int,
        billed_seconds: int = 0,
        video_resolution: VideoResolution | None = None,
        session: AsyncSession,
    ) -> int:
        """Total token cost: ``(token_cost + input_token_cost * k + per_second_token_cost * s) * n``.

        Args:
            n: Output count.
            source_media_count: ``k`` — number of input assets.
            billed_seconds: ``s`` — video seconds the provider will bill (0 for images).
            video_resolution: Selects a resolution-specific rule for video jobs.

        Raises:
            ValueError: ``n < 1``, a negative count, or a per-second rule priced with
                ``billed_seconds == 0`` (a caller that forgot to pass the duration).
        """
        if n < 1:
            raise ValueError("n must be >= 1")
        if source_media_count < 0:
            raise ValueError("source_media_count must be >= 0")
        if billed_seconds < 0:
            raise ValueError("billed_seconds must be >= 0")

        rule = await self._get_rule(
            provider, generation_type, model, video_resolution, session=session
        )
        if rule.per_second_token_cost > 0 and billed_seconds == 0:
            raise ValueError("per-second rule requires billed_seconds")
        return (
            rule.token_cost
            + rule.input_token_cost * source_media_count
            + rule.per_second_token_cost * billed_seconds
        ) * n

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
        per_second_token_cost: int = 0,
        video_resolution: VideoResolution | None = None,
        notes: str | None,
        admin_id: UUID,
        session: AsyncSession,
    ) -> PricingRule:
        validate_rule_invariants(
            generation_type=generation_type,
            token_cost=token_cost,
            input_token_cost=input_token_cost,
            per_second_token_cost=per_second_token_cost,
            has_video_resolution=video_resolution is not None,
        )
        repo = BillingRepository(session)
        return await repo.create_pricing_rule(
            id=new_id(),
            provider=provider,
            generation_type=generation_type,
            model=model,
            token_cost=token_cost,
            input_token_cost=input_token_cost,
            per_second_token_cost=per_second_token_cost,
            video_resolution=video_resolution,
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
        per_second_token_cost: int | None = None,
        is_active: bool | None = None,
        effective_until: OptionalUpdate[datetime] = UNSET_OPTIONAL_UPDATE,
        notes: OptionalUpdate[str] = UNSET_OPTIONAL_UPDATE,
        session: AsyncSession,
    ) -> PricingRule:
        repo = BillingRepository(session)
        current = await repo.get_pricing_rule(rule_id)
        if current is None:
            raise PriceNotFoundError(f"Pricing rule {rule_id} not found")
        # A patch that only toggles/ends/annotates a rule (notably deactivating a
        # broken one) must never be blocked by that rule's existing state.
        touches_costs = any(
            value is not None for value in (token_cost, input_token_cost, per_second_token_cost)
        )
        if touches_costs or is_active is True:
            validate_rule_invariants(
                generation_type=current.generation_type,
                token_cost=current.token_cost if token_cost is None else token_cost,
                input_token_cost=(
                    current.input_token_cost if input_token_cost is None else input_token_cost
                ),
                per_second_token_cost=(
                    current.per_second_token_cost
                    if per_second_token_cost is None
                    else per_second_token_cost
                ),
                has_video_resolution=current.video_resolution is not None,
            )
        rule = await repo.update_pricing_rule(
            rule_id,
            token_cost=token_cost,
            input_token_cost=input_token_cost,
            per_second_token_cost=per_second_token_cost,
            is_active=is_active,
            effective_until=effective_until,
            notes=notes,
        )
        if rule is None:
            raise PriceNotFoundError(f"Pricing rule {rule_id} not found")
        return rule
