"""Provider discovery endpoint (v2).

Returns provider-grouped, capability-rich model catalog.
Auth-optional: unauthenticated callers get the full catalog;
authenticated callers additionally receive user_context and per-model session_state.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from uuid import UUID

import structlog
from litestar import Controller, get
from litestar.di import Provide
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.dependencies.auth import get_optional_user_id
from src.api.schemas.providers import (
    ImageConstraints,
    ModelInfo,
    ModelInputs,
    ProviderInfo,
    ProvidersResponse,
    SourceMediaConstraints,
    UserContext,
    VideoConstraints,
)
from src.api.security import optional_auth_guard
from src.api.services.generation.service import GenerationService
from src.core.enums import (
    GenerationType,
    GpuSessionStatus,
    ModelSessionState,
    ModelType,
    Provider,
    ProvisioningMode,
    session_state_from_status,
)
from src.core.model_registry import get_model_meta
from src.core.product import ProductConfig
from src.core.resolution import TIER_MEGAPIXELS
from src.db.repositories.generation_model import GenerationModelRepository
from src.db.repositories.gpu_session import GpuSessionRepository
from src.db.repositories.user import UserRepository

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = structlog.get_logger(__name__)

# Provider display names — single source of truth; a missing entry raises KeyError (completeness test guards this)
PROVIDER_DISPLAY_NAMES: dict[Provider, str] = {
    Provider.AISHA: "Aisha",
    Provider.GROK: "xAI Grok",
}


def _build_model_info(
    mt: ModelType,
    record: object,
    *,
    session_state: ModelSessionState | None,
) -> ModelInfo:
    """Build ModelInfo from ModelType enum properties + model registry metadata.

    Args:
        mt: The ModelType enum member.
        record: The GenerationModel DB record (has .name, .description, .is_enabled).
        session_state: Per-user readiness; None for always-on providers or unauthenticated.
    """
    meta = get_model_meta(mt)
    return ModelInfo(
        model_key=mt.value,
        name=record.name,  # type: ignore[attr-defined]
        description=record.description,  # type: ignore[attr-defined]
        capabilities=[gt.value for gt in GenerationType if mt.supports_generation_type(gt)],
        is_enabled=record.is_enabled,  # type: ignore[attr-defined]
        max_images=meta.max_concurrent_outputs,
        max_prompt_length=meta.max_prompt_length,
        supports_negative_prompt=meta.supports_negative_prompt,
        aspect_ratios=[ar.value for ar in meta.aspect_ratios],
        requires_age_verification=meta.requires_age_verification,
        inputs=ModelInputs(
            source_media=(
                SourceMediaConstraints(
                    min=meta.inputs.source_media.min,
                    max=meta.inputs.source_media.max,
                    media_types=sorted(
                        meta.inputs.source_media.media_types, key=lambda kind: kind.value
                    ),
                )
                if meta.inputs.source_media is not None
                else None
            )
        ),
        image=(
            ImageConstraints(
                min_height=meta.image.min_height,
                max_height=meta.image.max_height,
                default_height=meta.image.default_height,
                output_resolutions=(
                    list(meta.image.output_resolutions)
                    if meta.image.output_resolutions is not None
                    else None
                ),
                supported_tiers=(
                    [t.value for t in meta.image.supported_tiers]
                    if meta.image.supported_tiers is not None
                    else None
                ),
                default_tier=(
                    meta.image.default_tier.value if meta.image.default_tier is not None else None
                ),
                tier_megapixels=(
                    {t.value: mp for t, mp in TIER_MEGAPIXELS.items()}
                    if meta.image.supported_tiers is not None
                    else None
                ),
                edit_aspect_ratios=[ar.value for ar in meta.image.edit_aspect_ratios],
            )
            if meta.image is not None
            else None
        ),
        video=(
            VideoConstraints(
                max_duration=meta.video.max_duration,
                resolutions=[r.value for r in meta.video.resolutions],
            )
            if meta.video is not None
            else None
        ),
        session_state=session_state.value if session_state is not None else None,
    )


class ProvidersController(Controller):
    """Provider and model discovery (v2)."""

    path = "/v1/providers"
    tags: Sequence[str] | None = ("Providers",)
    guards = [optional_auth_guard]  # noqa: RUF012
    dependencies = {"current_user_id": Provide(get_optional_user_id)}  # noqa: RUF012

    @get("/")
    async def list_providers(
        self,
        session: AsyncSession,
        generation_service: GenerationService,
        current_user_id: UUID | None,
        product_config: ProductConfig,
        product_id: str,
    ) -> ProvidersResponse:
        """List available providers and their models.

        Returns provider-grouped model catalog with capability metadata.
        When authenticated, includes user_context with subscription tier and
        per-model session_state for on-demand providers.
        Models are filtered by the current product's allowlist/blocklist.
        """
        repo = GenerationModelRepository(session)
        db_models = await repo.list_enabled_for_product(product_config)

        # Build per-user session state map in a single bulk query (authenticated only)
        session_state_by_model: dict[str, ModelSessionState] = {}
        if current_user_id is not None:
            sessions = await GpuSessionRepository(session).list_by_user(
                current_user_id, product_id, include_terminal=False
            )
            # list is created_at DESC; prefer an active session, else keep the newest seen
            for s in sessions:
                s_state = session_state_from_status(GpuSessionStatus(s.status))
                existing = session_state_by_model.get(s.model_type)
                if existing is None or s_state is ModelSessionState.ACTIVE:
                    session_state_by_model[s.model_type] = s_state

        # Group models by provider
        configured = generation_service.configured_providers
        provider_models: dict[Provider, list[ModelInfo]] = {}
        for record in db_models:
            try:
                mt = ModelType(record.model_key)
            except ValueError:
                logger.warning(
                    "providers.unknown_model_key",
                    model_key=record.model_key,
                )
                continue

            mode = mt.provider.provisioning_mode
            state: ModelSessionState | None = (
                session_state_by_model.get(mt.value, ModelSessionState.NONE)
                if mode is ProvisioningMode.ON_DEMAND and current_user_id is not None
                else None
            )
            info = _build_model_info(mt, record, session_state=state)
            provider_models.setdefault(mt.provider, []).append(info)

        # Build provider list — include all known providers even with 0 models
        providers = [
            ProviderInfo(
                provider=p.value,
                name=PROVIDER_DISPLAY_NAMES[p],
                available=(p in configured),
                provisioning_mode=p.provisioning_mode.value,
                models=provider_models.get(p, []),
            )
            for p in Provider
        ]

        # Optional user context
        user_context: UserContext | None = None
        if current_user_id is not None:
            user_repo = UserRepository(session)
            user = await user_repo.get_active_user(current_user_id)
            if user is not None:
                tier = user.subscription_tier
                user_context = UserContext(
                    subscription_tier=tier.value if hasattr(tier, "value") else str(tier),
                )

        return ProvidersResponse(providers=providers, user_context=user_context)
