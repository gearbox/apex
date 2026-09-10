"""Provider discovery endpoint (v2).

Returns provider-grouped, capability-rich model catalog.
Auth-optional: unauthenticated callers get the full catalog;
authenticated callers additionally receive user_context and per-model runtime.
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
    GenerationModeInfo,
    ImageConstraints,
    ModelInfo,
    ModelInputs,
    ModelProvisioningHintResponse,
    ModelRuntimeResponse,
    ProviderInfo,
    ProvidersResponse,
    SourceMediaConstraints,
    SourceMediaModeConstraints,
    UserContext,
    VideoConstraints,
)
from src.api.security import optional_auth_guard
from src.api.services.generation.generation_modes import resolve_generation_modes
from src.api.services.generation.service import GenerationService
from src.api.services.workflow.contract import (
    DIMENSION_REQUEST_PARAMETERS,
    DIMENSION_WRITABLE_PARAMETERS,
    NEGATIVE_PROMPT_PARAMETER,
    REQUEST_PARAMETER_BINDINGS,
    BundleCapabilities,
)
from src.core.enums import (
    DeploymentStatus,
    GenerationType,
    GpuSessionStatus,
    ModelType,
    Provider,
    ProvisioningMode,
    RuntimeState,
    runtime_state_from_session_status,
)
from src.core.model_registry import get_model_meta
from src.core.product import ProductConfig
from src.core.resolution import TIER_MEGAPIXELS
from src.db.repositories.generation_model import GenerationModelRepository
from src.db.repositories.gpu_session_deployment import GpuSessionDeploymentRepository
from src.db.repositories.user import UserRepository

if TYPE_CHECKING:
    from collections.abc import Sequence

    from src.core.generation_mode import SourceMediaConstraints as ModeSourceMediaConstraints

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
    runtime: ModelRuntimeResponse | None,
    capabilities: BundleCapabilities | None = None,
) -> ModelInfo:
    """Build ModelInfo from ModelType enum properties + model registry metadata.

    Args:
        mt: The ModelType enum member.
        record: The GenerationModel DB record (has .name, .description, .is_enabled).
        runtime: Per-user runtime; None for always-on providers or unauthenticated.
    """
    meta = get_model_meta(mt)
    requires_indexed_workflow = mt.provider.provisioning_mode is ProvisioningMode.ON_DEMAND
    has_indexed_workflow = isinstance(capabilities, BundleCapabilities)
    modes = resolve_generation_modes(mt, capabilities=capabilities)
    ordered = [generation_type for generation_type in GenerationType if generation_type in modes]
    generation_types = [generation_type.value for generation_type in ordered]
    if requires_indexed_workflow and isinstance(capabilities, BundleCapabilities):
        max_images = min(meta.max_concurrent_outputs, capabilities.max_batch_size)
        supports_negative = meta.supports_negative_prompt and capabilities.supports_negative_prompt
        writable = capabilities.writable
        unsupported_parameters = sorted(
            binding.parameter
            for binding in REQUEST_PARAMETER_BINDINGS
            if binding.writable not in writable
        )
        if not supports_negative:
            unsupported_parameters.append(NEGATIVE_PROMPT_PARAMETER)
        if not DIMENSION_WRITABLE_PARAMETERS.issubset(writable):
            unsupported_parameters.extend(DIMENSION_REQUEST_PARAMETERS)
        unsupported_parameters.sort()
    else:
        max_images = meta.max_concurrent_outputs
        supports_negative = meta.supports_negative_prompt
        unsupported_parameters = []

    contracts: list[ModeSourceMediaConstraints] = []
    for generation_type in ordered:
        contract = modes[generation_type].source_media
        if contract is not None:
            contracts.append(contract)
    legacy_source_media = (
        SourceMediaConstraints(
            min=min(contract.min for contract in contracts),
            max=max(contract.max for contract in contracts),
            media_types=sorted(
                {kind for contract in contracts for kind in contract.media_types},
                key=lambda kind: kind.value,
            ),
            required_for=[
                generation_type.value
                for generation_type in ordered
                if (contract := modes[generation_type].source_media) is not None
                and contract.min >= 1
            ],
        )
        if contracts
        else None
    )
    return ModelInfo(
        model_key=mt.value,
        name=record.name,  # type: ignore[attr-defined]
        description=record.description,  # type: ignore[attr-defined]
        capabilities=generation_types,
        generation_modes={
            generation_type.value: GenerationModeInfo(
                source_media=(
                    SourceMediaModeConstraints(
                        min=contract.min,
                        max=contract.max,
                        media_types=sorted(contract.media_types, key=lambda kind: kind.value),
                        roles=list(contract.roles) if contract.roles else None,
                    )
                    if (contract := modes[generation_type].source_media) is not None
                    else None
                )
            )
            for generation_type in ordered
        },
        is_enabled=record.is_enabled and (not requires_indexed_workflow or has_indexed_workflow),  # type: ignore[attr-defined]
        max_images=max_images,
        max_prompt_length=meta.max_prompt_length,
        supports_negative_prompt=supports_negative,
        unsupported_parameters=unsupported_parameters,
        aspect_ratios=[ar.value for ar in meta.aspect_ratios],
        requires_age_verification=meta.requires_age_verification,
        inputs=ModelInputs(source_media=legacy_source_media),
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
        runtime=runtime,
        provisioning=(
            ModelProvisioningHintResponse(
                typical_bootstrap_seconds=meta.typical_bootstrap_seconds,
                typical_attach_seconds=meta.typical_attach_seconds,
            )
            if requires_indexed_workflow
            else None
        ),
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
        per-model runtime for on-demand providers.
        Models are filtered by the current product's allowlist/blocklist.
        """
        repo = GenerationModelRepository(session)
        db_models = await repo.list_enabled_for_product(product_config)

        # One deployment-led query across all models. A GpuSession's legacy
        # model_type is deliberately ignored: sibling deployments own runtime.
        runtime_by_model: dict[str, ModelRuntimeResponse] = {}
        if current_user_id is not None:
            runtime_rows = await GpuSessionDeploymentRepository(session).list_live_runtime_for_user(
                current_user_id, product_id
            )
            for deployment, gpu_session, operation_id in runtime_rows:
                state = runtime_state_from_session_status(GpuSessionStatus(gpu_session.status))
                if state == RuntimeState.active:
                    if deployment.status == DeploymentStatus.deploying:
                        state = RuntimeState.provisioning
                    elif deployment.status == DeploymentStatus.removing:
                        state = RuntimeState.removing
                    elif (
                        deployment.status == DeploymentStatus.active
                        and deployment.routing_suspended
                    ):
                        state = RuntimeState.suspended
                runtime_by_model[deployment.model_type] = ModelRuntimeResponse(
                    state=state,
                    session_id=gpu_session.id,
                    deployment_id=deployment.id,
                    operation_id=operation_id,
                )

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
            runtime: ModelRuntimeResponse | None = (
                runtime_by_model.get(
                    mt.value,
                    ModelRuntimeResponse(
                        state=RuntimeState.none,
                        session_id=None,
                        deployment_id=None,
                        operation_id=None,
                    ),
                )
                if mode is ProvisioningMode.ON_DEMAND and current_user_id is not None
                else None
            )
            capabilities = generation_service.get_aisha_capabilities(mt)
            info = _build_model_info(
                mt,
                record,
                runtime=runtime,
                capabilities=capabilities,
            )
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
