"""Resolve the per-mode generation contract for a model."""

from __future__ import annotations

from typing import TYPE_CHECKING

import structlog

from src.core.model_registry import get_model_meta

if TYPE_CHECKING:
    from src.api.services.workflow.contract import BundleCapabilities
    from src.core.enums import ModelType
    from src.core.generation_mode import GenerationModes

logger = structlog.get_logger(__name__)


def resolve_generation_modes(
    model: ModelType,
    *,
    capabilities: BundleCapabilities | None,
) -> GenerationModes:
    """The single authority on what each mode of ``model`` accepts.

    Two declarations meet here and mean different things:

    * the registry contract states what the **provider implementation** can
      execute for this model;
    * the bundle contract states what **this workflow** declares.

    The effective contract is their intersection. Neither side may widen the
    other, so any request that satisfies an advertised contract is executable
    by the provider. A mode whose intersection is unsatisfiable is not offered.
    """
    provider_modes = get_model_meta(model).generation_modes
    if capabilities is None:
        return provider_modes

    resolved = {}
    for generation_type, bundle_mode in capabilities.generation_modes.items():
        provider_mode = provider_modes.get(generation_type)
        if provider_mode is None:
            continue
        effective = provider_mode.intersect(bundle_mode)
        if effective is None:
            logger.error(
                "generation_mode.unsatisfiable",
                model=model.value,
                generation_type=generation_type.value,
                provider_contract=str(provider_mode.source_media),
                bundle_contract=str(bundle_mode.source_media),
            )
            continue
        resolved[generation_type] = effective
    return resolved
