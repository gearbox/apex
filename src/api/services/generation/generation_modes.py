"""Resolve the per-mode generation contract for a model."""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.core.model_registry import get_model_meta

if TYPE_CHECKING:
    from src.api.services.workflow.contract import BundleCapabilities
    from src.core.enums import ModelType
    from src.core.generation_mode import GenerationModes


def resolve_generation_modes(
    model: ModelType,
    *,
    capabilities: BundleCapabilities | None,
) -> GenerationModes:
    """Return the single authority for what each model mode accepts.

    Static registry declarations serve always-on models. An indexed on-demand
    bundle may narrow the offered modes and supplies the contract for every
    mode it declares, but it cannot widen the static model registration.
    """
    static = get_model_meta(model).generation_modes
    if capabilities is None:
        return static
    return {
        generation_type: mode
        for generation_type, mode in capabilities.generation_modes.items()
        if generation_type in static
    }
