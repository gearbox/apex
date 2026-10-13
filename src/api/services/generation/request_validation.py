"""Model-independent request checks driven by registry metadata.

Pure functions over ``ModelMeta`` so each rule is unit-testable without the
orchestrator. ``GenerationService.generate`` calls them before any billing work.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.api.schemas.unified_generation import UnifiedGenerationRequest
    from src.core.model_registry import ModelMeta


def validate_seed(request: UnifiedGenerationRequest, meta: ModelMeta) -> None:
    """Reject a seed outside the model's declared range.

    ``None`` (omitted) always means "random" and is never checked; ``-1`` is not a second
    spelling of it. A model with no ``seed_range`` accepts any integer.

    Raises:
        ValueError: ``request.seed`` is set and outside ``meta.seed_range``.
    """
    bounds = meta.seed_range
    if bounds is None or request.seed is None:
        return
    if not bounds.min <= request.seed <= bounds.max:
        raise ValueError(
            f"Model '{request.model.value}' accepts seeds {bounds.min}-{bounds.max}; "
            "omit seed for a random one"
        )
