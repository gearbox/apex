"""Resolution-tier → concrete (width, height) resolution for image generation."""

from __future__ import annotations

import math
from dataclasses import dataclass

import structlog

from src.core.enums import AspectRatio, Resolution

logger = structlog.get_logger(__name__)

# Global tier → target megapixels. Single source of truth for tier budgets.
TIER_MEGAPIXELS: dict[Resolution, float] = {
    Resolution.DRAFT: 0.25,
    Resolution.STANDARD: 1.0,
    Resolution.HIGH: 2.0,
    Resolution.ULTRA: 4.0,
}


@dataclass(frozen=True, slots=True)
class ResolvedDimensions:
    width: int
    height: int

    @property
    def megapixels(self) -> float:
        return (self.width * self.height) / 1_000_000


def _snap(value: float, multiple: int) -> int:
    """Round value to the nearest positive multiple of `multiple`."""
    snapped = round(value / multiple) * multiple
    return max(multiple, snapped)


def _floor_multiple(value: int, multiple: int) -> int:
    return max(multiple, (value // multiple) * multiple)


def _resolve_edge(value: float, multiple: int, max_edge: int) -> int:
    """Snap value to nearest multiple and hard-clamp to max_edge."""
    snapped = _snap(value, multiple)
    cap = (max_edge // multiple) * multiple  # largest multiple ≤ max_edge
    # max_edge smaller than one latent tile — honor hard cap literally
    return max_edge if cap < multiple else min(snapped, cap)


def _clamp_area(w: float, h: float, max_megapixels: float) -> tuple[float, float]:
    """Scale (w, h) down proportionally if w*h exceeds the MP cap."""
    max_area = max_megapixels * 1_000_000
    area = w * h
    if area <= max_area:
        return w, h
    scale = math.sqrt(max_area / area)
    return w * scale, h * scale


def resolve_dimensions(
    *,
    aspect_ratio: AspectRatio,
    max_megapixels: float,
    latent_multiple: int,
    max_edge: int,
    tier: Resolution | None = None,
    explicit_width: int | None = None,
    explicit_height: int | None = None,
) -> ResolvedDimensions:
    """Resolve concrete (width, height).

    Exactly one of (tier) or (explicit_width AND explicit_height) drives sizing;
    callers must enforce mutual exclusion before this point. If explicit dims are
    given they are snapped + clamped (forgiving). If a tier is given, W*H is
    computed to hit the tier's MP budget (clamped to max_megapixels) for the
    requested aspect ratio.

    All outputs are snapped to `latent_multiple` and clamped so neither edge
    exceeds `max_edge` and total area does not exceed `max_megapixels`.
    """
    if explicit_width is not None and explicit_height is not None:
        w: float = explicit_width
        h: float = explicit_height
    else:
        chosen = tier or Resolution.STANDARD
        target_mp = min(TIER_MEGAPIXELS[chosen], max_megapixels)
        rw, rh = aspect_ratio.as_fraction()
        area = target_mp * 1_000_000
        # w/h = rw/rh ; w*h = area  ->  h = sqrt(area * rh / rw)
        h = math.sqrt(area * rh / rw)
        w = h * rw / rh

    w, h = _clamp_area(w, h, max_megapixels)
    width = _resolve_edge(w, latent_multiple, max_edge)
    height = _resolve_edge(h, latent_multiple, max_edge)
    return ResolvedDimensions(width=width, height=height)
