"""Per-generation-mode input contracts.

One vocabulary shared by the static registry, bundle-derived capabilities,
provider discovery, and request validation. Nothing else may describe what a
mode accepts.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

from src.core.enums import MEDIA_SLOT_KINDS, MediaKind, MediaSlot

if TYPE_CHECKING:
    from src.core.enums import GenerationType


@dataclass(frozen=True, slots=True)
class SourceMediaConstraints:
    """Ordered owned-library input limits for one generation mode.

    ``roles`` is empty when positions are interchangeable (ordered but
    unnamed). When non-empty it is positional: ``roles[i]`` names the slot
    the asset at ``source_media[i]`` fills, and ``len(roles) == min == max``.
    """

    min: int
    max: int
    media_types: frozenset[MediaKind]
    roles: tuple[MediaSlot, ...] = ()

    def kind_at(self, position: int) -> frozenset[MediaKind]:
        """Accepted kinds at one position — role-specific when roles exist."""
        if not self.roles:
            return self.media_types
        return frozenset({MEDIA_SLOT_KINDS[self.roles[position]]})


@dataclass(frozen=True, slots=True)
class GenerationModeMeta:
    """What one ``GenerationType`` accepts on one model."""

    source_media: SourceMediaConstraints | None = None


GenerationModes = Mapping["GenerationType", GenerationModeMeta]
