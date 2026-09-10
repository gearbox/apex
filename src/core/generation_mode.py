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

    def __post_init__(self) -> None:
        """Reject structurally impossible contracts at construction."""
        if not 1 <= self.min <= self.max:
            raise ValueError(
                f"source media cardinality must satisfy 1 <= min <= max, got {self.min}..{self.max}"
            )
        if not self.media_types:
            raise ValueError("source media contract must accept at least one media kind")
        if self.roles:
            if len(self.roles) != self.min or self.min != self.max:
                raise ValueError(
                    "positional roles require a fixed cardinality: "
                    f"{len(self.roles)} roles for {self.min}..{self.max}"
                )
            for role in self.roles:
                if MEDIA_SLOT_KINDS[role] not in self.media_types:
                    raise ValueError(
                        f"role {role.value!r} needs media kind "
                        f"{MEDIA_SLOT_KINDS[role].value!r}, which is not in media_types"
                    )

    def kind_at(self, position: int) -> frozenset[MediaKind]:
        """Accepted kinds at one position — role-specific when roles exist."""
        if not self.roles:
            return self.media_types
        return frozenset({MEDIA_SLOT_KINDS[self.roles[position]]})

    def intersect(self, other: SourceMediaConstraints) -> SourceMediaConstraints | None:
        """Narrow to what both contracts accept, or None when nothing does."""
        low = max(self.min, other.min)
        high = min(self.max, other.max)
        kinds = self.media_types & other.media_types
        if low > high or not kinds:
            return None
        if self.roles and other.roles and self.roles != other.roles:
            return None
        roles = self.roles or other.roles
        if roles and (len(roles) != low or low != high):
            return None
        return SourceMediaConstraints(min=low, max=high, media_types=kinds, roles=roles)


@dataclass(frozen=True, slots=True)
class GenerationModeMeta:
    """What one ``GenerationType`` accepts on one model."""

    source_media: SourceMediaConstraints | None = None

    def intersect(self, other: GenerationModeMeta) -> GenerationModeMeta | None:
        """Both sides must agree on whether the mode takes media at all."""
        if (self.source_media is None) != (other.source_media is None):
            return None
        if self.source_media is None or other.source_media is None:
            return GenerationModeMeta()
        narrowed = self.source_media.intersect(other.source_media)
        return None if narrowed is None else GenerationModeMeta(narrowed)


GenerationModes = Mapping["GenerationType", GenerationModeMeta]
