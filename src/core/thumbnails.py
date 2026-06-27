"""Thumbnail size specifications — single source of truth.

All thumbnail generation code reads from THUMBNAIL_SPECS. Never hardcode
150 or 512 anywhere else.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ThumbnailSpec:
    label: str
    max_edge: int


# Ascending by max_edge. Add new sizes here only.
THUMBNAIL_SPECS: tuple[ThumbnailSpec, ...] = (
    ThumbnailSpec("sm", 150),
    ThumbnailSpec("md", 512),
)

LABEL_BY_MAX_EDGE: dict[int, str] = {s.max_edge: s.label for s in THUMBNAIL_SPECS}


def label_for_max_edge(max_edge: int | None) -> str | None:
    """Return the label for a thumbnail_max_edge value, or None if unknown/None."""
    return None if max_edge is None else LABEL_BY_MAX_EDGE.get(max_edge)
