"""Renderer contracts and the *projections* renderers are allowed to see.

A renderer receives a :class:`RenderContext`, never a stored document or a database row. The
projection types below simply do not have fields for anything that must not be exported
(references, ideas, free-text notes, personas, aesthetic references), so leaking them is
impossible by construction (P1-12) — not merely avoided by convention.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping
    from uuid import UUID

    from src.core.enums import StudioExportTarget, StudioSectionKind
    from src.studio.documents import LyricsDocument, StyleSpec

__all__ = [
    "ArtistRenderInput",
    "ExportWarning",
    "FieldReport",
    "RenderContext",
    "RenderablePlan",
    "RenderableSection",
    "RenderedExport",
    "RenderedField",
    "TargetRenderer",
    "dedupe_ci",
]


@dataclass(frozen=True, slots=True)
class RenderableSection:
    """A plan section as a renderer may see it (no notes)."""

    id: UUID
    kind: StudioSectionKind
    label: str | None
    duration_seconds: int | None
    style: StyleSpec
    delivery: tuple[str, ...]
    instrumental: bool


@dataclass(frozen=True, slots=True)
class RenderablePlan:
    """A plan as a renderer may see it: title, tempo, instrumental flag, global style, sections."""

    title: str
    tempo_bpm: int | None
    instrumental: bool
    global_style: StyleSpec
    sections: tuple[RenderableSection, ...]


@dataclass(frozen=True, slots=True)
class ArtistRenderInput:
    """The exportable part of an artist: nothing but style tags and the global avoid list."""

    style_tags: tuple[str, ...]
    global_avoid: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class RenderContext:
    plan: RenderablePlan
    lyrics: LyricsDocument | None
    artist: ArtistRenderInput | None


@dataclass(frozen=True, slots=True)
class ExportWarning:
    """Something the target cannot express, reported instead of silently dropped."""

    code: str
    section_id: UUID | None = None


@dataclass(frozen=True, slots=True)
class FieldReport:
    """Size of one rendered field against the target's (unofficial) limit."""

    length: int
    limit: int | None
    within_limit: bool


@dataclass(frozen=True, slots=True)
class RenderedField:
    text: str
    report: FieldReport


@dataclass(frozen=True, slots=True)
class RenderedExport:
    """Output of a renderer. Text is always complete — never truncated, never an error.

    ``instrumental`` is the track-level flag of the plan. It is reported next to the fields
    because the target has no text for it (Suno takes an "Instrumental" toggle, not a tag).
    """

    target: StudioExportTarget
    renderer_version: str
    fields: Mapping[str, RenderedField]
    warnings: tuple[ExportWarning, ...]
    instrumental: bool


class TargetRenderer(Protocol):
    """A deterministic renderer: the same context yields byte-identical output."""

    target: StudioExportTarget
    version: str

    def render(self, ctx: RenderContext) -> RenderedExport: ...


def dedupe_ci(items: Iterable[str]) -> list[str]:
    """Drop case-insensitive duplicates (``casefold``); the first occurrence wins."""
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        key = item.casefold()
        if key not in seen:
            seen.add(key)
            result.append(item)
    return result
