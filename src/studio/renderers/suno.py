"""Deterministic renderer for Suno's three text boxes: Style, Exclude and Lyrics (plus a title).

Rendering is pure: the same context yields byte-identical output. It never raises on length —
every field reports ``length`` / ``limit`` / ``within_limit`` and the full text is returned, so
the UI can show and count what to cut (P1-7).

What Suno cannot express is reported as a structured warning instead of being dropped silently.
References, ideas, notes and personas are not in the render context at all (see ``base.py``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final, Protocol

from src.core.enums import StudioExportTarget, StudioSectionKind
from src.studio.renderers.base import (
    ExportWarning,
    FieldReport,
    RenderContext,
    RenderedExport,
    RenderedField,
    dedupe_ci,
)
from src.studio.section_names import rendered_labels
from src.studio.tempo import tempo_token

if TYPE_CHECKING:
    from uuid import UUID

    from src.studio.documents import StyleSpec
    from src.studio.renderers.base import RenderableSection

__all__ = [
    "EXCLUDE_LIMIT",
    "LYRICS_LIMIT",
    "LYRICS_RECOMMENDED_LENGTH",
    "STYLE_LIMIT",
    "VERSION",
    "SunoRenderer",
    "render_header",
]

VERSION: Final = "suno-v6.2026-11"

# Suno publishes no limits. These are unofficial measurements of the v6 web UI taken from the
# P1 design (Style 1000, Exclude 1000, Lyrics 5000). They drive `within_limit`, never an error.
# Verified on the live Suno v6 web UI on 2026-11-12.
STYLE_LIMIT: Final = 1000
EXCLUDE_LIMIT: Final = 1000
LYRICS_LIMIT: Final = 5000
LYRICS_RECOMMENDED_LENGTH: Final = 3000
"""Suno tends to rush through longer lyrics; above this a warning is raised (not an error)."""


class _HeaderSource(Protocol):
    """What a section header is made of: a ``RenderableSection`` or a ``PlanSection``."""

    @property
    def kind(self) -> StudioSectionKind: ...

    @property
    def style(self) -> StyleSpec: ...

    @property
    def delivery(self) -> tuple[str, ...]: ...

    @property
    def instrumental(self) -> bool: ...


def _field(text: str, limit: int | None) -> RenderedField:
    length = len(text)
    return RenderedField(
        text=text,
        report=FieldReport(
            length=length, limit=limit, within_limit=limit is None or length <= limit
        ),
    )


_INSTRUMENTAL_CUE: Final = "instrumental"


def _instrumental_cue(section: _HeaderSource) -> list[str]:
    """The ``instrumental`` descriptor for a flagged section whose header does not say so.

    A section of kind ``instrumental`` is already headed ``[Instrumental]``; any other kind with
    ``instrumental=True`` (a drop, a verse, ...) would otherwise render as a bare ``[Drop]`` and
    Suno would be free to sing over it. The cue goes **first** so a long tag list cannot push it
    out of sight (and counts toward the lyrics length like any other text).
    """
    if section.instrumental and section.kind is not StudioSectionKind.INSTRUMENTAL:
        return [_INSTRUMENTAL_CUE]
    return []


def _meta_tag(label: str, descriptors: list[str]) -> str:
    return f"[{label}: {', '.join(descriptors)}]" if descriptors else f"[{label}]"


def render_header(section: _HeaderSource, label: str) -> str:
    """The whole header line of a section, e.g. ``[Drop: instrumental, riser]``.

    The one place a header is written: the renderer uses it to write lyrics, the importer to
    recognise a header it would write itself, so the two can never drift apart.

    Args:
        section: A plan section (or its renderable projection).
        label: The header label, i.e. the explicit label or the default one.
    """
    descriptors = dedupe_ci([*_instrumental_cue(section), *section.delivery, *section.style.tags])
    return _meta_tag(label, descriptors)


class SunoRenderer:
    """Renders a track for Suno's Style / Exclude / Lyrics boxes."""

    target: StudioExportTarget = StudioExportTarget.SUNO
    version: str = VERSION

    def render(self, ctx: RenderContext) -> RenderedExport:
        plan = ctx.plan
        artist = ctx.artist

        # Plan tags first (earlier tags weigh more in Suno), then the artist's.
        style_tags = dedupe_ci(
            [*plan.global_style.tags, *(artist.style_tags if artist is not None else ())]
        )
        style_items = [*style_tags]
        if plan.tempo_bpm:
            style_items.append(tempo_token(plan.tempo_bpm))
        exclude_items = dedupe_ci(
            [*plan.global_style.avoid, *(artist.global_avoid if artist is not None else ())]
        )

        lines_by_section = (
            {section.section_id: section.lines for section in ctx.lyrics.sections}
            if ctx.lyrics is not None
            else {}
        )
        labels = rendered_labels((section.kind, section.label) for section in plan.sections)
        blocks: list[str] = []
        for section, label in zip(plan.sections, labels, strict=True):
            block = [render_header(section, label)]
            for line in lines_by_section.get(section.id, ()):
                if line.delivery:
                    block.append(f"[{', '.join(line.delivery)}]")
                block.append(line.text)
            blocks.append("\n".join(block))
        lyrics_text = "\n\n".join(blocks)

        style_text = ", ".join(style_items)
        exclude_text = ", ".join(exclude_items)

        return RenderedExport(
            target=StudioExportTarget.SUNO,
            renderer_version=VERSION,
            fields={
                "title": _field(plan.title, None),
                "style": _field(style_text, STYLE_LIMIT),
                "exclude": _field(exclude_text, EXCLUDE_LIMIT),
                "lyrics": _field(lyrics_text, LYRICS_LIMIT),
            },
            warnings=self._warnings(
                plan.sections,
                style_tags=style_tags,
                exclude_items=exclude_items,
                lyrics_length=len(lyrics_text),
            ),
            instrumental=plan.instrumental,
        )

    @staticmethod
    def _warnings(
        sections: tuple[RenderableSection, ...],
        *,
        style_tags: list[str],
        exclude_items: list[str],
        lyrics_length: int,
    ) -> tuple[ExportWarning, ...]:
        warnings: list[ExportWarning] = []
        excluded = {item.casefold() for item in exclude_items}
        if any(tag.casefold() in excluded for tag in style_tags):
            warnings.append(ExportWarning("style_exclude_overlap"))
        if lyrics_length > LYRICS_RECOMMENDED_LENGTH:
            warnings.append(ExportWarning("lyrics_above_recommended_length"))
        for section in sections:
            warnings.extend(_section_warnings(section.id, section))
        return tuple(warnings)


def _section_warnings(section_id: UUID, section: RenderableSection) -> list[ExportWarning]:
    found: list[ExportWarning] = []
    if section.style.avoid:
        found.append(ExportWarning("local_avoid_not_exportable", section_id))
    if section.duration_seconds is not None:
        found.append(ExportWarning("section_duration_not_exportable", section_id))
    return found
