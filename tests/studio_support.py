"""Shared builders for studio tests (documents, request bodies)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from uuid import UUID

import msgspec

from src.core.enums import StudioSectionKind
from src.core.uid import new_id
from src.studio.documents import (
    LyricLine,
    LyricsDocument,
    LyricsSection,
    PlanDocument,
    PlanSection,
    StyleSpec,
)

if TYPE_CHECKING:
    from collections.abc import Sequence


DEFAULT_SECTION_ID = UUID("00000000-0000-7000-8000-000000000001")
"""Id of the section ``make_plan`` adds when given none — so equal calls build equal plans."""


def make_section(
    kind: StudioSectionKind = StudioSectionKind.VERSE,
    *,
    section_id: UUID | None = None,
    label: str | None = None,
    duration_seconds: int | None = None,
    tags: Sequence[str] = (),
    avoid: Sequence[str] = (),
    delivery: Sequence[str] = (),
    instrumental: bool = False,
    notes: str | None = None,
) -> PlanSection:
    return PlanSection(
        id=section_id or new_id(),
        kind=kind,
        label=label,
        duration_seconds=duration_seconds,
        style=StyleSpec(tags=tuple(tags), avoid=tuple(avoid)),
        delivery=tuple(delivery),
        instrumental=instrumental,
        notes=notes,
    )


def make_plan(
    *sections: PlanSection,
    title: str = "Song",
    tempo_bpm: int | None = None,
    tags: Sequence[str] = (),
    avoid: Sequence[str] = (),
    idea: str | None = None,
    references: Sequence[str] = (),
) -> PlanDocument:
    return PlanDocument(
        title=title,
        idea=idea,
        tempo_bpm=tempo_bpm,
        global_style=StyleSpec(tags=tuple(tags), avoid=tuple(avoid)),
        references=tuple(references),
        sections=sections or (make_section(section_id=DEFAULT_SECTION_ID),),
    )


def make_lyrics(
    *entries: tuple[UUID, Sequence[str | LyricLine]], language: str = "en"
) -> LyricsDocument:
    sections = tuple(
        LyricsSection(
            section_id=section_id,
            lines=tuple(
                line if isinstance(line, LyricLine) else LyricLine(text=line) for line in lines
            ),
        )
        for section_id, lines in entries
    )
    return LyricsDocument(language=language, sections=sections)


def to_json(document: Any) -> dict[str, Any]:
    """JSON-mode body for an HTTP request (UUIDs / enums as strings, tuples as lists)."""
    return msgspec.json.decode(msgspec.json.encode(document))  # type: ignore[no-any-return]
