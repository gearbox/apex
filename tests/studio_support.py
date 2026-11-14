"""Shared builders for studio tests (documents, request bodies)."""

from __future__ import annotations

from dataclasses import dataclass
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
from src.studio.renderers.base import ArtistRenderInput
from src.studio.validation import normalize_lyrics, normalize_plan

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
    instrumental: bool = False,
    empty: bool = False,
) -> PlanDocument:
    """A plan; with no sections given it has one default section, unless ``empty`` is set."""
    default = () if empty else (make_section(section_id=DEFAULT_SECTION_ID),)
    return PlanDocument(
        title=title,
        idea=idea,
        tempo_bpm=tempo_bpm,
        instrumental=instrumental,
        global_style=StyleSpec(tags=tuple(tags), avoid=tuple(avoid)),
        references=tuple(references),
        sections=sections or default,
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


@dataclass(frozen=True, slots=True)
class GoldenSong:
    """The song of the P1 renderer golden test: ten sections, repeats, delivery, an artist."""

    plan: PlanDocument
    lyrics: LyricsDocument
    artist: ArtistRenderInput
    drop: PlanSection
    """The first drop: it carries a local avoid list and a planned duration (not exportable)."""


GOLDEN_STYLE = "synthwave, female vocals, Dreamy, Warm Reverb, 118 BPM"
GOLDEN_EXCLUDE = "heavy metal, autotune, screaming"
GOLDEN_LYRICS = (
    "[Intro: sparse pads]\n"
    "\n"
    "[Verse 1: breathy, minimal]\n"
    "First line\n"
    "[whispered]\n"
    "Second line\n"
    "\n"
    "[Pre-Chorus]\n"
    "Pre line\n"
    "\n"
    "[Chorus]\n"
    "Chorus one\n"
    "Chorus two\n"
    "\n"
    "[Verse 2]\n"
    "Verse b line\n"
    "\n"
    "[Chorus]\n"
    "Chorus again\n"
    "\n"
    "[Drop 1: heavy bass]\n"
    "\n"
    "[Drop 2: instrumental]\n"
    "\n"
    "[Break It Down]\n"
    "\n"
    "[Outro]"
)
"""What ``SunoRenderer`` writes for :func:`golden_song` (``tests/unit/studio/test_suno_renderer``)."""


def golden_song() -> GoldenSong:
    """Build the golden song (normalized, so it is exactly what a stored version holds)."""
    kinds = StudioSectionKind
    intro = make_section(kinds.INTRO, tags=("sparse pads",), notes="the hush")
    verse_a = make_section(
        kinds.VERSE, delivery=("breathy",), tags=("minimal",), duration_seconds=30
    )
    pre = make_section(kinds.PRE_CHORUS)
    chorus_1 = make_section(kinds.CHORUS)
    verse_b = make_section(kinds.VERSE)
    chorus_2 = make_section(kinds.CHORUS)
    drop = make_section(
        kinds.DROP, tags=("heavy bass",), avoid=("distortion",), duration_seconds=20
    )
    instrumental_drop = make_section(kinds.DROP, instrumental=True)
    bridge = make_section(kinds.BRIDGE, label="Break It Down")
    outro = make_section(kinds.OUTRO)
    plan = normalize_plan(
        make_plan(
            intro,
            verse_a,
            pre,
            chorus_1,
            verse_b,
            chorus_2,
            drop,
            instrumental_drop,
            bridge,
            outro,
            title="Neon Rain",
            tempo_bpm=118,
            tags=("synthwave", "female vocals", "Dreamy"),
            avoid=("heavy metal", "autotune"),
            idea="An idea the export never shows",
            references=("A reference the export never shows",),
        )
    )
    lyrics = normalize_lyrics(
        make_lyrics(
            (
                verse_a.id,
                ["First line", LyricLine(text="Second line", delivery=("whispered",))],
            ),
            (pre.id, ["Pre line"]),
            (chorus_1.id, ["Chorus one", "Chorus two"]),
            (verse_b.id, ["Verse b line"]),
            (chorus_2.id, ["Chorus again"]),
        ),
        plan,
    )
    artist = ArtistRenderInput(
        style_tags=("dreamy", "Warm Reverb"), global_avoid=("Autotune", "screaming")
    )
    return GoldenSong(plan=plan, lyrics=lyrics, artist=artist, drop=drop)
