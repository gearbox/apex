"""Fixtures for the lyrics-run unit tests: a plan with stable section ids, a context, documents."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING
from uuid import UUID

from src.core.enums import StudioSectionKind as K
from src.studio.documents import LyricLine, LyricsDocument, LyricsSection, PlanDocument
from src.studio.planner.context import ArtistSnapshot, RunContextSnapshot
from src.studio.validation import normalize_lyrics, normalize_plan
from tests.studio_support import make_plan, make_section

if TYPE_CHECKING:
    from collections.abc import Sequence

    from src.studio.documents import SyllableRange

__all__ = [
    "ARTIST",
    "CONTEXT",
    "PLAN",
    "S1",
    "S2",
    "S3",
    "S4",
    "doc",
    "line",
    "make_context",
]

S1 = UUID(int=0x1001)
S2 = UUID(int=0x1002)
S3 = UUID(int=0x1003)
S4 = UUID(int=0x1004)

PLAN: PlanDocument = normalize_plan(
    make_plan(
        make_section(K.VERSE, section_id=S1, label="Opening", delivery=["soft"]),
        make_section(K.CHORUS, section_id=S2, tags=["anthemic"], notes="big and wide"),
        make_section(K.INSTRUMENTAL, section_id=S3),
        make_section(K.VERSE, section_id=S4),
        title="Neon Rain",
        idea="a song about driving home",
        tempo_bpm=100,
        tags=["synthwave"],
    )
)
"""Four sections: s1 verse, s2 chorus, s3 instrumental, s4 verse."""


def make_context(
    *, rating: str = "sfw", artist: ArtistSnapshot | None = None
) -> RunContextSnapshot:
    return RunContextSnapshot(
        track_title="Track",
        artist=artist,
        concept=None,
        content_rating=rating,
        plan_version_id=None,
        plan_content_hash=None,
        current_plan=None,
        lyrics_version_id=None,
        lyrics_content_hash=None,
    )


CONTEXT = make_context()

ARTIST = ArtistSnapshot(
    id=UUID(int=1),
    name="Nova",
    persona=None,
    voice_notes=None,
    default_language="en",
    style_tags=(),
    global_avoid=(),
    aesthetic_references=(),
    updated_at=datetime(2026, 1, 1, tzinfo=UTC),
)


def line(text: str, *delivery: str) -> LyricLine:
    return LyricLine(text=text, delivery=tuple(delivery))


def doc(
    *sections: tuple[UUID, Sequence[str | LyricLine]],
    language: str = "en",
    line_syllables: SyllableRange | None = None,
    plan: PlanDocument = PLAN,
) -> LyricsDocument:
    """A normalized lyrics document over ``plan`` (plain strings become delivery-less lines)."""
    built = LyricsDocument(
        language=language,
        line_syllables=line_syllables,
        sections=tuple(
            LyricsSection(
                section_id=section_id,
                lines=tuple(x if isinstance(x, LyricLine) else LyricLine(text=x) for x in lines),
            )
            for section_id, lines in sections
        ),
    )
    return normalize_lyrics(built, plan)
