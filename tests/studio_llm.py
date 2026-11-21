"""Builders for the studio planner's plan drafts (what a model would return).

The scripted fake model itself is generic and lives in ``tests/llm_support.py``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.core.enums import StudioSectionKind
from src.studio.planner.drafts import (
    DraftRefusal,
    LinePatchDraft,
    LyricLineDraft,
    LyricsDraft,
    LyricsPatchDraft,
    LyricsRevisionDraft,
    LyricsSectionDraft,
    PlanDraft,
    PlanSectionDraft,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = [
    "make_draft",
    "make_line_patch",
    "make_lyrics_draft",
    "make_lyrics_patch",
    "make_lyrics_revision",
    "make_section_draft",
    "refusal_draft",
    "refusal_lyrics_draft",
    "refusal_lyrics_patch",
    "refusal_lyrics_revision",
    "section_lines",
]

K = StudioSectionKind


def make_section_draft(
    key: str = "s1",
    kind: StudioSectionKind = K.VERSE,
    *,
    label: str | None = None,
    duration_seconds: int | None = None,
    tags: Sequence[str] = (),
    delivery: Sequence[str] = (),
    instrumental: bool = False,
    notes: str | None = None,
) -> PlanSectionDraft:
    return PlanSectionDraft(
        key=key,
        kind=kind,
        label=label,
        duration_seconds=duration_seconds,
        tags=list(tags),
        delivery=list(delivery),
        instrumental=instrumental,
        notes=notes,
    )


def make_draft(
    *sections: PlanSectionDraft,
    title: str = "Neon Rain",
    tempo_bpm: int | None = 120,
    instrumental: bool = False,
    tags: Sequence[str] = ("synthwave",),
    avoid: Sequence[str] = ("autotune",),
    refusal: DraftRefusal | None = None,
    refusal_memories: Sequence[str] = (),
) -> PlanDraft:
    """A valid draft (two sections unless given some)."""
    return PlanDraft(
        refusal=refusal,
        refusal_memories=list(refusal_memories),
        title=title,
        tempo_bpm=tempo_bpm,
        instrumental=instrumental,
        tags=list(tags),
        avoid=list(avoid),
        sections=list(
            sections
            or (make_section_draft("s1", K.VERSE), make_section_draft("s2", K.CHORUS, tags=["big"]))
        ),
    )


def refusal_draft(reason: DraftRefusal, memories: Sequence[str] = ()) -> PlanDraft:
    """What a model returns to decline: the reason, the memory keys it names, empty values."""
    return PlanDraft(
        refusal=reason,
        refusal_memories=list(memories),
        title="",
        tempo_bpm=None,
        instrumental=False,
        tags=[],
        avoid=[],
        sections=[],
    )


# -- lyrics drafts -----------------------------------------------------------------------------


def _lines(lines: Sequence[str | LyricLineDraft]) -> list[LyricLineDraft]:
    return [
        line if isinstance(line, LyricLineDraft) else LyricLineDraft(text=line, delivery=[])
        for line in lines
    ]


def section_lines(key: str, *lines: str | LyricLineDraft) -> LyricsSectionDraft:
    """One section's lines as the model writes them (plain strings become delivery-less lines)."""
    return LyricsSectionDraft(key=key, lines=_lines(lines))


def make_lyrics_draft(
    *sections: LyricsSectionDraft, refusal: DraftRefusal | None = None
) -> LyricsDraft:
    """A lyrics reply made of the given sections."""
    return LyricsDraft(refusal=refusal, refusal_memories=[], sections=list(sections))


def refusal_lyrics_draft(reason: DraftRefusal, memories: Sequence[str] = ()) -> LyricsDraft:
    """What a model returns to decline writing lyrics (naming the memory keys it blames)."""
    return LyricsDraft(refusal=reason, refusal_memories=list(memories), sections=[])


def make_line_patch(key: str, line: int, text: str, delivery: Sequence[str] = ()) -> LinePatchDraft:
    """The replacement of one line (``line`` is 1-based, as the model numbers it)."""
    return LinePatchDraft(key=key, line=line, text=text, delivery=list(delivery))


def make_lyrics_patch(*lines: LinePatchDraft) -> LyricsPatchDraft:
    """A reply to a patch request."""
    return LyricsPatchDraft(refusal=None, refusal_memories=[], lines=list(lines))


def refusal_lyrics_patch(reason: DraftRefusal, memories: Sequence[str] = ()) -> LyricsPatchDraft:
    return LyricsPatchDraft(refusal=reason, refusal_memories=list(memories), lines=[])


def make_lyrics_revision(
    *,
    sections: Sequence[LyricsSectionDraft] = (),
    lines: Sequence[LinePatchDraft] = (),
) -> LyricsRevisionDraft:
    """The first reply of a revision: new sections and/or replaced lines."""
    return LyricsRevisionDraft(
        refusal=None, refusal_memories=[], sections=list(sections), lines=list(lines)
    )


def refusal_lyrics_revision(
    reason: DraftRefusal, memories: Sequence[str] = ()
) -> LyricsRevisionDraft:
    return LyricsRevisionDraft(
        refusal=reason, refusal_memories=list(memories), sections=[], lines=[]
    )
