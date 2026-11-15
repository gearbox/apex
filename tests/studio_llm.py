"""Builders for the studio planner's plan drafts (what a model would return).

The scripted fake model itself is generic and lives in ``tests/llm_support.py``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.core.enums import StudioSectionKind
from src.studio.planner.drafts import DraftRefusal, PlanDraft, PlanSectionDraft

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = [
    "make_draft",
    "make_section_draft",
    "refusal_draft",
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
) -> PlanDraft:
    """A valid draft (two sections unless given some)."""
    return PlanDraft(
        refusal=refusal,
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


def refusal_draft(reason: DraftRefusal) -> PlanDraft:
    """What a model returns to decline: the reason and empty values."""
    return PlanDraft(
        refusal=reason,
        title="",
        tempo_bpm=None,
        instrumental=False,
        tags=[],
        avoid=[],
        sections=[],
    )
