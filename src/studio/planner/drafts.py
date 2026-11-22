"""What the model is asked to return: draft Structs, and nothing else.

A draft is **not** a document. It is shaped for a model (short section keys, no ids), carries a
required ``refusal`` so the model can decline in-band (and a required list of the keys of the artist
memories that made it decline: empty unless the run had memories and the reason is
``content_policy``), and is mapped and validated by the server
(``mapping.py``, ``validate.py``) before anything is stored. Draft Structs declare **no defaults**:
a defaulted field would be optional in the strict JSON schema (see ``to_strict_schema``).
"""

from __future__ import annotations

from enum import StrEnum

from msgspec import Struct

from src.core.enums import StudioSectionKind

__all__ = [
    "DraftRefusal",
    "LinePatchDraft",
    "LyricLineDraft",
    "LyricsDraft",
    "LyricsPatchDraft",
    "LyricsRevisionDraft",
    "LyricsSectionDraft",
    "PlanDraft",
    "PlanSectionDraft",
]


class DraftRefusal(StrEnum):
    """Why the model declined to write."""

    CONTENT_POLICY = "content_policy"
    BRIEF_UNUSABLE = "brief_unusable"


class PlanSectionDraft(Struct, frozen=True, kw_only=True, forbid_unknown_fields=True):
    """One section as the model writes it."""

    key: str
    """``s1``, ``s2``, ... in song order. The model never sees (or invents) UUIDs."""
    kind: StudioSectionKind
    label: str | None
    duration_seconds: int | None
    tags: list[str]
    delivery: list[str]
    instrumental: bool
    notes: str | None


class PlanDraft(Struct, frozen=True, kw_only=True, forbid_unknown_fields=True):
    """The model's plan."""

    refusal: DraftRefusal | None
    refusal_memories: list[str]
    """The ``m<n>`` keys of the memories that caused a ``content_policy`` refusal; otherwise empty."""
    title: str
    tempo_bpm: int | None
    instrumental: bool
    tags: list[str]
    avoid: list[str]
    sections: list[PlanSectionDraft]


class LyricLineDraft(Struct, frozen=True, kw_only=True, forbid_unknown_fields=True):
    """One sung line as the model writes it."""

    text: str
    delivery: list[str]
    """Short performance cues for this line; an empty list when there are none."""


class LyricsSectionDraft(Struct, frozen=True, kw_only=True, forbid_unknown_fields=True):
    """The lines of one plan section."""

    key: str
    """``s<n>``: the n-th section of the plan (1-based, instrumental ones counted)."""
    lines: list[LyricLineDraft]


class LyricsDraft(Struct, frozen=True, kw_only=True, forbid_unknown_fields=True):
    """The model's lyrics for a whole plan."""

    refusal: DraftRefusal | None
    refusal_memories: list[str]
    """The ``m<n>`` keys of the memories that caused a ``content_policy`` refusal; otherwise empty."""
    sections: list[LyricsSectionDraft]


class LinePatchDraft(Struct, frozen=True, kw_only=True, forbid_unknown_fields=True):
    """The replacement of one line."""

    key: str
    line: int
    """1-based, as the model was shown it. The API and every issue path are 0-based."""
    text: str
    delivery: list[str]


class LyricsPatchDraft(Struct, frozen=True, kw_only=True, forbid_unknown_fields=True):
    """The in-run fix of rejected lines: replacements for exactly the lines that were asked for."""

    refusal: DraftRefusal | None
    refusal_memories: list[str]
    """The ``m<n>`` keys of the memories that caused a ``content_policy`` refusal; otherwise empty."""
    lines: list[LinePatchDraft]


class LyricsRevisionDraft(Struct, frozen=True, kw_only=True, forbid_unknown_fields=True):
    """The first reply of a ``revise_lyrics`` run."""

    refusal: DraftRefusal | None
    refusal_memories: list[str]
    """The ``m<n>`` keys of the memories that caused a ``content_policy`` refusal; otherwise empty."""
    sections: list[LyricsSectionDraft]
    """New lines for each whole-section target."""
    lines: list[LinePatchDraft]
    """One replacement for each single-line target."""
