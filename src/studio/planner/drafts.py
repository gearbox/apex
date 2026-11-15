"""What the model is asked to return: draft Structs, and nothing else.

A draft is **not** a document. It is shaped for a model (short section keys, no ids), carries a
required ``refusal`` so the model can decline in-band, and is mapped and validated by the server
(``mapping.py``, ``validate.py``) before anything is stored. Draft Structs declare **no defaults**:
a defaulted field would be optional in the strict JSON schema (see ``to_strict_schema``).
"""

from __future__ import annotations

from enum import StrEnum

from msgspec import Struct

from src.core.enums import StudioSectionKind

__all__ = ["DraftRefusal", "PlanDraft", "PlanSectionDraft"]


class DraftRefusal(StrEnum):
    """Why the model declined to plan."""

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
    title: str
    tempo_bpm: int | None
    instrumental: bool
    tags: list[str]
    avoid: list[str]
    sections: list[PlanSectionDraft]
