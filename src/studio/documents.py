"""Versioned studio documents: the *shape* of a Plan and of Lyrics.

These are the payloads stored (normalized) in ``studio_document_versions.content`` and accepted
by the version-create endpoint. They are immutable ``msgspec`` Structs; collections are tuples.

msgspec enforces the *wire* constraints (lengths, ranges, unknown fields). Whitespace-only
strings pass those checks, so semantic validation and normalization live in
``src/studio/validation.py``.

Changing a field in a way old rows cannot decode requires bumping ``CURRENT_SCHEMA_VERSION`` and
an Alembic data migration — there are no read-time shims (D5).
"""

from __future__ import annotations

from collections.abc import Mapping  # noqa: TC003 — runtime: Final[Mapping[...]] is evaluated
from typing import Annotated, Final
from uuid import UUID

import msgspec
from msgspec import Meta, Struct

from src.core.enums import StudioDocumentKind, StudioSectionKind
from src.studio.errors import StudioDocumentSchemaError
from src.studio.tempo import TEMPO_MAX, TEMPO_MIN

__all__ = [
    "CURRENT_SCHEMA_VERSION",
    "DeliveryTag",
    "LanguageTag",
    "LyricLine",
    "LyricsDocument",
    "LyricsSection",
    "PlanDocument",
    "PlanSection",
    "StyleSpec",
    "SyllableRange",
    "Tag",
    "decode_stored",
]

Tag = Annotated[str, Meta(min_length=1, max_length=60)]
DeliveryTag = Annotated[str, Meta(min_length=1, max_length=40)]
LanguageTag = Annotated[str, Meta(pattern=r"^[a-z]{2,3}(-[A-Za-z0-9]{2,8})*$", max_length=35)]


class StyleSpec(Struct, frozen=True, kw_only=True, forbid_unknown_fields=True):
    """Style tags to include and to avoid (``tags`` / ``avoid`` must not overlap)."""

    tags: Annotated[tuple[Tag, ...], Meta(max_length=40)] = ()
    avoid: Annotated[tuple[Tag, ...], Meta(max_length=40)] = ()


class PlanSection(Struct, frozen=True, kw_only=True, forbid_unknown_fields=True):
    """One section of a song. ``id`` is stable across plan versions; lyrics refer to it."""

    id: UUID
    kind: StudioSectionKind
    label: Annotated[str, Meta(min_length=1, max_length=60)] | None = None
    duration_seconds: Annotated[int, Meta(ge=1, le=600)] | None = None  # planned, not exported
    style: StyleSpec = StyleSpec()
    delivery: Annotated[tuple[DeliveryTag, ...], Meta(max_length=12)] = ()
    instrumental: bool = False
    notes: Annotated[str, Meta(max_length=1000)] | None = None  # never exported


class PlanDocument(Struct, frozen=True, kw_only=True, forbid_unknown_fields=True):
    """The structural plan of a track (schema_version 2).

    ``instrumental`` is a property of the *track*: independent of the sections and of the lyrics,
    versioned with the plan. A plan may have no sections at all (a track that is only a title and
    a style). The field order is the canonical encoding order frozen by migration 056.
    """

    title: Annotated[str, Meta(min_length=1, max_length=200)]
    idea: Annotated[str, Meta(max_length=4000)] | None = None  # never exported
    tempo_bpm: Annotated[int, Meta(ge=TEMPO_MIN, le=TEMPO_MAX)] | None = None
    instrumental: bool = False
    global_style: StyleSpec = StyleSpec()
    references: Annotated[
        tuple[Annotated[str, Meta(min_length=1, max_length=200)], ...], Meta(max_length=20)
    ] = ()  # NEVER exported (real-artist names are not sent to providers)
    sections: Annotated[tuple[PlanSection, ...], Meta(min_length=0, max_length=64)] = ()


class SyllableRange(Struct, frozen=True, kw_only=True, forbid_unknown_fields=True):
    min: Annotated[int, Meta(ge=1, le=32)]
    max: Annotated[int, Meta(ge=1, le=32)]


class LyricLine(Struct, frozen=True, kw_only=True, forbid_unknown_fields=True):
    text: Annotated[str, Meta(min_length=1, max_length=300)]
    delivery: Annotated[tuple[DeliveryTag, ...], Meta(max_length=6)] = ()


class LyricsSection(Struct, frozen=True, kw_only=True, forbid_unknown_fields=True):
    """Lines for one plan section (``section_id`` refers to ``PlanSection.id``)."""

    section_id: UUID
    lines: Annotated[tuple[LyricLine, ...], Meta(min_length=1, max_length=64)]


class LyricsDocument(Struct, frozen=True, kw_only=True, forbid_unknown_fields=True):
    """The lyrics of a track (schema_version 1); always based on one plan version."""

    language: LanguageTag
    line_syllables: SyllableRange | None = None  # stored now, enforced by the P2 validator
    sections: Annotated[tuple[LyricsSection, ...], Meta(max_length=64)] = ()


CURRENT_SCHEMA_VERSION: Final[Mapping[StudioDocumentKind, int]] = {
    StudioDocumentKind.PLAN: 2,
    StudioDocumentKind.LYRICS: 1,
}


def decode_stored(
    kind: StudioDocumentKind, schema_version: int, content: Mapping[str, object]
) -> PlanDocument | LyricsDocument:
    """Decode a stored ``content`` JSONB back into its document type.

    Args:
        kind: Document kind of the version row.
        schema_version: ``schema_version`` of the version row.
        content: The stored JSONB value.

    Returns:
        The decoded document.

    Raises:
        StudioDocumentSchemaError: ``schema_version`` is not one this code understands. Never
            falls back to a best-effort decode.
        msgspec.ValidationError: The stored content does not match the schema (a data bug).
    """
    if schema_version != CURRENT_SCHEMA_VERSION[kind]:
        raise StudioDocumentSchemaError(
            f"unsupported {kind.value} schema_version {schema_version} "
            f"(this build understands {CURRENT_SCHEMA_VERSION[kind]})"
        )
    if kind is StudioDocumentKind.PLAN:
        return msgspec.convert(content, PlanDocument, strict=True)
    return msgspec.convert(content, LyricsDocument, strict=True)
