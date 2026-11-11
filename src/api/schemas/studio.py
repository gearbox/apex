"""Request / response DTOs for ``/v1/studio`` (msgspec; see ``docs/contracts/studio-contract.md``).

Request structs forbid unknown fields; PATCH structs are tri-state (``msgspec.UNSET`` = leave
alone, ``None`` = clear where the field is nullable). Document payloads reuse the Structs from
``src/studio/documents.py`` directly.
"""

from __future__ import annotations

from datetime import datetime
from typing import Annotated
from uuid import UUID

import msgspec
from msgspec import Meta

from src.core.enums import (
    StudioDocumentKind,
    StudioDocumentOrigin,
    StudioExportTarget,
    StudioIncompatibleLyricsPolicy,
    StudioReleaseKind,
)
from src.studio.documents import (
    LanguageTag,
    LyricsDocument,
    PlanDocument,
    Tag,
)

__all__ = [
    "ArtistCreate",
    "ArtistOut",
    "ArtistPatch",
    "CurrentPut",
    "ExportFieldOut",
    "ExportWarningOut",
    "LyricsVersionCreate",
    "LyricsVersionDetail",
    "PlanVersionCreate",
    "PlanVersionDetail",
    "ProjectProfileOut",
    "ProjectProfilePut",
    "SunoExportOut",
    "TrackCreate",
    "TrackDetail",
    "TrackOrderPut",
    "TrackOut",
    "TrackPatch",
    "VersionCreate",
    "VersionCreateOut",
    "VersionDetail",
    "VersionSummary",
]

_ArtistName = Annotated[str, Meta(min_length=1, max_length=100)]
_Persona = Annotated[str, Meta(max_length=4000)]
_VoiceNotes = Annotated[str, Meta(max_length=2000)]
_AestheticRefs = Annotated[
    list[Annotated[str, Meta(min_length=1, max_length=200)]], Meta(max_length=20)
]
_Tags = Annotated[list[Tag], Meta(max_length=40)]
_TrackTitle = Annotated[str, Meta(min_length=1, max_length=200)]
_Concept = Annotated[str, Meta(max_length=8000)]

# ---------------------------------------------------------------------------
# Artists
# ---------------------------------------------------------------------------


class ArtistCreate(msgspec.Struct, forbid_unknown_fields=True, kw_only=True):
    name: _ArtistName
    persona: _Persona | None = None
    voice_notes: _VoiceNotes | None = None
    default_language: LanguageTag | None = None
    aesthetic_references: _AestheticRefs = msgspec.field(default_factory=list)
    """Planner-only context. Never exported."""
    style_tags: _Tags = msgspec.field(default_factory=list)
    """Exportable (e.g. vocal type, delivery); merged into Suno Style after the plan's tags."""
    global_avoid: _Tags = msgspec.field(default_factory=list)
    """Exportable; merged into Suno Exclude after the plan's own avoid list."""


class ArtistPatch(msgspec.Struct, forbid_unknown_fields=True, kw_only=True):
    name: _ArtistName | msgspec.UnsetType = msgspec.UNSET
    persona: _Persona | msgspec.UnsetType | None = msgspec.UNSET
    voice_notes: _VoiceNotes | msgspec.UnsetType | None = msgspec.UNSET
    default_language: LanguageTag | msgspec.UnsetType | None = msgspec.UNSET
    aesthetic_references: _AestheticRefs | msgspec.UnsetType = msgspec.UNSET
    style_tags: _Tags | msgspec.UnsetType = msgspec.UNSET
    global_avoid: _Tags | msgspec.UnsetType = msgspec.UNSET


class ArtistOut(msgspec.Struct, kw_only=True):
    id: UUID
    name: str
    persona: str | None
    voice_notes: str | None
    default_language: str | None
    aesthetic_references: list[str]
    style_tags: list[str]
    global_avoid: list[str]
    created_at: datetime
    updated_at: datetime


# ---------------------------------------------------------------------------
# Projects (studio profile of a library project) and tracks
# ---------------------------------------------------------------------------


class ProjectProfilePut(msgspec.Struct, forbid_unknown_fields=True, kw_only=True):
    """Full replace of the studio profile (PUT). Omitted nullable fields become ``null``."""

    release_kind: StudioReleaseKind
    artist_id: UUID | None = None
    concept: _Concept | None = None


class ProjectProfileOut(msgspec.Struct, kw_only=True):
    library_project_id: UUID
    name: str
    """The library project's name."""
    release_kind: StudioReleaseKind
    artist_id: UUID | None
    concept: str | None
    track_count: int
    created_at: datetime
    updated_at: datetime


class TrackCreate(msgspec.Struct, forbid_unknown_fields=True, kw_only=True):
    title: _TrackTitle


class TrackPatch(msgspec.Struct, forbid_unknown_fields=True, kw_only=True):
    title: _TrackTitle | msgspec.UnsetType = msgspec.UNSET


class TrackOrderPut(msgspec.Struct, forbid_unknown_fields=True, kw_only=True):
    track_ids: Annotated[list[UUID], Meta(max_length=64)]
    """Every track of the project exactly once, in the new order."""


class TrackOut(msgspec.Struct, kw_only=True):
    id: UUID
    library_project_id: UUID
    position: int
    title: str
    current_plan_version_id: UUID | None
    current_lyrics_version_id: UUID | None
    created_at: datetime
    updated_at: datetime


# ---------------------------------------------------------------------------
# Versions
# ---------------------------------------------------------------------------


class PlanVersionCreate(
    msgspec.Struct, tag="plan", tag_field="kind", forbid_unknown_fields=True, kw_only=True
):
    content: PlanDocument
    parent_version_id: UUID | None = None
    make_current: bool = True
    incompatible_lyrics: StudioIncompatibleLyricsPolicy = StudioIncompatibleLyricsPolicy.REJECT
    """``reject`` (default): 409 if the current lyrics no longer fit. ``detach``: clear the
    current-lyrics pointer instead (the lyrics version itself is kept)."""


class LyricsVersionCreate(
    msgspec.Struct, tag="lyrics", tag_field="kind", forbid_unknown_fields=True, kw_only=True
):
    content: LyricsDocument
    based_on_plan_version_id: UUID
    parent_version_id: UUID | None = None
    make_current: bool = True


VersionCreate = PlanVersionCreate | LyricsVersionCreate
"""Tagged union decoded from the ``kind`` discriminant (same declaration style as BulkOperation)."""


class VersionSummary(msgspec.Struct, kw_only=True):
    id: UUID
    kind: StudioDocumentKind
    version_number: int
    parent_version_id: UUID | None
    based_on_plan_version_id: UUID | None
    origin: StudioDocumentOrigin
    schema_version: int
    content_hash: str
    created_at: datetime
    is_current: bool


class PlanVersionDetail(msgspec.Struct, tag="plan", tag_field="kind", kw_only=True):
    id: UUID
    track_id: UUID
    version_number: int
    parent_version_id: UUID | None
    origin: StudioDocumentOrigin
    schema_version: int
    content_hash: str
    created_at: datetime
    is_current: bool
    content: PlanDocument


class LyricsVersionDetail(msgspec.Struct, tag="lyrics", tag_field="kind", kw_only=True):
    id: UUID
    track_id: UUID
    version_number: int
    parent_version_id: UUID | None
    based_on_plan_version_id: UUID
    origin: StudioDocumentOrigin
    schema_version: int
    content_hash: str
    created_at: datetime
    is_current: bool
    content: LyricsDocument


VersionDetail = PlanVersionDetail | LyricsVersionDetail


class VersionCreateOut(msgspec.Struct, kw_only=True):
    """Response of version create: ``201`` when a row was written, ``200`` on an idempotent replay."""

    version: PlanVersionDetail | LyricsVersionDetail
    lyrics_detached: bool = False
    """``true`` only when a plan create used ``incompatible_lyrics=detach`` and the current
    lyrics no longer fit, so the current-lyrics pointer was cleared."""


class CurrentPut(msgspec.Struct, forbid_unknown_fields=True, kw_only=True):
    """Set the track's current pair. Omitted fields are left alone; ``lyrics_version_id: null``
    clears the lyrics pointer."""

    plan_version_id: UUID | msgspec.UnsetType = msgspec.UNSET
    lyrics_version_id: UUID | msgspec.UnsetType | None = msgspec.UNSET


class TrackDetail(TrackOut, kw_only=True):
    """A track with its current documents — one call opens a track."""

    current_plan: PlanVersionDetail | None
    current_lyrics: LyricsVersionDetail | None


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------


class ExportFieldOut(msgspec.Struct, kw_only=True):
    length: int
    limit: int | None
    within_limit: bool


class ExportWarningOut(msgspec.Struct, kw_only=True):
    code: str
    section_id: UUID | None = None


class SunoExportOut(msgspec.Struct, kw_only=True):
    target: StudioExportTarget
    renderer_version: str
    plan_version_id: UUID
    lyrics_version_id: UUID | None
    title: str
    style: str
    exclude: str
    lyrics: str
    fields: dict[str, ExportFieldOut]
    """Per-field size report for ``title`` / ``style`` / ``exclude`` / ``lyrics``. The limits are
    unofficial measurements: ``within_limit=false`` is information, never an error."""
    warnings: list[ExportWarningOut]
