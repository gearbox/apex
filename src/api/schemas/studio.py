"""Request / response DTOs for ``/v1/studio`` (msgspec; see ``docs/contracts/studio-contract.md``).

Request structs forbid unknown fields; PATCH structs are tri-state (``msgspec.UNSET`` = leave
alone, ``None`` = clear where the field is nullable). Document payloads reuse the Structs from
``src/studio/documents.py`` directly.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Annotated
from uuid import UUID

import msgspec
from msgspec import Meta

from src.api.schemas.media import MediaObject
from src.core.enums import (
    StudioDocumentKind,
    StudioDocumentOrigin,
    StudioExportTarget,
    StudioIncompatibleLyricsPolicy,
    StudioMemoryOrigin,
    StudioMemoryStatus,
    StudioReleaseKind,
)
from src.studio.constraints import (
    CAST_APPEARANCE_MAX,
    CAST_DESCRIPTION_MAX,
    CAST_LABEL_MAX,
    CAST_LOOK_NAME_MAX,
    CAST_NAME_MAX,
    MAX_CAST_REFERENCES,
    MEMORY_BODY_MAX,
    MEMORY_IMAGE_MAX,
    MEMORY_IMAGERY_MAX,
    MEMORY_THEME_MAX,
    MEMORY_THEMES_MAX,
    MEMORY_TITLE_MAX,
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
    "CastLookIn",
    "CastLookOut",
    "CastLookPatch",
    "CastMemberCreate",
    "CastMemberOut",
    "CastMemberPatch",
    "CastMemberSummary",
    "CastReferenceIn",
    "CastReferenceOrder",
    "CastReferenceOut",
    "CastReferencePatch",
    "CurrentPut",
    "ExportFieldOut",
    "ExportWarningOut",
    "ImportWarningOut",
    "LyricsVersionCreate",
    "LyricsVersionDetail",
    "MemoryCreate",
    "MemoryOut",
    "MemoryPatch",
    "PlanVersionCreate",
    "PlanVersionDetail",
    "ProjectProfileOut",
    "ProjectProfilePut",
    "RoundTripOut",
    "SunoExportOut",
    "SunoImportOut",
    "SunoImportRequest",
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
_MemoryTitle = Annotated[str, Meta(min_length=1, max_length=MEMORY_TITLE_MAX)]
_MemoryBody = Annotated[str, Meta(min_length=1, max_length=MEMORY_BODY_MAX)]
_MemoryThemes = Annotated[
    list[Annotated[str, Meta(max_length=MEMORY_THEME_MAX)]], Meta(max_length=MEMORY_THEMES_MAX)
]
_MemoryImagery = Annotated[
    list[Annotated[str, Meta(max_length=MEMORY_IMAGE_MAX)]], Meta(max_length=MEMORY_IMAGERY_MAX)
]
_Salience = Annotated[int, Meta(ge=1, le=5)]
_CastName = Annotated[str, Meta(min_length=1, max_length=CAST_NAME_MAX)]
_CastDescription = Annotated[str, Meta(max_length=CAST_DESCRIPTION_MAX)]
_CastAppearance = Annotated[str, Meta(max_length=CAST_APPEARANCE_MAX)]
_CastLabel = Annotated[str, Meta(min_length=1, max_length=CAST_LABEL_MAX)]
_CastLookName = Annotated[str, Meta(min_length=1, max_length=CAST_LOOK_NAME_MAX)]
_AssetRefText = Annotated[str, Meta(min_length=1, max_length=100)]
"""A wire asset ref (``"upload:<uuid>"`` / ``"output:<uuid>"``): parsed by ``parse_asset_ref``."""

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
    performer_cast_member_id: UUID | None = None
    """Optional: the cast member who performs this artist (404 ``cast_member_not_found`` if the
    caller has no such member)."""


class ArtistPatch(msgspec.Struct, forbid_unknown_fields=True, kw_only=True):
    name: _ArtistName | msgspec.UnsetType = msgspec.UNSET
    persona: _Persona | msgspec.UnsetType | None = msgspec.UNSET
    voice_notes: _VoiceNotes | msgspec.UnsetType | None = msgspec.UNSET
    default_language: LanguageTag | msgspec.UnsetType | None = msgspec.UNSET
    aesthetic_references: _AestheticRefs | msgspec.UnsetType = msgspec.UNSET
    style_tags: _Tags | msgspec.UnsetType = msgspec.UNSET
    global_avoid: _Tags | msgspec.UnsetType = msgspec.UNSET
    performer_cast_member_id: UUID | msgspec.UnsetType | None = msgspec.UNSET
    """Absent = leave unchanged. ``null`` = unlink. UUID = link (must be the caller's member)."""


class ArtistOut(msgspec.Struct, kw_only=True):
    id: UUID
    name: str
    persona: str | None
    voice_notes: str | None
    default_language: str | None
    aesthetic_references: list[str]
    style_tags: list[str]
    global_avoid: list[str]
    performer_cast_member_id: UUID | None
    created_at: datetime
    updated_at: datetime


# ---------------------------------------------------------------------------
# Cast (S1-P3)
# ---------------------------------------------------------------------------


class CastReferenceIn(msgspec.Struct, forbid_unknown_fields=True, kw_only=True):
    """One reference image of a cast member: a user label and a library asset to pin."""

    label: _CastLabel
    asset: _AssetRefText
    """Wire asset ref, ``"upload:<uuid>"`` / ``"output:<uuid>"``; an image of the caller's."""


class CastMemberCreate(msgspec.Struct, forbid_unknown_fields=True, kw_only=True):
    name: _CastName
    description: _CastDescription | None = None
    appearance: _CastAppearance | None = None
    """A short descriptor, used verbatim as a prompt fragment from S4 on."""
    references: Annotated[list[CastReferenceIn], Meta(max_length=MAX_CAST_REFERENCES)] = (
        msgspec.field(default_factory=list)
    )
    """Created atomically with the member, in this order (``position`` 0..n-1)."""


class CastMemberPatch(msgspec.Struct, forbid_unknown_fields=True, kw_only=True):
    name: _CastName | msgspec.UnsetType = msgspec.UNSET
    description: _CastDescription | msgspec.UnsetType | None = msgspec.UNSET
    appearance: _CastAppearance | msgspec.UnsetType | None = msgspec.UNSET


class CastReferencePatch(msgspec.Struct, forbid_unknown_fields=True, kw_only=True):
    label: _CastLabel | msgspec.UnsetType = msgspec.UNSET
    asset: _AssetRefText | msgspec.UnsetType = msgspec.UNSET
    """Changing it moves the pin: the new asset is held, the old one released."""


class CastReferenceOrder(msgspec.Struct, forbid_unknown_fields=True, kw_only=True):
    reference_ids: Annotated[list[UUID], Meta(max_length=MAX_CAST_REFERENCES)]
    """Exactly the member's reference ids, each once, in the new order (else ``invalid_order``)."""


class CastLookIn(msgspec.Struct, forbid_unknown_fields=True, kw_only=True):
    name: _CastLookName
    reference_ids: Annotated[list[UUID], Meta(max_length=MAX_CAST_REFERENCES)]
    """At least one, all of this member's, each once."""


class CastLookPatch(msgspec.Struct, forbid_unknown_fields=True, kw_only=True):
    name: _CastLookName | msgspec.UnsetType = msgspec.UNSET
    reference_ids: (
        Annotated[list[UUID], Meta(max_length=MAX_CAST_REFERENCES)] | msgspec.UnsetType
    ) = msgspec.UNSET
    """Replaces the whole set (at least one)."""


class CastReferenceOut(msgspec.Struct, kw_only=True):
    id: UUID
    label: str
    position: int
    asset: str
    """Wire asset ref of the pinned library asset."""
    media: MediaObject


class CastLookOut(msgspec.Struct, kw_only=True):
    id: UUID
    name: str
    reference_ids: list[UUID]
    """In reference position order. Empty when every reference of the look was deleted."""


class CastMemberSummary(msgspec.Struct, kw_only=True):
    id: UUID
    name: str
    appearance: str | None
    reference_count: int
    look_count: int
    cover: MediaObject | None
    """The reference at position 0, if any."""
    created_at: datetime
    updated_at: datetime


class CastMemberOut(msgspec.Struct, kw_only=True):
    id: UUID
    name: str
    description: str | None
    appearance: str | None
    references: list[CastReferenceOut]
    """By position."""
    looks: list[CastLookOut]
    """By name."""
    created_at: datetime
    updated_at: datetime


# ---------------------------------------------------------------------------
# Artist memories (P2b)
# ---------------------------------------------------------------------------


class MemoryCreate(msgspec.Struct, forbid_unknown_fields=True, kw_only=True):
    """One episode of the artist's fictional life. ``origin`` is never taken from the client."""

    title: _MemoryTitle
    body: _MemoryBody
    occurred_on: date | None = None
    """A fictional date: used for ordering and display only."""
    themes: _MemoryThemes = msgspec.field(default_factory=list)
    imagery: _MemoryImagery = msgspec.field(default_factory=list)
    """Concrete sensory details the planner may borrow obliquely."""
    salience: _Salience = 3
    """1-5: how readily the planner reaches for this memory when it fills the budget itself."""
    from_draft_id: UUID | None = None
    """The memory draft this memory was written from (``POST /v1/studio/chat-messages/{id}/memory-draft``).
    Set, the server checks the draft is the caller's, ``complete`` and from a chat of **this**
    artist, and stores the memory with ``origin = chat``; the client still sends the (edited) fields.
    Omitted, ``origin = user``. Never a way to set ``origin`` directly."""


class MemoryPatch(msgspec.Struct, forbid_unknown_fields=True, kw_only=True):
    """Tri-state update: omitted keeps the stored value; ``occurred_on: null`` clears the date."""

    title: _MemoryTitle | msgspec.UnsetType = msgspec.UNSET
    body: _MemoryBody | msgspec.UnsetType = msgspec.UNSET
    occurred_on: date | msgspec.UnsetType | None = msgspec.UNSET
    themes: _MemoryThemes | msgspec.UnsetType = msgspec.UNSET
    imagery: _MemoryImagery | msgspec.UnsetType = msgspec.UNSET
    salience: _Salience | msgspec.UnsetType = msgspec.UNSET
    status: StudioMemoryStatus | msgspec.UnsetType = msgspec.UNSET
    """``archived`` keeps the memory but takes it out of every run; ``active`` brings it back."""


class MemoryOut(msgspec.Struct, kw_only=True):
    id: UUID
    artist_id: UUID
    title: str
    body: str
    occurred_on: date | None
    themes: list[str]
    imagery: list[str]
    salience: int
    status: StudioMemoryStatus
    origin: StudioMemoryOrigin
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
    instrumental: bool
    """The plan's track-level flag. Suno has no text for it: switch its "Instrumental" toggle on
    and paste the Lyrics field (section headers only) as is."""
    fields: dict[str, ExportFieldOut]
    """Per-field size report for ``title`` / ``style`` / ``exclude`` / ``lyrics``. The limits are
    unofficial measurements: ``within_limit=false`` is information, never an error."""
    warnings: list[ExportWarningOut]


# ---------------------------------------------------------------------------
# Import
# ---------------------------------------------------------------------------


class SunoImportRequest(msgspec.Struct, forbid_unknown_fields=True, kw_only=True):
    """The three text boxes of a Suno song, pasted back into the studio."""

    title: _TrackTitle | None = None
    """Plan title; omitted keeps the current plan's title (a track without a plan: the track's).
    The track's own title is never changed by an import."""
    style: Annotated[str, Meta(max_length=1000)] = ""
    exclude: Annotated[str, Meta(max_length=1000)] = ""
    lyrics: Annotated[str, Meta(max_length=20_000)] = ""
    instrumental: bool | None = None
    """Omitted (or ``null``) keeps the current plan's flag (a track without a plan: ``false``).
    Never inferred from the text: a Lyrics box with headers only is not an instrumental, and
    pasting lyric lines over an instrumental track needs an explicit ``false``."""
    language: LanguageTag | None = None
    """Lyrics language; defaults to the current lyrics', then the artist's default."""
    dry_run: bool = False
    """Parse and report, write nothing."""


class ImportWarningOut(msgspec.Struct, kw_only=True):
    code: str
    line: int | None
    """1-based line of the pasted Lyrics; ``null`` for Style / Exclude / whole-import warnings."""


class RoundTripOut(msgspec.Struct, kw_only=True):
    """Whether rendering the imported documents reproduces each pasted box (canonicalised)."""

    style: bool
    exclude: bool
    lyrics: bool


class SunoImportOut(msgspec.Struct, kw_only=True):
    dry_run: bool
    plan: PlanDocument
    """The normalized plan the text stands for (what is, or on a dry run would be, stored)."""
    lyrics: LyricsDocument | None
    """``null`` when the pasted Lyrics have no lyric line."""
    plan_unchanged: bool
    lyrics_unchanged: bool
    warnings: list[ImportWarningOut]
    round_trip: RoundTripOut
    plan_version: VersionSummary | None
    """The plan version that is current after the import; ``null`` on a dry run."""
    lyrics_version: VersionSummary | None
    """The lyrics version that is current after the import; ``null`` on a dry run, or when the
    song has no lyrics."""
    created: bool
    """Whether any version row was written."""
