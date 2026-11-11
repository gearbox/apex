"""ORM row → API DTO mapping shared by the studio services."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from src.api.schemas.studio import (
    ArtistOut,
    LyricsVersionDetail,
    PlanVersionDetail,
    ProjectProfileOut,
    TrackOut,
    VersionSummary,
)
from src.core.enums import StudioDocumentKind, StudioDocumentOrigin, StudioReleaseKind
from src.studio.documents import LyricsDocument, PlanDocument, decode_stored

if TYPE_CHECKING:
    from src.db.models.studio import StudioArtist, StudioDocumentVersion, StudioTrack
    from src.db.repositories.studio import ProjectListRow

__all__ = [
    "artist_out",
    "decode_lyrics",
    "decode_plan",
    "is_current",
    "lyrics_version_detail",
    "plan_version_detail",
    "project_out",
    "track_out",
    "version_detail",
    "version_summary",
]


def artist_out(artist: StudioArtist) -> ArtistOut:
    return ArtistOut(
        id=artist.id,
        name=artist.name,
        persona=artist.persona,
        voice_notes=artist.voice_notes,
        default_language=artist.default_language,
        aesthetic_references=list(artist.aesthetic_references),
        style_tags=list(artist.style_tags),
        global_avoid=list(artist.global_avoid),
        created_at=artist.created_at,
        updated_at=artist.updated_at,
    )


def project_out(row: ProjectListRow) -> ProjectProfileOut:
    profile = row.profile
    return ProjectProfileOut(
        library_project_id=profile.library_project_id,
        name=row.name,
        release_kind=StudioReleaseKind(profile.release_kind),
        artist_id=profile.artist_id,
        concept=profile.concept,
        track_count=row.track_count,
        created_at=profile.created_at,
        updated_at=profile.updated_at,
    )


def track_out(track: StudioTrack) -> TrackOut:
    return TrackOut(
        id=track.id,
        library_project_id=track.library_project_id,
        position=track.position,
        title=track.title,
        current_plan_version_id=track.current_plan_version_id,
        current_lyrics_version_id=track.current_lyrics_version_id,
        created_at=track.created_at,
        updated_at=track.updated_at,
    )


def is_current(version: StudioDocumentVersion, track: StudioTrack) -> bool:
    return version.id in (track.current_plan_version_id, track.current_lyrics_version_id)


def decode_plan(version: StudioDocumentVersion) -> PlanDocument:
    """Decode a stored plan version (raises loudly on an unknown ``schema_version``)."""
    return cast(
        "PlanDocument",
        decode_stored(StudioDocumentKind.PLAN, version.schema_version, version.content),
    )


def decode_lyrics(version: StudioDocumentVersion) -> LyricsDocument:
    """Decode a stored lyrics version (raises loudly on an unknown ``schema_version``)."""
    return cast(
        "LyricsDocument",
        decode_stored(StudioDocumentKind.LYRICS, version.schema_version, version.content),
    )


def version_summary(version: StudioDocumentVersion, track: StudioTrack) -> VersionSummary:
    return VersionSummary(
        id=version.id,
        kind=StudioDocumentKind(version.kind),
        version_number=version.version_number,
        parent_version_id=version.parent_version_id,
        based_on_plan_version_id=version.based_on_plan_version_id,
        origin=StudioDocumentOrigin(version.origin),
        schema_version=version.schema_version,
        content_hash=version.content_hash,
        created_at=version.created_at,
        is_current=is_current(version, track),
    )


def plan_version_detail(
    version: StudioDocumentVersion, track: StudioTrack, document: PlanDocument | None = None
) -> PlanVersionDetail:
    """Plan detail DTO; decodes the stored content unless ``document`` is supplied."""
    return PlanVersionDetail(
        id=version.id,
        track_id=version.track_id,
        version_number=version.version_number,
        parent_version_id=version.parent_version_id,
        origin=StudioDocumentOrigin(version.origin),
        schema_version=version.schema_version,
        content_hash=version.content_hash,
        created_at=version.created_at,
        is_current=is_current(version, track),
        content=document if document is not None else decode_plan(version),
    )


def lyrics_version_detail(
    version: StudioDocumentVersion, track: StudioTrack, document: LyricsDocument | None = None
) -> LyricsVersionDetail:
    """Lyrics detail DTO; decodes the stored content unless ``document`` is supplied."""
    assert version.based_on_plan_version_id is not None  # noqa: S101 — DB CHECK guarantees it
    lyrics = document if document is not None else decode_lyrics(version)
    return LyricsVersionDetail(
        id=version.id,
        track_id=version.track_id,
        version_number=version.version_number,
        parent_version_id=version.parent_version_id,
        based_on_plan_version_id=version.based_on_plan_version_id,
        origin=StudioDocumentOrigin(version.origin),
        schema_version=version.schema_version,
        content_hash=version.content_hash,
        created_at=version.created_at,
        is_current=is_current(version, track),
        content=lyrics,
    )


def version_detail(
    version: StudioDocumentVersion,
    track: StudioTrack,
    document: PlanDocument | LyricsDocument | None = None,
) -> PlanVersionDetail | LyricsVersionDetail:
    """Detail DTO for a version of either kind (dispatches on ``version.kind``)."""
    if version.kind == StudioDocumentKind.PLAN.value:
        return plan_version_detail(
            version, track, document if isinstance(document, PlanDocument) else None
        )
    return lyrics_version_detail(
        version, track, document if isinstance(document, LyricsDocument) else None
    )
