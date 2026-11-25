"""ORM row → API DTO mapping shared by the studio services."""

from __future__ import annotations

from typing import TYPE_CHECKING, Final, cast
from uuid import UUID

from src.api.schemas.studio import (
    ArtistOut,
    LyricsVersionDetail,
    MemoryOut,
    PlanVersionDetail,
    ProjectProfileOut,
    TrackOut,
    VersionSummary,
)
from src.api.schemas.studio_runs import RunOut, RunWarningOut
from src.core.enums import (
    StudioDocumentKind,
    StudioDocumentOrigin,
    StudioMemoryOrigin,
    StudioMemoryStatus,
    StudioReleaseKind,
    StudioRunKind,
    StudioRunStage,
    StudioRunStatus,
)
from src.studio.documents import LyricsDocument, PlanDocument, decode_stored
from src.studio.planner.context import ArtistSnapshot, MemorySnapshot

if TYPE_CHECKING:
    from src.db.models.studio import (
        StudioArtist,
        StudioArtistMemory,
        StudioDocumentVersion,
        StudioRun,
        StudioTrack,
    )
    from src.db.repositories.studio import ProjectListRow

__all__ = [
    "DANGLING_CURRENT_POINTER",
    "artist_out",
    "artist_snapshot",
    "decode_lyrics",
    "decode_plan",
    "is_current",
    "lyrics_version_detail",
    "memory_ids_used",
    "memory_out",
    "memory_snapshot",
    "plan_version_detail",
    "project_out",
    "require_current_row",
    "require_current_rows",
    "run_out",
    "track_out",
    "version_detail",
    "version_summary",
]

DANGLING_CURRENT_POINTER: Final = "studio current pointer without a row"


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
        performer_cast_member_id=artist.performer_cast_member_id,
        created_at=artist.created_at,
        updated_at=artist.updated_at,
    )


def memory_out(memory: StudioArtistMemory) -> MemoryOut:
    return MemoryOut(
        id=memory.id,
        artist_id=memory.artist_id,
        title=memory.title,
        body=memory.body,
        occurred_on=memory.occurred_on,
        themes=list(memory.themes),
        imagery=list(memory.imagery),
        salience=memory.salience,
        status=StudioMemoryStatus(memory.status),
        origin=StudioMemoryOrigin(memory.origin),
        created_at=memory.created_at,
        updated_at=memory.updated_at,
    )


def artist_snapshot(artist: StudioArtist) -> ArtistSnapshot:
    """Every planner-relevant field of the artist, as a run or a chat records it."""
    return ArtistSnapshot(
        id=artist.id,
        name=artist.name,
        persona=artist.persona,
        voice_notes=artist.voice_notes,
        default_language=artist.default_language,
        style_tags=tuple(artist.style_tags),
        global_avoid=tuple(artist.global_avoid),
        aesthetic_references=tuple(artist.aesthetic_references),
        updated_at=artist.updated_at,
    )


def memory_snapshot(memory: StudioArtistMemory) -> MemorySnapshot:
    """A memory as a run records it (P2-M4): the full text and the fields the prompt shows."""
    return MemorySnapshot(
        id=memory.id,
        title=memory.title,
        body=memory.body,
        themes=tuple(memory.themes),
        imagery=tuple(memory.imagery),
        occurred_on=memory.occurred_on,
        salience=memory.salience,
        updated_at=memory.updated_at,
    )


def memory_ids_used(row: StudioRun) -> list[UUID]:
    """The ids of the memories the run used, in prompt order, from its context snapshot.

    Ids only: the snapshot holds the memories' text, which never leaves the run. A run that has
    not started has no snapshot yet (nothing used); neither has a run without memories.
    """
    snapshot = row.context_snapshot
    if snapshot is None:
        return []
    return [UUID(str(memory["id"])) for memory in snapshot.get("memories", [])]


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
        current_audio_version_id=track.current_audio_version_id,
        created_at=track.created_at,
        updated_at=track.updated_at,
    )


def is_current(version: StudioDocumentVersion, track: StudioTrack) -> bool:
    return version.id in (track.current_plan_version_id, track.current_lyrics_version_id)


def require_current_row(row: StudioDocumentVersion | None) -> StudioDocumentVersion:
    """The version row a track's **non-null** current pointer refers to.

    The composite same-track foreign keys make a dangling pointer impossible, so a missing row
    means an internal invariant is broken. It must fail loudly (the catch-all 500 handler logs
    it), never read as "no document": that would skip the pair-compatibility check or hide the
    track's current document from the UI (RD5).

    Raises:
        RuntimeError: ``row`` is ``None``.
    """
    if row is None:
        raise RuntimeError(DANGLING_CURRENT_POINTER)
    return row


def require_current_rows(
    track: StudioTrack,
    plan_row: StudioDocumentVersion | None,
    lyrics_row: StudioDocumentVersion | None,
) -> tuple[StudioDocumentVersion | None, StudioDocumentVersion | None]:
    """The rows of a track's current pointers, as read together with the track in one statement.

    A row is ``None`` exactly when its pointer is. Because the pointers and the rows came from
    one snapshot (``StudioTrackRepository.get_with_current_versions``), a non-null pointer without
    its row is the broken invariant of :func:`require_current_row`, not a race.

    Raises:
        RuntimeError: A non-null pointer has no row.
    """
    if track.current_plan_version_id is not None:
        plan_row = require_current_row(plan_row)
    if track.current_lyrics_version_id is not None:
        lyrics_row = require_current_row(lyrics_row)
    return plan_row, lyrics_row


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


def run_out(row: StudioRun) -> RunOut:
    """The API view of a run row - ids, codes and timestamps only."""
    return RunOut(
        id=row.id,
        track_id=row.track_id,
        kind=StudioRunKind(row.kind),
        status=StudioRunStatus(row.status),
        stage=StudioRunStage(row.stage) if row.stage is not None else None,
        revision=row.revision,
        price_tokens=row.price_tokens,
        model_key=row.model_key,
        result_plan_version_id=row.result_plan_version_id,
        result_lyrics_version_id=row.result_lyrics_version_id,
        warnings=[
            RunWarningOut(code=str(w["code"]), path=w.get("path"), count=w.get("count"))
            for w in row.warnings
        ],
        memory_ids_used=memory_ids_used(row),
        error_code=row.error_code,
        error_detail=row.error_detail,
        llm_calls=row.llm_calls,
        created_at=row.created_at,
        started_at=row.started_at,
        finished_at=row.finished_at,
    )
