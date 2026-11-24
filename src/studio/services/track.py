"""Tracks of a studio project (``/v1/studio/projects/{id}/tracks``, ``/v1/studio/tracks/{id}``)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

import msgspec
import structlog

from src.api.schemas.studio import TrackCreate, TrackDetail, TrackOut, TrackPatch
from src.db.repositories.studio import (
    StudioProjectRepository,
    StudioRunRepository,
    StudioTrackRepository,
)
from src.db.repositories.studio_track_audio import StudioTrackAudioRepository
from src.studio.constraints import collect_length_issues
from src.studio.errors import (
    DocumentIssue,
    RunInProgress,
    StudioDocumentInvalid,
    StudioNotFound,
    TrackLimitReached,
    TrackOrderMismatch,
)
from src.studio.services.mappers import (
    lyrics_version_detail,
    plan_version_detail,
    require_current_rows,
    track_out,
)
from src.studio.services.track_audio import build_audio_outs
from src.studio.validation import normalize_text

if TYPE_CHECKING:
    from collections.abc import Sequence
    from uuid import UUID

    from sqlalchemy.ext.asyncio import AsyncSession

    from src.api.services.library_holds import AssetHoldService

__all__ = ["MAX_TRACKS_PER_PROJECT", "TrackService"]

logger = structlog.get_logger(__name__)

MAX_TRACKS_PER_PROJECT: Final = 64


def _normalize_title(raw: str, request_type: type[TrackCreate] | type[TrackPatch]) -> str:
    """NFC / trim / collapse ``raw``; refuse an empty result or one past the wire limit (RD6).

    The limit is read from ``request_type`` (the Struct that declares ``title``), never repeated
    here: NFC can expand a title past ``varchar(200)`` after the wire check passed.
    """
    title = normalize_text(raw)
    issues: list[DocumentIssue] = []
    if not title:
        issues.append(DocumentIssue("title", "empty_after_normalization"))
    else:
        collect_length_issues(request_type(title=title), issues)
    if issues:
        raise StudioDocumentInvalid(tuple(issues))
    return title


class TrackService:
    """Create, order, rename and delete tracks; open one with its current documents and audio."""

    def __init__(self, session: AsyncSession, holds: AssetHoldService) -> None:
        self._session = session
        self._holds = holds
        self._profiles = StudioProjectRepository(session)
        self._tracks = StudioTrackRepository(session)
        self._runs = StudioRunRepository(session)
        self._audio = StudioTrackAudioRepository(session)

    async def create(
        self, library_project_id: UUID, user_id: UUID, product_id: str, title: str
    ) -> TrackOut:
        normalized = _normalize_title(title, TrackCreate)
        # The profile lock serializes position allocation and the cap against concurrent creates,
        # reorders and project deletion.
        profile = await self._profiles.lock(
            library_project_id, user_id=user_id, product_id=product_id
        )
        if profile is None:
            raise StudioNotFound
        if (
            await self._tracks.count_for_project(
                library_project_id, user_id=user_id, product_id=product_id
            )
            >= MAX_TRACKS_PER_PROJECT
        ):
            raise TrackLimitReached
        track = await self._tracks.insert(
            user_id=user_id,
            product_id=product_id,
            library_project_id=library_project_id,
            position=await self._tracks.next_position(library_project_id),
            title=normalized,
        )
        logger.info(
            "studio.track_created",
            track_id=str(track.id),
            project_id=str(library_project_id),
            position=track.position,
        )
        return track_out(track)

    async def list_for_project(
        self, library_project_id: UUID, user_id: UUID, product_id: str
    ) -> list[TrackOut]:
        profile = await self._profiles.get(
            library_project_id, user_id=user_id, product_id=product_id
        )
        if profile is None:
            raise StudioNotFound
        tracks = await self._tracks.list_for_project(
            library_project_id, user_id=user_id, product_id=product_id
        )
        return [track_out(t) for t in tracks]

    async def reorder(
        self,
        library_project_id: UUID,
        user_id: UUID,
        product_id: str,
        track_ids: Sequence[UUID],
    ) -> list[TrackOut]:
        """Apply a new order. ``track_ids`` must be exactly the project's track set."""
        profile = await self._profiles.lock(
            library_project_id, user_id=user_id, product_id=product_id
        )
        if profile is None:
            raise StudioNotFound
        existing = await self._tracks.list_for_project(
            library_project_id, user_id=user_id, product_id=product_id
        )
        if len(track_ids) != len(existing) or set(track_ids) != {t.id for t in existing}:
            raise TrackOrderMismatch
        await self._tracks.reorder(
            library_project_id, track_ids, user_id=user_id, product_id=product_id
        )
        logger.info(
            "studio.track_reordered", project_id=str(library_project_id), count=len(track_ids)
        )
        reordered = await self._tracks.list_for_project(
            library_project_id, user_id=user_id, product_id=product_id
        )
        return [track_out(t) for t in reordered]

    async def get_detail(self, track_id: UUID, user_id: UUID, product_id: str) -> TrackDetail:
        # The track and the rows its pointers name come from ONE statement, so a concurrent
        # "switch current + delete the old version" cannot leave a pointer without its row.
        found = await self._tracks.get_with_current_versions(
            track_id, user_id=user_id, product_id=product_id
        )
        if found is None:
            raise StudioNotFound
        track = found[0]
        # A non-null pointer whose row cannot be read is a broken invariant: raise (RD5), do not
        # return a track that silently claims to have no current document.
        plan_row, lyrics_row = require_current_rows(track, found[1], found[2])
        current_plan = plan_version_detail(plan_row, track) if plan_row is not None else None
        current_lyrics = (
            lyrics_version_detail(lyrics_row, track) if lyrics_row is not None else None
        )
        # The current audio is read through the track's pointer in ONE statement (its own
        # snapshot): the pointer reported below is the one whose row was read, never a dangling one.
        audio_row = await self._audio.get_current(track_id, user_id=user_id, product_id=product_id)
        current_audio = (
            (await build_audio_outs(self._session, [audio_row], user_id))[0]
            if audio_row is not None
            else None
        )
        base = track_out(track)
        return TrackDetail(
            id=base.id,
            library_project_id=base.library_project_id,
            position=base.position,
            title=base.title,
            current_plan_version_id=base.current_plan_version_id,
            current_lyrics_version_id=base.current_lyrics_version_id,
            current_audio_version_id=current_audio.id if current_audio is not None else None,
            created_at=base.created_at,
            updated_at=base.updated_at,
            current_plan=current_plan,
            current_lyrics=current_lyrics,
            current_audio=current_audio,
        )

    async def patch(
        self, track_id: UUID, user_id: UUID, product_id: str, data: TrackPatch
    ) -> TrackOut:
        track = await self._tracks.get(track_id, user_id=user_id, product_id=product_id)
        if track is None:
            raise StudioNotFound
        if data.title is msgspec.UNSET:
            return track_out(track)
        updated = await self._tracks.update_title(
            track_id,
            user_id=user_id,
            product_id=product_id,
            title=_normalize_title(data.title, TrackPatch),
        )
        if updated is None:
            raise StudioNotFound
        logger.info("studio.track_renamed", track_id=str(track_id))
        return track_out(updated)

    async def delete(self, track_id: UUID, user_id: UUID, product_id: str) -> None:
        """Delete a track and (by cascade) all its versions, audio versions and finished runs.

        Positions of the others are kept. The holds the audio versions took are opaque to the
        database (no foreign key from a hold to the studio), so they are released explicitly,
        in the same transaction: collected under the track lock, released once the cascade has
        removed the rows that referenced them (a hold cannot be deleted under its audio version).

        Raises:
            StudioNotFound: No such track for the caller.
            RunInProgress: The track has a queued or running run.
        """
        # Lock order is always profile -> track, so a delete cannot interleave with a reorder.
        track = await self._tracks.get(track_id, user_id=user_id, product_id=product_id)
        if track is None:
            raise StudioNotFound
        await self._profiles.lock(track.library_project_id, user_id=user_id, product_id=product_id)
        track = await self._tracks.lock(track_id, user_id=user_id, product_id=product_id)
        if track is None:  # deleted while waiting for the profile lock
            raise StudioNotFound
        # A cascade would delete a queued / running run - and its paid reservation - unrefunded.
        # Submit needs this track's lock too, so no run can become active after this check.
        if await self._runs.has_active_for_track(track_id):
            raise RunInProgress
        # Taken under the track lock, which every attach needs too: no hold can appear after this.
        hold_ids = await self._audio.hold_ids_of_track(track_id)
        await self._tracks.delete(track_id, user_id=user_id, product_id=product_id)
        await self._holds.release(hold_ids)
        logger.info(
            "studio.track_deleted",
            track_id=str(track_id),
            project_id=str(track.library_project_id),
            audio_version_count=len(hold_ids),
        )
