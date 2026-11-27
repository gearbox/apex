"""Repository for the studio's track audio versions (S2a).

Conventions (those of ``studio.py``, P1-9, and ``studio_cast.py``):

* Every query filters ``product_id`` **and** ``user_id`` (the row's own scope), so a foreign or
  cross-product id is simply "not found" - the same 404 as an unknown one.
* Writes are Core ``insert/update/delete … RETURNING``; this repository never ``flush()``es or
  ``commit()``s. Locking reads use ``populate_existing``.
* A version's asset lives only on its hold row: the reads here **join** ``library_asset_holds`` to
  learn it (the row stores no copy). The join is the only place the studio touches the library's
  table; the studio never writes it (that is ``AssetHoldService``'s job).
* A version's timeline facts (S3a: the current pointer, how many timelines it has and whether the
  current one is complete) come from the **same statement** - one counted subquery and one outer
  join, so a page of versions is one SELECT, with no per-item query.
* Callers that need atomicity lock the **track** first (``StudioTrackRepository.lock``): the
  20-version cap, the numbering and every mutation hang off that one lock.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import msgspec
from sqlalchemy import delete, func, select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import aliased

from src.core.library_ref import LibraryAssetSource
from src.db.models.library import LibraryAssetHold
from src.db.models.studio import StudioAudioTimeline, StudioTrack, StudioTrackAudio

if TYPE_CHECKING:
    from collections.abc import Sequence
    from datetime import datetime
    from uuid import UUID

    from sqlalchemy import ColumnElement, Select
    from sqlalchemy.ext.asyncio import AsyncSession

__all__ = ["StudioTrackAudioRepository", "TrackAudioRow"]


@dataclass(frozen=True, slots=True)
class TrackAudioRow:
    """An audio version together with the asset its hold pins (one join, no second copy)."""

    id: UUID
    track_id: UUID
    version_number: int
    label: str | None
    origin: str
    hold_id: UUID
    plan_version_id: UUID | None
    lyrics_version_id: UUID | None
    duration_ms: int
    created_at: datetime
    updated_at: datetime
    is_current: bool
    """Whether the track's current-audio pointer names this version (read in the same statement)."""
    current_timeline_id: UUID | None
    timeline_count: int
    timeline_complete: bool | None
    """``is_complete`` of the current timeline; ``None`` when there is no current timeline."""
    source: LibraryAssetSource
    asset_id: UUID


def _row(row: object) -> TrackAudioRow:
    """Build a row from a ``(audio columns…, hold.upload_id, hold.output_id)`` result row."""
    upload_id = row.upload_id  # type: ignore[attr-defined]
    return TrackAudioRow(
        id=row.id,  # type: ignore[attr-defined]
        track_id=row.track_id,  # type: ignore[attr-defined]
        version_number=row.version_number,  # type: ignore[attr-defined]
        label=row.label,  # type: ignore[attr-defined]
        origin=row.origin,  # type: ignore[attr-defined]
        hold_id=row.hold_id,  # type: ignore[attr-defined]
        plan_version_id=row.plan_version_id,  # type: ignore[attr-defined]
        lyrics_version_id=row.lyrics_version_id,  # type: ignore[attr-defined]
        duration_ms=row.duration_ms,  # type: ignore[attr-defined]
        created_at=row.created_at,  # type: ignore[attr-defined]
        updated_at=row.updated_at,  # type: ignore[attr-defined]
        is_current=bool(row.is_current),  # type: ignore[attr-defined]
        current_timeline_id=row.current_timeline_id,  # type: ignore[attr-defined]
        timeline_count=row.timeline_count,  # type: ignore[attr-defined]
        timeline_complete=row.timeline_complete,  # type: ignore[attr-defined]
        source=LibraryAssetSource.UPLOAD if upload_id is not None else LibraryAssetSource.OUTPUT,
        asset_id=upload_id if upload_id is not None else row.output_id,  # type: ignore[attr-defined]
    )


_CURRENT_TIMELINE = aliased(StudioAudioTimeline, name="current_timeline")

_TIMELINE_COUNT = (
    select(func.count())
    .select_from(StudioAudioTimeline)
    .where(
        StudioAudioTimeline.audio_version_id == StudioTrackAudio.id,
        StudioAudioTimeline.user_id == StudioTrackAudio.user_id,
        StudioAudioTimeline.product_id == StudioTrackAudio.product_id,
    )
    .correlate(StudioTrackAudio)
    .scalar_subquery()
)

_COLUMNS = (
    StudioTrackAudio.id,
    StudioTrackAudio.track_id,
    StudioTrackAudio.version_number,
    StudioTrackAudio.label,
    StudioTrackAudio.origin,
    StudioTrackAudio.hold_id,
    StudioTrackAudio.plan_version_id,
    StudioTrackAudio.lyrics_version_id,
    StudioTrackAudio.duration_ms,
    StudioTrackAudio.created_at,
    StudioTrackAudio.updated_at,
    # IS NOT DISTINCT FROM: a track with no current audio is never "current" (never NULL).
    StudioTrack.current_audio_version_id.is_not_distinct_from(StudioTrackAudio.id).label(
        "is_current"
    ),
    StudioTrackAudio.current_timeline_id,
    _TIMELINE_COUNT.label("timeline_count"),
    _CURRENT_TIMELINE.is_complete.label("timeline_complete"),
    LibraryAssetHold.upload_id,
    LibraryAssetHold.output_id,
)


def _select_rows() -> Select[*tuple[Any, ...]]:
    """The rows' SELECT: the version, its track (for ``is_current``), its hold (the asset) and
    its current timeline, if any (for ``timeline_complete``)."""
    return (
        select(*_COLUMNS)
        .select_from(StudioTrackAudio)
        .join(StudioTrack, StudioTrack.id == StudioTrackAudio.track_id)
        .join(LibraryAssetHold, LibraryAssetHold.id == StudioTrackAudio.hold_id)
        .outerjoin(_CURRENT_TIMELINE, _CURRENT_TIMELINE.id == StudioTrackAudio.current_timeline_id)
    )


class StudioTrackAudioRepository:
    """Data access for ``studio_track_audio``."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    @staticmethod
    def _owned(audio_id: UUID, user_id: UUID, product_id: str) -> tuple[ColumnElement[bool], ...]:
        return (
            StudioTrackAudio.id == audio_id,
            StudioTrackAudio.user_id == user_id,
            StudioTrackAudio.product_id == product_id,
        )

    # -- numbering -----------------------------------------------------------------------------

    async def allocate_version_number(self, track_id: UUID) -> int:
        """Take the next audio version number from the track's counter.

        Numbers are never reused (not even after a delete). The caller holds the track lock and
        calls this only when a row is actually going to be inserted.
        """
        result = await self._session.execute(
            update(StudioTrack)
            .where(StudioTrack.id == track_id)
            .values(next_audio_version_number=StudioTrack.next_audio_version_number + 1)
            .returning((StudioTrack.next_audio_version_number - 1).label("allocated"))
            .execution_options(synchronize_session=False)
        )
        return result.scalar_one()

    # -- writes --------------------------------------------------------------------------------

    async def insert(
        self,
        *,
        audio_id: UUID,
        user_id: UUID,
        product_id: str,
        track_id: UUID,
        version_number: int,
        label: str | None,
        origin: str,
        hold_id: UUID,
        plan_version_id: UUID | None,
        lyrics_version_id: UUID | None,
        duration_ms: int,
    ) -> None:
        """Insert a version whose id was generated first (the hold names it as its holder).

        Raises:
            sqlalchemy.exc.IntegrityError: A link names another track's version (composite FK),
                the version number repeats, or the track is gone; call inside
                ``session.begin_nested()``.
        """
        await self._session.execute(
            pg_insert(StudioTrackAudio).values(
                id=audio_id,
                user_id=user_id,
                product_id=product_id,
                track_id=track_id,
                version_number=version_number,
                label=label,
                origin=origin,
                hold_id=hold_id,
                plan_version_id=plan_version_id,
                lyrics_version_id=lyrics_version_id,
                duration_ms=duration_ms,
            )
        )

    async def update(
        self,
        audio_id: UUID,
        *,
        user_id: UUID,
        product_id: str,
        label: str | msgspec.UnsetType | None = msgspec.UNSET,
        plan_version_id: UUID | msgspec.UnsetType | None = msgspec.UNSET,
        lyrics_version_id: UUID | msgspec.UnsetType | None = msgspec.UNSET,
    ) -> bool:
        """Change the editable fields (unset ones are left alone) and bump ``updated_at``.

        Raises:
            sqlalchemy.exc.IntegrityError: A link names another track's version; call inside
                ``session.begin_nested()``.
        """
        values: dict[str, object] = {"updated_at": func.now()}
        if not isinstance(label, msgspec.UnsetType):
            values["label"] = label
        if not isinstance(plan_version_id, msgspec.UnsetType):
            values["plan_version_id"] = plan_version_id
        if not isinstance(lyrics_version_id, msgspec.UnsetType):
            values["lyrics_version_id"] = lyrics_version_id
        result = await self._session.execute(
            update(StudioTrackAudio)
            .where(*self._owned(audio_id, user_id, product_id))
            .values(**values)
            .returning(StudioTrackAudio.id)
            .execution_options(synchronize_session=False)
        )
        return result.scalar_one_or_none() is not None

    async def delete(self, audio_id: UUID, *, user_id: UUID, product_id: str) -> bool:
        """Delete a version row (its hold is released by the caller, in the same savepoint).

        Raises:
            sqlalchemy.exc.IntegrityError: It became the track's current audio after the caller
                checked (NO ACTION FK); call inside ``session.begin_nested()``.
        """
        result = await self._session.execute(
            delete(StudioTrackAudio)
            .where(*self._owned(audio_id, user_id, product_id))
            .returning(StudioTrackAudio.id)
        )
        return result.scalar_one_or_none() is not None

    # -- reads ---------------------------------------------------------------------------------

    async def get(self, audio_id: UUID, *, user_id: UUID, product_id: str) -> TrackAudioRow | None:
        result = await self._session.execute(
            _select_rows().where(*self._owned(audio_id, user_id, product_id))
        )
        row = result.one_or_none()
        return None if row is None else _row(row)

    async def get_current(
        self, track_id: UUID, *, user_id: UUID, product_id: str
    ) -> TrackAudioRow | None:
        """The version the track's current-audio pointer names, read in ONE statement.

        The pointer and its row come from one ``READ COMMITTED`` snapshot, so a concurrent
        "switch current + delete the old version" can never leave a pointer naming a row that is
        gone. ``None`` when the track has no current audio (or is not the caller's).
        """
        result = await self._session.execute(
            _select_rows().where(
                StudioTrack.id == track_id,
                StudioTrack.user_id == user_id,
                StudioTrack.product_id == product_id,
                StudioTrack.current_audio_version_id == StudioTrackAudio.id,
            )
        )
        row = result.one_or_none()
        return None if row is None else _row(row)

    async def get_on_track(
        self, audio_id: UUID, track_id: UUID, *, user_id: UUID, product_id: str
    ) -> TrackAudioRow | None:
        """The version if it exists **on this track** (another track's is simply not found)."""
        result = await self._session.execute(
            _select_rows().where(
                *self._owned(audio_id, user_id, product_id), StudioTrackAudio.track_id == track_id
            )
        )
        row = result.one_or_none()
        return None if row is None else _row(row)

    async def lock(
        self, audio_id: UUID, *, user_id: UUID, product_id: str
    ) -> StudioTrackAudio | None:
        """``SELECT … FOR UPDATE`` a version (after the track lock: track → audio version)."""
        result = await self._session.execute(
            select(StudioTrackAudio)
            .where(*self._owned(audio_id, user_id, product_id))
            .with_for_update()
            .execution_options(populate_existing=True)
        )
        return result.scalar_one_or_none()

    async def count_for_track(self, track_id: UUID) -> int:
        result = await self._session.execute(
            select(func.count())
            .select_from(StudioTrackAudio)
            .where(StudioTrackAudio.track_id == track_id)
        )
        return result.scalar_one()

    async def list_page(
        self,
        track_id: UUID,
        *,
        user_id: UUID,
        product_id: str,
        limit: int,
        before_number: int | None = None,
    ) -> Sequence[TrackAudioRow]:
        """Newest first by ``version_number``: ``limit + 1`` rows below the cursor's number."""
        query = _select_rows().where(
            StudioTrackAudio.track_id == track_id,
            StudioTrackAudio.user_id == user_id,
            StudioTrackAudio.product_id == product_id,
        )
        if before_number is not None:
            query = query.where(StudioTrackAudio.version_number < before_number)
        result = await self._session.execute(
            query.order_by(StudioTrackAudio.version_number.desc()).limit(limit + 1)
        )
        return [_row(row) for row in result.all()]

    async def hold_ids_of_track(self, track_id: UUID) -> list[UUID]:
        """Every hold the track's audio versions took (released before a track delete)."""
        result = await self._session.execute(
            select(StudioTrackAudio.hold_id)
            .where(StudioTrackAudio.track_id == track_id)
            .order_by(StudioTrackAudio.id)
        )
        return list(result.scalars().all())
