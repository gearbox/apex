"""Repository for the studio's realized audio timelines (S3a).

Conventions (those of ``studio.py``, P1-9, and ``studio_track_audio.py``):

* Every query on timelines filters the **timeline's own** ``user_id`` and ``product_id``, so a
  foreign, cross-product or unknown id is simply "not found" - one 404.
* Writes are Core ``insert/update/delete … RETURNING``; this repository never ``flush()``es or
  ``commit()``s.
* Rows are immutable: the only mutations are insert, delete, and the audio version's counter and
  current pointer.
* ``is_current`` is read from the audio version in the **same statement** as the timeline.
* Callers that need atomicity lock first, in the order **track → audio version → timeline**: the
  50-version cap, the numbering and every write hang off those locks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from sqlalchemy import delete, exists, func, select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert

from src.core.uid import new_id
from src.db.models.studio import StudioAudioTimeline, StudioDocumentVersion, StudioTrackAudio

if TYPE_CHECKING:
    from collections.abc import Sequence
    from datetime import datetime
    from uuid import UUID

    from sqlalchemy import ColumnElement, Row, Select
    from sqlalchemy.ext.asyncio import AsyncSession

__all__ = [
    "StudioAudioTimelineRepository",
    "TimelineExportRow",
    "TimelineRow",
    "TimelineSummaryRow",
]


@dataclass(frozen=True, slots=True)
class TimelineSummaryRow:
    """A stored timeline without its document, plus whether its audio version's pointer names
    it."""

    id: UUID
    audio_version_id: UUID
    track_id: UUID
    version_number: int
    parent_id: UUID | None
    origin: str
    is_complete: bool
    created_at: datetime
    is_current: bool


@dataclass(frozen=True, slots=True)
class TimelineRow(TimelineSummaryRow):
    """A stored timeline with its document."""

    schema_version: int
    content: dict[str, Any]
    content_hash: str


@dataclass(frozen=True, slots=True)
class TimelineExportRow:
    """A timeline with the lyrics its audio version links, read in **one** statement."""

    timeline: TimelineRow
    lyrics_version_id: UUID | None
    lyrics_row: StudioDocumentVersion | None
    """``None`` exactly when ``lyrics_version_id`` is (anything else is a broken invariant)."""


_SUMMARY_COLUMNS = (
    StudioAudioTimeline.id,
    StudioAudioTimeline.audio_version_id,
    StudioAudioTimeline.track_id,
    StudioAudioTimeline.version_number,
    StudioAudioTimeline.parent_id,
    StudioAudioTimeline.origin,
    StudioAudioTimeline.is_complete,
    StudioAudioTimeline.created_at,
    # IS NOT DISTINCT FROM: an audio version with no current timeline never yields NULL.
    StudioTrackAudio.current_timeline_id.is_not_distinct_from(StudioAudioTimeline.id).label(
        "is_current"
    ),
)

_DOCUMENT_COLUMNS = (
    StudioAudioTimeline.schema_version,
    StudioAudioTimeline.content,
    StudioAudioTimeline.content_hash,
)


def _select_rows(*, with_content: bool) -> Select[*tuple[Any, ...]]:
    """The rows' SELECT: the timeline and its audio version (for ``is_current``).

    ``with_content=False`` leaves the document (and its schema version and hash) unread: a
    document may be 128 KiB, and a page holds up to 50 timelines.
    """
    columns = (*_SUMMARY_COLUMNS, *_DOCUMENT_COLUMNS) if with_content else _SUMMARY_COLUMNS
    return (
        select(*columns)
        .select_from(StudioAudioTimeline)
        .join(StudioTrackAudio, StudioTrackAudio.id == StudioAudioTimeline.audio_version_id)
    )


def _summary_row(row: Row[*tuple[Any, ...]]) -> TimelineSummaryRow:
    return TimelineSummaryRow(
        id=row.id,
        audio_version_id=row.audio_version_id,
        track_id=row.track_id,
        version_number=row.version_number,
        parent_id=row.parent_id,
        origin=row.origin,
        is_complete=row.is_complete,
        created_at=row.created_at,
        is_current=bool(row.is_current),
    )


def _row(row: Row[*tuple[Any, ...]]) -> TimelineRow:
    return TimelineRow(
        id=row.id,
        audio_version_id=row.audio_version_id,
        track_id=row.track_id,
        version_number=row.version_number,
        parent_id=row.parent_id,
        origin=row.origin,
        is_complete=row.is_complete,
        created_at=row.created_at,
        is_current=bool(row.is_current),
        schema_version=row.schema_version,
        content=row.content,
        content_hash=row.content_hash,
    )


class StudioAudioTimelineRepository:
    """Data access for ``studio_audio_timelines`` (and the audio version's counter and pointer)."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    @staticmethod
    def _scope(user_id: UUID, product_id: str) -> tuple[ColumnElement[bool], ...]:
        """Where-clauses: the timeline belongs to ``user_id`` in ``product_id``."""
        return (
            StudioAudioTimeline.user_id == user_id,
            StudioAudioTimeline.product_id == product_id,
        )

    # -- numbering and the pointer (both live on the audio version) ----------------------------

    async def allocate_number(self, audio_id: UUID) -> int:
        """Take the next timeline number from the audio version's counter.

        Numbers are never reused (not even after a delete). The caller holds the track and audio
        version locks and calls this only when a row is actually going to be inserted.
        """
        result = await self._session.execute(
            update(StudioTrackAudio)
            .where(StudioTrackAudio.id == audio_id)
            .values(next_timeline_number=StudioTrackAudio.next_timeline_number + 1)
            .returning((StudioTrackAudio.next_timeline_number - 1).label("allocated"))
            .execution_options(synchronize_session=False)
        )
        return result.scalar_one()

    async def set_current(self, audio_id: UUID, timeline_id: UUID | None) -> None:
        """Point the audio version at a timeline, or clear the pointer.

        Raises:
            sqlalchemy.exc.IntegrityError: The timeline is not one of this audio version's
                (composite FK); call inside ``session.begin_nested()``.
        """
        await self._session.execute(
            update(StudioTrackAudio)
            .where(StudioTrackAudio.id == audio_id)
            .values(current_timeline_id=timeline_id)
            .execution_options(synchronize_session=False)
        )

    # -- writes --------------------------------------------------------------------------------

    async def insert(
        self,
        *,
        user_id: UUID,
        product_id: str,
        track_id: UUID,
        audio_version_id: UUID,
        version_number: int,
        parent_id: UUID | None,
        origin: str,
        schema_version: int,
        content: dict[str, object],
        content_hash: str,
        is_complete: bool,
    ) -> UUID:
        """Insert a timeline version and return its id.

        Raises:
            sqlalchemy.exc.IntegrityError: An identical sibling exists, the number repeats, or the
                audio version or parent is not what it claims to be; call inside
                ``session.begin_nested()``.
        """
        result = await self._session.execute(
            pg_insert(StudioAudioTimeline)
            .values(
                id=new_id(),
                user_id=user_id,
                product_id=product_id,
                track_id=track_id,
                audio_version_id=audio_version_id,
                version_number=version_number,
                parent_id=parent_id,
                origin=origin,
                schema_version=schema_version,
                content=content,
                content_hash=content_hash,
                is_complete=is_complete,
            )
            .returning(StudioAudioTimeline.id)
        )
        return result.scalar_one()

    async def delete(self, timeline_id: UUID, *, user_id: UUID, product_id: str) -> bool:
        """Delete a timeline version.

        Raises:
            sqlalchemy.exc.IntegrityError: It became current or got a child after the caller
                checked (NO ACTION FKs); call inside ``session.begin_nested()``.
        """
        result = await self._session.execute(
            delete(StudioAudioTimeline)
            .where(StudioAudioTimeline.id == timeline_id, *self._scope(user_id, product_id))
            .returning(StudioAudioTimeline.id)
        )
        return result.scalar_one_or_none() is not None

    # -- reads ---------------------------------------------------------------------------------

    async def get(self, timeline_id: UUID, *, user_id: UUID, product_id: str) -> TimelineRow | None:
        result = await self._session.execute(
            _select_rows(with_content=True).where(
                StudioAudioTimeline.id == timeline_id, *self._scope(user_id, product_id)
            )
        )
        row = result.one_or_none()
        return None if row is None else _row(row)

    async def get_for_export(
        self, timeline_id: UUID, *, user_id: UUID, product_id: str
    ) -> TimelineExportRow | None:
        """The timeline **and** the lyrics its audio version links, from one snapshot.

        One statement is one ``READ COMMITTED`` snapshot. Read apart, a concurrent "delete the
        timeline, then relink the audio" could pair these times with another version's text.
        """
        result = await self._session.execute(
            _select_rows(with_content=True)
            .add_columns(StudioTrackAudio.lyrics_version_id, StudioDocumentVersion)
            .outerjoin(
                StudioDocumentVersion,
                StudioDocumentVersion.id == StudioTrackAudio.lyrics_version_id,
            )
            .where(StudioAudioTimeline.id == timeline_id, *self._scope(user_id, product_id))
        )
        row = result.one_or_none()
        if row is None:
            return None
        return TimelineExportRow(
            timeline=_row(row),
            lyrics_version_id=row.lyrics_version_id,
            lyrics_row=row.StudioDocumentVersion,
        )

    async def find_sibling(
        self,
        audio_id: UUID,
        parent_id: UUID | None,
        content_hash: str,
        *,
        user_id: UUID,
        product_id: str,
    ) -> UUID | None:
        """The id of the version with this parent and content, if there is one (replay)."""
        result = await self._session.execute(
            select(StudioAudioTimeline.id).where(
                StudioAudioTimeline.audio_version_id == audio_id,
                StudioAudioTimeline.parent_id.is_not_distinct_from(parent_id),
                StudioAudioTimeline.content_hash == content_hash,
                *self._scope(user_id, product_id),
            )
        )
        return result.scalar_one_or_none()

    async def count_for_audio(self, audio_id: UUID, *, user_id: UUID, product_id: str) -> int:
        result = await self._session.execute(
            select(func.count())
            .select_from(StudioAudioTimeline)
            .where(
                StudioAudioTimeline.audio_version_id == audio_id,
                *self._scope(user_id, product_id),
            )
        )
        return result.scalar_one()

    async def has_children(self, timeline_id: UUID, *, user_id: UUID, product_id: str) -> bool:
        result = await self._session.execute(
            select(
                exists().where(
                    StudioAudioTimeline.parent_id == timeline_id,
                    *self._scope(user_id, product_id),
                )
            )
        )
        return bool(result.scalar_one())

    async def list_page(
        self,
        audio_id: UUID,
        *,
        user_id: UUID,
        product_id: str,
        limit: int,
        before_number: int | None = None,
    ) -> Sequence[TimelineSummaryRow]:
        """Newest first by ``version_number``: ``limit + 1`` rows below the cursor's number,
        without their documents."""
        query = _select_rows(with_content=False).where(
            StudioAudioTimeline.audio_version_id == audio_id,
            *self._scope(user_id, product_id),
        )
        if before_number is not None:
            query = query.where(StudioAudioTimeline.version_number < before_number)
        result = await self._session.execute(
            query.order_by(StudioAudioTimeline.version_number.desc()).limit(limit + 1)
        )
        return [_summary_row(row) for row in result.all()]
