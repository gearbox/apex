"""Repositories for the Media Project Studio.

Conventions (P1-9):

* Every query filters ``product_id`` **and** ``user_id``.
* Writes are Core ``insert/update/delete … RETURNING``; repositories never ``flush()`` or
  ``commit()``. The request transaction is owned by ``get_db_session``.
* Row locks (``lock``) and refreshing reads use ``populate_existing`` so a row that was waited on
  is never served from a stale identity-map entry.
* Callers that need atomicity take the right lock first: the **profile** row before
  ``next_position`` / ``reorder`` / project deletion, the **track** row before allocating a
  version number or touching its pointers.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from typing import TYPE_CHECKING, Any

import msgspec
from sqlalchemy import (
    Boolean,
    Integer,
    and_,
    delete,
    func,
    literal,
    literal_column,
    or_,
    select,
    text,
    tuple_,
    update,
)
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import aliased

from src.core.enums import (
    StudioDocumentKind,
    StudioMemoryOrigin,
    StudioMemoryStatus,
    StudioRunStage,
    StudioRunStatus,
)
from src.core.uid import new_id
from src.db.models.library import LibraryProject
from src.db.models.studio import (
    StudioArtist,
    StudioArtistMemory,
    StudioDocumentVersion,
    StudioProjectProfile,
    StudioRun,
    StudioTrack,
)

if TYPE_CHECKING:
    from collections.abc import Collection, Mapping, Sequence
    from datetime import date, datetime
    from uuid import UUID

    from sqlalchemy import ColumnElement, ScalarSelect, Select
    from sqlalchemy.ext.asyncio import AsyncSession
    from sqlalchemy.orm import InstrumentedAttribute

__all__ = [
    "SIBLING_CONSTRAINT",
    "StudioArtistMemoryRepository",
    "StudioArtistRepository",
    "StudioDocumentVersionRepository",
    "StudioProjectRepository",
    "StudioRunRepository",
    "StudioTrackRepository",
    "VersionDeletionBlockers",
]

SIBLING_CONSTRAINT = "uq_studio_doc_versions_sibling_content"
"""Name of the UNIQUE NULLS NOT DISTINCT constraint that makes version create idempotent."""

_FRESH: Mapping[str, Any] = {"populate_existing": True}

_RUN_QUOTA_LOCK_NAMESPACE = 0x57_0D_10
"""Advisory-lock namespace of the per-user run quota (first key of ``pg_advisory_xact_lock``)."""


def _keyset[*Ts](
    query: Select[*Ts],
    created_at: InstrumentedAttribute[datetime],
    id_: InstrumentedAttribute[UUID],
    cursor_ts: datetime | None,
    cursor_id: UUID | None,
    limit: int,
) -> Select[*Ts]:
    """Newest-first keyset page: ``limit + 1`` rows after the cursor (``has_more`` = len > limit)."""
    if cursor_ts is not None and cursor_id is not None:
        query = query.where(
            tuple_(created_at, id_) < tuple_(literal(cursor_ts), literal(cursor_id))
        )
    return query.order_by(created_at.desc(), id_.desc()).limit(limit + 1)


class StudioArtistRepository:
    """Data access for ``studio_artists``."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def insert(
        self,
        *,
        user_id: UUID,
        product_id: str,
        name: str,
        persona: str | None,
        voice_notes: str | None,
        default_language: str | None,
        aesthetic_references: Sequence[str],
        style_tags: Sequence[str],
        global_avoid: Sequence[str],
    ) -> StudioArtist:
        """Insert an artist.

        Raises:
            sqlalchemy.exc.IntegrityError: ``name`` collides case-insensitively for this owner;
                call inside ``session.begin_nested()``.
        """
        result = await self._session.execute(
            pg_insert(StudioArtist)
            .values(
                id=new_id(),
                user_id=user_id,
                product_id=product_id,
                name=name,
                persona=persona,
                voice_notes=voice_notes,
                default_language=default_language,
                aesthetic_references=list(aesthetic_references),
                style_tags=list(style_tags),
                global_avoid=list(global_avoid),
            )
            .returning(StudioArtist)
        )
        return result.scalar_one()

    async def get(self, artist_id: UUID, *, user_id: UUID, product_id: str) -> StudioArtist | None:
        result = await self._session.execute(
            select(StudioArtist)
            .where(
                StudioArtist.id == artist_id,
                StudioArtist.user_id == user_id,
                StudioArtist.product_id == product_id,
            )
            .execution_options(**_FRESH)
        )
        return result.scalar_one_or_none()

    async def lock(self, artist_id: UUID, *, user_id: UUID, product_id: str) -> StudioArtist | None:
        """``SELECT … FOR UPDATE`` the artist: serializes read-modify-write PATCHes.

        ``populate_existing`` matters: without it the identity map could hand back a stale
        artist that an earlier read in the same session loaded, defeating the lock.
        """
        result = await self._session.execute(
            select(StudioArtist)
            .where(
                StudioArtist.id == artist_id,
                StudioArtist.user_id == user_id,
                StudioArtist.product_id == product_id,
            )
            .with_for_update()
            .execution_options(**_FRESH)
        )
        return result.scalar_one_or_none()

    async def list_page(
        self,
        *,
        user_id: UUID,
        product_id: str,
        limit: int,
        cursor_ts: datetime | None = None,
        cursor_id: UUID | None = None,
    ) -> Sequence[StudioArtist]:
        query = select(StudioArtist).where(
            StudioArtist.user_id == user_id, StudioArtist.product_id == product_id
        )
        result = await self._session.execute(
            _keyset(query, StudioArtist.created_at, StudioArtist.id, cursor_ts, cursor_id, limit)
        )
        return result.scalars().all()

    async def count_for_owner(self, *, user_id: UUID, product_id: str) -> int:
        result = await self._session.execute(
            select(func.count())
            .select_from(StudioArtist)
            .where(StudioArtist.user_id == user_id, StudioArtist.product_id == product_id)
        )
        return result.scalar_one()

    async def update(
        self,
        artist_id: UUID,
        *,
        user_id: UUID,
        product_id: str,
        changes: Mapping[str, object],
    ) -> StudioArtist | None:
        """Apply ``changes`` (column name → value). An empty mapping just re-reads the row.

        Raises:
            sqlalchemy.exc.IntegrityError: A renamed artist collides case-insensitively; call
                inside ``session.begin_nested()``.
        """
        if not changes:
            return await self.get(artist_id, user_id=user_id, product_id=product_id)
        result = await self._session.execute(
            update(StudioArtist)
            .where(
                StudioArtist.id == artist_id,
                StudioArtist.user_id == user_id,
                StudioArtist.product_id == product_id,
            )
            .values(**changes)
            .returning(StudioArtist)
            .execution_options(**_FRESH, synchronize_session=False)
        )
        return result.scalar_one_or_none()

    async def delete(self, artist_id: UUID, *, user_id: UUID, product_id: str) -> bool:
        """Delete an artist.

        Raises:
            sqlalchemy.exc.IntegrityError: A profile still uses it (NO ACTION FK); call inside
                ``session.begin_nested()``.
        """
        result = await self._session.execute(
            delete(StudioArtist)
            .where(
                StudioArtist.id == artist_id,
                StudioArtist.user_id == user_id,
                StudioArtist.product_id == product_id,
            )
            .returning(StudioArtist.id)
        )
        return result.scalar_one_or_none() is not None

    async def count_profiles_using(self, artist_id: UUID, *, user_id: UUID, product_id: str) -> int:
        result = await self._session.execute(
            select(func.count())
            .select_from(StudioProjectProfile)
            .where(
                StudioProjectProfile.artist_id == artist_id,
                StudioProjectProfile.user_id == user_id,
                StudioProjectProfile.product_id == product_id,
            )
        )
        return result.scalar_one()


class StudioArtistMemoryRepository:
    """Data access for ``studio_artist_memories``.

    A memory carries its own ``user_id`` / ``product_id`` (the database ties them to its artist's
    with a composite FK), and **every** method filters them on the memory row itself - by-id reads
    and by-artist reads alike. A caller that passes an artist id it did not resolve through its own
    scope gets nothing back (``None`` / empty / ``0``) instead of another tenant's rows.

    Lock order: **artist, then memory** (never the reverse). ``lock`` takes the memory row only
    (``FOR UPDATE OF`` the memory table): the artist lock is the caller's, taken first.
    """

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    @staticmethod
    def _scope(user_id: UUID, product_id: str) -> tuple[Any, ...]:
        """Where-clauses: the memory belongs to ``user_id`` in ``product_id``."""
        return (
            StudioArtistMemory.user_id == user_id,
            StudioArtistMemory.product_id == product_id,
        )

    async def insert(
        self,
        *,
        user_id: UUID,
        product_id: str,
        artist_id: UUID,
        title: str,
        body: str,
        occurred_on: date | None,
        themes: Sequence[str],
        imagery: Sequence[str],
        salience: int,
        origin: StudioMemoryOrigin,
    ) -> StudioArtistMemory:
        """Insert an ``active`` memory. The caller holds the artist lock and has counted.

        ``user_id`` / ``product_id`` are the **locked artist row's** values; the composite FK
        refuses a pair that is not the artist's.
        """
        result = await self._session.execute(
            pg_insert(StudioArtistMemory)
            .values(
                id=new_id(),
                user_id=user_id,
                product_id=product_id,
                artist_id=artist_id,
                title=title,
                body=body,
                occurred_on=occurred_on,
                themes=list(themes),
                imagery=list(imagery),
                salience=salience,
                status=StudioMemoryStatus.ACTIVE.value,
                origin=origin.value,
            )
            .returning(StudioArtistMemory)
        )
        return result.scalar_one()

    async def get(
        self, memory_id: UUID, *, user_id: UUID, product_id: str
    ) -> StudioArtistMemory | None:
        """The memory, if it is the caller's (``None`` otherwise: no probing)."""
        result = await self._session.execute(
            select(StudioArtistMemory)
            .where(StudioArtistMemory.id == memory_id, *self._scope(user_id, product_id))
            .execution_options(**_FRESH)
        )
        return result.scalar_one_or_none()

    async def lock(
        self, memory_id: UUID, *, user_id: UUID, product_id: str
    ) -> StudioArtistMemory | None:
        """``SELECT … FOR UPDATE`` the memory row (not the artist's): the caller locked the artist."""
        result = await self._session.execute(
            select(StudioArtistMemory)
            .where(StudioArtistMemory.id == memory_id, *self._scope(user_id, product_id))
            .with_for_update(of=StudioArtistMemory)
            .execution_options(**_FRESH)
        )
        return result.scalar_one_or_none()

    async def list_page(
        self,
        artist_id: UUID,
        *,
        user_id: UUID,
        product_id: str,
        status: StudioMemoryStatus,
        limit: int,
        cursor_occurred_on: date | None = None,
        cursor_ts: datetime | None = None,
        cursor_id: UUID | None = None,
    ) -> Sequence[StudioArtistMemory]:
        """``limit + 1`` memories of ``status`` in **diary order**, after the cursor.

        Order: ``occurred_on DESC NULLS LAST, created_at DESC, id DESC`` (the episode date, undated
        last, then the newest written). The cursor is ``(cursor_occurred_on, cursor_ts,
        cursor_id)`` of the last row of the previous page; there is one only when ``cursor_ts`` and
        ``cursor_id`` are both given (``cursor_occurred_on`` is ``None`` for an undated row).
        """
        query = select(StudioArtistMemory).where(
            StudioArtistMemory.artist_id == artist_id,
            StudioArtistMemory.status == status.value,
            *self._scope(user_id, product_id),
        )
        if cursor_ts is not None and cursor_id is not None:
            query = query.where(_after_in_diary_order(cursor_occurred_on, cursor_ts, cursor_id))
        result = await self._session.execute(
            query.order_by(
                StudioArtistMemory.occurred_on.desc().nulls_last(),
                StudioArtistMemory.created_at.desc(),
                StudioArtistMemory.id.desc(),
            ).limit(limit + 1)
        )
        return result.scalars().all()

    async def count_active(self, artist_id: UUID, *, user_id: UUID, product_id: str) -> int:
        """Active memories of the artist. Callers that enforce the cap hold the artist lock."""
        result = await self._session.execute(
            select(func.count())
            .select_from(StudioArtistMemory)
            .where(
                StudioArtistMemory.artist_id == artist_id,
                StudioArtistMemory.status == StudioMemoryStatus.ACTIVE.value,
                *self._scope(user_id, product_id),
            )
        )
        return result.scalar_one()

    async def update(
        self,
        memory_id: UUID,
        *,
        artist_id: UUID,
        user_id: UUID,
        product_id: str,
        changes: Mapping[str, object],
    ) -> StudioArtistMemory | None:
        """Apply ``changes`` (column name → value) to a memory of ``artist_id``.

        An empty mapping just re-reads the row. A scope that is not the row's changes nothing.
        """
        where = (
            StudioArtistMemory.id == memory_id,
            StudioArtistMemory.artist_id == artist_id,
            *self._scope(user_id, product_id),
        )
        if not changes:
            result = await self._session.execute(
                select(StudioArtistMemory).where(*where).execution_options(**_FRESH)
            )
            return result.scalar_one_or_none()
        result = await self._session.execute(
            update(StudioArtistMemory)
            .where(*where)
            .values(**changes)
            .returning(StudioArtistMemory)
            .execution_options(**_FRESH, synchronize_session=False)
        )
        return result.scalar_one_or_none()

    async def delete(self, memory_id: UUID, *, user_id: UUID, product_id: str) -> UUID | None:
        """Delete a memory of the caller's; its ``artist_id``, or ``None`` if there was none.

        Always allowed: runs hold their own copy (P2-M5). The artist id is returned so the caller
        needs no prior read just to log it.
        """
        result = await self._session.execute(
            delete(StudioArtistMemory)
            .where(StudioArtistMemory.id == memory_id, *self._scope(user_id, product_id))
            .returning(StudioArtistMemory.artist_id)
        )
        return result.scalar_one_or_none()

    async def get_many_owned(
        self,
        memory_ids: Collection[UUID],
        *,
        artist_id: UUID,
        user_id: UUID,
        product_id: str,
    ) -> Sequence[StudioArtistMemory]:
        """The memories among ``memory_ids`` that belong to ``artist_id`` (any status).

        An id that is unknown, or another artist's, is simply absent from the result - the caller
        cannot tell the two apart, which is the point.
        """
        if not memory_ids:
            return []
        result = await self._session.execute(
            select(StudioArtistMemory)
            .where(
                StudioArtistMemory.artist_id == artist_id,
                StudioArtistMemory.id.in_(list(memory_ids)),
                *self._scope(user_id, product_id),
            )
            .execution_options(**_FRESH)
        )
        return result.scalars().all()

    async def active_in_selection_order(
        self,
        artist_id: UUID,
        *,
        user_id: UUID,
        product_id: str,
        exclude: Collection[UUID],
        limit: int,
    ) -> Sequence[StudioArtistMemory]:
        """Active memories in the P2-M2 order: ``salience DESC, occurred_on DESC NULLS LAST, id``."""
        query = select(StudioArtistMemory).where(
            StudioArtistMemory.artist_id == artist_id,
            StudioArtistMemory.status == StudioMemoryStatus.ACTIVE.value,
            *self._scope(user_id, product_id),
        )
        if exclude:
            query = query.where(StudioArtistMemory.id.not_in(list(exclude)))
        result = await self._session.execute(
            query.order_by(
                StudioArtistMemory.salience.desc(),
                StudioArtistMemory.occurred_on.desc().nulls_last(),
                StudioArtistMemory.id,
            )
            .limit(limit)
            .execution_options(**_FRESH)
        )
        return result.scalars().all()


def _after_in_diary_order(
    occurred_on: date | None, created_at: datetime, id_: UUID
) -> ColumnElement[bool]:
    """The rows strictly after ``(occurred_on, created_at, id)`` in diary order.

    The order is ``occurred_on DESC NULLS LAST, created_at DESC, id DESC``, so with a dated cursor
    the later rows are: an earlier date; the same date and an earlier ``(created_at, id)``; and
    every undated row. With an undated cursor only undated rows follow, by ``(created_at, id)``.
    """
    column = StudioArtistMemory.occurred_on
    older_in_day = tuple_(StudioArtistMemory.created_at, StudioArtistMemory.id) < tuple_(
        literal(created_at), literal(id_)
    )
    if occurred_on is None:
        return and_(column.is_(None), older_in_day)
    return or_(column < occurred_on, and_(column == occurred_on, older_in_day), column.is_(None))


@dataclass(frozen=True, slots=True)
class ProjectListRow:
    """A profile with its library project's name and its track count."""

    profile: StudioProjectProfile
    name: str
    track_count: int


class StudioProjectRepository:
    """Data access for ``studio_project_profiles``."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def upsert(
        self,
        library_project_id: UUID,
        *,
        user_id: UUID,
        product_id: str,
        artist_id: UUID | None,
        release_kind: str,
        concept: str | None,
    ) -> tuple[StudioProjectProfile, bool]:
        """Create the profile or replace its editable fields, atomically.

        The caller has already verified that the library project exists and is owned.

        Returns:
            ``(profile, created)``.
        """
        stmt = pg_insert(StudioProjectProfile).values(
            library_project_id=library_project_id,
            user_id=user_id,
            product_id=product_id,
            artist_id=artist_id,
            release_kind=release_kind,
            concept=concept,
        )
        upsert = stmt.on_conflict_do_update(
            index_elements=[StudioProjectProfile.library_project_id],
            set_={
                "artist_id": stmt.excluded.artist_id,
                "release_kind": stmt.excluded.release_kind,
                "concept": stmt.excluded.concept,
                "updated_at": func.now(),
            },
            # A row can only be conflicted by its own owner (the library project was checked), but
            # never rewrite another owner's profile if that invariant were ever broken.
            where=(StudioProjectProfile.user_id == user_id)
            & (StudioProjectProfile.product_id == product_id),
        ).returning(StudioProjectProfile, literal_column("(xmax = 0)", Boolean).label("created"))
        row = (
            await self._session.execute(
                upsert.execution_options(**_FRESH, synchronize_session=False)
            )
        ).one()
        return row[0], bool(row[1])

    async def get(
        self, library_project_id: UUID, *, user_id: UUID, product_id: str
    ) -> StudioProjectProfile | None:
        result = await self._session.execute(
            select(StudioProjectProfile)
            .where(
                StudioProjectProfile.library_project_id == library_project_id,
                StudioProjectProfile.user_id == user_id,
                StudioProjectProfile.product_id == product_id,
            )
            .execution_options(**_FRESH)
        )
        return result.scalar_one_or_none()

    async def lock(
        self, library_project_id: UUID, *, user_id: UUID, product_id: str
    ) -> StudioProjectProfile | None:
        """``SELECT … FOR UPDATE`` the profile: serializes track create / reorder / delete."""
        result = await self._session.execute(
            select(StudioProjectProfile)
            .where(
                StudioProjectProfile.library_project_id == library_project_id,
                StudioProjectProfile.user_id == user_id,
                StudioProjectProfile.product_id == product_id,
            )
            .with_for_update()
            .execution_options(**_FRESH)
        )
        return result.scalar_one_or_none()

    def _track_count(self) -> ScalarSelect[int]:
        return (
            select(func.count())
            .select_from(StudioTrack)
            .where(StudioTrack.library_project_id == StudioProjectProfile.library_project_id)
            .correlate(StudioProjectProfile)
            .scalar_subquery()
        )

    async def get_row(
        self, library_project_id: UUID, *, user_id: UUID, product_id: str
    ) -> ProjectListRow | None:
        result = await self._session.execute(
            select(StudioProjectProfile, LibraryProject.name, self._track_count())
            .join(LibraryProject, LibraryProject.id == StudioProjectProfile.library_project_id)
            .where(
                StudioProjectProfile.library_project_id == library_project_id,
                StudioProjectProfile.user_id == user_id,
                StudioProjectProfile.product_id == product_id,
            )
            .execution_options(**_FRESH)
        )
        row = result.one_or_none()
        return None if row is None else ProjectListRow(row[0], row[1], row[2])

    async def list_page(
        self,
        *,
        user_id: UUID,
        product_id: str,
        limit: int,
        cursor_ts: datetime | None = None,
        cursor_id: UUID | None = None,
    ) -> Sequence[ProjectListRow]:
        query = (
            select(StudioProjectProfile, LibraryProject.name, self._track_count())
            .join(LibraryProject, LibraryProject.id == StudioProjectProfile.library_project_id)
            .where(
                StudioProjectProfile.user_id == user_id,
                StudioProjectProfile.product_id == product_id,
            )
        )
        result = await self._session.execute(
            _keyset(
                query,
                StudioProjectProfile.created_at,
                StudioProjectProfile.library_project_id,
                cursor_ts,
                cursor_id,
                limit,
            )
        )
        return [ProjectListRow(row[0], row[1], row[2]) for row in result.all()]

    async def delete(self, library_project_id: UUID, *, user_id: UUID, product_id: str) -> bool:
        """Delete the profile (the caller verified, under the profile lock, there are no tracks)."""
        result = await self._session.execute(
            delete(StudioProjectProfile)
            .where(
                StudioProjectProfile.library_project_id == library_project_id,
                StudioProjectProfile.user_id == user_id,
                StudioProjectProfile.product_id == product_id,
            )
            .returning(StudioProjectProfile.library_project_id)
        )
        return result.scalar_one_or_none() is not None


class StudioTrackRepository:
    """Data access for ``studio_tracks``."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    def _owned(self, track_id: UUID, user_id: UUID, product_id: str) -> tuple[Any, ...]:
        return (
            StudioTrack.id == track_id,
            StudioTrack.user_id == user_id,
            StudioTrack.product_id == product_id,
        )

    async def insert(
        self,
        *,
        user_id: UUID,
        product_id: str,
        library_project_id: UUID,
        position: int,
        title: str,
    ) -> StudioTrack:
        result = await self._session.execute(
            pg_insert(StudioTrack)
            .values(
                id=new_id(),
                user_id=user_id,
                product_id=product_id,
                library_project_id=library_project_id,
                position=position,
                title=title,
            )
            .returning(StudioTrack)
        )
        return result.scalar_one()

    async def get(self, track_id: UUID, *, user_id: UUID, product_id: str) -> StudioTrack | None:
        result = await self._session.execute(
            select(StudioTrack)
            .where(*self._owned(track_id, user_id, product_id))
            .execution_options(**_FRESH)
        )
        return result.scalar_one_or_none()

    async def get_with_current_versions(
        self, track_id: UUID, *, user_id: UUID, product_id: str
    ) -> tuple[StudioTrack, StudioDocumentVersion | None, StudioDocumentVersion | None] | None:
        """The track and the rows its current pointers name, read in ONE statement.

        One statement is one ``READ COMMITTED`` snapshot, so the pointers and the rows always
        agree: a concurrent "switch current + delete the old version" can never leave a pointer
        read here naming a row that is gone. No lock is taken (the lock order is untouched).

        Returns:
            ``(track, current plan row, current lyrics row)``; a row is ``None`` exactly when its
            pointer is ``None`` (the foreign keys make a dangling pointer impossible, so a caller
            that sees a non-null pointer with a ``None`` row has found a broken invariant).
            ``None`` when the track is not the caller's.
        """
        plan = aliased(StudioDocumentVersion)
        lyrics = aliased(StudioDocumentVersion)
        result = await self._session.execute(
            select(StudioTrack, plan, lyrics)
            .outerjoin(plan, plan.id == StudioTrack.current_plan_version_id)
            .outerjoin(lyrics, lyrics.id == StudioTrack.current_lyrics_version_id)
            .where(*self._owned(track_id, user_id, product_id))
            .execution_options(**_FRESH)
        )
        row = result.one_or_none()
        return None if row is None else (row[0], row[1], row[2])

    async def lock(self, track_id: UUID, *, user_id: UUID, product_id: str) -> StudioTrack | None:
        """``SELECT … FOR UPDATE`` the track: serializes version numbers and pointer changes."""
        result = await self._session.execute(
            select(StudioTrack)
            .where(*self._owned(track_id, user_id, product_id))
            .with_for_update()
            .execution_options(**_FRESH)
        )
        return result.scalar_one_or_none()

    async def list_for_project(
        self, library_project_id: UUID, *, user_id: UUID, product_id: str
    ) -> Sequence[StudioTrack]:
        result = await self._session.execute(
            select(StudioTrack)
            .where(
                StudioTrack.library_project_id == library_project_id,
                StudioTrack.user_id == user_id,
                StudioTrack.product_id == product_id,
            )
            .order_by(StudioTrack.position)
            .execution_options(**_FRESH)
        )
        return result.scalars().all()

    async def count_for_project(
        self, library_project_id: UUID, *, user_id: UUID, product_id: str
    ) -> int:
        result = await self._session.execute(
            select(func.count())
            .select_from(StudioTrack)
            .where(
                StudioTrack.library_project_id == library_project_id,
                StudioTrack.user_id == user_id,
                StudioTrack.product_id == product_id,
            )
        )
        return result.scalar_one()

    async def next_position(self, library_project_id: UUID) -> int:
        """Next free position (``max + 1``, or 0). The caller holds the profile lock."""
        result = await self._session.execute(
            select(func.coalesce(func.max(StudioTrack.position) + 1, 0)).where(
                StudioTrack.library_project_id == library_project_id
            )
        )
        return result.scalar_one()

    async def reorder(
        self,
        library_project_id: UUID,
        ordered_ids: Sequence[UUID],
        *,
        user_id: UUID,
        product_id: str,
    ) -> None:
        """Set positions ``0..n-1`` in the given order with one ``UPDATE … FROM unnest(…)``.

        The ``(project, position)`` unique is deferred, so a swap is legal inside the statement.
        The caller holds the profile lock and has verified ``ordered_ids`` is exactly the
        project's track set.
        """
        pairs = (
            func.unnest(
                literal(list(ordered_ids), ARRAY(PG_UUID(as_uuid=True))),
                literal(list(range(len(ordered_ids))), ARRAY(Integer)),
            )
            .table_valued("track_id", "new_position")
            .render_derived(name="pairs")
        )
        await self._session.execute(
            update(StudioTrack)
            .where(
                StudioTrack.id == pairs.c.track_id,
                StudioTrack.library_project_id == library_project_id,
                StudioTrack.user_id == user_id,
                StudioTrack.product_id == product_id,
            )
            .values(position=pairs.c.new_position)
            .execution_options(synchronize_session=False)
        )

    async def update_title(
        self, track_id: UUID, *, user_id: UUID, product_id: str, title: str
    ) -> StudioTrack | None:
        result = await self._session.execute(
            update(StudioTrack)
            .where(*self._owned(track_id, user_id, product_id))
            .values(title=title)
            .returning(StudioTrack)
            .execution_options(**_FRESH, synchronize_session=False)
        )
        return result.scalar_one_or_none()

    async def delete(self, track_id: UUID, *, user_id: UUID, product_id: str) -> bool:
        """Delete a track; its versions go with it (``ON DELETE CASCADE``) in the same statement."""
        result = await self._session.execute(
            delete(StudioTrack)
            .where(*self._owned(track_id, user_id, product_id))
            .returning(StudioTrack.id)
        )
        return result.scalar_one_or_none() is not None

    async def allocate_version_number(self, track_id: UUID, kind: StudioDocumentKind) -> int:
        """Take the next version number of ``kind`` from the track's counter.

        Numbers are never reused (not even after a delete). The caller holds the track lock and
        calls this only when a row is actually going to be inserted.
        """
        column = (
            StudioTrack.next_plan_version_number
            if kind is StudioDocumentKind.PLAN
            else StudioTrack.next_lyrics_version_number
        )
        result = await self._session.execute(
            update(StudioTrack)
            .where(StudioTrack.id == track_id)
            .values({column.key: column + 1})
            .returning((column - 1).label("allocated"))
            .execution_options(synchronize_session=False)
        )
        return result.scalar_one()

    async def set_current(
        self,
        track_id: UUID,
        *,
        plan_version_id: UUID | msgspec.UnsetType | None = msgspec.UNSET,
        lyrics_version_id: UUID | msgspec.UnsetType | None = msgspec.UNSET,
    ) -> StudioTrack:
        """Update the current pointers in one statement (unset arguments are left alone).

        The caller holds the track lock and has verified kind and compatibility; the database
        enforces same-track (composite FK) and "lyrics need a plan" (CHECK).
        """
        values: dict[str, UUID | None] = {}
        if not isinstance(plan_version_id, msgspec.UnsetType):
            values["current_plan_version_id"] = plan_version_id
        if not isinstance(lyrics_version_id, msgspec.UnsetType):
            values["current_lyrics_version_id"] = lyrics_version_id
        result = await self._session.execute(
            update(StudioTrack)
            .where(StudioTrack.id == track_id)
            .values(**values)
            .returning(StudioTrack)
            .execution_options(**_FRESH, synchronize_session=False)
        )
        return result.scalar_one()


@dataclass(frozen=True, slots=True)
class VersionDeletionBlockers:
    """What keeps a version from being deleted (both zero → deletable if not current)."""

    child_count: int
    lyrics_base_count: int


class StudioDocumentVersionRepository:
    """Data access for ``studio_document_versions`` (rows are immutable; only delete mutates)."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def get(
        self, version_id: UUID, *, user_id: UUID, product_id: str
    ) -> StudioDocumentVersion | None:
        result = await self._session.execute(
            select(StudioDocumentVersion).where(
                StudioDocumentVersion.id == version_id,
                StudioDocumentVersion.user_id == user_id,
                StudioDocumentVersion.product_id == product_id,
            )
        )
        return result.scalar_one_or_none()

    async def get_on_track(
        self, version_id: UUID, track_id: UUID, *, user_id: UUID, product_id: str
    ) -> StudioDocumentVersion | None:
        result = await self._session.execute(
            select(StudioDocumentVersion).where(
                StudioDocumentVersion.id == version_id,
                StudioDocumentVersion.track_id == track_id,
                StudioDocumentVersion.user_id == user_id,
                StudioDocumentVersion.product_id == product_id,
            )
        )
        return result.scalar_one_or_none()

    async def count(self, track_id: UUID, kind: StudioDocumentKind) -> int:
        result = await self._session.execute(
            select(func.count())
            .select_from(StudioDocumentVersion)
            .where(
                StudioDocumentVersion.track_id == track_id,
                StudioDocumentVersion.kind == kind.value,
            )
        )
        return result.scalar_one()

    async def find_sibling(
        self,
        *,
        track_id: UUID,
        kind: StudioDocumentKind,
        parent_version_id: UUID | None,
        content_hash: str,
    ) -> StudioDocumentVersion | None:
        """The existing version with the same ``(track, kind, parent, content_hash)``, if any."""
        result = await self._session.execute(
            select(StudioDocumentVersion).where(
                StudioDocumentVersion.track_id == track_id,
                StudioDocumentVersion.kind == kind.value,
                StudioDocumentVersion.parent_version_id.is_not_distinct_from(parent_version_id),
                StudioDocumentVersion.content_hash == content_hash,
            )
        )
        return result.scalar_one_or_none()

    async def insert_or_get_sibling(
        self,
        *,
        user_id: UUID,
        product_id: str,
        track_id: UUID,
        kind: StudioDocumentKind,
        version_number: int,
        parent_version_id: UUID | None,
        based_on_plan_version_id: UUID | None,
        schema_version: int,
        origin: str,
        content: Mapping[str, object],
        content_hash: str,
    ) -> tuple[StudioDocumentVersion, bool]:
        """``INSERT … ON CONFLICT ON CONSTRAINT <sibling> DO NOTHING RETURNING``; else read it.

        Returns:
            ``(version, created)`` — ``created`` is ``False`` when an identical sibling existed.
        """
        result = await self._session.execute(
            pg_insert(StudioDocumentVersion)
            .values(
                id=new_id(),
                user_id=user_id,
                product_id=product_id,
                track_id=track_id,
                kind=kind.value,
                version_number=version_number,
                parent_version_id=parent_version_id,
                based_on_plan_version_id=based_on_plan_version_id,
                schema_version=schema_version,
                origin=origin,
                content=dict(content),
                content_hash=content_hash,
            )
            .on_conflict_do_nothing(constraint=SIBLING_CONSTRAINT)
            .returning(StudioDocumentVersion)
        )
        inserted = result.scalar_one_or_none()
        if inserted is not None:
            return inserted, True
        existing = await self.find_sibling(
            track_id=track_id,
            kind=kind,
            parent_version_id=parent_version_id,
            content_hash=content_hash,
        )
        if existing is None:  # pragma: no cover - a conflict row vanished mid-transaction
            raise RuntimeError("studio version sibling conflicted but cannot be read back")
        return existing, False

    async def list_page(
        self,
        track_id: UUID,
        kind: StudioDocumentKind | None,
        *,
        user_id: UUID,
        product_id: str,
        limit: int,
        cursor_ts: datetime | None = None,
        cursor_id: UUID | None = None,
    ) -> Sequence[StudioDocumentVersion]:
        query = select(StudioDocumentVersion).where(
            StudioDocumentVersion.track_id == track_id,
            StudioDocumentVersion.user_id == user_id,
            StudioDocumentVersion.product_id == product_id,
        )
        if kind is not None:
            query = query.where(StudioDocumentVersion.kind == kind.value)
        result = await self._session.execute(
            _keyset(
                query,
                StudioDocumentVersion.created_at,
                StudioDocumentVersion.id,
                cursor_ts,
                cursor_id,
                limit,
            )
        )
        return result.scalars().all()

    async def deletion_blockers(
        self, version_id: UUID, *, track_id: UUID, kind: StudioDocumentKind
    ) -> VersionDeletionBlockers:
        """Children (versions derived from it) and lyrics versions based on it — one query."""
        children = (
            select(func.count())
            .select_from(StudioDocumentVersion)
            .where(
                StudioDocumentVersion.track_id == track_id,
                StudioDocumentVersion.kind == kind.value,
                StudioDocumentVersion.parent_version_id == version_id,
            )
            .scalar_subquery()
        )
        lyrics_based = (
            select(func.count())
            .select_from(StudioDocumentVersion)
            .where(
                StudioDocumentVersion.track_id == track_id,
                StudioDocumentVersion.based_on_plan_version_id == version_id,
            )
            .scalar_subquery()
        )
        row = (await self._session.execute(select(children, lyrics_based))).one()
        return VersionDeletionBlockers(child_count=row[0], lyrics_base_count=row[1])

    async def delete(self, version_id: UUID, *, user_id: UUID, product_id: str) -> bool:
        """Delete a version.

        Raises:
            sqlalchemy.exc.IntegrityError: It became current, a parent or a lyrics base after the
                caller checked (NO ACTION FKs); call inside ``session.begin_nested()``.
        """
        result = await self._session.execute(
            delete(StudioDocumentVersion)
            .where(
                StudioDocumentVersion.id == version_id,
                StudioDocumentVersion.user_id == user_id,
                StudioDocumentVersion.product_id == product_id,
            )
            .returning(StudioDocumentVersion.id)
        )
        return result.scalar_one_or_none() is not None


class StudioRunRepository:
    """Data access for ``studio_runs``.

    Every state change is one guarded ``UPDATE … WHERE status = <expected> RETURNING`` that bumps
    ``revision``; a method that returns ``None`` lost the race (the run is already past the state
    the caller expected), which is how a run stays *terminal once*. No method commits or flushes.
    Lock order across the studio: track row, then run rows.
    """

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    @staticmethod
    def _owned(run_id: UUID, user_id: UUID, product_id: str) -> tuple[Any, ...]:
        return (
            StudioRun.id == run_id,
            StudioRun.user_id == user_id,
            StudioRun.product_id == product_id,
        )

    # -- create / read -----------------------------------------------------------------------

    async def insert(
        self,
        *,
        run_id: UUID,
        user_id: UUID,
        product_id: str,
        track_id: UUID,
        kind: str,
        idempotency_key: str,
        request_hash: str,
        request: Mapping[str, object],
        model_key: str,
        price_tokens: int,
        inputs: Mapping[str, object] | None,
    ) -> StudioRun:
        """Insert a ``queued`` run.

        ``inputs`` is what the run works on, resolved at submit: ``None`` for a ``plan`` run and
        required for every other kind (``ck_studio_runs_inputs_by_kind``). It has no default, so a
        caller cannot forget it.

        Raises:
            sqlalchemy.exc.IntegrityError: Another active run holds the track
                (``uq_studio_runs_one_active_per_track``), the idempotency key was taken
                (``uq_studio_runs_user_idempotency``) or ``inputs`` does not fit ``kind``; call
                inside ``session.begin_nested()``.
        """
        result = await self._session.execute(
            pg_insert(StudioRun)
            .values(
                id=run_id,
                user_id=user_id,
                product_id=product_id,
                track_id=track_id,
                kind=kind,
                status=StudioRunStatus.QUEUED.value,
                idempotency_key=idempotency_key,
                request_hash=request_hash,
                request=dict(request),
                inputs=dict(inputs) if inputs is not None else None,
                model_key=model_key,
                price_tokens=price_tokens,
            )
            .returning(StudioRun)
            .execution_options(**_FRESH)
        )
        return result.scalar_one()

    async def set_billing_account(self, run_id: UUID, account_id: UUID) -> None:
        """Record the account a run reserved on (set in the submit transaction, no revision bump)."""
        await self._session.execute(
            update(StudioRun)
            .where(StudioRun.id == run_id)
            .values(billing_account_id=account_id)
            .execution_options(synchronize_session=False)
        )

    async def get(self, run_id: UUID, *, user_id: UUID, product_id: str) -> StudioRun | None:
        result = await self._session.execute(
            select(StudioRun)
            .where(*self._owned(run_id, user_id, product_id))
            .execution_options(**_FRESH)
        )
        return result.scalar_one_or_none()

    async def get_by_key(
        self, *, user_id: UUID, product_id: str, idempotency_key: str
    ) -> StudioRun | None:
        result = await self._session.execute(
            select(StudioRun)
            .where(
                StudioRun.user_id == user_id,
                StudioRun.product_id == product_id,
                StudioRun.idempotency_key == idempotency_key,
            )
            .execution_options(**_FRESH)
        )
        return result.scalar_one_or_none()

    async def list_page(
        self,
        track_id: UUID,
        *,
        user_id: UUID,
        product_id: str,
        limit: int,
        cursor_ts: datetime | None,
        cursor_id: UUID | None,
    ) -> Sequence[StudioRun]:
        """Newest-first keyset page of a track's runs (``limit + 1`` rows)."""
        query = select(StudioRun).where(
            StudioRun.track_id == track_id,
            StudioRun.user_id == user_id,
            StudioRun.product_id == product_id,
        )
        result = await self._session.execute(
            _keyset(
                query, StudioRun.created_at, StudioRun.id, cursor_ts, cursor_id, limit
            ).execution_options(**_FRESH)
        )
        return result.scalars().all()

    # -- limits ------------------------------------------------------------------------------

    async def lock_user_quota(self, user_id: UUID) -> None:
        """Serialize the per-user limit checks of concurrent submits (released at commit).

        Taken after the track lock; nothing takes it before one, so it joins no cycle. A hash
        collision only serializes two unrelated users.
        """
        await self._session.execute(
            text("SELECT pg_advisory_xact_lock(:ns, hashtext(:uid))"),
            {"ns": _RUN_QUOTA_LOCK_NAMESPACE, "uid": str(user_id)},
        )

    async def count_created_in_last_hour(self, *, user_id: UUID, product_id: str) -> int:
        result = await self._session.execute(
            select(func.count())
            .select_from(StudioRun)
            .where(
                StudioRun.user_id == user_id,
                StudioRun.product_id == product_id,
                StudioRun.created_at > func.now() - timedelta(hours=1),
            )
        )
        return result.scalar_one()

    async def count_active(self, *, user_id: UUID, product_id: str) -> int:
        result = await self._session.execute(
            select(func.count())
            .select_from(StudioRun)
            .where(
                StudioRun.user_id == user_id,
                StudioRun.product_id == product_id,
                StudioRun.status.in_([s.value for s in StudioRunStatus.active()]),
            )
        )
        return result.scalar_one()

    async def has_active_for_track(self, track_id: UUID) -> bool:
        result = await self._session.execute(
            select(StudioRun.id)
            .where(
                StudioRun.track_id == track_id,
                StudioRun.status.in_([s.value for s in StudioRunStatus.active()]),
            )
            .limit(1)
        )
        return result.first() is not None

    async def active_run_uses_version(self, track_id: UUID, version_id: UUID) -> bool:
        """Whether an active run of the track takes ``version_id`` as its plan or lyrics input.

        At most one run is active per track (``uq_studio_runs_one_active_per_track``), so this reads
        one row. The caller holds the track lock, as ``submit`` does, so no new run can slip in
        between the answer and the action it guards.
        """
        needle = str(version_id)
        result = await self._session.execute(
            select(StudioRun.id)
            .where(
                StudioRun.track_id == track_id,
                StudioRun.status.in_([s.value for s in StudioRunStatus.active()]),
                or_(
                    StudioRun.inputs["plan_version_id"].astext == needle,
                    StudioRun.inputs["lyrics_version_id"].astext == needle,
                ),
            )
            .limit(1)
        )
        return result.first() is not None

    # -- transitions (all guarded; ``None`` = the run was no longer in the expected state) ---

    async def cancel_queued(
        self, run_id: UUID, *, user_id: UUID, product_id: str
    ) -> StudioRun | None:
        """``queued → cancelled``."""
        result = await self._session.execute(
            update(StudioRun)
            .where(
                *self._owned(run_id, user_id, product_id),
                StudioRun.status == StudioRunStatus.QUEUED.value,
            )
            .values(
                status=StudioRunStatus.CANCELLED.value,
                stage=None,
                finished_at=func.now(),
                revision=StudioRun.revision + 1,
            )
            .returning(StudioRun)
            .execution_options(**_FRESH, synchronize_session=False)
        )
        return result.scalar_one_or_none()

    async def claim_queued(self, limit: int) -> Sequence[StudioRun]:
        """Lock up to ``limit`` queued runs, oldest first, skipping rows another worker holds."""
        result = await self._session.execute(
            select(StudioRun)
            .where(StudioRun.status == StudioRunStatus.QUEUED.value)
            .order_by(StudioRun.created_at, StudioRun.id)
            .limit(limit)
            .with_for_update(skip_locked=True)
            .execution_options(**_FRESH)
        )
        return result.scalars().all()

    async def mark_running(
        self,
        run_id: UUID,
        *,
        lease_seconds: int,
        model_id: str,
        prompt_version: str,
        context_snapshot: Mapping[str, object],
    ) -> StudioRun | None:
        """``queued → running`` (stage ``context``), stamping the lease and the snapshot."""
        result = await self._session.execute(
            update(StudioRun)
            .where(StudioRun.id == run_id, StudioRun.status == StudioRunStatus.QUEUED.value)
            .values(
                status=StudioRunStatus.RUNNING.value,
                stage=StudioRunStage.CONTEXT.value,
                started_at=func.now(),
                lease_expires_at=func.now() + timedelta(seconds=lease_seconds),
                model_id=model_id,
                prompt_version=prompt_version,
                context_snapshot=dict(context_snapshot),
                revision=StudioRun.revision + 1,
            )
            .returning(StudioRun)
            .execution_options(**_FRESH, synchronize_session=False)
        )
        return result.scalar_one_or_none()

    async def set_stage(self, run_id: UUID, stage: StudioRunStage) -> StudioRun | None:
        """Move a ``running`` run to ``stage``; ``None`` when it is no longer running."""
        result = await self._session.execute(
            update(StudioRun)
            .where(StudioRun.id == run_id, StudioRun.status == StudioRunStatus.RUNNING.value)
            .values(stage=stage.value, revision=StudioRun.revision + 1)
            .returning(StudioRun)
            .execution_options(**_FRESH, synchronize_session=False)
        )
        return result.scalar_one_or_none()

    async def finish_succeeded(
        self,
        run_id: UUID,
        *,
        result_plan_version_id: UUID | None,
        warnings: Sequence[Mapping[str, object]],
        llm_calls: int,
        prompt_tokens: int,
        completion_tokens: int,
        result_lyrics_version_id: UUID | None,
    ) -> StudioRun | None:
        """``running → succeeded``; a plan run sets the plan result, the others the lyrics one.

        Both result ids are required (no default): exactly the kind's own is set
        (``ck_studio_runs_result_by_kind``).
        """
        result = await self._session.execute(
            update(StudioRun)
            .where(StudioRun.id == run_id, StudioRun.status == StudioRunStatus.RUNNING.value)
            .values(
                status=StudioRunStatus.SUCCEEDED.value,
                stage=None,
                result_plan_version_id=result_plan_version_id,
                result_lyrics_version_id=result_lyrics_version_id,
                warnings=[dict(w) for w in warnings],
                llm_calls=llm_calls,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                finished_at=func.now(),
                revision=StudioRun.revision + 1,
            )
            .returning(StudioRun)
            .execution_options(**_FRESH, synchronize_session=False)
        )
        return result.scalar_one_or_none()

    async def finish_failed(
        self,
        run_id: UUID,
        *,
        error_code: str,
        error_detail: Mapping[str, object] | None,
        llm_calls: int,
        prompt_tokens: int,
        completion_tokens: int,
    ) -> StudioRun | None:
        """``running → failed``."""
        result = await self._session.execute(
            update(StudioRun)
            .where(StudioRun.id == run_id, StudioRun.status == StudioRunStatus.RUNNING.value)
            .values(
                status=StudioRunStatus.FAILED.value,
                stage=None,
                error_code=error_code,
                error_detail=dict(error_detail) if error_detail is not None else None,
                llm_calls=llm_calls,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                finished_at=func.now(),
                revision=StudioRun.revision + 1,
            )
            .returning(StudioRun)
            .execution_options(**_FRESH, synchronize_session=False)
        )
        return result.scalar_one_or_none()

    async def fail_queued(self, run_id: UUID, *, error_code: str) -> StudioRun | None:
        """``queued → failed`` - a run the worker could not even start (context build failed)."""
        result = await self._session.execute(
            update(StudioRun)
            .where(StudioRun.id == run_id, StudioRun.status == StudioRunStatus.QUEUED.value)
            .values(
                status=StudioRunStatus.FAILED.value,
                stage=None,
                error_code=error_code,
                finished_at=func.now(),
                revision=StudioRun.revision + 1,
            )
            .returning(StudioRun)
            .execution_options(**_FRESH, synchronize_session=False)
        )
        return result.scalar_one_or_none()

    async def sweep_expired(self, *, error_code: str) -> Sequence[StudioRun]:
        """``running`` runs past their lease → ``failed`` (one statement; takes only run locks)."""
        result = await self._session.execute(
            update(StudioRun)
            .where(
                StudioRun.status == StudioRunStatus.RUNNING.value,
                StudioRun.lease_expires_at < func.now(),
            )
            .values(
                status=StudioRunStatus.FAILED.value,
                stage=None,
                error_code=error_code,
                finished_at=func.now(),
                revision=StudioRun.revision + 1,
            )
            .returning(StudioRun)
            .execution_options(**_FRESH, synchronize_session=False)
        )
        return result.scalars().all()
