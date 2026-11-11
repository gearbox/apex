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
from typing import TYPE_CHECKING, Any

import msgspec
from sqlalchemy import (
    Boolean,
    Integer,
    delete,
    func,
    literal,
    literal_column,
    select,
    tuple_,
    update,
)
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.dialects.postgresql import insert as pg_insert

from src.core.enums import StudioDocumentKind
from src.core.uid import new_id
from src.db.models.library import LibraryProject
from src.db.models.studio import (
    StudioArtist,
    StudioDocumentVersion,
    StudioProjectProfile,
    StudioTrack,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from datetime import datetime
    from uuid import UUID

    from sqlalchemy import ScalarSelect, Select
    from sqlalchemy.ext.asyncio import AsyncSession
    from sqlalchemy.orm import InstrumentedAttribute

__all__ = [
    "SIBLING_CONSTRAINT",
    "StudioArtistRepository",
    "StudioDocumentVersionRepository",
    "StudioProjectRepository",
    "StudioTrackRepository",
    "VersionDeletionBlockers",
]

SIBLING_CONSTRAINT = "uq_studio_doc_versions_sibling_content"
"""Name of the UNIQUE NULLS NOT DISTINCT constraint that makes version create idempotent."""

_FRESH: Mapping[str, Any] = {"populate_existing": True}


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

    async def get_by_ids(
        self, artist_ids: Sequence[UUID], *, user_id: UUID, product_id: str
    ) -> dict[UUID, StudioArtist]:
        if not artist_ids:
            return {}
        result = await self._session.execute(
            select(StudioArtist).where(
                StudioArtist.id.in_(artist_ids),
                StudioArtist.user_id == user_id,
                StudioArtist.product_id == product_id,
            )
        )
        return {artist.id: artist for artist in result.scalars()}

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
