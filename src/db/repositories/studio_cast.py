"""Repositories for the studio cast (S1-P3).

Conventions (those of ``studio.py``, P1-9):

* Every query filters ``product_id`` **and** ``user_id`` (the row's own scope), so a foreign or
  cross-product id is simply "not found" - the same 404 as an unknown one.
* Writes are Core ``insert/update/delete … RETURNING``; repositories never ``flush()`` or
  ``commit()``. Locking reads use ``populate_existing``.
* A reference's asset lives only on its hold row: the reads here **join** ``library_asset_holds``
  to learn it (the reference stores no copy). The join is the only place the cast touches the
  library's table; the cast never writes it (that is ``AssetHoldService``'s job).
* Callers that need atomicity lock the member first (``StudioCastMemberRepository.lock``); every
  member-level mutation does, which is what makes the reference and look caps hard.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from sqlalchemy import SmallInteger, case, cast, delete, exists, func, select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert

from src.core.library_ref import LibraryAssetSource
from src.core.uid import new_id
from src.db.models.library import LibraryAssetHold
from src.db.models.studio import StudioArtist
from src.db.models.studio_cast import (
    CAST_REFERENCE_HOLDER_KIND,
    StudioCastLook,
    StudioCastLookReference,
    StudioCastMember,
    StudioCastReference,
)
from src.db.repositories.studio import _FRESH, _keyset

if TYPE_CHECKING:
    from collections.abc import Collection, Mapping, Sequence
    from datetime import datetime
    from uuid import UUID

    from sqlalchemy import ColumnElement
    from sqlalchemy.ext.asyncio import AsyncSession

    from src.core.library_ref import AssetRef

__all__ = [
    "CastReferenceRow",
    "StudioCastLookRepository",
    "StudioCastMemberRepository",
    "StudioCastReferenceRepository",
]


@dataclass(frozen=True, slots=True)
class CastReferenceRow:
    """A reference together with the asset its hold pins (one join, no second copy)."""

    id: UUID
    cast_member_id: UUID
    label: str
    position: int
    hold_id: UUID
    source: LibraryAssetSource
    asset_id: UUID


def _reference_row(row: object) -> CastReferenceRow:
    """Build a row from a ``(reference columns…, hold.upload_id, hold.output_id)`` result row."""
    upload_id = row.upload_id  # type: ignore[attr-defined]
    return CastReferenceRow(
        id=row.id,  # type: ignore[attr-defined]
        cast_member_id=row.cast_member_id,  # type: ignore[attr-defined]
        label=row.label,  # type: ignore[attr-defined]
        position=row.position,  # type: ignore[attr-defined]
        hold_id=row.hold_id,  # type: ignore[attr-defined]
        source=LibraryAssetSource.UPLOAD if upload_id is not None else LibraryAssetSource.OUTPUT,
        asset_id=upload_id if upload_id is not None else row.output_id,  # type: ignore[attr-defined]
    )


_REFERENCE_COLUMNS = (
    StudioCastReference.id,
    StudioCastReference.cast_member_id,
    StudioCastReference.label,
    StudioCastReference.position,
    StudioCastReference.hold_id,
    LibraryAssetHold.upload_id,
    LibraryAssetHold.output_id,
)


class StudioCastMemberRepository:
    """Data access for ``studio_cast_members``."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    @staticmethod
    def _owned(member_id: UUID, user_id: UUID, product_id: str) -> tuple[ColumnElement[bool], ...]:
        return (
            StudioCastMember.id == member_id,
            StudioCastMember.user_id == user_id,
            StudioCastMember.product_id == product_id,
        )

    async def insert(
        self,
        *,
        user_id: UUID,
        product_id: str,
        name: str,
        description: str | None,
        appearance: str | None,
    ) -> StudioCastMember:
        """Insert a member.

        Raises:
            sqlalchemy.exc.IntegrityError: ``name`` collides case-insensitively for this owner;
                call inside ``session.begin_nested()``.
        """
        result = await self._session.execute(
            pg_insert(StudioCastMember)
            .values(
                id=new_id(),
                user_id=user_id,
                product_id=product_id,
                name=name,
                description=description,
                appearance=appearance,
            )
            .returning(StudioCastMember)
        )
        return result.scalar_one()

    async def get(
        self, member_id: UUID, *, user_id: UUID, product_id: str
    ) -> StudioCastMember | None:
        result = await self._session.execute(
            select(StudioCastMember)
            .where(*self._owned(member_id, user_id, product_id))
            .execution_options(**_FRESH)
        )
        return result.scalar_one_or_none()

    async def lock(
        self, member_id: UUID, *, user_id: UUID, product_id: str
    ) -> StudioCastMember | None:
        """``SELECT … FOR UPDATE`` the member: serializes every member-level mutation.

        Lock order: artist → cast member → reference / look → (assets, in the library).
        """
        result = await self._session.execute(
            select(StudioCastMember)
            .where(*self._owned(member_id, user_id, product_id))
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
        holding: AssetRef | None = None,
    ) -> Sequence[StudioCastMember]:
        """Newest-first keyset page; ``holding`` keeps the members that reference that asset.

        The filter is one join through ``library_asset_holds``: a reference counts only if its
        hold is *its own* (``holder_kind`` / ``holder_id``), so a hold of another context on the
        same asset never makes a member match.
        """
        query = select(StudioCastMember).where(
            StudioCastMember.user_id == user_id, StudioCastMember.product_id == product_id
        )
        if holding is not None:
            asset_column = (
                LibraryAssetHold.upload_id
                if holding.source == LibraryAssetSource.UPLOAD
                else LibraryAssetHold.output_id
            )
            query = query.where(
                exists().where(
                    StudioCastReference.cast_member_id == StudioCastMember.id,
                    LibraryAssetHold.id == StudioCastReference.hold_id,
                    LibraryAssetHold.holder_kind == CAST_REFERENCE_HOLDER_KIND,
                    LibraryAssetHold.holder_id == StudioCastReference.id,
                    LibraryAssetHold.user_id == user_id,
                    LibraryAssetHold.product_id == product_id,
                    asset_column == holding.asset_id,
                )
            )
        result = await self._session.execute(
            _keyset(
                query,
                StudioCastMember.created_at,
                StudioCastMember.id,
                cursor_ts,
                cursor_id,
                limit,
            )
        )
        return result.scalars().all()

    async def count_for_owner(self, *, user_id: UUID, product_id: str) -> int:
        result = await self._session.execute(
            select(func.count())
            .select_from(StudioCastMember)
            .where(StudioCastMember.user_id == user_id, StudioCastMember.product_id == product_id)
        )
        return result.scalar_one()

    async def update(
        self,
        member_id: UUID,
        *,
        user_id: UUID,
        product_id: str,
        changes: Mapping[str, object],
    ) -> StudioCastMember | None:
        """Apply ``changes`` (column name → value). An empty mapping just re-reads the row.

        Raises:
            sqlalchemy.exc.IntegrityError: A renamed member collides case-insensitively; call
                inside ``session.begin_nested()``.
        """
        if not changes:
            return await self.get(member_id, user_id=user_id, product_id=product_id)
        result = await self._session.execute(
            update(StudioCastMember)
            .where(*self._owned(member_id, user_id, product_id))
            .values(**changes)
            .returning(StudioCastMember)
            .execution_options(**_FRESH, synchronize_session=False)
        )
        return result.scalar_one_or_none()

    async def touch(self, member_id: UUID) -> None:
        """Bump ``updated_at``: a reference or a look of the member changed."""
        await self._session.execute(
            update(StudioCastMember)
            .where(StudioCastMember.id == member_id)
            .values(updated_at=func.now())
            .execution_options(synchronize_session=False)
        )

    async def delete(self, member_id: UUID, *, user_id: UUID, product_id: str) -> bool:
        """Delete a member; its references, looks and look links go with it (cascade).

        Raises:
            sqlalchemy.exc.IntegrityError: An artist still names it as performer
                (``fk_studio_artists_performer``, NO ACTION); call inside
                ``session.begin_nested()``.
        """
        result = await self._session.execute(
            delete(StudioCastMember)
            .where(*self._owned(member_id, user_id, product_id))
            .returning(StudioCastMember.id)
        )
        return result.scalar_one_or_none() is not None

    async def count_performed_artists(
        self, member_id: UUID, *, user_id: UUID, product_id: str
    ) -> int:
        """How many of the owner's artists name the member as their performer."""
        result = await self._session.execute(
            select(func.count())
            .select_from(StudioArtist)
            .where(
                StudioArtist.performer_cast_member_id == member_id,
                StudioArtist.user_id == user_id,
                StudioArtist.product_id == product_id,
            )
        )
        return result.scalar_one()

    async def exists_for_owner(self, member_id: UUID, *, user_id: UUID, product_id: str) -> bool:
        """Whether the caller has such a member (a plain read: the FK's KEY SHARE is the guard)."""
        result = await self._session.execute(
            select(exists().where(*self._owned(member_id, user_id, product_id)))
        )
        return bool(result.scalar_one())


class StudioCastReferenceRepository:
    """Data access for ``studio_cast_references`` (joined to the hold for the asset)."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    @staticmethod
    def _owned(
        reference_id: UUID, member_id: UUID, user_id: UUID, product_id: str
    ) -> tuple[ColumnElement[bool], ...]:
        return (
            StudioCastReference.id == reference_id,
            StudioCastReference.cast_member_id == member_id,
            StudioCastReference.user_id == user_id,
            StudioCastReference.product_id == product_id,
        )

    async def insert(
        self,
        *,
        reference_id: UUID,
        member_id: UUID,
        user_id: UUID,
        product_id: str,
        label: str,
        position: int,
        hold_id: UUID,
    ) -> None:
        """Insert a reference (its id was generated first so the hold could name it).

        Raises:
            sqlalchemy.exc.IntegrityError: ``label`` collides case-insensitively within the
                member; call inside ``session.begin_nested()``.
        """
        await self._session.execute(
            pg_insert(StudioCastReference).values(
                id=reference_id,
                cast_member_id=member_id,
                user_id=user_id,
                product_id=product_id,
                label=label,
                position=position,
                hold_id=hold_id,
            )
        )

    async def get(
        self, reference_id: UUID, member_id: UUID, *, user_id: UUID, product_id: str
    ) -> CastReferenceRow | None:
        result = await self._session.execute(
            select(*_REFERENCE_COLUMNS)
            .join(LibraryAssetHold, LibraryAssetHold.id == StudioCastReference.hold_id)
            .where(*self._owned(reference_id, member_id, user_id, product_id))
        )
        row = result.one_or_none()
        return None if row is None else _reference_row(row)

    async def list_for_members(
        self, member_ids: Collection[UUID], *, user_id: UUID, product_id: str
    ) -> Sequence[CastReferenceRow]:
        """Every reference of the given members, by member then position."""
        if not member_ids:
            return []
        result = await self._session.execute(
            select(*_REFERENCE_COLUMNS)
            .join(LibraryAssetHold, LibraryAssetHold.id == StudioCastReference.hold_id)
            .where(
                StudioCastReference.cast_member_id.in_(member_ids),
                StudioCastReference.user_id == user_id,
                StudioCastReference.product_id == product_id,
            )
            .order_by(StudioCastReference.cast_member_id, StudioCastReference.position)
        )
        return [_reference_row(row) for row in result.all()]

    async def list_covers(
        self, member_ids: Collection[UUID], *, user_id: UUID, product_id: str
    ) -> Mapping[UUID, CastReferenceRow]:
        """The reference at position 0 of each given member (members without one are absent)."""
        if not member_ids:
            return {}
        result = await self._session.execute(
            select(*_REFERENCE_COLUMNS)
            .join(LibraryAssetHold, LibraryAssetHold.id == StudioCastReference.hold_id)
            .where(
                StudioCastReference.cast_member_id.in_(member_ids),
                StudioCastReference.user_id == user_id,
                StudioCastReference.product_id == product_id,
                StudioCastReference.position == 0,
            )
        )
        rows = [_reference_row(row) for row in result.all()]
        return {row.cast_member_id: row for row in rows}

    async def counts_for_members(self, member_ids: Collection[UUID]) -> Mapping[UUID, int]:
        """Reference counts per member (members without references are absent)."""
        if not member_ids:
            return {}
        result = await self._session.execute(
            select(StudioCastReference.cast_member_id, func.count())
            .where(StudioCastReference.cast_member_id.in_(member_ids))
            .group_by(StudioCastReference.cast_member_id)
        )
        return {row[0]: row[1] for row in result.all()}

    async def count_for_member(self, member_id: UUID) -> int:
        result = await self._session.execute(
            select(func.count())
            .select_from(StudioCastReference)
            .where(StudioCastReference.cast_member_id == member_id)
        )
        return result.scalar_one()

    async def hold_ids_of_member(self, member_id: UUID) -> list[UUID]:
        result = await self._session.execute(
            select(StudioCastReference.hold_id).where(
                StudioCastReference.cast_member_id == member_id
            )
        )
        return list(result.scalars().all())

    async def ids_of_member(self, member_id: UUID) -> list[UUID]:
        """The member's reference ids by position."""
        result = await self._session.execute(
            select(StudioCastReference.id)
            .where(StudioCastReference.cast_member_id == member_id)
            .order_by(StudioCastReference.position)
        )
        return list(result.scalars().all())

    async def update(
        self,
        reference_id: UUID,
        member_id: UUID,
        *,
        user_id: UUID,
        product_id: str,
        changes: Mapping[str, object],
    ) -> None:
        """Apply ``changes`` (column name → value).

        Raises:
            sqlalchemy.exc.IntegrityError: A new label collides case-insensitively; call inside
                ``session.begin_nested()``.
        """
        if not changes:
            return
        await self._session.execute(
            update(StudioCastReference)
            .where(*self._owned(reference_id, member_id, user_id, product_id))
            .values(**changes)
            .execution_options(synchronize_session=False)
        )

    async def delete(
        self, reference_id: UUID, member_id: UUID, *, user_id: UUID, product_id: str
    ) -> UUID | None:
        """Delete a reference (its look links cascade); return its hold id, ``None`` if absent."""
        result = await self._session.execute(
            delete(StudioCastReference)
            .where(*self._owned(reference_id, member_id, user_id, product_id))
            .returning(StudioCastReference.hold_id)
        )
        return result.scalar_one_or_none()

    async def compact_positions(self, member_id: UUID) -> None:
        """Renumber the member's references ``0..n-1`` in their current order.

        One statement; the position uniqueness is ``DEFERRABLE INITIALLY DEFERRED``, so rows may
        pass through duplicate values inside it.
        """
        ranked = (
            select(
                StudioCastReference.id.label("id"),
                cast(
                    func.row_number().over(order_by=StudioCastReference.position) - 1,
                    SmallInteger,
                ).label("rank"),
            )
            .where(StudioCastReference.cast_member_id == member_id)
            .subquery()
        )
        await self._session.execute(
            update(StudioCastReference)
            .where(
                StudioCastReference.id == ranked.c.id, StudioCastReference.position != ranked.c.rank
            )
            .values(position=ranked.c.rank)
            .execution_options(synchronize_session=False)
        )

    async def set_order(self, member_id: UUID, ordered_ids: Sequence[UUID]) -> None:
        """Give the references the positions ``0..n-1`` in ``ordered_ids`` order (one statement)."""
        if not ordered_ids:
            return
        positions = case(
            {reference_id: index for index, reference_id in enumerate(ordered_ids)},
            value=StudioCastReference.id,
        )
        await self._session.execute(
            update(StudioCastReference)
            .where(
                StudioCastReference.cast_member_id == member_id,
                StudioCastReference.id.in_(ordered_ids),
            )
            .values(position=positions)
            .execution_options(synchronize_session=False)
        )


class StudioCastLookRepository:
    """Data access for ``studio_cast_looks`` and their reference links."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    @staticmethod
    def _owned(
        look_id: UUID, member_id: UUID, user_id: UUID, product_id: str
    ) -> tuple[ColumnElement[bool], ...]:
        return (
            StudioCastLook.id == look_id,
            StudioCastLook.cast_member_id == member_id,
            StudioCastLook.user_id == user_id,
            StudioCastLook.product_id == product_id,
        )

    async def insert(
        self, *, member_id: UUID, user_id: UUID, product_id: str, name: str
    ) -> StudioCastLook:
        """Insert a look.

        Raises:
            sqlalchemy.exc.IntegrityError: ``name`` collides case-insensitively within the
                member; call inside ``session.begin_nested()``.
        """
        result = await self._session.execute(
            pg_insert(StudioCastLook)
            .values(
                id=new_id(),
                cast_member_id=member_id,
                user_id=user_id,
                product_id=product_id,
                name=name,
            )
            .returning(StudioCastLook)
        )
        return result.scalar_one()

    async def get(
        self, look_id: UUID, member_id: UUID, *, user_id: UUID, product_id: str
    ) -> StudioCastLook | None:
        result = await self._session.execute(
            select(StudioCastLook)
            .where(*self._owned(look_id, member_id, user_id, product_id))
            .execution_options(**_FRESH)
        )
        return result.scalar_one_or_none()

    async def list_for_members(
        self, member_ids: Collection[UUID], *, user_id: UUID, product_id: str
    ) -> Sequence[StudioCastLook]:
        """Every look of the given members, by member then name (case-insensitive)."""
        if not member_ids:
            return []
        result = await self._session.execute(
            select(StudioCastLook)
            .where(
                StudioCastLook.cast_member_id.in_(member_ids),
                StudioCastLook.user_id == user_id,
                StudioCastLook.product_id == product_id,
            )
            .order_by(
                StudioCastLook.cast_member_id, func.lower(StudioCastLook.name), StudioCastLook.id
            )
            .execution_options(**_FRESH)
        )
        return result.scalars().all()

    async def counts_for_members(self, member_ids: Collection[UUID]) -> Mapping[UUID, int]:
        """Look counts per member (members without looks are absent)."""
        if not member_ids:
            return {}
        result = await self._session.execute(
            select(StudioCastLook.cast_member_id, func.count())
            .where(StudioCastLook.cast_member_id.in_(member_ids))
            .group_by(StudioCastLook.cast_member_id)
        )
        return {row[0]: row[1] for row in result.all()}

    async def count_for_member(self, member_id: UUID) -> int:
        result = await self._session.execute(
            select(func.count())
            .select_from(StudioCastLook)
            .where(StudioCastLook.cast_member_id == member_id)
        )
        return result.scalar_one()

    async def rename(
        self, look_id: UUID, member_id: UUID, *, user_id: UUID, product_id: str, name: str
    ) -> None:
        """Rename a look.

        Raises:
            sqlalchemy.exc.IntegrityError: The name collides case-insensitively within the
                member; call inside ``session.begin_nested()``.
        """
        await self._session.execute(
            update(StudioCastLook)
            .where(*self._owned(look_id, member_id, user_id, product_id))
            .values(name=name)
            .execution_options(synchronize_session=False)
        )

    async def touch(self, look_id: UUID) -> None:
        """Bump ``updated_at``: the look's reference set changed."""
        await self._session.execute(
            update(StudioCastLook)
            .where(StudioCastLook.id == look_id)
            .values(updated_at=func.now())
            .execution_options(synchronize_session=False)
        )

    async def delete(
        self, look_id: UUID, member_id: UUID, *, user_id: UUID, product_id: str
    ) -> bool:
        result = await self._session.execute(
            delete(StudioCastLook)
            .where(*self._owned(look_id, member_id, user_id, product_id))
            .returning(StudioCastLook.id)
        )
        return result.scalar_one_or_none() is not None

    async def replace_links(
        self, look_id: UUID, member_id: UUID, reference_ids: Sequence[UUID]
    ) -> None:
        """Make the look's reference set exactly ``reference_ids``.

        Every link is stamped with ``member_id``: both composite foreign keys then require the look
        and each reference to belong to that member, so a reference of another member is refused by
        the database (the service checks first and reports it as a ``422``).

        Raises:
            sqlalchemy.exc.IntegrityError: A reference (or the look) is not ``member_id``'s.
        """
        await self._session.execute(
            delete(StudioCastLookReference).where(StudioCastLookReference.look_id == look_id)
        )
        if reference_ids:
            await self._session.execute(
                pg_insert(StudioCastLookReference).values(
                    [
                        {"look_id": look_id, "reference_id": rid, "cast_member_id": member_id}
                        for rid in reference_ids
                    ]
                )
            )

    async def links_for_looks(self, look_ids: Collection[UUID]) -> Mapping[UUID, list[UUID]]:
        """Reference ids per look, in reference position order (an emptied look is absent)."""
        if not look_ids:
            return {}
        result = await self._session.execute(
            select(StudioCastLookReference.look_id, StudioCastLookReference.reference_id)
            .join(
                StudioCastReference,
                StudioCastReference.id == StudioCastLookReference.reference_id,
            )
            .where(StudioCastLookReference.look_id.in_(look_ids))
            .order_by(StudioCastLookReference.look_id, StudioCastReference.position)
        )
        links: dict[UUID, list[UUID]] = {}
        for look_id, reference_id in result.all():
            links.setdefault(look_id, []).append(reference_id)
        return links
