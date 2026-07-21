"""Repository for user-created library tags and their asset assignments.

Mirrors LibraryProjectRepository's owner-scoping discipline — every method
takes both ``user_id`` and ``product_id`` — except the two purge helpers
(``delete_tags_for_assets``), which are deliberately unscoped like
``LibraryRepository.delete_metadata_for_assets``: callers already establish
ownership/scope before collecting the pairs to purge.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING

from sqlalchemy import and_, func, literal, or_, select, tuple_
from sqlalchemy import delete as sa_delete
from sqlalchemy.dialects.postgresql import insert as pg_insert

from src.core.uid import new_id
from src.db.models.library import LibraryAssetTag, LibraryTag
from src.db.repositories.library import UNSET_UPDATE, OptionalUpdate, _UnsetUpdate

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from datetime import datetime
    from uuid import UUID

    from sqlalchemy.ext.asyncio import AsyncSession

    from src.core.library_ref import AssetRef

# Conflict target for asset-tag upserts — matches the library_asset_tags
# primary key (tag_id, asset_type, asset_id); works regardless of the PK's
# generated constraint name since these columns form a unique index.
_ASSET_TAG_CONFLICT_COLS = ("tag_id", "asset_type", "asset_id")


@dataclass(frozen=True)
class TagRef:
    """Lightweight (id, name) pair for hydrating an asset's tag chips."""

    id: UUID
    name: str


class LibraryTagRepository:
    """Data access layer for library_tags and library_asset_tags."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    # -------------------------------------------------------------------------
    # Tag CRUD
    # -------------------------------------------------------------------------

    async def create(
        self,
        *,
        user_id: UUID,
        product_id: str,
        name: str,
    ) -> LibraryTag:
        """Insert a new tag row and flush it.

        Raises:
            sqlalchemy.exc.IntegrityError: If ``name`` collides
                case-insensitively with an existing tag for this
                (product_id, user_id) — caller is expected to run this
                inside a ``session.begin_nested()`` block (see
                LibraryTagService).
        """
        tag = LibraryTag(
            id=new_id(),
            product_id=product_id,
            user_id=user_id,
            name=name,
        )
        self._session.add(tag)
        await self._session.flush()
        return tag

    async def get(
        self,
        tag_id: UUID,
        *,
        user_id: UUID,
        product_id: str,
    ) -> LibraryTag | None:
        """Fetch a single tag, scoped to its owner and product."""
        result = await self._session.execute(
            select(LibraryTag).where(
                LibraryTag.id == tag_id,
                LibraryTag.user_id == user_id,
                LibraryTag.product_id == product_id,
            )
        )
        return result.scalar_one_or_none()

    async def get_many(
        self,
        tag_ids: Sequence[UUID],
        *,
        user_id: UUID,
        product_id: str,
    ) -> dict[UUID, LibraryTag]:
        """Batch-fetch tags by id, ownership-scoped — one query, not per-id.

        Used by PATCH/bulk callers to validate tag_ids: the caller compares
        ``len(result)`` against the requested id count to detect
        foreign/missing tags.
        """
        if not tag_ids:
            return {}

        result = await self._session.execute(
            select(LibraryTag).where(
                LibraryTag.id.in_(tag_ids),
                LibraryTag.user_id == user_id,
                LibraryTag.product_id == product_id,
            )
        )
        return {tag.id: tag for tag in result.scalars().all()}

    async def list_by_user(
        self,
        user_id: UUID,
        product_id: str,
        *,
        limit: int,
        cursor_ts: datetime | None = None,
        cursor_id: UUID | None = None,
    ) -> Sequence[LibraryTag]:
        """List a user's tags, newest first, keyset-paginated.

        Uses the limit+1 fetch pattern — caller checks
        ``len(result) > limit`` to determine ``has_more``.
        """
        query = select(LibraryTag).where(
            LibraryTag.user_id == user_id,
            LibraryTag.product_id == product_id,
        )

        if cursor_ts is not None and cursor_id is not None:
            query = query.where(
                tuple_(LibraryTag.created_at, LibraryTag.id)
                < tuple_(literal(cursor_ts), literal(cursor_id))
            )

        result = await self._session.execute(
            query.order_by(
                LibraryTag.created_at.desc(),
                LibraryTag.id.desc(),
            ).limit(limit + 1)
        )
        return result.scalars().all()

    async def update(
        self,
        tag_id: UUID,
        *,
        user_id: UUID,
        product_id: str,
        name: OptionalUpdate[str] = UNSET_UPDATE,
    ) -> LibraryTag | None:
        """Apply a tri-state rename. ``UNSET_UPDATE`` leaves the name unchanged.

        Raises:
            sqlalchemy.exc.IntegrityError: If the new ``name`` collides
                case-insensitively with another tag for this owner — caller
                is expected to run this inside a ``session.begin_nested()``
                block (see LibraryTagService).
        """
        tag = await self.get(tag_id, user_id=user_id, product_id=product_id)
        if tag is None:
            return None

        if not isinstance(name, _UnsetUpdate):
            tag.name = name

        await self._session.flush()
        return tag

    async def delete(
        self,
        tag_id: UUID,
        *,
        user_id: UUID,
        product_id: str,
    ) -> bool:
        """Delete a tag. Asset assignments cascade via ON DELETE CASCADE.

        Returns:
            True if deleted; False if not found / not owned.
        """
        tag = await self.get(tag_id, user_id=user_id, product_id=product_id)
        if tag is None:
            return False
        await self._session.delete(tag)
        await self._session.flush()
        return True

    async def batch_asset_counts(
        self,
        tag_ids: Sequence[UUID],
        *,
        user_id: UUID,
        product_id: str,
    ) -> dict[UUID, int]:
        """Count assets assigned to each of a batch of tags — one grouped query.

        Args:
            tag_ids: Tag ids appearing in the current page.
            user_id: Owner scope.
            product_id: Product scope.

        Returns:
            Mapping from tag_id to asset count. Tags with zero assigned
            assets are absent from the result.
        """
        if not tag_ids:
            return {}

        result = await self._session.execute(
            select(LibraryAssetTag.tag_id, func.count())
            .where(
                LibraryAssetTag.tag_id.in_(tag_ids),
                LibraryAssetTag.user_id == user_id,
                LibraryAssetTag.product_id == product_id,
            )
            .group_by(LibraryAssetTag.tag_id)
        )
        return dict(result.tuples().all())

    # -------------------------------------------------------------------------
    # Asset-tag operations
    # -------------------------------------------------------------------------

    async def batch_tags_for_assets(
        self,
        pairs: Sequence[tuple[str, UUID]],
        *,
        user_id: UUID,
        product_id: str,
    ) -> Mapping[tuple[str, UUID], list[TagRef]]:
        """Fetch tags for a batch of (asset_type, asset_id) pairs — one query.

        Ordered by ``lower(name)`` in SQL so pagination-stable chips render
        identically everywhere (not left to Python-side sorting).

        Args:
            pairs: (asset_type, asset_id) tuples appearing on the current page.
            user_id: Owner scope.
            product_id: Product scope.

        Returns:
            Mapping from (asset_type, asset_id) to its tags, name-ordered.
            Pairs with no tags are absent from the result.
        """
        if not pairs:
            return {}

        conditions = [
            and_(LibraryAssetTag.asset_type == asset_type, LibraryAssetTag.asset_id == asset_id)
            for asset_type, asset_id in pairs
        ]
        result = await self._session.execute(
            select(
                LibraryAssetTag.asset_type,
                LibraryAssetTag.asset_id,
                LibraryTag.id,
                LibraryTag.name,
            )
            .select_from(LibraryAssetTag)
            .join(LibraryTag, LibraryTag.id == LibraryAssetTag.tag_id)
            .where(
                LibraryAssetTag.user_id == user_id,
                LibraryAssetTag.product_id == product_id,
                or_(*conditions),
            )
            .order_by(func.lower(LibraryTag.name))
        )

        grouped: dict[tuple[str, UUID], list[TagRef]] = defaultdict(list)
        for asset_type, asset_id, tag_id, name in result.tuples().all():
            grouped[(asset_type, asset_id)].append(TagRef(id=tag_id, name=name))
        return dict(grouped)

    async def set_asset_tags(
        self,
        source: str,
        asset_id: UUID,
        tag_ids: Sequence[UUID],
        *,
        user_id: UUID,
        product_id: str,
    ) -> None:
        """Replace the full tag set on a single asset.

        Two statements, no per-tag round trips: delete rows no longer in
        ``tag_ids``, then upsert the requested set. With an empty
        ``tag_ids``, the delete has no ``NOT IN`` predicate (an empty IN/NOT
        IN list is a classic SQLAlchemy footgun) — it simply removes every
        tag from the asset.

        Ownership of every id in ``tag_ids`` must already be verified by the
        caller (see LibraryService.patch_asset).
        """
        scope = (
            LibraryAssetTag.asset_type == source,
            LibraryAssetTag.asset_id == asset_id,
            LibraryAssetTag.user_id == user_id,
            LibraryAssetTag.product_id == product_id,
        )

        if not tag_ids:
            await self._session.execute(sa_delete(LibraryAssetTag).where(*scope))
            await self._session.flush()
            return

        await self._session.execute(
            sa_delete(LibraryAssetTag).where(*scope, LibraryAssetTag.tag_id.notin_(tag_ids))
        )

        rows = [
            {
                "tag_id": tag_id,
                "asset_type": source,
                "asset_id": asset_id,
                "product_id": product_id,
                "user_id": user_id,
            }
            for tag_id in tag_ids
        ]
        stmt = (
            pg_insert(LibraryAssetTag)
            .values(rows)
            .on_conflict_do_nothing(index_elements=_ASSET_TAG_CONFLICT_COLS)
        )
        await self._session.execute(stmt)
        await self._session.flush()

    async def bulk_add_tags(
        self,
        refs: Sequence[AssetRef],
        tag_ids: Sequence[UUID],
        *,
        user_id: UUID,
        product_id: str,
    ) -> None:
        """Add a batch of tags to a batch of assets in one statement (cross-product).

        ``ON CONFLICT DO NOTHING`` makes re-adds idempotent and tolerates
        intra-statement duplicates — callers should still dedupe ``refs``
        and ``tag_ids`` beforehand so logged counts stay deterministic (see
        LibraryService.bulk_apply).
        """
        if not refs or not tag_ids:
            return

        rows = [
            {
                "tag_id": tag_id,
                "asset_type": ref.source.value,
                "asset_id": ref.asset_id,
                "product_id": product_id,
                "user_id": user_id,
            }
            for ref in refs
            for tag_id in tag_ids
        ]
        stmt = (
            pg_insert(LibraryAssetTag)
            .values(rows)
            .on_conflict_do_nothing(index_elements=_ASSET_TAG_CONFLICT_COLS)
        )
        await self._session.execute(stmt)
        await self._session.flush()

    async def bulk_remove_tags(
        self,
        refs: Sequence[AssetRef],
        tag_ids: Sequence[UUID],
        *,
        user_id: UUID,
        product_id: str,
    ) -> None:
        """Remove a batch of tags from a batch of assets in one statement."""
        if not refs or not tag_ids:
            return

        conditions = [
            and_(
                LibraryAssetTag.asset_type == ref.source.value,
                LibraryAssetTag.asset_id == ref.asset_id,
            )
            for ref in refs
        ]
        await self._session.execute(
            sa_delete(LibraryAssetTag).where(
                LibraryAssetTag.user_id == user_id,
                LibraryAssetTag.product_id == product_id,
                LibraryAssetTag.tag_id.in_(tag_ids),
                or_(*conditions),
            )
        )
        await self._session.flush()

    async def delete_tags_for_assets(self, pairs: Sequence[tuple[str, UUID]]) -> int:
        """Bulk-delete asset-tag rows matching (asset_type, asset_id) pairs.

        Deliberately not scoped to (user_id, product_id) — callers (asset
        deletion, retention sweep) already establish ownership/scope before
        collecting the pairs to purge. Mirrors
        ``LibraryRepository.delete_metadata_for_assets``.

        Args:
            pairs: (asset_type, asset_id) tuples to purge.

        Returns:
            Number of rows deleted.
        """
        if not pairs:
            return 0

        conditions = [
            and_(LibraryAssetTag.asset_type == asset_type, LibraryAssetTag.asset_id == asset_id)
            for asset_type, asset_id in pairs
        ]
        result = await self._session.execute(sa_delete(LibraryAssetTag).where(or_(*conditions)))
        return result.rowcount or 0  # type: ignore[attr-defined]
