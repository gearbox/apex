"""Repository for user-created library projects.

Every method takes both ``user_id`` and ``product_id`` — cross-product /
cross-user project assignment is structurally impossible because a project
can never be fetched (and therefore never assigned) without both matching
(P8 in the Phase 2 design).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from sqlalchemy import func, literal, select, tuple_

from src.core.uid import new_id
from src.db.models.library import LibraryAssetMetadata, LibraryProject
from src.db.repositories.library import UNSET_UPDATE, OptionalUpdate, _UnsetUpdate

if TYPE_CHECKING:
    from collections.abc import Sequence
    from datetime import datetime
    from uuid import UUID

    from sqlalchemy.ext.asyncio import AsyncSession


class LibraryProjectRepository:
    """Data access layer for library_projects."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def create(
        self,
        *,
        user_id: UUID,
        product_id: str,
        name: str,
        description: str | None,
    ) -> LibraryProject:
        """Insert a new project row and flush it.

        Raises:
            sqlalchemy.exc.IntegrityError: If ``name`` collides
                case-insensitively with an existing project for this
                (product_id, user_id) — caller is expected to run this
                inside a ``session.begin_nested()`` block to isolate the
                failure (see LibraryProjectService).
        """
        project = LibraryProject(
            id=new_id(),
            product_id=product_id,
            user_id=user_id,
            name=name,
            description=description,
        )
        self._session.add(project)
        await self._session.flush()
        return project

    async def get(
        self,
        project_id: UUID,
        *,
        user_id: UUID,
        product_id: str,
    ) -> LibraryProject | None:
        """Fetch a single project, scoped to its owner and product."""
        result = await self._session.execute(
            select(LibraryProject).where(
                LibraryProject.id == project_id,
                LibraryProject.user_id == user_id,
                LibraryProject.product_id == product_id,
            )
        )
        return result.scalar_one_or_none()

    async def list_by_user(
        self,
        user_id: UUID,
        product_id: str,
        *,
        limit: int,
        cursor_ts: datetime | None = None,
        cursor_id: UUID | None = None,
    ) -> Sequence[LibraryProject]:
        """List a user's projects, newest first, keyset-paginated.

        Uses the limit+1 fetch pattern — caller checks
        ``len(result) > limit`` to determine ``has_more``.
        """
        query = select(LibraryProject).where(
            LibraryProject.user_id == user_id,
            LibraryProject.product_id == product_id,
        )

        if cursor_ts is not None and cursor_id is not None:
            query = query.where(
                tuple_(LibraryProject.created_at, LibraryProject.id)
                < tuple_(literal(cursor_ts), literal(cursor_id))
            )

        result = await self._session.execute(
            query.order_by(
                LibraryProject.created_at.desc(),
                LibraryProject.id.desc(),
            ).limit(limit + 1)
        )
        return result.scalars().all()

    async def update(
        self,
        project_id: UUID,
        *,
        user_id: UUID,
        product_id: str,
        name: OptionalUpdate[str] = UNSET_UPDATE,
        description: OptionalUpdate[str | None] = UNSET_UPDATE,
    ) -> LibraryProject | None:
        """Apply a tri-state partial update. ``UNSET_UPDATE`` leaves a field unchanged.

        Raises:
            sqlalchemy.exc.IntegrityError: If the new ``name`` collides
                case-insensitively with another project for this owner —
                caller is expected to run this inside a
                ``session.begin_nested()`` block (see LibraryProjectService).
        """
        project = await self.get(project_id, user_id=user_id, product_id=product_id)
        if project is None:
            return None

        if not isinstance(name, _UnsetUpdate):
            project.name = name
        if not isinstance(description, _UnsetUpdate):
            project.description = description

        await self._session.flush()
        return project

    async def delete(
        self,
        project_id: UUID,
        *,
        user_id: UUID,
        product_id: str,
    ) -> bool:
        """Delete a project. Assets referencing it are unassigned via ON DELETE SET NULL (P2).

        Returns:
            True if deleted; False if not found / not owned.
        """
        project = await self.get(project_id, user_id=user_id, product_id=product_id)
        if project is None:
            return False
        await self._session.delete(project)
        await self._session.flush()
        return True

    async def batch_asset_counts(
        self,
        project_ids: Sequence[UUID],
        *,
        user_id: UUID,
        product_id: str,
    ) -> dict[UUID, int]:
        """Count assets assigned to each of a batch of projects — one grouped query.

        Args:
            project_ids: Project ids appearing in the current page.
            user_id: Owner scope.
            product_id: Product scope.

        Returns:
            Mapping from project_id to asset count. Projects with zero
            assigned assets are absent from the result.
        """
        if not project_ids:
            return {}

        result = await self._session.execute(
            select(LibraryAssetMetadata.project_id, func.count(LibraryAssetMetadata.id))
            .where(
                LibraryAssetMetadata.project_id.in_(project_ids),
                LibraryAssetMetadata.user_id == user_id,
                LibraryAssetMetadata.product_id == product_id,
            )
            .group_by(LibraryAssetMetadata.project_id)
        )
        # The WHERE clause above already excludes NULL project_id rows; the
        # `if pid is not None` here is only to satisfy mypy's column typing.
        return {pid: count for pid, count in result.tuples().all() if pid is not None}

    async def batch_names(
        self,
        project_ids: Sequence[UUID],
        *,
        user_id: UUID,
        product_id: str,
    ) -> dict[UUID, str]:
        """Resolve project names for a batch of ids — one query, not per-row.

        Args:
            project_ids: Project ids to resolve.
            user_id: Owner scope.
            product_id: Product scope.

        Returns:
            Mapping from project_id to name. Missing/foreign ids are absent.
        """
        if not project_ids:
            return {}

        result = await self._session.execute(
            select(LibraryProject.id, LibraryProject.name).where(
                LibraryProject.id.in_(project_ids),
                LibraryProject.user_id == user_id,
                LibraryProject.product_id == product_id,
            )
        )
        return dict(result.tuples().all())
