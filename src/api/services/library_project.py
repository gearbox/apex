"""Library project service — CRUD for user-created asset groupings.

Kept as a separate service from ``LibraryService`` (SRP, per the Phase 2
design) — project lifecycle (create/rename/delete) is a distinct concern
from asset read/mutation, even though the two are related via
``library_asset_metadata.project_id``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import msgspec
import structlog
from sqlalchemy.exc import IntegrityError

from src.api.schemas.library import LibraryProject, LibraryProjectListItem, LibraryProjectPatch
from src.api.schemas.pagination import CursorPage, decode_cursor, encode_cursor
from src.api.services.owner_scoped_names import normalize_owner_scoped_name
from src.db.repositories.library import UNSET_UPDATE, OptionalUpdate
from src.db.repositories.library_project import LibraryProjectRepository

if TYPE_CHECKING:
    from collections.abc import Sequence
    from uuid import UUID

    from sqlalchemy.ext.asyncio import AsyncSession

    from src.api.services.library_project_guards import (
        ProjectDeletionBlocker,
        ProjectDeletionGuard,
    )

logger = structlog.get_logger(__name__)


class LibraryProjectValidationError(Exception):
    """Raised when a project name fails validation after normalization."""


class LibraryProjectNameConflictError(Exception):
    """Raised when a project name collides case-insensitively for this owner. → HTTP 409"""

    def __init__(self, name: str) -> None:
        self.name = name
        super().__init__(f"A project named {name!r} already exists")


class LibraryProjectInUseError(Exception):
    """Raised when another context still depends on the project. → HTTP 409 ``project_in_use``"""

    def __init__(self, blockers: Sequence[ProjectDeletionBlocker]) -> None:
        self.blockers = tuple(blockers)
        super().__init__("The project is in use and cannot be deleted")


class LibraryProjectService:
    """Business logic for /v1/library/projects."""

    def __init__(
        self, session: AsyncSession, *, deletion_guards: Sequence[ProjectDeletionGuard]
    ) -> None:
        """Create the service.

        Args:
            session: Request-scoped session.
            deletion_guards: Ports that may veto a delete (required, no default — a construction
                site must decide, so a guard can never be forgotten by omission).
        """
        self._session = session
        self._deletion_guards = tuple(deletion_guards)

    async def list_projects(
        self,
        user_id: UUID,
        product_id: str,
        *,
        session: AsyncSession,
        limit: int = 30,
        cursor: str | None = None,
    ) -> CursorPage[LibraryProjectListItem]:
        """Return a paginated list of a user's projects, newest first."""
        repo = LibraryProjectRepository(session)

        cursor_ts, cursor_id = decode_cursor(cursor) if cursor is not None else (None, None)

        rows = await repo.list_by_user(
            user_id, product_id, limit=limit, cursor_ts=cursor_ts, cursor_id=cursor_id
        )
        has_more = len(rows) > limit
        page_rows = list(rows[:limit])

        counts = await repo.batch_asset_counts(
            [p.id for p in page_rows], user_id=user_id, product_id=product_id
        )
        items = [
            LibraryProjectListItem(
                id=p.id,
                name=p.name,
                description=p.description,
                created_at=p.created_at,
                updated_at=p.updated_at,
                asset_count=counts.get(p.id, 0),
            )
            for p in page_rows
        ]

        next_cursor: str | None = None
        if has_more and page_rows:
            last = page_rows[-1]
            next_cursor = encode_cursor(last.created_at, last.id)

        return CursorPage(items=items, limit=limit, has_more=has_more, next_cursor=next_cursor)

    async def get(
        self,
        project_id: UUID,
        user_id: UUID,
        product_id: str,
        *,
        session: AsyncSession,
    ) -> LibraryProject | None:
        """Fetch a single project, or None if missing/not owned."""
        repo = LibraryProjectRepository(session)
        project = await repo.get(project_id, user_id=user_id, product_id=product_id)
        if project is None:
            return None
        return LibraryProject(
            id=project.id,
            name=project.name,
            description=project.description,
            created_at=project.created_at,
            updated_at=project.updated_at,
        )

    async def create(
        self,
        user_id: UUID,
        product_id: str,
        name: str,
        description: str | None,
        *,
        session: AsyncSession,
    ) -> LibraryProject:
        """Create a new project.

        Raises:
            LibraryProjectValidationError: If ``name`` normalizes to empty.
            LibraryProjectNameConflictError: If the normalized name collides
                case-insensitively with an existing project for this owner.
        """
        normalized = self._normalize_name(name)
        repo = LibraryProjectRepository(session)

        try:
            async with session.begin_nested():
                project = await repo.create(
                    user_id=user_id,
                    product_id=product_id,
                    name=normalized,
                    description=description,
                )
        except IntegrityError as exc:
            raise LibraryProjectNameConflictError(normalized) from exc

        logger.info("library.project_created", project_id=str(project.id), user_id=str(user_id))
        return LibraryProject(
            id=project.id,
            name=project.name,
            description=project.description,
            created_at=project.created_at,
            updated_at=project.updated_at,
        )

    async def patch(
        self,
        project_id: UUID,
        patch: LibraryProjectPatch,
        user_id: UUID,
        product_id: str,
        *,
        session: AsyncSession,
    ) -> LibraryProject | None:
        """Apply a tri-state rename/redescribe.

        Returns:
            Updated LibraryProject, or None if not found/not owned.

        Raises:
            LibraryProjectValidationError: If a provided ``name`` normalizes
                to empty.
            LibraryProjectNameConflictError: If the new name collides
                case-insensitively with another project for this owner.
        """
        repo = LibraryProjectRepository(session)

        # Only a name change can trip the case-insensitive uniqueness
        # constraint, so the IntegrityError branch below is reachable only
        # when name_update holds a normalized string (never UNSET_UPDATE).
        name_update: OptionalUpdate[str] = UNSET_UPDATE
        if patch.name is not msgspec.UNSET:
            name_update = self._normalize_name(patch.name)

        try:
            async with session.begin_nested():
                project = await repo.update(
                    project_id,
                    user_id=user_id,
                    product_id=product_id,
                    name=name_update,
                    description=(
                        patch.description
                        if patch.description is not msgspec.UNSET
                        else UNSET_UPDATE
                    ),
                )
        except IntegrityError as exc:
            if not isinstance(name_update, str):
                raise  # pragma: no cover - unreachable: only a name change trips this constraint
            raise LibraryProjectNameConflictError(name_update) from exc

        if project is None:
            return None

        logger.info("library.project_updated", project_id=str(project_id), user_id=str(user_id))
        return LibraryProject(
            id=project.id,
            name=project.name,
            description=project.description,
            created_at=project.created_at,
            updated_at=project.updated_at,
        )

    async def delete(
        self,
        project_id: UUID,
        user_id: UUID,
        product_id: str,
        *,
        session: AsyncSession,
    ) -> bool:
        """Delete a project. Assigned assets are unassigned via ON DELETE SET NULL (P2).

        Returns:
            True if deleted; False if not found / not owned.

        Raises:
            LibraryProjectInUseError: A deletion guard reported blockers (e.g. studio tracks), or
                one appeared between the guard check and the delete (the FK backstop fired).
        """
        repo = LibraryProjectRepository(session)
        if await repo.get(project_id, user_id=user_id, product_id=product_id) is None:
            return False

        await self._raise_if_blocked(project_id, user_id, product_id, session)
        try:
            async with session.begin_nested():
                deleted = await repo.delete(project_id, user_id=user_id, product_id=product_id)
        except IntegrityError:
            # Race: a dependent row appeared after the guards ran and a NO ACTION FK refused the
            # delete. Re-evaluate to report the precise blockers — never a 500.
            await self._raise_if_blocked(project_id, user_id, product_id, session)
            raise
        if deleted:
            logger.info("library.project_deleted", project_id=str(project_id), user_id=str(user_id))
        return deleted

    async def _raise_if_blocked(
        self, project_id: UUID, user_id: UUID, product_id: str, session: AsyncSession
    ) -> None:
        blockers: list[ProjectDeletionBlocker] = []
        for guard in self._deletion_guards:
            blockers.extend(
                await guard.blockers(
                    project_id, user_id=user_id, product_id=product_id, session=session
                )
            )
        if blockers:
            logger.info(
                "library.project_delete_blocked",
                project_id=str(project_id),
                user_id=str(user_id),
                blockers=[(b.code, b.count) for b in blockers],
            )
            raise LibraryProjectInUseError(blockers)

    @staticmethod
    def _normalize_name(raw: str) -> str:
        """Trim + collapse inner whitespace; re-validate length post-normalization."""
        try:
            return normalize_owner_scoped_name(raw, max_length=100)
        except ValueError as exc:
            raise LibraryProjectValidationError(str(exc)) from exc
