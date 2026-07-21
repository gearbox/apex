"""Library tag service — CRUD for user-created, per-user/per-product tags.

Kept as a separate service from ``LibraryService``/``LibraryProjectService``
(SRP) — tag lifecycle (create/rename/delete) is a distinct concern from
asset read/mutation, even though the two meet via ``library_asset_tags``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import msgspec
import structlog
from sqlalchemy.exc import IntegrityError

from src.api.schemas.library import LibraryTag, LibraryTagListItem, LibraryTagPatch
from src.api.schemas.pagination import CursorPage, decode_cursor, encode_cursor
from src.api.services.owner_scoped_names import normalize_owner_scoped_name
from src.db.repositories.library import UNSET_UPDATE, OptionalUpdate
from src.db.repositories.library_tag import LibraryTagRepository

if TYPE_CHECKING:
    from uuid import UUID

    from sqlalchemy.ext.asyncio import AsyncSession

logger = structlog.get_logger(__name__)

_TAG_NAME_MAX_LENGTH = 50


class LibraryTagValidationError(Exception):
    """Raised when a tag name fails validation after normalization."""


class LibraryTagNameConflictError(Exception):
    """Raised when a tag name collides case-insensitively for this owner. → HTTP 409"""

    def __init__(self, name: str) -> None:
        self.name = name
        super().__init__(f"A tag named {name!r} already exists")


class LibraryTagService:
    """Business logic for /v1/library/tags."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def list_tags(
        self,
        user_id: UUID,
        product_id: str,
        *,
        session: AsyncSession,
        limit: int = 30,
        cursor: str | None = None,
    ) -> CursorPage[LibraryTagListItem]:
        """Return a paginated list of a user's tags, newest first."""
        repo = LibraryTagRepository(session)

        cursor_ts, cursor_id = decode_cursor(cursor) if cursor is not None else (None, None)

        rows = await repo.list_by_user(
            user_id, product_id, limit=limit, cursor_ts=cursor_ts, cursor_id=cursor_id
        )
        has_more = len(rows) > limit
        page_rows = list(rows[:limit])

        counts = await repo.batch_asset_counts(
            [t.id for t in page_rows], user_id=user_id, product_id=product_id
        )
        items = [
            LibraryTagListItem(
                id=t.id,
                name=t.name,
                created_at=t.created_at,
                updated_at=t.updated_at,
                asset_count=counts.get(t.id, 0),
            )
            for t in page_rows
        ]

        next_cursor: str | None = None
        if has_more and page_rows:
            last = page_rows[-1]
            next_cursor = encode_cursor(last.created_at, last.id)

        return CursorPage(items=items, limit=limit, has_more=has_more, next_cursor=next_cursor)

    async def get(
        self,
        tag_id: UUID,
        user_id: UUID,
        product_id: str,
        *,
        session: AsyncSession,
    ) -> LibraryTag | None:
        """Fetch a single tag, or None if missing/not owned."""
        repo = LibraryTagRepository(session)
        tag = await repo.get(tag_id, user_id=user_id, product_id=product_id)
        if tag is None:
            return None
        return LibraryTag(
            id=tag.id, name=tag.name, created_at=tag.created_at, updated_at=tag.updated_at
        )

    async def create(
        self,
        user_id: UUID,
        product_id: str,
        name: str,
        *,
        session: AsyncSession,
    ) -> LibraryTag:
        """Create a new tag.

        Raises:
            LibraryTagValidationError: If ``name`` normalizes to empty.
            LibraryTagNameConflictError: If the normalized name collides
                case-insensitively with an existing tag for this owner.
        """
        normalized = self._normalize_name(name)
        repo = LibraryTagRepository(session)

        try:
            async with session.begin_nested():
                tag = await repo.create(user_id=user_id, product_id=product_id, name=normalized)
        except IntegrityError as exc:
            raise LibraryTagNameConflictError(normalized) from exc

        logger.info("library.tag_created", tag_id=str(tag.id), user_id=str(user_id))
        return LibraryTag(
            id=tag.id, name=tag.name, created_at=tag.created_at, updated_at=tag.updated_at
        )

    async def patch(
        self,
        tag_id: UUID,
        patch: LibraryTagPatch,
        user_id: UUID,
        product_id: str,
        *,
        session: AsyncSession,
    ) -> LibraryTag | None:
        """Apply a tri-state rename.

        Returns:
            Updated LibraryTag, or None if not found/not owned.

        Raises:
            LibraryTagValidationError: If a provided ``name`` normalizes to
                empty.
            LibraryTagNameConflictError: If the new name collides
                case-insensitively with another tag for this owner.
        """
        repo = LibraryTagRepository(session)

        name_update: OptionalUpdate[str] = UNSET_UPDATE
        if patch.name is not msgspec.UNSET:
            name_update = self._normalize_name(patch.name)

        try:
            async with session.begin_nested():
                tag = await repo.update(
                    tag_id, user_id=user_id, product_id=product_id, name=name_update
                )
        except IntegrityError as exc:
            if not isinstance(name_update, str):
                raise  # pragma: no cover - unreachable: only a name change trips this constraint
            raise LibraryTagNameConflictError(name_update) from exc

        if tag is None:
            return None

        logger.info("library.tag_updated", tag_id=str(tag_id), user_id=str(user_id))
        return LibraryTag(
            id=tag.id, name=tag.name, created_at=tag.created_at, updated_at=tag.updated_at
        )

    async def delete(
        self,
        tag_id: UUID,
        user_id: UUID,
        product_id: str,
        *,
        session: AsyncSession,
    ) -> bool:
        """Delete a tag. Asset assignments cascade via ON DELETE CASCADE."""
        repo = LibraryTagRepository(session)
        deleted = await repo.delete(tag_id, user_id=user_id, product_id=product_id)
        if deleted:
            logger.info("library.tag_deleted", tag_id=str(tag_id), user_id=str(user_id))
        return deleted

    @staticmethod
    def _normalize_name(raw: str) -> str:
        """Trim + collapse inner whitespace; re-validate length post-normalization."""
        try:
            return normalize_owner_scoped_name(raw, max_length=_TAG_NAME_MAX_LENGTH)
        except ValueError as exc:
            raise LibraryTagValidationError(str(exc)) from exc
