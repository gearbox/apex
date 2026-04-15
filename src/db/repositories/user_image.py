"""Repository for user image (upload) database operations."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.models.storage import UserImage
from src.db.repositories.base import BaseRepository

if TYPE_CHECKING:
    from collections.abc import Sequence
    from uuid import UUID


class UserImageRepository(BaseRepository[UserImage]):
    """Data access layer for UserImage records."""

    _model = UserImage

    def __init__(self, session: AsyncSession) -> None:
        super().__init__(session)

    async def create(
        self,
        *,
        id: UUID,
        user_id: UUID,
        storage_key: str,
        original_filename: str,
        content_type: str,
        size_bytes: int,
        format: str,
        expires_at: datetime,
        product_id: str,
    ) -> UserImage:
        """Create a new user upload record.

        Args:
            id: Unique upload ID (matches R2 file ID).
            user_id: Owner of the upload.
            storage_key: Full R2 storage key.
            original_filename: Original uploaded filename.
            content_type: MIME type.
            size_bytes: File size.
            format: Image format (png, jpeg, webp).
            expires_at: When the upload should be cleaned up.
            product_id: Product this upload belongs to.

        Returns:
            Created UserImage instance.
        """
        upload = UserImage(
            id=id,
            user_id=user_id,
            storage_key=storage_key,
            original_filename=original_filename,
            content_type=content_type,
            size_bytes=size_bytes,
            format=format,
            expires_at=expires_at,
            product_id=product_id,
        )
        self._session.add(upload)
        await self._session.flush()
        return upload

    async def get(
        self,
        image_id: UUID,
        *,
        user_id: UUID | None = None,
    ) -> UserImage | None:
        """Get a user image by ID, optionally scoped to a user.

        Args:
            image_id: Image ID to look up.
            user_id: When provided, only returns the image if owned
                by this user. ``None`` skips the ownership check.

        Returns:
            UserImage if found, None otherwise.
        """
        return await self._get_with_optional_owner(image_id, user_id=user_id)

    async def get_by_key(self, storage_key: str) -> UserImage | None:
        """Get a user image by storage key.

        Args:
            storage_key: R2 storage key.

        Returns:
            UserImage if found, None otherwise.
        """
        result = await self._session.execute(
            select(UserImage).where(UserImage.storage_key == storage_key)
        )
        return result.scalar_one_or_none()

    async def list_by_user(
        self,
        user_id: UUID,
        *,
        limit: int = 100,
        cursor_ts: datetime | None = None,
        cursor_id: UUID | None = None,
    ) -> Sequence[UserImage]:
        """List images for a user with cursor-based pagination.

        Uses limit+1 fetch pattern. Caller checks ``len(result) > limit``
        to determine ``has_more``.

        Args:
            user_id: User to list images for.
            limit: Maximum results to return (fetch limit+1 for has_more).
            cursor_ts: ``created_at`` of the last item on the previous page.
            cursor_id: ``id`` of the last item on the previous page.

        Returns:
            List of UserImage instances ordered by ``created_at DESC, id DESC``.
        """
        return await self._list_by_user_cursor(
            user_id, limit=limit, cursor_ts=cursor_ts, cursor_id=cursor_id
        )

    async def delete(
        self,
        image_id: UUID,
        *,
        user_id: UUID | None = None,
    ) -> bool:
        """Delete a user image record.

        Args:
            image_id: Image ID to delete.
            user_id: When provided, only deletes if owned by this user.

        Returns:
            True if deleted, False if not found.
        """
        image = await self.get(image_id, user_id=user_id)
        if image is None:
            return False
        await self._session.delete(image)
        await self._session.flush()
        return True

    async def get_expired(
        self,
        before: datetime | None = None,
        limit: int = 1000,
    ) -> Sequence[UserImage]:
        """Get images past their expiration date.

        Args:
            before: Consider expired if expires_at < before (default: now).
            limit: Maximum results to return.

        Returns:
            List of expired UserImage instances.
        """
        if before is None:
            before = datetime.now(UTC)

        result = await self._session.execute(
            select(UserImage)
            .where(UserImage.expires_at < before)
            .order_by(UserImage.expires_at)
            .limit(limit)
        )
        return result.scalars().all()

    async def count_and_sum_by_user(self, user_id: UUID) -> tuple[int, int]:
        """Count uploads and sum their size for a user.

        Used by storage stats aggregation.

        Args:
            user_id: User to aggregate for.

        Returns:
            Tuple of (count, total_bytes).
        """
        result = await self._session.execute(
            select(
                func.count(UserImage.id),
                func.coalesce(func.sum(UserImage.size_bytes), 0),
            ).where(UserImage.user_id == user_id)
        )
        count, total_bytes = result.one()
        return int(count), int(total_bytes)
