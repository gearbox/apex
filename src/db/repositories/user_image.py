"""Repository for user image (upload) database operations."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, cast
from uuid import UUID

from sqlalchemy import func, literal, select, tuple_
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.models import GenerationOutput, UserImage

if TYPE_CHECKING:
    from collections.abc import Sequence


class UserImageRepository:
    """Data access layer for UserImage records."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

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
        if user_id is None:
            return cast(
                UserImage | None,
                await self._session.get(UserImage, image_id),
            )
        result = await self._session.execute(
            select(UserImage).where(
                UserImage.id == image_id,
                UserImage.user_id == user_id,
            )
        )
        return result.scalar_one_or_none()

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
        query = select(UserImage).where(UserImage.user_id == user_id)

        if cursor_ts is not None and cursor_id is not None:
            query = query.where(
                tuple_(UserImage.created_at, UserImage.id)
                < tuple_(literal(cursor_ts), literal(cursor_id))
            )

        result = await self._session.execute(
            query.order_by(UserImage.created_at.desc(), UserImage.id.desc()).limit(limit + 1)
        )
        return result.scalars().all()

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

    async def get_storage_stats(self, user_id: UUID) -> dict[str, int]:
        """Get storage statistics for a user.

        Args:
            user_id: User to get stats for.

        Returns:
            Dict with upload_count, output_count, total_bytes.
        """
        upload_result = await self._session.execute(
            select(
                func.count(UserImage.id),
                func.coalesce(func.sum(UserImage.size_bytes), 0),
            ).where(UserImage.user_id == user_id)
        )
        upload_count, upload_bytes = upload_result.one()

        output_result = await self._session.execute(
            select(
                func.count(GenerationOutput.id),
                func.coalesce(func.sum(GenerationOutput.size_bytes), 0),
            ).where(GenerationOutput.user_id == user_id)
        )
        output_count, output_bytes = output_result.one()

        return {
            "upload_count": upload_count,
            "output_count": output_count,
            "total_bytes": upload_bytes + output_bytes,
        }
