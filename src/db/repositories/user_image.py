"""Repository for user image (upload) database operations."""

from __future__ import annotations

from collections import defaultdict
from datetime import UTC, datetime
from typing import TYPE_CHECKING, cast

from sqlalchemy import CursorResult, and_, func, literal, or_, select, tuple_, update

from src.db.models.storage import UserImage
from src.db.repositories.base import BaseRepository

if TYPE_CHECKING:
    from collections.abc import Sequence
    from uuid import UUID

    from sqlalchemy.ext.asyncio import AsyncSession


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
        is_thumbnail: bool = False,
        parent_image_id: UUID | None = None,
        thumbnail_max_edge: int | None = None,
        width: int | None = None,
        height: int | None = None,
        source_output_id: UUID | None = None,
        source_upload_id: UUID | None = None,
        source_timestamp_ms: int | None = None,
        duration_ms: int | None = None,
    ) -> UserImage:
        """Create a new user upload record.

        Args:
            id: Unique upload ID (matches R2 file ID).
            user_id: Owner of the upload.
            storage_key: Full R2 storage key.
            original_filename: Original uploaded filename.
            content_type: MIME type.
            size_bytes: File size.
            format: Image format (png, jpeg, webp) or video format.
            expires_at: When the upload should be cleaned up.
            product_id: Product this upload belongs to.
            is_thumbnail: Whether this is a derived thumbnail row.
            parent_image_id: Parent upload this thumbnail derives from.
            thumbnail_max_edge: Size bucket (150=sm, 512=md). None on full rows.
            width: Pixel width.
            height: Pixel height.
            source_output_id: Source generation output this frame was
                extracted from (at most one of source_output_id /
                source_upload_id may be set).
            source_upload_id: Source uploaded video this frame was extracted
                from.
            source_timestamp_ms: Timestamp within the source video this frame
                was extracted at.
            duration_ms: Video duration (uploaded videos only; display only —
                extraction math always re-probes the file).

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
            is_thumbnail=is_thumbnail,
            parent_image_id=parent_image_id,
            thumbnail_max_edge=thumbnail_max_edge,
            width=width,
            height=height,
            source_output_id=source_output_id,
            source_upload_id=source_upload_id,
            source_timestamp_ms=source_timestamp_ms,
            duration_ms=duration_ms,
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

        Returns thumbnail rows too (content proxy needs to resolve by id).

        Args:
            image_id: Image ID to look up.
            user_id: When provided, only returns the image if owned
                by this user. ``None`` skips the ownership check.

        Returns:
            UserImage if found, None otherwise.
        """
        return await self._get_with_optional_owner(image_id, user_id=user_id)

    async def get_many(
        self,
        ids: Sequence[UUID],
        *,
        user_id: UUID,
    ) -> dict[UUID, UserImage]:
        """Batch-fetch user images by ID, ownership-scoped.

        Args:
            ids: Image IDs to look up.
            user_id: Owner — rows of other users are excluded.

        Returns:
            Mapping from id to UserImage for rows that exist and are
            owned by ``user_id``. Missing/foreign ids are simply absent
            from the result (caller decides how to handle misses).
        """
        if not ids:
            return {}

        result = await self._session.execute(
            select(UserImage).where(UserImage.id.in_(ids), UserImage.user_id == user_id)
        )
        return {image.id: image for image in result.scalars().all()}

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
        """List full (non-thumbnail) images for a user with cursor-based pagination.

        Uses limit+1 fetch pattern. Caller checks ``len(result) > limit``
        to determine ``has_more``.

        Args:
            user_id: User to list images for.
            limit: Maximum results to return (fetch limit+1 for has_more).
            cursor_ts: ``created_at`` of the last item on the previous page.
            cursor_id: ``id`` of the last item on the previous page.

        Returns:
            List of non-thumbnail UserImage instances ordered by
            ``created_at DESC, id DESC``.
        """
        query = select(UserImage).where(
            UserImage.user_id == user_id,
            UserImage.is_thumbnail.is_(False),
        )

        if cursor_ts is not None and cursor_id is not None:
            query = query.where(
                tuple_(UserImage.created_at, UserImage.id)
                < tuple_(literal(cursor_ts), literal(cursor_id))
            )

        result = await self._session.execute(
            query.order_by(
                UserImage.created_at.desc(),
                UserImage.id.desc(),
            ).limit(limit + 1)
        )
        return result.scalars().all()

    async def list_derivatives(self, parent_image_id: UUID) -> Sequence[UserImage]:
        """List derivative uploads (thumbnails) for a given parent upload.

        Args:
            parent_image_id: Parent upload ID.

        Returns:
            List of derivative UserImage instances.
        """
        result = await self._session.execute(
            select(UserImage).where(UserImage.parent_image_id == parent_image_id)
        )
        return result.scalars().all()

    async def batch_derivatives(
        self,
        parent_ids: Sequence[UUID],
    ) -> dict[UUID, list[UserImage]]:
        """Fetch all derivatives for a batch of parent upload IDs.

        Args:
            parent_ids: Parent upload IDs to look up.

        Returns:
            Mapping from parent_image_id to list of derivative UserImage rows.
        """
        if not parent_ids:
            return {}

        result = await self._session.execute(
            select(UserImage).where(UserImage.parent_image_id.in_(parent_ids))
        )
        rows = result.scalars().all()

        grouped: dict[UUID, list[UserImage]] = defaultdict(list)
        for row in rows:
            if row.parent_image_id is not None:
                grouped[row.parent_image_id].append(row)
        return dict(grouped)

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
        """Get full (non-thumbnail) images past their expiration date.

        Thumbnails are excluded — they cascade-delete with their parent.

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
            .where(
                UserImage.expires_at < before,
                UserImage.is_thumbnail.is_(False),
            )
            .order_by(UserImage.expires_at)
            .limit(limit)
        )
        return result.scalars().all()

    async def touch_expiry(
        self,
        image_id: UUID,
        *,
        user_id: UUID,
        expires_at: datetime,
    ) -> bool:
        """Reset the retention window on a full upload and its thumbnails.

        Ownership-scoped: only rows belonging to ``user_id`` are touched.
        Matching a thumbnail row directly by ``image_id`` is rejected —
        the sliding window applies to full uploads only; derivatives are
        bumped via their ``parent_image_id`` link.

        Args:
            image_id: Full (non-thumbnail) upload ID.
            user_id: Owner — rows of other users are never touched.
            expires_at: New expiry timestamp (caller computes
                ``now + retention_days``).

        Returns:
            True if the full upload row was updated; False if the image
            does not exist, is not owned by ``user_id``, or is a
            thumbnail row.
        """
        result = cast(
            "CursorResult[tuple[()]]",
            await self._session.execute(
                update(UserImage)
                .where(
                    UserImage.user_id == user_id,
                    or_(
                        and_(UserImage.id == image_id, UserImage.is_thumbnail.is_(False)),
                        UserImage.parent_image_id == image_id,
                    ),
                )
                .values(expires_at=expires_at)
                .execution_options(synchronize_session=False)
            ),
        )
        return (result.rowcount or 0) > 0

    async def count_and_sum_by_user(self, user_id: UUID) -> tuple[int, int]:
        """Count full uploads and sum their size for a user.

        Excludes thumbnail rows — only full uploads are counted.

        Args:
            user_id: User to aggregate for.

        Returns:
            Tuple of (count, total_bytes).
        """
        result = await self._session.execute(
            select(
                func.count(UserImage.id),
                func.coalesce(func.sum(UserImage.size_bytes), 0),
            ).where(
                UserImage.user_id == user_id,
                UserImage.is_thumbnail.is_(False),
            )
        )
        count, total_bytes = result.one()
        return int(count), int(total_bytes)
