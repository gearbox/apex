"""Repository for generation output database operations."""

from __future__ import annotations

from collections import defaultdict
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from sqlalchemy import delete, func, literal, select, tuple_

from src.db.models.storage import GenerationOutput
from src.db.repositories.base import BaseRepository

if TYPE_CHECKING:
    from collections.abc import Sequence
    from uuid import UUID

    from sqlalchemy.ext.asyncio import AsyncSession


class OutputRepository(BaseRepository[GenerationOutput]):
    """Data access layer for GenerationOutput records."""

    _model = GenerationOutput

    def __init__(self, session: AsyncSession) -> None:
        super().__init__(session)

    async def create(
        self,
        *,
        id: UUID,
        user_id: UUID,
        job_id: UUID,
        storage_key: str,
        content_type: str,
        size_bytes: int,
        format: str,
        output_index: int,
        expires_at: datetime,
        product_id: str,
        input_image_id: UUID | None = None,
        is_thumbnail: bool = False,
        parent_output_id: UUID | None = None,
        thumbnail_max_edge: int | None = None,
        width: int | None = None,
        height: int | None = None,
    ) -> GenerationOutput:
        """Create a new generation output record.

        Args:
            id: Unique output ID (matches R2 file ID).
            user_id: Owner of the output.
            job_id: Associated generation job.
            storage_key: Full R2 storage key.
            content_type: MIME type.
            size_bytes: File size.
            format: Image format.
            output_index: Index in batch (0-based).
            expires_at: When the output should be cleaned up.
            product_id: Product this output belongs to.
            input_image_id: Associated input image (for i2i).
            is_thumbnail: Whether this output is a derived preview.
            parent_output_id: Full output this thumbnail derives from.
            thumbnail_max_edge: Size bucket (150=sm, 512=md). None on full rows.
            width: Pixel width (stored for full and thumbnail rows).
            height: Pixel height (stored for full and thumbnail rows).

        Returns:
            Created GenerationOutput instance.
        """
        output = GenerationOutput(
            id=id,
            user_id=user_id,
            job_id=job_id,
            storage_key=storage_key,
            content_type=content_type,
            size_bytes=size_bytes,
            format=format,
            output_index=output_index,
            expires_at=expires_at,
            input_image_id=input_image_id,
            is_thumbnail=is_thumbnail,
            parent_output_id=parent_output_id,
            thumbnail_max_edge=thumbnail_max_edge,
            width=width,
            height=height,
            product_id=product_id,
        )
        self._session.add(output)
        await self._session.flush()
        return output

    async def get(
        self,
        output_id: UUID,
        *,
        user_id: UUID | None = None,
    ) -> GenerationOutput | None:
        """Get an output by ID, optionally scoped to a user.

        Args:
            output_id: Output ID to look up.
            user_id: When provided, only returns if owned by this user.

        Returns:
            GenerationOutput if found, None otherwise.
        """
        return await self._get_with_optional_owner(output_id, user_id=user_id)

    async def list_by_job(self, job_id: UUID) -> Sequence[GenerationOutput]:
        """List full (non-thumbnail) outputs for a job, ordered by output_index.

        Args:
            job_id: Job to list outputs for.

        Returns:
            List of non-thumbnail GenerationOutput instances ordered by index.
        """
        result = await self._session.execute(
            select(GenerationOutput)
            .where(
                GenerationOutput.job_id == job_id,
                GenerationOutput.is_thumbnail.is_(False),
            )
            .order_by(GenerationOutput.output_index)
        )
        return result.scalars().all()

    async def list_by_user(
        self,
        user_id: UUID,
        *,
        limit: int = 100,
        cursor_ts: datetime | None = None,
        cursor_id: UUID | None = None,
    ) -> Sequence[GenerationOutput]:
        """List full (non-thumbnail) outputs for a user with cursor-based pagination.

        Uses limit+1 fetch pattern. Caller checks ``len(result) > limit``
        to determine ``has_more``.

        Args:
            user_id: User to list outputs for.
            limit: Maximum results to return (fetch limit+1 for has_more).
            cursor_ts: ``created_at`` of the last item on the previous page.
            cursor_id: ``id`` of the last item on the previous page.

        Returns:
            List of non-thumbnail GenerationOutput instances ordered by
            ``created_at DESC, id DESC``.
        """
        query = select(GenerationOutput).where(
            GenerationOutput.user_id == user_id,
            GenerationOutput.is_thumbnail.is_(False),
        )

        if cursor_ts is not None and cursor_id is not None:
            query = query.where(
                tuple_(GenerationOutput.created_at, GenerationOutput.id)
                < tuple_(literal(cursor_ts), literal(cursor_id))
            )

        result = await self._session.execute(
            query.order_by(
                GenerationOutput.created_at.desc(),
                GenerationOutput.id.desc(),
            ).limit(limit + 1)
        )
        return result.scalars().all()

    async def list_derivatives(self, parent_output_id: UUID) -> Sequence[GenerationOutput]:
        """List derivative outputs (thumbnails) for a given parent output.

        Args:
            parent_output_id: Parent full output ID.

        Returns:
            List of derivative GenerationOutput instances.
        """
        result = await self._session.execute(
            select(GenerationOutput).where(GenerationOutput.parent_output_id == parent_output_id)
        )
        return result.scalars().all()

    async def batch_derivatives(
        self,
        parent_ids: Sequence[UUID],
    ) -> dict[UUID, list[GenerationOutput]]:
        """Fetch all derivatives for a batch of parent output IDs.

        Args:
            parent_ids: Parent output IDs to look up.

        Returns:
            Mapping from parent_output_id to list of derivative GenerationOutput rows.
            Parent IDs with no derivatives are absent from the result.
        """
        if not parent_ids:
            return {}

        result = await self._session.execute(
            select(GenerationOutput).where(GenerationOutput.parent_output_id.in_(parent_ids))
        )
        rows = result.scalars().all()

        grouped: dict[UUID, list[GenerationOutput]] = defaultdict(list)
        for row in rows:
            if row.parent_output_id is not None:
                grouped[row.parent_output_id].append(row)
        return dict(grouped)

    async def get_expired(
        self,
        before: datetime | None = None,
        limit: int = 1000,
    ) -> Sequence[GenerationOutput]:
        """Get outputs past their expiration date.

        Args:
            before: Consider expired if expires_at < before (default: now).
            limit: Maximum results to return.

        Returns:
            List of expired GenerationOutput instances.
        """
        if before is None:
            before = datetime.now(UTC)

        result = await self._session.execute(
            select(GenerationOutput)
            .where(
                GenerationOutput.expires_at < before,
                GenerationOutput.is_thumbnail.is_(False),
            )
            .order_by(GenerationOutput.expires_at)
            .limit(limit)
        )
        return result.scalars().all()

    async def delete(
        self,
        output_id: UUID,
        *,
        user_id: UUID | None = None,
    ) -> bool:
        """Delete an output record.

        Args:
            output_id: Output ID to delete.
            user_id: When provided, only deletes if owned by this user.

        Returns:
            True if deleted, False if not found.
        """
        output = await self.get(output_id, user_id=user_id)
        if output is None:
            return False
        await self._session.delete(output)
        await self._session.flush()
        return True

    async def delete_batch(self, output_ids: list[UUID]) -> int:
        """Delete multiple output records in a single round-trip.

        Uses the DELETE statement's ``rowcount`` instead of a separate
        COUNT query — avoids an extra DB round-trip and eliminates
        drift under concurrent deletes.

        Args:
            output_ids: List of output IDs to delete.

        Returns:
            Number of records deleted.
        """
        if not output_ids:
            return 0

        result = await self._session.execute(
            delete(GenerationOutput).where(GenerationOutput.id.in_(output_ids))
        )
        return result.rowcount or 0  # type: ignore[attr-defined]

    async def count_and_sum_by_user(self, user_id: UUID) -> tuple[int, int]:
        """Count full outputs and sum their size for a user.

        Excludes thumbnail rows — only full outputs are counted.

        Args:
            user_id: User to aggregate for.

        Returns:
            Tuple of (count, total_bytes).
        """
        result = await self._session.execute(
            select(
                func.count(GenerationOutput.id),
                func.coalesce(func.sum(GenerationOutput.size_bytes), 0),
            ).where(
                GenerationOutput.user_id == user_id,
                GenerationOutput.is_thumbnail.is_(False),
            )
        )
        count, total_bytes = result.one()
        return int(count), int(total_bytes)
