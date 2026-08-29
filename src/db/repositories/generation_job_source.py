"""Persistence helpers for ordered generation input provenance."""

from __future__ import annotations

from typing import TYPE_CHECKING

from sqlalchemy import select

from src.core.library_ref import LibraryAssetSource
from src.db.models.storage import GenerationJobSource

if TYPE_CHECKING:
    from collections.abc import Sequence
    from uuid import UUID

    from sqlalchemy.ext.asyncio import AsyncSession

    from src.api.services.generation.source_media import ResolvedSourceMedia


class GenerationJobSourceRepository:
    """Create and read ordered ``generation_job_sources`` rows."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def create_many(
        self,
        *,
        job_id: UUID,
        product_id: str,
        sources: Sequence[ResolvedSourceMedia],
    ) -> list[GenerationJobSource]:
        """Persist all resolved sources in their request order."""
        rows = [
            GenerationJobSource(
                job_id=job_id,
                position=source.position,
                product_id=product_id,
                source_upload_id=(
                    source.ref.asset_id if source.ref.source is LibraryAssetSource.UPLOAD else None
                ),
                source_output_id=(
                    source.ref.asset_id if source.ref.source is LibraryAssetSource.OUTPUT else None
                ),
                asset_ref=source.asset_ref,
                media_kind=source.media_kind.value,
            )
            for source in sources
        ]
        self._session.add_all(rows)
        await self._session.flush()
        return rows

    async def list_for_job(
        self,
        job_id: UUID,
        *,
        product_id: str,
    ) -> Sequence[GenerationJobSource]:
        """Return a job's source rows in their persisted 0-based order."""
        result = await self._session.execute(
            select(GenerationJobSource)
            .where(
                GenerationJobSource.job_id == job_id,
                GenerationJobSource.product_id == product_id,
            )
            .order_by(GenerationJobSource.position)
        )
        return result.scalars().all()
