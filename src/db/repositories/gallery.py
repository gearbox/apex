"""Gallery-specific database queries."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from uuid import UUID

import structlog
from sqlalchemy import literal, select, tuple_
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from src.core.enums import GenerationType, JobStatus, OutputMediaType
from src.db.models.storage import GenerationJob, GenerationOutput

logger = structlog.get_logger(__name__)

_VIDEO_TYPES: list[str] = [gt.value for gt in GenerationType if gt.is_video]
_IMAGE_TYPES: list[str] = [gt.value for gt in GenerationType if not gt.is_video]


@dataclass(frozen=True)
class CoverData:
    """Cover resolution data for a single gallery job."""

    cover_output_id: UUID | None = None
    video_output_id: UUID | None = None
    thumbnail_output_id: UUID | None = None
    output_count: int = 0


class GalleryRepository:
    """Gallery-specific database queries."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def list_gallery_jobs(
        self,
        user_id: UUID,
        product_id: str,
        *,
        limit: int = 20,
        cursor_ts: datetime | None = None,
        cursor_id: UUID | None = None,
        media_type: OutputMediaType | None = None,
        generation_type: GenerationType | None = None,
        model: str | None = None,
    ) -> Sequence[GenerationJob]:
        """List paginated completed jobs for gallery grid.

        Uses limit+1 fetch pattern. Caller checks len(result) > limit for has_more.

        Args:
            user_id: Owner of the jobs.
            product_id: Product scope.
            limit: Page size (fetch limit+1).
            cursor_ts: created_at of last item on previous page.
            cursor_id: id of last item on previous page.
            media_type: Filter to IMAGE or VIDEO generation types.
            generation_type: Filter to a specific generation type.
            model: Filter to a specific model.

        Returns:
            Sequence of GenerationJob ordered by created_at DESC, id DESC.
        """
        query = select(GenerationJob).where(
            GenerationJob.user_id == user_id,
            GenerationJob.product_id == product_id,
            GenerationJob.status == JobStatus.COMPLETED,
            GenerationJob.is_deleted.is_(False),
        )

        if media_type == OutputMediaType.VIDEO:
            query = query.where(GenerationJob.generation_type.in_(_VIDEO_TYPES))
        elif media_type == OutputMediaType.IMAGE:
            query = query.where(GenerationJob.generation_type.in_(_IMAGE_TYPES))

        if generation_type is not None:
            query = query.where(GenerationJob.generation_type == generation_type.value)

        if model is not None:
            query = query.where(GenerationJob.model == model)

        if cursor_ts is not None and cursor_id is not None:
            query = query.where(
                tuple_(GenerationJob.created_at, GenerationJob.id)
                < tuple_(literal(cursor_ts), literal(cursor_id))
            )

        result = await self._session.execute(
            query.order_by(
                GenerationJob.created_at.desc(),
                GenerationJob.id.desc(),
            ).limit(limit + 1)
        )
        return result.scalars().all()

    async def get_gallery_job(
        self,
        job_id: UUID,
        user_id: UUID,
        product_id: str,
    ) -> GenerationJob | None:
        """Get a single completed job with eager-loaded relationships.

        Args:
            job_id: Job to fetch.
            user_id: Owner check.
            product_id: Product scope check.

        Returns:
            GenerationJob with outputs, source_job, input_image loaded; None if not found.
        """
        result = await self._session.execute(
            select(GenerationJob)
            .where(
                GenerationJob.id == job_id,
                GenerationJob.user_id == user_id,
                GenerationJob.product_id == product_id,
                GenerationJob.status == JobStatus.COMPLETED,
                GenerationJob.is_deleted.is_(False),
            )
            .options(
                selectinload(GenerationJob.outputs),
                selectinload(GenerationJob.source_job),
                selectinload(GenerationJob.input_image),
            )
        )
        return result.scalar_one_or_none()

    async def batch_cover_data(
        self,
        job_ids: list[UUID],
    ) -> dict[UUID, CoverData]:
        """Fetch all outputs for a batch of jobs and build cover data.

        Args:
            job_ids: List of job IDs (bounded by page size, max 25).

        Returns:
            Mapping from job_id to CoverData.
        """
        if not job_ids:
            return {}

        result = await self._session.execute(
            select(GenerationOutput)
            .where(GenerationOutput.job_id.in_(job_ids))
            .order_by(GenerationOutput.job_id, GenerationOutput.output_index)
        )
        outputs = result.scalars().all()

        # Group outputs by job_id
        by_job: dict[UUID, list[GenerationOutput]] = defaultdict(list)
        for output in outputs:
            by_job[output.job_id].append(output)

        cover_map: dict[UUID, CoverData] = {}
        for job_id in job_ids:
            job_outputs = by_job.get(job_id, [])
            thumbnail_output_id: UUID | None = None
            video_output_id: UUID | None = None
            cover_output_id: UUID | None = None
            real_count = 0

            for out in job_outputs:
                if out.is_thumbnail:
                    thumbnail_output_id = out.id
                else:
                    real_count += 1
                    if out.content_type.startswith("video/"):
                        # Take the first video output found
                        if video_output_id is None:
                            video_output_id = out.id
                    else:
                        # Last real image output (highest index) becomes cover
                        cover_output_id = out.id

            cover_map[job_id] = CoverData(
                cover_output_id=cover_output_id,
                video_output_id=video_output_id,
                thumbnail_output_id=thumbnail_output_id,
                output_count=real_count,
            )

        return cover_map
