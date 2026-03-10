"""Output gallery endpoint — enriched outputs with generation context.

Provides a single endpoint that joins ``GenerationOutput`` records with
their parent ``GenerationJob`` so the frontend can render a gallery grid
without extra API calls.

Endpoint:
  GET /v1/storage/gallery  — paginated gallery with job context + presigned URLs

Add this method to the existing ``StorageController`` in
``src/api/routes/storage.py``.
"""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

import msgspec
import structlog
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from src.api.services.storage import R2StorageService
from src.core.enums import GenerationType, JobStatus
from src.db.models.storage import GenerationJob, GenerationOutput

logger = structlog.get_logger(__name__)

_URL_TTL = 3600  # 1 hour presigned URL lifetime


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class GalleryOutputItem(msgspec.Struct, kw_only=True):
    """A single gallery item — one output enriched with job context."""

    output_id: UUID
    """GenerationOutput UUID."""

    job_id: UUID
    """Parent GenerationJob UUID."""

    job_name: str
    """Human-readable job name."""

    provider: str
    """Provider that generated this output."""

    model: str | None = None
    """Model identifier."""

    generation_type: GenerationType
    """Workflow type."""

    status: JobStatus
    """Parent job status."""

    prompt: str
    """Original prompt used for this generation."""

    content_type: str
    """MIME type of the output file."""

    format: str
    """File format string."""

    size_bytes: int
    """Output file size."""

    is_thumbnail: bool
    """True for video poster frames."""

    url: str
    """Presigned download URL (valid ~1 hour)."""

    thumbnail_url: str | None = None
    """Presigned thumbnail URL, if output is a video with an extracted frame."""

    created_at: datetime
    """When the output was created (≈ job completion time)."""

    token_cost: int | None = None
    """Tokens charged for the parent job."""


class GalleryResponse(msgspec.Struct, kw_only=True):
    """Paginated gallery response."""

    items: list[GalleryOutputItem]
    total: int
    limit: int
    offset: int


# ---------------------------------------------------------------------------
# Gallery query helper (used by StorageController)
# ---------------------------------------------------------------------------


async def build_gallery_response(
    user_id: UUID,
    storage: R2StorageService,
    session: AsyncSession,
    *,
    limit: int = 20,
    offset: int = 0,
    generation_type: GenerationType | None = None,
    provider: str | None = None,
) -> GalleryResponse:
    """Build the gallery response for a user.

    Joins outputs with jobs, generates presigned URLs, and handles
    thumbnail lookup for video outputs.

    Args:
        user_id: Authenticated user.
        storage: R2 storage service.
        session: DB session.
        limit: Page size (max 100).
        offset: Page offset.
        generation_type: Optional filter.
        provider: Optional provider filter.

    Returns:
        Paginated ``GalleryResponse``.
    """
    limit = min(limit, 100)

    # Base filter: user's non-thumbnail outputs, joined with completed jobs
    base_filter = [
        GenerationOutput.user_id == user_id,
        GenerationOutput.is_thumbnail == False,  # noqa: E712 — show real outputs only
        GenerationJob.status == JobStatus.COMPLETED.value,
        # Exclude hidden jobs (soft-deleted via sentinel error_message)
        GenerationJob.error_message != "__hidden__",
    ]

    if generation_type is not None:
        base_filter.append(GenerationJob.generation_type == generation_type.value)

    if provider is not None:
        base_filter.append(GenerationJob.provider == provider)

    # Count
    count_q = (
        select(func.count())
        .select_from(GenerationOutput)
        .join(GenerationJob, GenerationOutput.job_id == GenerationJob.id)
        .where(*base_filter)
    )
    total_result = await session.execute(count_q)
    total: int = total_result.scalar_one()

    # Data query — load outputs with parent job eagerly
    data_q = (
        select(GenerationOutput)
        .join(GenerationJob, GenerationOutput.job_id == GenerationJob.id)
        .where(*base_filter)
        .options(selectinload(GenerationOutput.job))
        .order_by(GenerationOutput.created_at.desc())
        .limit(limit)
        .offset(offset)
    )

    outputs_result = await session.execute(data_q)
    outputs = list(outputs_result.scalars().all())

    # Collect job IDs that need thumbnail lookup (video outputs)
    video_job_ids = {out.job_id for out in outputs if out.content_type.startswith("video/")}

    # Fetch thumbnails for video jobs in one query
    thumbnail_map: dict[UUID, str] = {}
    if video_job_ids:
        thumb_q = select(GenerationOutput).where(
            GenerationOutput.job_id.in_(video_job_ids),
            GenerationOutput.is_thumbnail == True,  # noqa: E712
        )
        thumb_result = await session.execute(thumb_q)
        thumbs = thumb_result.scalars().all()
        for thumb in thumbs:
            try:
                r = await storage.get_presigned_url(thumb.storage_key, expires_in=_URL_TTL)
                thumbnail_map[thumb.job_id] = r.presigned_url
            except Exception:
                logger.warning("gallery.thumbnail_url_failed", output_id=str(thumb.id))

    # Build items
    items: list[GalleryOutputItem] = []
    for out in outputs:
        job: GenerationJob = out.job  # eager-loaded

        try:
            url_result = await storage.get_presigned_url(out.storage_key, expires_in=_URL_TTL)
            presigned = url_result.presigned_url
        except Exception:
            logger.warning("gallery.presigned_url_failed", output_id=str(out.id))
            continue

        items.append(
            GalleryOutputItem(
                output_id=out.id,
                job_id=job.id,
                job_name=job.name,
                provider=job.provider,
                model=job.model,
                generation_type=GenerationType(job.generation_type),
                status=JobStatus(job.status),
                prompt=job.prompt,
                content_type=out.content_type,
                format=out.format,
                size_bytes=out.size_bytes,
                is_thumbnail=out.is_thumbnail,
                url=presigned,
                thumbnail_url=thumbnail_map.get(out.job_id),
                created_at=out.created_at,
                token_cost=job.token_cost,
            )
        )

    return GalleryResponse(items=items, total=total, limit=limit, offset=offset)
