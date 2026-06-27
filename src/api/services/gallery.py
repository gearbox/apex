"""Gallery service — cover resolution, badge, lineage, content URL building."""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING
from uuid import UUID

import structlog

from src.api.schemas.gallery import (
    GalleryGridItem,
    GalleryGroupDetail,
    GalleryLineage,
    GalleryOutputItem,
)
from src.api.schemas.pagination import CursorPage, decode_cursor, encode_cursor
from src.api.services.media import build_output_media, build_upload_media
from src.core.enums import GalleryBadge, GallerySourceType, GenerationType, OutputMediaType
from src.db.repositories.gallery import CoverData, GalleryRepository

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from src.db.models.storage import GenerationJob, GenerationOutput

logger = structlog.get_logger(__name__)


class GalleryService:
    """Business logic for gallery endpoints."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    async def list_gallery(
        self,
        user_id: UUID,
        product_id: str,
        *,
        session: AsyncSession,
        limit: int = 20,
        cursor: str | None = None,
        media_type: OutputMediaType | None = None,
        generation_type: GenerationType | None = None,
        model: str | None = None,
    ) -> CursorPage[GalleryGridItem]:
        """Return paginated gallery grid.

        Args:
            user_id: Requesting user.
            product_id: Product scope.
            session: Database session.
            limit: Page size.
            cursor: Opaque cursor from previous page.
            media_type: Optional media type filter.
            generation_type: Optional generation type filter.
            model: Optional model filter.

        Returns:
            CursorPage of GalleryGridItem.
        """
        repo = GalleryRepository(session)

        cursor_ts = None
        cursor_id = None
        if cursor is not None:
            cursor_ts, cursor_id = decode_cursor(cursor)

        rows = await repo.list_gallery_jobs(
            user_id,
            product_id,
            limit=limit,
            cursor_ts=cursor_ts,
            cursor_id=cursor_id,
            media_type=media_type,
            generation_type=generation_type,
            model=model,
        )

        has_more = len(rows) > limit
        page_rows = list(rows[:limit])

        job_ids = [job.id for job in page_rows]
        cover_map = await repo.batch_cover_data(job_ids)

        items: list[GalleryGridItem] = []
        for job in page_rows:
            cover_data = cover_map.get(job.id, CoverData())
            if cover_data.primary_output is None:
                logger.warning("gallery.job_has_no_outputs", job_id=str(job.id))
                continue

            gt = GenerationType(job.generation_type)
            cover = build_output_media(cover_data.primary_output, cover_data.primary_derivatives)
            items.append(
                GalleryGridItem(
                    job_id=job.id,
                    cover=cover,
                    badge=self._resolve_badge(gt),
                    output_count=cover_data.output_count,
                    generation_type=gt,
                    model=job.model,
                    aspect_ratio=job.aspect_ratio,
                    prompt_snippet=self._prompt_snippet(job.prompt),
                    created_at=job.created_at,
                )
            )

        next_cursor = None
        if has_more and page_rows:
            last = page_rows[-1]
            next_cursor = encode_cursor(last.created_at, last.id)

        return CursorPage(
            items=items,
            limit=limit,
            has_more=has_more,
            next_cursor=next_cursor,
        )

    async def get_gallery_detail(
        self,
        job_id: UUID,
        user_id: UUID,
        product_id: str,
        *,
        session: AsyncSession,
    ) -> GalleryGroupDetail | None:
        """Return full detail view for a generation group.

        Args:
            job_id: Generation job ID.
            user_id: Requesting user.
            product_id: Product scope.
            session: Database session.

        Returns:
            GalleryGroupDetail or None if not found.
        """
        repo = GalleryRepository(session)
        job = await repo.get_gallery_job(job_id, user_id, product_id)
        if job is None:
            return None

        gt = GenerationType(job.generation_type)

        # Separate full outputs from thumbnails and build derivatives map
        full_outputs = [o for o in job.outputs if not o.is_thumbnail]
        derivatives_map: dict[UUID, list[GenerationOutput]] = defaultdict(list)
        for o in job.outputs:
            if o.is_thumbnail and o.parent_output_id is not None:
                derivatives_map[o.parent_output_id].append(o)

        output_items = [
            self._build_output_item(o, list(derivatives_map.get(o.id, [])))
            for o in sorted(full_outputs, key=lambda o: o.output_index)
        ]

        # Resolve input_media for detail view header
        input_media = None
        if job.source_output is not None:
            # Derivatives were eagerly loaded on source_output
            input_media = build_output_media(job.source_output, list(job.source_output.derivatives))
        elif job.input_image is not None:
            # Derivatives were eagerly loaded on input_image
            input_media = build_upload_media(job.input_image, list(job.input_image.derivatives))

        return GalleryGroupDetail(
            job_id=job.id,
            badge=self._resolve_badge(gt),
            input_media=input_media,
            prompt=job.prompt,
            negative_prompt=job.negative_prompt,
            outputs=output_items,
            media_type=self._resolve_media_type(gt),
            model=job.model,
            provider=job.provider,
            generation_type=gt,
            aspect_ratio=job.aspect_ratio,
            token_cost=job.token_cost,
            created_at=job.created_at,
            completed_at=job.completed_at,
            lineage=self._build_lineage(job),
        )

    # -------------------------------------------------------------------------
    # Private helpers
    # -------------------------------------------------------------------------

    @staticmethod
    def _resolve_badge(gt: GenerationType) -> GalleryBadge:
        """Resolve gallery badge from generation type."""
        if gt.requires_image_input or gt.requires_video_input:
            return GalleryBadge.IMAGE
        return GalleryBadge.PROMPT

    @staticmethod
    def _resolve_media_type(gt: GenerationType) -> OutputMediaType:
        """Resolve output media type from generation type."""
        return OutputMediaType.VIDEO if gt.is_video else OutputMediaType.IMAGE

    def _build_lineage(self, job: GenerationJob) -> GalleryLineage | None:
        """Build lineage info from a job's source fields."""
        if job.source_job_id is not None:
            source_job = job.source_job
            return GalleryLineage(
                source_type=GallerySourceType.GENERATION,
                source_job_id=job.source_job_id,
                source_job_name=source_job.name if source_job is not None else None,
                source_output_id=job.source_output_id,
            )
        if job.input_image_id is not None:
            return GalleryLineage(
                source_type=GallerySourceType.UPLOAD,
                source_upload_id=job.input_image_id,
            )
        return None

    @staticmethod
    def _prompt_snippet(prompt: str, max_len: int = 100) -> str:
        """Truncate prompt at word boundary."""
        if len(prompt) <= max_len:
            return prompt
        truncated = prompt[:max_len].rsplit(" ", 1)[0]
        return f"{truncated}…"

    @staticmethod
    def _build_output_item(
        output: GenerationOutput,
        derivatives: list[GenerationOutput],
    ) -> GalleryOutputItem:
        """Build a GalleryOutputItem from a GenerationOutput and its derivatives."""
        return GalleryOutputItem(
            id=output.id,
            output_index=output.output_index,
            created_at=output.created_at,
            media=build_output_media(output, derivatives),
        )
