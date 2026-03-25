"""Gallery service — cover resolution, badge, lineage, content URL building."""

from __future__ import annotations

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
            cover_url, video_url = self._resolve_cover(job, cover_data)
            gt = GenerationType(job.generation_type)
            items.append(
                GalleryGridItem(
                    job_id=job.id,
                    cover_url=cover_url,
                    video_url=video_url,
                    badge=self._resolve_badge(gt),
                    media_type=self._resolve_media_type(gt),
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
        non_thumbnail_outputs = [o for o in job.outputs if not o.is_thumbnail]
        thumbnail_map = {o.job_id: o for o in job.outputs if o.is_thumbnail}

        output_items = [
            self._build_output_item(o, thumbnail_map.get(o.job_id))
            for o in sorted(non_thumbnail_outputs, key=lambda o: o.output_index)
        ]

        # Resolve input image URL for detail view header
        input_url: str | None = None
        if job.source_output_id is not None:
            input_url = self._output_url(job.source_output_id)
        elif job.input_image_id is not None:
            input_url = self._upload_url(job.input_image_id)

        return GalleryGroupDetail(
            job_id=job.id,
            badge=self._resolve_badge(gt),
            input_image_url=input_url,
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

    @staticmethod
    def _output_media_type(content_type: str) -> OutputMediaType:
        """Classify media type from MIME content_type."""
        return OutputMediaType.VIDEO if content_type.startswith("video/") else OutputMediaType.IMAGE

    @staticmethod
    def _output_url(output_id: UUID) -> str:
        return f"/v1/content/outputs/{output_id}"

    @staticmethod
    def _upload_url(image_id: UUID) -> str:
        return f"/v1/content/uploads/{image_id}"

    def _resolve_cover(
        self,
        job: GenerationJob,
        cover_data: CoverData,
    ) -> tuple[str, str | None]:
        """Resolve cover_url and video_url for a grid item.

        Returns:
            (cover_url, video_url)
        """
        gt = GenerationType(job.generation_type)
        video_url: str | None = None

        # For ALL video types, set video_url if we have a video output
        if gt.is_video and cover_data.video_output_id is not None:
            video_url = self._output_url(cover_data.video_output_id)

        if gt.requires_image_input:
            # i2i, i2v, flf2v: prefer source output, then upload, then cover
            if job.source_output_id is not None:
                return self._output_url(job.source_output_id), video_url
            if job.input_image_id is not None:
                return self._upload_url(job.input_image_id), video_url
            if cover_data.cover_output_id is not None:
                return self._output_url(cover_data.cover_output_id), video_url
            # Fallback: use thumbnail if available
            if cover_data.thumbnail_output_id is not None:
                return self._output_url(cover_data.thumbnail_output_id), video_url

        elif gt.requires_video_input:
            # v2v: prefer source output, then thumbnail
            if job.source_output_id is not None:
                return self._output_url(job.source_output_id), video_url
            if cover_data.thumbnail_output_id is not None:
                return self._output_url(cover_data.thumbnail_output_id), video_url

        elif gt == GenerationType.T2V:
            # t2v: cover = thumbnail, video = video output
            cover = (
                self._output_url(cover_data.thumbnail_output_id)
                if cover_data.thumbnail_output_id is not None
                else self._output_url(cover_data.video_output_id)
                if cover_data.video_output_id is not None
                else "/v1/content/outputs/unknown"
            )
            return cover, video_url

        else:
            # t2i: last generated output
            if cover_data.cover_output_id is not None:
                return self._output_url(cover_data.cover_output_id), None

        # Final fallback
        return "/v1/content/outputs/unknown", video_url

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
        return truncated + "\u2026"

    def _build_output_item(
        self,
        output: GenerationOutput,
        _thumbnail: GenerationOutput | None = None,
    ) -> GalleryOutputItem:
        """Build a GalleryOutputItem from a GenerationOutput."""
        thumbnail_url: str | None = None
        if (
            self._output_media_type(output.content_type) == OutputMediaType.VIDEO
            and _thumbnail is not None
        ):
            thumbnail_url = self._output_url(_thumbnail.id)

        return GalleryOutputItem(
            id=output.id,
            url=self._output_url(output.id),
            thumbnail_url=thumbnail_url,
            content_type=output.content_type,
            media_type=self._output_media_type(output.content_type),
            format=output.format,
            size_bytes=output.size_bytes,
            output_index=output.output_index,
            created_at=output.created_at,
        )
