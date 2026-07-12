"""FrameExtractionService — creates/reads video frame extraction jobs.

Request-scoped. Job execution itself happens in FrameExtractionWorker; this
service only validates the request, inserts the queued job row, and — on
read — presigns preview URLs / assembles MediaObjects for completed jobs.
No idempotency key, no billing: frame extraction is free and side-effect
bounded (a duplicate job just wastes CPU).
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any
from uuid import UUID

import structlog

from src.api.schemas.frames import (
    ExtractedFrame,
    FrameExtractResult,
    FrameJobResponse,
    FrameJobSource,
    FramePreviewResult,
    PreviewFrame,
)
from src.api.services.media import build_upload_media
from src.core.enums import FrameExtractionKind, FrameExtractionStatus
from src.core.uid import new_id
from src.db.repositories.frame_extraction import FrameExtractionJobRepository
from src.db.repositories.output import OutputRepository
from src.db.repositories.user_image import UserImageRepository

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from src.api.services.storage.r2 import R2StorageService
    from src.db.models.frame_extraction import FrameExtractionJob

logger = structlog.get_logger(__name__)


class FrameExtractionError(Exception):
    """Base exception for frame extraction service errors."""


class FrameSourceNotFoundError(FrameExtractionError):
    """Raised when the requested source output/upload doesn't exist or isn't owned by the caller."""


class FrameSourceNotVideoError(FrameExtractionError):
    """Raised when the requested source exists but isn't a video."""


class FrameJobNotFoundError(FrameExtractionError):
    """Raised when the requested frame extraction job doesn't exist or isn't owned by the caller."""


class FrameExtractionService:
    """Creates and reads video frame extraction jobs."""

    def __init__(
        self,
        session: AsyncSession,
        r2_storage: R2StorageService,
        *,
        product_id: str,
        preview_url_ttl_seconds: int,
        retention_days: int,
    ) -> None:
        self._session = session
        self._storage = r2_storage
        self._product_id = product_id
        self._preview_url_ttl_seconds = preview_url_ttl_seconds
        self._retention_days = retention_days
        self._job_repo = FrameExtractionJobRepository(session)
        self._image_repo = UserImageRepository(session)
        self._output_repo = OutputRepository(session)

    async def create_preview_job(
        self,
        user_id: UUID,
        *,
        source_output_id: UUID | None,
        source_upload_id: UUID | None,
        frame_count: int,
    ) -> FrameExtractionJob:
        """Validate the source and insert a queued preview job."""
        source_kind, source_id = await self._resolve_and_validate_source(
            user_id,
            source_output_id=source_output_id,
            source_upload_id=source_upload_id,
        )
        if source_kind == "upload":
            await self._touch_source_expiry(user_id, source_id)
        job = await self._job_repo.create(
            id=new_id(),
            user_id=user_id,
            product_id=self._product_id,
            kind=FrameExtractionKind.PREVIEW.value,
            params={"frame_count": frame_count},
            source_output_id=source_id if source_kind == "output" else None,
            source_upload_id=source_id if source_kind == "upload" else None,
        )
        logger.info(
            "frames.preview_job_created",
            job_id=str(job.id),
            user_id=str(user_id),
            frame_count=frame_count,
        )
        return job

    async def create_extract_job(
        self,
        user_id: UUID,
        *,
        source_output_id: UUID | None,
        source_upload_id: UUID | None,
        timestamps_ms: list[int],
    ) -> FrameExtractionJob:
        """Validate the source and insert a queued extract job."""
        source_kind, source_id = await self._resolve_and_validate_source(
            user_id,
            source_output_id=source_output_id,
            source_upload_id=source_upload_id,
        )
        if source_kind == "upload":
            await self._touch_source_expiry(user_id, source_id)
        job = await self._job_repo.create(
            id=new_id(),
            user_id=user_id,
            product_id=self._product_id,
            kind=FrameExtractionKind.EXTRACT.value,
            params={"timestamps_ms": timestamps_ms},
            source_output_id=source_id if source_kind == "output" else None,
            source_upload_id=source_id if source_kind == "upload" else None,
        )
        logger.info(
            "frames.extract_job_created",
            job_id=str(job.id),
            user_id=str(user_id),
            timestamp_count=len(timestamps_ms),
        )
        return job

    async def get_job(self, user_id: UUID, job_id: UUID) -> FrameJobResponse:
        """Fetch a job (ownership-checked) and assemble its full response.

        Presigns preview URLs / builds extracted-frame MediaObjects at read
        time — the persisted ``result`` JSON stores only keys and IDs.
        """
        job = await self._job_repo.get(job_id, user_id=user_id)
        if job is None:
            raise FrameJobNotFoundError(f"Job not found: {job_id}")

        source_id = (
            job.source_output_id if job.source_output_id is not None else job.source_upload_id
        )
        if source_id is None:
            raise FrameExtractionError(f"Job {job.id} has no source set (data corruption)")
        source = FrameJobSource(
            type="output" if job.source_output_id is not None else "upload",
            id=source_id,
        )

        preview: FramePreviewResult | None = None
        extracted: FrameExtractResult | None = None

        if job.status == FrameExtractionStatus.COMPLETED.value and job.result is not None:
            if job.kind == FrameExtractionKind.PREVIEW.value:
                preview = await self._build_preview_result(job.result)
            elif job.kind == FrameExtractionKind.EXTRACT.value:
                extracted = await self._build_extract_result(user_id, job.result)

        return FrameJobResponse(
            job_id=job.id,
            kind=job.kind,
            status=job.status,
            created_at=job.created_at,
            started_at=job.started_at,
            finished_at=job.finished_at,
            error=job.error,
            source=source,
            preview=preview,
            extracted=extracted,
        )

    async def _touch_source_expiry(self, user_id: UUID, source_upload_id: UUID) -> None:
        """Reset the source upload's retention window — using it as a frame
        extraction source is "active use", same as i2i input (see
        GenerationService.submit).
        """
        touched = await self._image_repo.touch_expiry(
            source_upload_id,
            user_id=user_id,
            expires_at=datetime.now(UTC) + timedelta(days=self._retention_days),
        )
        if touched:
            logger.info(
                "frames.source_expiry_extended",
                image_id=str(source_upload_id),
                user_id=str(user_id),
                retention_days=self._retention_days,
            )

    async def _resolve_and_validate_source(
        self,
        user_id: UUID,
        *,
        source_output_id: UUID | None,
        source_upload_id: UUID | None,
    ) -> tuple[str, UUID]:
        """Ownership + product_id check, then confirm the source is a video.

        Returns:
            (kind, id) where kind is "output" or "upload".

        Raises:
            FrameSourceNotFoundError: Source doesn't exist, isn't owned by
                ``user_id``, or belongs to a different product.
            FrameSourceNotVideoError: Source exists but isn't a video.
        """
        if source_output_id is not None and source_upload_id is not None:
            raise ValueError("Exactly one of source_output_id/source_upload_id is required")

        if source_output_id is not None:
            output = await self._output_repo.get(source_output_id, user_id=user_id)
            if output is None or output.product_id != self._product_id:
                raise FrameSourceNotFoundError(f"Output not found: {source_output_id}")
            if not output.content_type.startswith("video/"):
                raise FrameSourceNotVideoError(f"Output {source_output_id} is not a video")
            return "output", source_output_id

        if source_upload_id is None:
            raise ValueError("Exactly one of source_output_id/source_upload_id is required")

        upload = await self._image_repo.get(source_upload_id, user_id=user_id)
        if upload is None or upload.product_id != self._product_id:
            raise FrameSourceNotFoundError(f"Upload not found: {source_upload_id}")
        if not upload.content_type.startswith("video/"):
            raise FrameSourceNotVideoError(f"Upload {source_upload_id} is not a video")
        return "upload", source_upload_id

    async def _build_preview_result(self, result: dict[str, Any]) -> FramePreviewResult:
        entries: list[dict[str, Any]] = result["frames"]
        frames = [
            PreviewFrame(
                index=entry["index"],
                timestamp_ms=entry["timestamp_ms"],
                url=await self._storage.sign_key(
                    entry["key"],
                    expires_in=self._preview_url_ttl_seconds,
                ),
            )
            for entry in entries
        ]
        return FramePreviewResult(frames=frames, expires_in_seconds=self._preview_url_ttl_seconds)

    async def _build_extract_result(
        self, user_id: UUID, result: dict[str, Any]
    ) -> FrameExtractResult:
        entries: list[dict[str, Any]] = result["frames"]
        upload_ids = [UUID(entry["upload_id"]) for entry in entries]
        derivatives_map = await self._image_repo.batch_derivatives(upload_ids)

        frames: list[ExtractedFrame] = []
        for entry in entries:
            upload_id = UUID(entry["upload_id"])
            upload = await self._image_repo.get(upload_id, user_id=user_id)
            if upload is None:
                # Extracted frame was deleted (retention sweep, manual delete)
                # since the job completed — drop it from the response rather
                # than error the whole job read.
                continue
            media = build_upload_media(upload, derivatives_map.get(upload_id, []))
            frames.append(
                ExtractedFrame(
                    timestamp_ms=entry["timestamp_ms"],
                    upload_id=upload_id,
                    media=media,
                )
            )
        return FrameExtractResult(frames=frames)
