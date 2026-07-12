"""Video frame extraction API routes.

Two-phase flow: POST /preview and POST /extract both return 202 with a job
id; the client polls GET /jobs/{id} until status is completed|failed. No
idempotency key, no billing (frame extraction is free — see CLAUDE.md
idempotency-pattern rule 8, which does not apply here).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from uuid import UUID

import structlog
from litestar import Controller, Response, get, post
from litestar.di import Provide
from litestar.status_codes import (
    HTTP_200_OK,
    HTTP_202_ACCEPTED,
    HTTP_400_BAD_REQUEST,
    HTTP_404_NOT_FOUND,
)

from src.api.dependencies.auth import get_current_user_id
from src.api.schemas.errors import ErrorEnvelope
from src.api.schemas.frames import (
    FrameExtractRequest,
    FrameJobCreatedResponse,
    FrameJobResponse,
    FramePreviewRequest,
)
from src.api.security import auth_guard
from src.api.services.frames.service import (
    FrameExtractionService,
    FrameJobNotFoundError,
    FrameSourceNotFoundError,
    FrameSourceNotVideoError,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from src.db.models.frame_extraction import FrameExtractionJob

logger = structlog.get_logger(__name__)

JOB_NOT_FOUND = "Frame extraction job not found"


def _invalid_source_response() -> Response[Any]:
    return Response(
        content=ErrorEnvelope(
            error="invalid_source",
            message="Exactly one of source_output_id/source_upload_id is required",
            status_code=HTTP_400_BAD_REQUEST,
        ),
        status_code=HTTP_400_BAD_REQUEST,
    )


def _job_created_response(job: FrameExtractionJob) -> Response[Any]:
    return Response(
        content=FrameJobCreatedResponse(job_id=job.id, status=job.status),
        status_code=HTTP_202_ACCEPTED,
    )


class FramesController(Controller):
    """Video frame preview/extraction endpoints.

    Sources: either a ``GenerationOutput`` video or a user-uploaded video
    (``UserImage`` row with a video content type). Extraction is free — no
    token charge, no idempotency-key requirement (D6).
    """

    path = "/v1/frames"
    tags: Sequence[str] | None = ("Frames",)
    guards = [auth_guard]  # noqa: RUF012
    dependencies = {"current_user_id": Provide(get_current_user_id)}  # noqa: RUF012

    @post("/preview")
    async def create_preview(
        self,
        current_user_id: UUID,
        frame_extraction_service: FrameExtractionService,
        data: FramePreviewRequest,
    ) -> Response[FrameJobCreatedResponse | ErrorEnvelope]:
        """Request a low-res, N-frame preview strip (async job).

        Exactly one of ``source_output_id`` / ``source_upload_id`` must be
        provided. Poll ``GET /jobs/{job_id}`` until ``status`` is
        ``completed`` or ``failed``.
        """
        if (data.source_output_id is None) == (data.source_upload_id is None):
            return _invalid_source_response()

        try:
            job = await frame_extraction_service.create_preview_job(
                current_user_id,
                source_output_id=data.source_output_id,
                source_upload_id=data.source_upload_id,
                frame_count=data.frame_count,
            )
        except FrameSourceNotFoundError as e:
            logger.warning("frames.source_not_found", error=str(e))
            return Response(
                content=ErrorEnvelope(
                    error="not_found", message=str(e), status_code=HTTP_404_NOT_FOUND
                ),
                status_code=HTTP_404_NOT_FOUND,
            )
        except FrameSourceNotVideoError as e:
            logger.warning("frames.not_a_video", error=str(e))
            return Response(
                content=ErrorEnvelope(
                    error="not_a_video", message=str(e), status_code=HTTP_400_BAD_REQUEST
                ),
                status_code=HTTP_400_BAD_REQUEST,
            )

        return _job_created_response(job)

    @post("/extract")
    async def create_extract(
        self,
        current_user_id: UUID,
        frame_extraction_service: FrameExtractionService,
        data: FrameExtractRequest,
    ) -> Response[FrameJobCreatedResponse | ErrorEnvelope]:
        """Request full-resolution frame extraction at specific timestamps (async job).

        Exactly one of ``source_output_id`` / ``source_upload_id`` must be
        provided. Each saved frame becomes a ``UserImage`` upload with
        lineage, immediately usable for i2i/i2v and downloadable via the
        content proxy. Each timestamp must be within ``[0, duration_ms)`` —
        the upper bound is exclusive, since an end-of-stream seek frequently
        decodes nothing.
        """
        if (data.source_output_id is None) == (data.source_upload_id is None):
            return _invalid_source_response()

        try:
            job = await frame_extraction_service.create_extract_job(
                current_user_id,
                source_output_id=data.source_output_id,
                source_upload_id=data.source_upload_id,
                timestamps_ms=data.timestamps_ms,
            )
        except FrameSourceNotFoundError as e:
            logger.warning("frames.source_not_found", error=str(e))
            return Response(
                content=ErrorEnvelope(
                    error="not_found", message=str(e), status_code=HTTP_404_NOT_FOUND
                ),
                status_code=HTTP_404_NOT_FOUND,
            )
        except FrameSourceNotVideoError as e:
            logger.warning("frames.not_a_video", error=str(e))
            return Response(
                content=ErrorEnvelope(
                    error="not_a_video", message=str(e), status_code=HTTP_400_BAD_REQUEST
                ),
                status_code=HTTP_400_BAD_REQUEST,
            )

        return _job_created_response(job)

    @get("/jobs/{job_id:uuid}")
    async def get_job(
        self,
        current_user_id: UUID,
        frame_extraction_service: FrameExtractionService,
        job_id: UUID,
    ) -> Response[FrameJobResponse | ErrorEnvelope]:
        """Poll a frame extraction job's status/result.

        Ownership-checked — returns 404 for a foreign user's job.
        """
        try:
            response = await frame_extraction_service.get_job(current_user_id, job_id)
        except FrameJobNotFoundError:
            return Response(
                content=ErrorEnvelope(
                    error="not_found", message=JOB_NOT_FOUND, status_code=HTTP_404_NOT_FOUND
                ),
                status_code=HTTP_404_NOT_FOUND,
            )

        return Response(content=response, status_code=HTTP_200_OK)
