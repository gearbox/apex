"""Grok API routes for image and video generation."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from datetime import datetime, timezone
from uuid import UUID

from litestar import Controller, Response, get, post
from litestar.status_codes import (
    HTTP_200_OK,
    HTTP_201_CREATED,
    HTTP_400_BAD_REQUEST,
    HTTP_404_NOT_FOUND,
    HTTP_503_SERVICE_UNAVAILABLE,
)
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.schemas.grok import (
    GROK_MODELS,
    GrokImageEditRequest,
    GrokImageRequest,
    GrokJobResponse,
    GrokJobStatusResponse,
    GrokProviderInfo,
    GrokVideoEditRequest,
    GrokVideoFromImageRequest,
    GrokVideoRequest,
)
from src.api.services.grok.job_service import GrokJobError, GrokJobService
from src.api.services.storage import R2StorageService
from src.core.enums import GenerationType, JobStatus

logger = logging.getLogger(__name__)


class GrokProviderController(Controller):
    """Grok provider information endpoints."""

    path = "/api/v1/grok"
    tags: Sequence[str] | None = ["Grok Provider"]

    @get("/")
    async def get_provider_info(
        self,
        grok_job_service: GrokJobService | None,
    ) -> GrokProviderInfo:
        """Get Grok provider information and available models.

        Returns provider configuration status and supported models.
        """
        return GrokProviderInfo(
            available=grok_job_service is not None,
            models=GROK_MODELS,
        )


class GrokImageController(Controller):
    """Grok image generation endpoints."""

    path = "/api/v1/grok/image"
    tags: Sequence[str] | None = ["Grok Image"]

    @post("/")
    async def generate_image(
        self,
        data: GrokImageRequest,
        session: AsyncSession,
        grok_job_service: GrokJobService | None,
    ) -> Response[GrokJobResponse]:
        """Generate images from text prompt.

        Creates and executes an image generation job using Grok.
        Results are stored in R2 and can be accessed via job status endpoint.

        Supports:
        - grok-imagine-image: Up to 10 images per request
        - grok-2-image-1212: Up to 10 images per request
        """
        if grok_job_service is None:
            return Response(
                content=GrokJobResponse(
                    job_id=UUID("00000000-0000-0000-0000-000000000000"),
                    status=JobStatus.FAILED,
                    name="Service Unavailable",
                    model=data.model,
                    generation_type=GenerationType.T2I,
                    created_at=datetime.now(timezone.utc),
                    message="Grok provider is not configured",
                ),
                status_code=HTTP_503_SERVICE_UNAVAILABLE,
            )

        try:
            # TODO: Get user_id from auth context
            # For now, use a placeholder
            # user_id = UUID("00000000-0000-0000-0000-000000000001")
            user_id = UUID("393132ed-0cb2-4642-877c-f8dc562bffb6")

            job = await grok_job_service.create_image_job(
                session=session,
                user_id=user_id,
                prompt=data.prompt,
                model=data.model,
                generation_type=GenerationType.T2I,
                n=data.n,
                aspect_ratio=data.aspect_ratio,
                name=data.name,
            )

            await session.commit()

            if job is None:
                return Response(
                    content=GrokJobResponse(
                        job_id=UUID("00000000-0000-0000-0000-000000000000"),
                        status=JobStatus.FAILED,
                        name=data.name or data.prompt[:50],
                        model=data.model,
                        generation_type=GenerationType.T2I,
                        created_at=datetime.now(timezone.utc),
                        message="Job creation failed",
                    ),
                    status_code=HTTP_400_BAD_REQUEST,
                )

            return Response(
                content=GrokJobResponse(
                    job_id=job.id,
                    status=JobStatus(job.status),
                    name=job.name,
                    model=data.model,
                    generation_type=GenerationType.T2I,
                    created_at=job.created_at,
                    message="Image generation completed"
                    if job.status == JobStatus.COMPLETED.value
                    else None,
                ),
                status_code=HTTP_201_CREATED,
            )

        except GrokJobError as e:
            logger.error(f"Grok image generation failed: {e}")
            return Response(
                content=GrokJobResponse(
                    job_id=UUID("00000000-0000-0000-0000-000000000000"),
                    status=JobStatus.FAILED,
                    name=data.name or data.prompt[:50],
                    model=data.model,
                    generation_type=GenerationType.T2I,
                    created_at=datetime.now(timezone.utc),
                    message=str(e),
                ),
                status_code=HTTP_400_BAD_REQUEST,
            )

    @post("/edit")
    async def edit_image(
        self,
        data: GrokImageEditRequest,
        session: AsyncSession,
        grok_job_service: GrokJobService | None,
        r2_storage: R2StorageService,
    ) -> Response[GrokJobResponse]:
        """Edit an existing image with a text prompt.

        Performs image-to-image editing using grok-imagine-image.
        The input image must be previously uploaded via the storage endpoints.
        """
        if grok_job_service is None:
            return Response(
                content=GrokJobResponse(
                    job_id=UUID("00000000-0000-0000-0000-000000000000"),
                    status=JobStatus.FAILED,
                    name="Service Unavailable",
                    model=data.model,
                    generation_type=GenerationType.I2I,
                    created_at=datetime.now(timezone.utc),
                    message="Grok provider is not configured",
                ),
                status_code=HTTP_503_SERVICE_UNAVAILABLE,
            )

        try:
            # TODO: Get user_id from auth context
            user_id = UUID("00000000-0000-0000-0000-000000000001")

            # Get input image URL from storage
            # TODO: Look up the image record and get its storage key
            from src.db import StorageRepository

            repo = StorageRepository(session)
            input_image = await repo.get_upload(data.input_image_id)

            if input_image is None:
                return Response(
                    content=GrokJobResponse(
                        job_id=UUID("00000000-0000-0000-0000-000000000000"),
                        status=JobStatus.FAILED,
                        name="Invalid Input",
                        model=data.model,
                        generation_type=GenerationType.I2I,
                        created_at=datetime.now(timezone.utc),
                        message=f"Input image {data.input_image_id} not found",
                    ),
                    status_code=HTTP_404_NOT_FOUND,
                )

            # Generate presigned URL for the input image
            url_result = await r2_storage.get_presigned_url(
                input_image.storage_key,
                expires_in=3600,
            )
            input_image_url = url_result.presigned_url

            job = await grok_job_service.create_image_job(
                session=session,
                user_id=user_id,
                prompt=data.prompt,
                model=data.model,
                generation_type=GenerationType.I2I,
                n=1,  # Editing produces single output
                name=data.name,
                input_image_url=input_image_url,
                input_image_id=data.input_image_id,
            )

            await session.commit()

            if job is None:
                return Response(
                    content=GrokJobResponse(
                        job_id=UUID("00000000-0000-0000-0000-000000000000"),
                        status=JobStatus.FAILED,
                        name=data.name or data.prompt[:50],
                        model=data.model,
                        generation_type=GenerationType.I2I,
                        created_at=datetime.now(timezone.utc),
                        message="Job creation failed",
                    ),
                    status_code=HTTP_400_BAD_REQUEST,
                )

            return Response(
                content=GrokJobResponse(
                    job_id=job.id,
                    status=JobStatus(job.status),
                    name=job.name,
                    model=data.model,
                    generation_type=GenerationType.I2I,
                    created_at=job.created_at,
                    message="Image editing completed"
                    if job.status == JobStatus.COMPLETED.value
                    else None,
                ),
                status_code=HTTP_201_CREATED,
            )

        except GrokJobError as e:
            logger.error(f"Grok image editing failed: {e}")
            return Response(
                content=GrokJobResponse(
                    job_id=UUID("00000000-0000-0000-0000-000000000000"),
                    status=JobStatus.FAILED,
                    name=data.name or data.prompt[:50],
                    model=data.model,
                    generation_type=GenerationType.I2I,
                    created_at=datetime.now(timezone.utc),
                    message=str(e),
                ),
                status_code=HTTP_400_BAD_REQUEST,
            )


class GrokVideoController(Controller):
    """Grok video generation endpoints."""

    path = "/api/v1/grok/video"
    tags: Sequence[str] | None = ["Grok Video"]

    @post("/")
    async def generate_video(
        self,
        data: GrokVideoRequest,
        session: AsyncSession,
        grok_job_service: GrokJobService | None,
    ) -> Response[GrokJobResponse]:
        """Generate video from text prompt.

        Starts an asynchronous video generation job. The job will be processed
        in the background and results can be polled via the job status endpoint.

        Video generation typically takes 30-120 seconds depending on duration
        and resolution.
        """
        if grok_job_service is None:
            return Response(
                content=GrokJobResponse(
                    job_id=UUID("00000000-0000-0000-0000-000000000000"),
                    status=JobStatus.FAILED,
                    name="Service Unavailable",
                    model=data.model,
                    generation_type=GenerationType.T2V,
                    created_at=datetime.now(timezone.utc),
                    message="Grok provider is not configured",
                ),
                status_code=HTTP_503_SERVICE_UNAVAILABLE,
            )

        try:
            # TODO: Get user_id from auth context
            user_id = UUID("00000000-0000-0000-0000-000000000001")

            job = await grok_job_service.start_video_job(
                session=session,
                user_id=user_id,
                prompt=data.prompt,
                model=data.model,
                generation_type=GenerationType.T2V,
                duration=data.duration,
                aspect_ratio=data.aspect_ratio,
                resolution=data.resolution,
                name=data.name,
            )

            await session.commit()

            if job is None:
                return Response(
                    content=GrokJobResponse(
                        job_id=UUID("00000000-0000-0000-0000-000000000000"),
                        status=JobStatus.FAILED,
                        name=data.name or data.prompt[:50],
                        model=data.model,
                        generation_type=GenerationType.T2V,
                        created_at=datetime.now(timezone.utc),
                        message="Job creation failed",
                    ),
                    status_code=HTTP_400_BAD_REQUEST,
                )

            return Response(
                content=GrokJobResponse(
                    job_id=job.id,
                    status=JobStatus(job.status),
                    name=job.name,
                    model=data.model,
                    generation_type=GenerationType.T2V,
                    created_at=job.created_at,
                    message="Video generation started. Poll job status for results.",
                ),
                status_code=HTTP_201_CREATED,
            )

        except GrokJobError as e:
            logger.error(f"Grok video generation failed to start: {e}")
            return Response(
                content=GrokJobResponse(
                    job_id=UUID("00000000-0000-0000-0000-000000000000"),
                    status=JobStatus.FAILED,
                    name=data.name or data.prompt[:50],
                    model=data.model,
                    generation_type=GenerationType.T2V,
                    created_at=datetime.now(timezone.utc),
                    message=str(e),
                ),
                status_code=HTTP_400_BAD_REQUEST,
            )

    @post("/from-image")
    async def generate_video_from_image(
        self,
        data: GrokVideoFromImageRequest,
        session: AsyncSession,
        grok_job_service: GrokJobService | None,
        r2_storage: R2StorageService,
    ) -> Response[GrokJobResponse]:
        """Generate video from an image (image-to-video).

        Animates an existing image into a video. The input image must be
        previously uploaded via the storage endpoints.
        """
        if grok_job_service is None:
            return Response(
                content=GrokJobResponse(
                    job_id=UUID("00000000-0000-0000-0000-000000000000"),
                    status=JobStatus.FAILED,
                    name="Service Unavailable",
                    model=data.model,
                    generation_type=GenerationType.I2V,
                    created_at=datetime.now(timezone.utc),
                    message="Grok provider is not configured",
                ),
                status_code=HTTP_503_SERVICE_UNAVAILABLE,
            )

        try:
            # TODO: Get user_id from auth context
            user_id = UUID("00000000-0000-0000-0000-000000000001")

            # Get input image URL from storage
            from src.db import StorageRepository

            repo = StorageRepository(session)
            input_image = await repo.get_upload(data.input_image_id)

            if input_image is None:
                return Response(
                    content=GrokJobResponse(
                        job_id=UUID("00000000-0000-0000-0000-000000000000"),
                        status=JobStatus.FAILED,
                        name="Invalid Input",
                        model=data.model,
                        generation_type=GenerationType.I2V,
                        created_at=datetime.now(timezone.utc),
                        message=f"Input image {data.input_image_id} not found",
                    ),
                    status_code=HTTP_404_NOT_FOUND,
                )

            # Generate presigned URL for the input image
            url_result = await r2_storage.get_presigned_url(
                input_image.storage_key,
                expires_in=3600,
            )
            input_image_url = url_result.presigned_url

            job = await grok_job_service.start_video_job(
                session=session,
                user_id=user_id,
                prompt=data.prompt,
                model=data.model,
                generation_type=GenerationType.I2V,
                duration=data.duration,
                aspect_ratio=data.aspect_ratio,
                resolution=data.resolution,
                name=data.name,
                input_image_url=input_image_url,
            )

            await session.commit()

            if job is None:
                return Response(
                    content=GrokJobResponse(
                        job_id=UUID("00000000-0000-0000-0000-000000000000"),
                        status=JobStatus.FAILED,
                        name=data.name or data.prompt[:50],
                        model=data.model,
                        generation_type=GenerationType.I2V,
                        created_at=datetime.now(timezone.utc),
                        message="Job creation failed",
                    ),
                    status_code=HTTP_400_BAD_REQUEST,
                )

            return Response(
                content=GrokJobResponse(
                    job_id=job.id,
                    status=JobStatus(job.status),
                    name=job.name,
                    model=data.model,
                    generation_type=GenerationType.I2V,
                    created_at=job.created_at,
                    message="Video generation started. Poll job status for results.",
                ),
                status_code=HTTP_201_CREATED,
            )

        except GrokJobError as e:
            logger.error(f"Grok I2V generation failed to start: {e}")
            return Response(
                content=GrokJobResponse(
                    job_id=UUID("00000000-0000-0000-0000-000000000000"),
                    status=JobStatus.FAILED,
                    name=data.name or data.prompt[:50],
                    model=data.model,
                    generation_type=GenerationType.I2V,
                    created_at=datetime.now(timezone.utc),
                    message=str(e),
                ),
                status_code=HTTP_400_BAD_REQUEST,
            )

    @post("/edit")
    async def edit_video(
        self,
        data: GrokVideoEditRequest,
        session: AsyncSession,
        grok_job_service: GrokJobService | None,
    ) -> Response[GrokJobResponse]:
        """Edit an existing video with a text prompt (V2V).

        Performs video-to-video editing using grok-imagine-video.
        The input video must be a publicly accessible URL.
        Maximum input video length is 8.7 seconds.
        """
        if grok_job_service is None:
            return Response(
                content=GrokJobResponse(
                    job_id=UUID("00000000-0000-0000-0000-000000000000"),
                    status=JobStatus.FAILED,
                    name="Service Unavailable",
                    model=data.model,
                    generation_type=GenerationType.V2V,
                    created_at=datetime.now(timezone.utc),
                    message="Grok provider is not configured",
                ),
                status_code=HTTP_503_SERVICE_UNAVAILABLE,
            )

        try:
            # TODO: Get user_id from auth context
            user_id = UUID("00000000-0000-0000-0000-000000000001")

            job = await grok_job_service.start_video_job(
                session=session,
                user_id=user_id,
                prompt=data.prompt,
                model=data.model,
                generation_type=GenerationType.V2V,
                name=data.name,
                input_video_url=data.input_video_url,
            )

            await session.commit()

            if job is None:
                return Response(
                    content=GrokJobResponse(
                        job_id=UUID("00000000-0000-0000-0000-000000000000"),
                        status=JobStatus.FAILED,
                        name=data.name or data.prompt[:50],
                        model=data.model,
                        generation_type=GenerationType.V2V,
                        created_at=datetime.now(timezone.utc),
                        message="Job creation failed",
                    ),
                    status_code=HTTP_400_BAD_REQUEST,
                )

            return Response(
                content=GrokJobResponse(
                    job_id=job.id,
                    status=JobStatus(job.status),
                    name=job.name,
                    model=data.model,
                    generation_type=GenerationType.V2V,
                    created_at=job.created_at,
                    message="Video editing started. Poll job status for results.",
                ),
                status_code=HTTP_201_CREATED,
            )

        except GrokJobError as e:
            logger.error(f"Grok V2V editing failed to start: {e}")
            return Response(
                content=GrokJobResponse(
                    job_id=UUID("00000000-0000-0000-0000-000000000000"),
                    status=JobStatus.FAILED,
                    name=data.name or data.prompt[:50],
                    model=data.model,
                    generation_type=GenerationType.V2V,
                    created_at=datetime.now(timezone.utc),
                    message=str(e),
                ),
                status_code=HTTP_400_BAD_REQUEST,
            )


class GrokJobController(Controller):
    """Grok job status endpoints."""

    path = "/api/v1/grok/jobs"
    tags: Sequence[str] | None = ["Grok Jobs"]

    @get("/{job_id:uuid}")
    async def get_job_status(
        self,
        job_id: UUID,
        session: AsyncSession,
        grok_job_service: GrokJobService | None,
    ) -> Response[GrokJobStatusResponse]:
        """Get current status of a Grok generation job.

        Returns job details including progress and output URLs.
        For video jobs, this may trigger a poll to check completion status.
        """
        if grok_job_service is None:
            return Response(
                content=GrokJobStatusResponse(
                    job_id=job_id,
                    status=JobStatus.FAILED,
                    name="Service Unavailable",
                    generation_type=GenerationType.T2I,
                    prompt="",
                    created_at=datetime.now(timezone.utc),
                    error="Grok provider is not configured",
                ),
                status_code=HTTP_503_SERVICE_UNAVAILABLE,
            )

        job = await grok_job_service.get_job(session, job_id)

        if job is None:
            return Response(
                content=GrokJobStatusResponse(
                    job_id=job_id,
                    status=JobStatus.FAILED,
                    name="Not Found",
                    generation_type=GenerationType.T2I,
                    prompt="",
                    created_at=datetime.now(timezone.utc),
                    error=f"Job {job_id} not found",
                ),
                status_code=HTTP_404_NOT_FOUND,
            )

        # Get output URLs if job is complete
        outputs: list[str] = []
        if job.status == JobStatus.COMPLETED.value:
            outputs = await grok_job_service.get_job_outputs(session, job_id)

        return Response(
            content=GrokJobStatusResponse(
                job_id=job.id,
                status=JobStatus(job.status),
                name=job.name,
                generation_type=GenerationType(job.generation_type),
                prompt=job.prompt,
                enhanced_prompt=job.enhanced_prompt,
                created_at=job.created_at,
                started_at=job.started_at,
                completed_at=job.completed_at,
                outputs=outputs,
                error=job.error_message,
            ),
            status_code=HTTP_200_OK,
        )
