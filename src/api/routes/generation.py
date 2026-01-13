"""Generation API routes."""

import logging
import uuid
from collections.abc import Sequence
from datetime import datetime, timezone
from typing import Annotated

from litestar import Controller, Response, get, post
from litestar.datastructures import UploadFile
from litestar.enums import RequestEncodingType
from litestar.params import Body
from litestar.status_codes import (
    HTTP_200_OK,
    HTTP_201_CREATED,
    HTTP_400_BAD_REQUEST,
    HTTP_404_NOT_FOUND,
)

from src.api.schemas.generation import (
    GenerationRequest,
    GenerationType,
    HealthResponse,
    ImageUploadResponse,
    JobResponse,
    JobStatus,
    JobStatusResponse,
)
from src.api.services.comfyui_client import ComfyUIClient, ComfyUIClientError
from src.api.services.job_manager import JobManager
from src.api.services.workflow_service import WorkflowError, WorkflowService

logger = logging.getLogger(__name__)


class HealthController(Controller):
    """Health check endpoints."""

    path = "/health"
    tags: Sequence[str] | None = ["Health"]

    @get("/")
    async def health_check(
        self,
        comfyui_client: ComfyUIClient,
    ) -> HealthResponse:
        """Check API and ComfyUI connectivity.

        Returns health status of the service and its dependencies.
        """
        comfyui_connected = await comfyui_client.health_check()

        return HealthResponse(
            status="healthy" if comfyui_connected else "unhealthy",
            comfyui_connected=comfyui_connected,
        )


class GenerationController(Controller):
    """Image generation endpoints."""

    path = "/api/v1/generate"
    tags: Sequence[str] | None = ["Generation"]

    @post("/")
    async def create_generation(
        self,
        data: GenerationRequest,
        comfyui_client: ComfyUIClient,
        job_manager: JobManager,
        workflow_service: WorkflowService,
    ) -> Response[JobResponse]:
        """Submit a new image generation job.

        Creates a generation job and queues it with ComfyUI.
        Returns immediately with job ID for status polling.
        """
        # Create job entry
        job = job_manager.create_job(data)

        try:
            # Load and configure workflow
            workflow = workflow_service.load_workflow(data.model_type)
            workflow_service.validate_workflow(workflow)

            # Apply request parameters
            configured_workflow = workflow_service.apply_parameters(
                workflow=workflow,
                request=data,
                filename_prefix=f"gen_{job.job_id[:8]}",
            )

            # Queue with ComfyUI
            result = await comfyui_client.queue_prompt(configured_workflow)
            prompt_id = result.get("prompt_id")

            if prompt_id:
                job_manager.set_queued(job.job_id, prompt_id)
            else:
                job_manager.set_failed(job.job_id, "No prompt_id returned from ComfyUI")

        except WorkflowError as e:
            logger.error(f"Workflow error: {e}")
            job_manager.set_failed(job.job_id, str(e))

        except ComfyUIClientError as e:
            logger.error(f"ComfyUI error: {e}")
            job_manager.set_failed(job.job_id, str(e))

        return Response(
            content=JobResponse(
                job_id=job.job_id,
                status=job.status,
                name=job.name,
                created_at=job.created_at,
                message="Job queued successfully" if job.status == JobStatus.QUEUED else job.error,
            ),
            status_code=HTTP_201_CREATED,
        )

    @post("/with-images")
    async def create_generation_with_images(
        self,
        comfyui_client: ComfyUIClient,
        job_manager: JobManager,
        workflow_service: WorkflowService,
        data: Annotated[
            GenerationRequest,
            Body(media_type=RequestEncodingType.MULTI_PART),
        ],
        image1: UploadFile | None = None,
        image2: UploadFile | None = None,
    ) -> Response[JobResponse]:
        """Submit a generation job with reference images (image-to-image).

        Accepts up to 2 reference images for image-to-image generation.
        Images are uploaded to ComfyUI and referenced in the workflow.

        Note: If generation_type is 't2i', uploaded images will be ignored.
        For i2i generation, at least one image is required.
        """
        # Validate i2i requires at least one image
        if data.generation_type == GenerationType.I2I and image1 is None and image2 is None:
            return Response(
                content=JobResponse(
                    job_id="",
                    status=JobStatus.FAILED,
                    name=data.name or "Failed",
                    created_at=datetime.now(timezone.utc),
                    message="Image-to-image (i2i) generation requires at least one input image",
                ),
                status_code=HTTP_400_BAD_REQUEST,
            )

        job = job_manager.create_job(data)

        uploaded_image_1: str | None = None
        uploaded_image_2: str | None = None

        try:
            # Upload images if provided
            if image1 is not None:
                image_data = await image1.read()
                # Generate unique filename
                ext = image1.filename.rsplit(".", 1)[-1] if image1.filename else "png"
                filename = f"input_{job.job_id[:8]}_1.{ext}"

                result = await comfyui_client.upload_image(image_data, filename)
                uploaded_image_1 = result.get("name")
                logger.debug(f"Uploaded image1: {uploaded_image_1}")

            if image2 is not None:
                image_data = await image2.read()
                ext = image2.filename.rsplit(".", 1)[-1] if image2.filename else "png"
                filename = f"input_{job.job_id[:8]}_2.{ext}"

                result = await comfyui_client.upload_image(image_data, filename)
                uploaded_image_2 = result.get("name")
                logger.debug(f"Uploaded image2: {uploaded_image_2}")

            # Load and configure workflow
            workflow = workflow_service.load_workflow(data.model_type)
            workflow_service.validate_workflow(workflow)

            # Apply parameters including uploaded images
            configured_workflow = workflow_service.apply_parameters(
                workflow=workflow,
                request=data,
                input_image_1=uploaded_image_1,
                input_image_2=uploaded_image_2,
                filename_prefix=f"gen_{job.job_id[:8]}",
            )

            # Queue with ComfyUI
            result = await comfyui_client.queue_prompt(configured_workflow)
            prompt_id = result.get("prompt_id")

            if prompt_id:
                job_manager.set_queued(job.job_id, prompt_id)
                job.input_images = [
                    img for img in [uploaded_image_1, uploaded_image_2] if img
                ]
            else:
                job_manager.set_failed(job.job_id, "No prompt_id returned from ComfyUI")

        except WorkflowError as e:
            logger.error(f"Workflow error: {e}")
            job_manager.set_failed(job.job_id, str(e))

        except ComfyUIClientError as e:
            logger.error(f"ComfyUI error: {e}")
            job_manager.set_failed(job.job_id, str(e))

        return Response(
            content=JobResponse(
                job_id=job.job_id,
                status=job.status,
                name=job.name,
                created_at=job.created_at,
                message="Job queued successfully" if job.status == JobStatus.QUEUED else job.error,
            ),
            status_code=HTTP_201_CREATED,
        )


class JobController(Controller):
    """Job status and management endpoints."""

    path = "/api/v1/jobs"
    tags: Sequence[str] | None = ["Jobs"]

    @get("/{job_id:str}")
    async def get_job_status(
        self,
        job_id: str,
        job_manager: JobManager,
    ) -> Response[JobStatusResponse]:
        """Get current status of a generation job.

        Polls ComfyUI for latest status if job is still processing.
        Returns job details including progress and result images.
        """
        # Poll for updates
        job = await job_manager.poll_job_status(job_id)

        if job is None:
            return Response(
                content=JobStatusResponse(
                    job_id=job_id,
                    status=JobStatus.FAILED,
                    name="Unknown",
                    created_at=None,  # type: ignore
                    error="Job not found",
                ),
                status_code=HTTP_404_NOT_FOUND,
            )

        return Response(
            content=JobStatusResponse(
                job_id=job.job_id,
                status=job.status,
                name=job.name,
                created_at=job.created_at,
                started_at=job.started_at,
                completed_at=job.completed_at,
                progress=job.progress,
                images=job.images,
                error=job.error,
            ),
            status_code=HTTP_200_OK,
        )

    @get("/")
    async def list_jobs(
        self,
        job_manager: JobManager,
        status: JobStatus | None = None,
        limit: int = 50,
    ) -> list[JobStatusResponse]:
        """List generation jobs.

        Optionally filter by status. Returns most recent jobs first.
        """
        jobs = job_manager.list_jobs(status=status, limit=limit)

        return [
            JobStatusResponse(
                job_id=job.job_id,
                status=job.status,
                name=job.name,
                created_at=job.created_at,
                started_at=job.started_at,
                completed_at=job.completed_at,
                progress=job.progress,
                images=job.images,
                error=job.error,
            )
            for job in jobs
        ]


class ImageController(Controller):
    """Image upload and retrieval endpoints."""

    path = "/api/v1/images"
    tags: Sequence[str] | None = ["Images"]

    @post("/upload")
    async def upload_image(
        self,
        comfyui_client: ComfyUIClient,
        file: UploadFile,
    ) -> Response[ImageUploadResponse]:
        """Upload an image to ComfyUI for use in generation.

        Returns the filename that can be referenced in generation requests.
        """
        try:
            image_data = await file.read()

            # Generate unique filename preserving extension
            ext = file.filename.rsplit(".", 1)[-1] if file.filename else "png"
            unique_filename = f"upload_{uuid.uuid4().hex[:12]}.{ext}"

            result = await comfyui_client.upload_image(image_data, unique_filename)

            return Response(
                content=ImageUploadResponse(
                    filename=result.get("name", unique_filename),
                    subfolder=result.get("subfolder", ""),
                    type=result.get("type", "input"),
                ),
                status_code=HTTP_201_CREATED,
            )

        except ComfyUIClientError as e:
            logger.error(f"Image upload failed: {e}")
            return Response(
                content=ImageUploadResponse(
                    filename="",
                    subfolder="",
                    type="input",
                ),
                status_code=500,
            )
