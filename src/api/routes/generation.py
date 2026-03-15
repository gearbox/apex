"""Generation API routes."""

from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Annotated
from uuid import UUID

import structlog
from litestar import Controller, Response, get, post
from litestar.datastructures import UploadFile
from litestar.di import Provide
from litestar.enums import RequestEncodingType
from litestar.params import Body
from litestar.status_codes import (
    HTTP_201_CREATED,
    HTTP_400_BAD_REQUEST,
)
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.dependencies.auth import get_current_user_id
from src.api.schemas.generation import (
    GenerationRequest,
    HealthResponse,
    ImageUploadResponse,
    JobResponse,
)
from src.api.security import auth_guard
from src.api.services.billing import BillingService
from src.api.services.comfyui_client import ComfyUIClient, ComfyUIClientError
from src.api.services.job_manager import JobManager
from src.api.services.pricing import PricingService
from src.api.services.workflow_service import WorkflowError, WorkflowService
from src.core.enums import GenerationType, JobStatus
from src.core.uid import new_id

logger = structlog.get_logger(__name__)


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


# TODO: JobManager has no concept of user_id and should be refactored to support user context for multi-user environments.
# WorkflowService may also need adjustments to support user-specific workflows or parameters in the future.
# For now, we will pass current_user_id to the controllers and services that need it,
# but it is not yet integrated into the JobManager or WorkflowService logic.
class LegacyGenerationController(Controller):
    """DEPRECATED: Image generation endpoints. Use POST /v1/generate with unified schema."""

    path = "/v1/legacy/generate"
    tags: Sequence[str] | None = ["Generation (Deprecated)"]
    guards = [auth_guard]
    dependencies = {"current_user_id": Provide(get_current_user_id)}

    @post("/")
    async def create_generation(
        self,
        current_user_id: UUID,  # will be used when persisting to DB
        data: GenerationRequest,
        session: AsyncSession,
        comfyui_client: ComfyUIClient,
        job_manager: JobManager,
        workflow_service: WorkflowService,
        billing_service: BillingService,
        pricing_service: PricingService,
    ) -> Response[JobResponse]:
        """Submit a new image generation job.

        Creates a generation job and queues it with ComfyUI.
        Follows the Saga pattern for token billing deduction.
        Returns immediately with job ID for status polling.
        """
        # --- SAGA: Pre-flight check ---
        account = await billing_service.resolve_account_for_user(current_user_id, session=session)
        token_cost = await pricing_service.get_price(
            "aisha", GenerationType.T2I.value, data.model_type.value, session=session
        )
        await billing_service.assert_sufficient_balance(account.id, token_cost, session=session)

        # Create job entry (in-memory job manager for now)
        job = job_manager.create_job(data)

        # We need a proper UUID to persist for token reservation, but job_manager gives strings.
        # Fall back to a UUID for DB purposes if job_manager doesn't generate one natively.
        try:
            db_job_id = UUID(job.job_id)
        except ValueError:
            db_job_id = new_id()
            # If the job manager changes later, we should ideally use its real UUID.

        txn = None
        try:
            # --- SAGA: Pre-flight deduction ---
            txn = await billing_service.check_and_reserve(
                account.id,
                token_cost,
                db_job_id,
                metadata={
                    "provider": "aisha",
                    "generation_type": data.generation_type.value,
                    "model": data.model_type.value,
                    "prompt": data.prompt[:100],
                },
                session=session,
            )
            await session.commit()

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
            if prompt_id := result.get("prompt_id"):
                job_manager.set_queued(job.job_id, prompt_id)
            else:
                job_manager.set_failed(job.job_id, "No prompt_id returned from ComfyUI")

        except WorkflowError as e:
            logger.error("workflow.error", error=str(e))
            job_manager.set_failed(job.job_id, str(e))
            if txn:
                await billing_service.refund(
                    db_job_id, description=f"Workflow error: {e}", session=session
                )
                await session.commit()

        except ComfyUIClientError as e:
            logger.error("comfyui.error", error=str(e))
            job_manager.set_failed(job.job_id, str(e))
            if txn:
                await billing_service.refund(
                    db_job_id, description=f"ComfyUI error: {e}", session=session
                )
                await session.commit()

        except Exception as e:
            logger.exception("generation.unexpected_error", job_id=job.job_id)
            job_manager.set_failed(job.job_id, str(e))
            if txn:
                try:
                    await billing_service.refund(
                        db_job_id, description=f"Unexpected generation error: {e}", session=session
                    )
                    await session.commit()
                except Exception as refund_error:
                    logger.exception(
                        "generation.refund_failed", job_id=job.job_id, error=str(refund_error)
                    )
            raise

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
        current_user_id: UUID,  # will be used when persisting to DB
        session: AsyncSession,
        comfyui_client: ComfyUIClient,
        job_manager: JobManager,
        workflow_service: WorkflowService,
        billing_service: BillingService,
        pricing_service: PricingService,
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
        Follows the Saga pattern for pre-flight token billing based on i2i.

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
                    created_at=datetime.now(UTC),
                    message="Image-to-image (i2i) generation requires at least one input image",
                ),
                status_code=HTTP_400_BAD_REQUEST,
            )

        # --- SAGA: Pre-flight check ---
        account = await billing_service.resolve_account_for_user(current_user_id, session=session)
        # Assuming we charge based on generation type requested (I2I vs T2I)
        token_cost = await pricing_service.get_price(
            "aisha", data.generation_type.value, data.model_type.value, session=session
        )
        await billing_service.assert_sufficient_balance(account.id, token_cost, session=session)

        job = job_manager.create_job(data)
        try:
            db_job_id = UUID(job.job_id)
        except ValueError:
            db_job_id = new_id()

        uploaded_image_1: str | None = None
        uploaded_image_2: str | None = None

        txn = None
        try:
            # --- SAGA: Pre-flight deduction ---
            txn = await billing_service.check_and_reserve(
                account.id,
                token_cost,
                db_job_id,
                metadata={
                    "provider": "aisha",
                    "generation_type": data.generation_type.value,
                    "model": data.model_type.value,
                    "prompt": data.prompt[:100],
                },
                session=session,
            )
            await session.commit()

            # Upload images if provided
            if image1 is not None:
                image_data = await image1.read()
                # Generate unique filename
                ext = image1.filename.rsplit(".", 1)[-1] if image1.filename else "png"
                filename = f"input_{job.job_id[:8]}_1.{ext}"

                result = await comfyui_client.upload_image(image_data, filename)
                uploaded_image_1 = result.get("name")
                logger.debug("image.uploaded", name="image1", result=str(uploaded_image_1))

            if image2 is not None:
                image_data = await image2.read()
                ext = image2.filename.rsplit(".", 1)[-1] if image2.filename else "png"
                filename = f"input_{job.job_id[:8]}_2.{ext}"

                result = await comfyui_client.upload_image(image_data, filename)
                uploaded_image_2 = result.get("name")
                logger.debug("image.uploaded", name="image2", result=str(uploaded_image_2))

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
            if prompt_id := result.get("prompt_id"):
                job_manager.set_queued(job.job_id, prompt_id)
                job.input_images = [img for img in [uploaded_image_1, uploaded_image_2] if img]
            else:
                job_manager.set_failed(job.job_id, "No prompt_id returned from ComfyUI")

        except WorkflowError as e:
            logger.error("workflow.error", error=str(e))
            job_manager.set_failed(job.job_id, str(e))
            if txn:
                await billing_service.refund(
                    db_job_id, description=f"Workflow error: {e}", session=session
                )
                await session.commit()

        except ComfyUIClientError as e:
            logger.error("comfyui.error", error=str(e))
            job_manager.set_failed(job.job_id, str(e))
            if txn:
                await billing_service.refund(
                    db_job_id, description=f"ComfyUI error: {e}", session=session
                )
                await session.commit()

        except Exception as e:
            logger.exception("generation.unexpected_error", job_id=job.job_id)
            job_manager.set_failed(job.job_id, str(e))
            if txn:
                try:
                    await billing_service.refund(
                        db_job_id, description=f"Unexpected generation error: {e}", session=session
                    )
                    await session.commit()
                except Exception as refund_error:
                    logger.exception(
                        "generation.refund_failed", job_id=job.job_id, error=str(refund_error)
                    )
            raise

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


class ImageController(Controller):
    """Image upload and retrieval endpoints."""

    path = "/v1/images"
    tags: Sequence[str] | None = ["Images"]
    guards = [auth_guard]
    dependencies = {"current_user_id": Provide(get_current_user_id)}

    @post("/upload")
    async def upload_image(
        self,
        current_user_id: UUID,  # will be used for tracking uploads
        comfyui_client: ComfyUIClient,
        data: Annotated[UploadFile, Body(media_type=RequestEncodingType.MULTI_PART)],
    ) -> Response[ImageUploadResponse]:
        """Upload an image to ComfyUI for use in generation.

        Returns the filename that can be referenced in generation requests.
        """
        try:
            image_data = await data.read()

            # Generate unique filename preserving extension
            ext = data.filename.rsplit(".", 1)[-1] if data.filename else "png"
            logger.debug("image.uploading", ext=ext)
            unique_filename = f"upload_{new_id().hex[:12]}.{ext}"

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
            logger.error("image.upload_failed", error=str(e))
            return Response(
                content=ImageUploadResponse(
                    filename="",
                    subfolder="",
                    type="input",
                ),
                status_code=500,
            )
