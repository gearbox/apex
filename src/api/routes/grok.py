"""Grok API routes for image and video generation."""

from __future__ import annotations

import contextlib
import logging
from collections.abc import Sequence
from datetime import UTC, datetime
from uuid import UUID

from litestar import Controller, Response, get, post
from litestar.di import Provide
from litestar.status_codes import (
    HTTP_200_OK,
    HTTP_201_CREATED,
    HTTP_400_BAD_REQUEST,
    HTTP_404_NOT_FOUND,
    HTTP_503_SERVICE_UNAVAILABLE,
)
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.dependencies.auth import get_current_user_id
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
from src.api.security import auth_guard
from src.api.services.billing import BillingService
from src.api.services.grok.job_service import GrokJobError, GrokJobService
from src.api.services.pricing import PricingService
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
    guards = [auth_guard]
    dependencies = {"current_user_id": Provide(get_current_user_id)}

    @post("/")
    async def generate_image(
        self,
        current_user_id: UUID,
        data: GrokImageRequest,
        session: AsyncSession,
        grok_job_service: GrokJobService | None,
        billing_service: BillingService,
        pricing_service: PricingService,
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
                    created_at=datetime.now(UTC),
                    message="Grok provider is not configured",
                ),
                status_code=HTTP_503_SERVICE_UNAVAILABLE,
            )

        # --- Billing pre-flight ---
        account = await billing_service.resolve_account_for_user(current_user_id, session=session)
        token_cost = await pricing_service.get_price(
            "grok", GenerationType.T2I.value, data.model.value, session=session
        )
        await billing_service.assert_sufficient_balance(account.id, token_cost, session=session)

        job = None
        try:
            job = await grok_job_service.create_image_job(
                session=session,
                user_id=current_user_id,
                prompt=data.prompt,
                model=data.model,
                generation_type=GenerationType.T2I,
                n=data.n,
                aspect_ratio=data.aspect_ratio,
                name=data.name,
            )

            if job is None:
                return Response(
                    content=GrokJobResponse(
                        job_id=UUID("00000000-0000-0000-0000-000000000000"),
                        status=JobStatus.FAILED,
                        name=data.name or data.prompt[:50],
                        model=data.model,
                        generation_type=GenerationType.T2I,
                        created_at=datetime.now(UTC),
                        message="Job creation failed",
                    ),
                    status_code=HTTP_400_BAD_REQUEST,
                )

            # Reserve tokens after job creation
            txn = await billing_service.check_and_reserve(
                account.id,
                token_cost,
                job.id,
                metadata={
                    "provider": "grok",
                    "generation_type": GenerationType.T2I.value,
                    "model": data.model.value,
                },
                session=session,
            )
            job.token_cost = token_cost
            job.debit_transaction_id = txn.id

            await session.commit()

            balance = await billing_service.get_balance(account.id, session=session)

            return Response(
                content=GrokJobResponse(
                    job_id=job.id,
                    status=JobStatus(job.status),
                    name=job.name,
                    model=data.model,
                    generation_type=GenerationType.T2I,
                    created_at=job.created_at,
                    tokens_charged=token_cost,
                    balance_remaining=balance,
                    message="Image generation completed"
                    if job.status == JobStatus.COMPLETED.value
                    else None,
                ),
                status_code=HTTP_201_CREATED,
            )

        except GrokJobError as e:
            logger.error("Grok image generation failed: %s", e)
            # Refund on provider error
            if job is not None:
                try:
                    await billing_service.refund(
                        job.id,
                        description=f"Provider error refund: {e}",
                        session=session,
                    )
                    await session.commit()
                except Exception:
                    logger.exception("Failed to refund for job %s", job.id)
            return Response(
                content=GrokJobResponse(
                    job_id=UUID("00000000-0000-0000-0000-000000000000"),
                    status=JobStatus.FAILED,
                    name=data.name or data.prompt[:50],
                    model=data.model,
                    generation_type=GenerationType.T2I,
                    created_at=datetime.now(UTC),
                    message=str(e),
                ),
                status_code=HTTP_400_BAD_REQUEST,
            )

    @post("/edit")
    async def edit_image(
        self,
        current_user_id: UUID,
        data: GrokImageEditRequest,
        session: AsyncSession,
        grok_job_service: GrokJobService | None,
        r2_storage: R2StorageService,
        billing_service: BillingService,
        pricing_service: PricingService,
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
                    created_at=datetime.now(UTC),
                    message="Grok provider is not configured",
                ),
                status_code=HTTP_503_SERVICE_UNAVAILABLE,
            )

        # --- Billing pre-flight ---
        account = await billing_service.resolve_account_for_user(current_user_id, session=session)
        token_cost = await pricing_service.get_price(
            "grok", GenerationType.I2I.value, data.model.value, session=session
        )
        await billing_service.assert_sufficient_balance(account.id, token_cost, session=session)

        job = None
        try:
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
                        generation_type=GenerationType.I2I,
                        created_at=datetime.now(UTC),
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
                user_id=current_user_id,
                prompt=data.prompt,
                model=data.model,
                generation_type=GenerationType.I2I,
                n=1,  # Editing produces single output
                name=data.name,
                input_image_url=input_image_url,
                input_image_id=data.input_image_id,
            )

            if job is None:
                return Response(
                    content=GrokJobResponse(
                        job_id=UUID("00000000-0000-0000-0000-000000000000"),
                        status=JobStatus.FAILED,
                        name=data.name or data.prompt[:50],
                        model=data.model,
                        generation_type=GenerationType.I2I,
                        created_at=datetime.now(UTC),
                        message="Job creation failed",
                    ),
                    status_code=HTTP_400_BAD_REQUEST,
                )

            # Reserve tokens
            txn = await billing_service.check_and_reserve(
                account.id,
                token_cost,
                job.id,
                metadata={
                    "provider": "grok",
                    "generation_type": GenerationType.I2I.value,
                    "model": data.model.value,
                },
                session=session,
            )
            job.token_cost = token_cost
            job.debit_transaction_id = txn.id

            await session.commit()
            balance = await billing_service.get_balance(account.id, session=session)

            return Response(
                content=GrokJobResponse(
                    job_id=job.id,
                    status=JobStatus(job.status),
                    name=job.name,
                    model=data.model,
                    generation_type=GenerationType.I2I,
                    created_at=job.created_at,
                    tokens_charged=token_cost,
                    balance_remaining=balance,
                    message="Image editing completed"
                    if job.status == JobStatus.COMPLETED.value
                    else None,
                ),
                status_code=HTTP_201_CREATED,
            )

        except GrokJobError as e:
            logger.error("Grok image editing failed: %s", e)
            if job is not None:
                try:
                    await billing_service.refund(
                        job.id,
                        description=f"Provider error refund: {e}",
                        session=session,
                    )
                    await session.commit()
                except Exception:
                    logger.exception("Failed to refund for job %s", job.id)
            return Response(
                content=GrokJobResponse(
                    job_id=UUID("00000000-0000-0000-0000-000000000000"),
                    status=JobStatus.FAILED,
                    name=data.name or data.prompt[:50],
                    model=data.model,
                    generation_type=GenerationType.I2I,
                    created_at=datetime.now(UTC),
                    message=str(e),
                ),
                status_code=HTTP_400_BAD_REQUEST,
            )


class GrokVideoController(Controller):
    """Grok video generation endpoints."""

    path = "/api/v1/grok/video"
    tags: Sequence[str] | None = ["Grok Video"]
    guards = [auth_guard]
    dependencies = {"current_user_id": Provide(get_current_user_id)}

    @post("/")
    async def generate_video(
        self,
        current_user_id: UUID,
        data: GrokVideoRequest,
        session: AsyncSession,
        grok_job_service: GrokJobService | None,
        billing_service: BillingService,
        pricing_service: PricingService,
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
                    created_at=datetime.now(UTC),
                    message="Grok provider is not configured",
                ),
                status_code=HTTP_503_SERVICE_UNAVAILABLE,
            )

        # --- Billing pre-flight ---
        account = await billing_service.resolve_account_for_user(current_user_id, session=session)
        token_cost = await pricing_service.get_price(
            "grok", GenerationType.T2V.value, data.model.value, session=session
        )
        await billing_service.assert_sufficient_balance(account.id, token_cost, session=session)

        job = None
        try:
            job = await grok_job_service.start_video_job(
                session=session,
                user_id=current_user_id,
                prompt=data.prompt,
                model=data.model,
                generation_type=GenerationType.T2V,
                duration=data.duration,
                aspect_ratio=data.aspect_ratio,
                resolution=data.resolution,
                name=data.name,
            )

            if job is None:
                return Response(
                    content=GrokJobResponse(
                        job_id=UUID("00000000-0000-0000-0000-000000000000"),
                        status=JobStatus.FAILED,
                        name=data.name or data.prompt[:50],
                        model=data.model,
                        generation_type=GenerationType.T2V,
                        created_at=datetime.now(UTC),
                        message="Job creation failed",
                    ),
                    status_code=HTTP_400_BAD_REQUEST,
                )

            # Reserve tokens
            txn = await billing_service.check_and_reserve(
                account.id,
                token_cost,
                job.id,
                metadata={
                    "provider": "grok",
                    "generation_type": GenerationType.T2V.value,
                    "model": data.model.value,
                },
                session=session,
            )
            job.token_cost = token_cost
            job.debit_transaction_id = txn.id

            await session.commit()
            balance = await billing_service.get_balance(account.id, session=session)

            return Response(
                content=GrokJobResponse(
                    job_id=job.id,
                    status=JobStatus(job.status),
                    name=job.name,
                    model=data.model,
                    generation_type=GenerationType.T2V,
                    created_at=job.created_at,
                    tokens_charged=token_cost,
                    balance_remaining=balance,
                    message="Video generation started. Poll job status for results.",
                ),
                status_code=HTTP_201_CREATED,
            )

        except GrokJobError as e:
            logger.error("Grok video generation failed to start: %s", e)
            if job is not None:
                try:
                    await billing_service.refund(
                        job.id,
                        description=f"Provider error refund: {e}",
                        session=session,
                    )
                    await session.commit()
                except Exception:
                    logger.exception("Failed to refund for job %s", job.id)
            return Response(
                content=GrokJobResponse(
                    job_id=UUID("00000000-0000-0000-0000-000000000000"),
                    status=JobStatus.FAILED,
                    name=data.name or data.prompt[:50],
                    model=data.model,
                    generation_type=GenerationType.T2V,
                    created_at=datetime.now(UTC),
                    message=str(e),
                ),
                status_code=HTTP_400_BAD_REQUEST,
            )

    @post("/from-image")
    async def generate_video_from_image(
        self,
        current_user_id: UUID,
        data: GrokVideoFromImageRequest,
        session: AsyncSession,
        grok_job_service: GrokJobService | None,
        r2_storage: R2StorageService,
        billing_service: BillingService,
        pricing_service: PricingService,
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
                    created_at=datetime.now(UTC),
                    message="Grok provider is not configured",
                ),
                status_code=HTTP_503_SERVICE_UNAVAILABLE,
            )

        # --- Billing pre-flight ---
        account = await billing_service.resolve_account_for_user(current_user_id, session=session)
        token_cost = await pricing_service.get_price(
            "grok", GenerationType.I2V.value, data.model.value, session=session
        )
        await billing_service.assert_sufficient_balance(account.id, token_cost, session=session)

        job = None
        try:
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
                        created_at=datetime.now(UTC),
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
                user_id=current_user_id,
                prompt=data.prompt,
                model=data.model,
                generation_type=GenerationType.I2V,
                duration=data.duration,
                aspect_ratio=data.aspect_ratio,
                resolution=data.resolution,
                name=data.name,
                input_image_url=input_image_url,
            )

            if job is None:
                return Response(
                    content=GrokJobResponse(
                        job_id=UUID("00000000-0000-0000-0000-000000000000"),
                        status=JobStatus.FAILED,
                        name=data.name or data.prompt[:50],
                        model=data.model,
                        generation_type=GenerationType.I2V,
                        created_at=datetime.now(UTC),
                        message="Job creation failed",
                    ),
                    status_code=HTTP_400_BAD_REQUEST,
                )

            # Reserve tokens
            txn = await billing_service.check_and_reserve(
                account.id,
                token_cost,
                job.id,
                metadata={
                    "provider": "grok",
                    "generation_type": GenerationType.I2V.value,
                    "model": data.model.value,
                },
                session=session,
            )
            job.token_cost = token_cost
            job.debit_transaction_id = txn.id

            await session.commit()
            balance = await billing_service.get_balance(account.id, session=session)

            return Response(
                content=GrokJobResponse(
                    job_id=job.id,
                    status=JobStatus(job.status),
                    name=job.name,
                    model=data.model,
                    generation_type=GenerationType.I2V,
                    created_at=job.created_at,
                    tokens_charged=token_cost,
                    balance_remaining=balance,
                    message="Video generation started. Poll job status for results.",
                ),
                status_code=HTTP_201_CREATED,
            )

        except GrokJobError as e:
            logger.error("Grok I2V generation failed to start: %s", e)
            if job is not None:
                try:
                    await billing_service.refund(
                        job.id,
                        description=f"Provider error refund: {e}",
                        session=session,
                    )
                    await session.commit()
                except Exception:
                    logger.exception("Failed to refund for job %s", job.id)
            return Response(
                content=GrokJobResponse(
                    job_id=UUID("00000000-0000-0000-0000-000000000000"),
                    status=JobStatus.FAILED,
                    name=data.name or data.prompt[:50],
                    model=data.model,
                    generation_type=GenerationType.I2V,
                    created_at=datetime.now(UTC),
                    message=str(e),
                ),
                status_code=HTTP_400_BAD_REQUEST,
            )

    @post("/edit")
    async def edit_video(
        self,
        current_user_id: UUID,
        data: GrokVideoEditRequest,
        session: AsyncSession,
        grok_job_service: GrokJobService | None,
        billing_service: BillingService,
        pricing_service: PricingService,
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
                    created_at=datetime.now(UTC),
                    message="Grok provider is not configured",
                ),
                status_code=HTTP_503_SERVICE_UNAVAILABLE,
            )

        # --- Billing pre-flight ---
        account = await billing_service.resolve_account_for_user(current_user_id, session=session)
        token_cost = await pricing_service.get_price(
            "grok", GenerationType.V2V.value, data.model.value, session=session
        )
        await billing_service.assert_sufficient_balance(account.id, token_cost, session=session)

        job = None
        try:
            job = await grok_job_service.start_video_job(
                session=session,
                user_id=current_user_id,
                prompt=data.prompt,
                model=data.model,
                generation_type=GenerationType.V2V,
                name=data.name,
                input_video_url=data.input_video_url,
            )

            if job is None:
                return Response(
                    content=GrokJobResponse(
                        job_id=UUID("00000000-0000-0000-0000-000000000000"),
                        status=JobStatus.FAILED,
                        name=data.name or data.prompt[:50],
                        model=data.model,
                        generation_type=GenerationType.V2V,
                        created_at=datetime.now(UTC),
                        message="Job creation failed",
                    ),
                    status_code=HTTP_400_BAD_REQUEST,
                )

            # Reserve tokens
            txn = await billing_service.check_and_reserve(
                account.id,
                token_cost,
                job.id,
                metadata={
                    "provider": "grok",
                    "generation_type": GenerationType.V2V.value,
                    "model": data.model.value,
                },
                session=session,
            )
            job.token_cost = token_cost
            job.debit_transaction_id = txn.id

            await session.commit()
            balance = await billing_service.get_balance(account.id, session=session)

            return Response(
                content=GrokJobResponse(
                    job_id=job.id,
                    status=JobStatus(job.status),
                    name=job.name,
                    model=data.model,
                    generation_type=GenerationType.V2V,
                    created_at=job.created_at,
                    tokens_charged=token_cost,
                    balance_remaining=balance,
                    message="Video editing started. Poll job status for results.",
                ),
                status_code=HTTP_201_CREATED,
            )

        except GrokJobError as e:
            logger.error("Grok V2V editing failed to start: %s", e)
            if job is not None:
                try:
                    await billing_service.refund(
                        job.id,
                        description=f"Provider error refund: {e}",
                        session=session,
                    )
                    await session.commit()
                except Exception:
                    logger.exception("Failed to refund for job %s", job.id)
            return Response(
                content=GrokJobResponse(
                    job_id=UUID("00000000-0000-0000-0000-000000000000"),
                    status=JobStatus.FAILED,
                    name=data.name or data.prompt[:50],
                    model=data.model,
                    generation_type=GenerationType.V2V,
                    created_at=datetime.now(UTC),
                    message=str(e),
                ),
                status_code=HTTP_400_BAD_REQUEST,
            )


class GrokJobController(Controller):
    """Grok job status endpoints."""

    path = "/api/v1/grok/jobs"
    tags: Sequence[str] | None = ["Grok Jobs"]
    guards = [auth_guard]
    dependencies = {"current_user_id": Provide(get_current_user_id)}

    @get("/{job_id:uuid}")
    async def get_job_status(
        self,
        current_user_id: UUID,
        job_id: UUID,
        session: AsyncSession,
        grok_job_service: GrokJobService | None,
        billing_service: BillingService,
    ) -> Response[GrokJobStatusResponse]:
        """Get current status of a Grok generation job.

        Returns job details including progress and output URLs.
        For video jobs, this may trigger a poll to check completion status.
        Only returns jobs owned by the authenticated user.
        """
        if grok_job_service is None:
            return Response(
                content=GrokJobStatusResponse(
                    job_id=job_id,
                    status=JobStatus.FAILED,
                    name="Service Unavailable",
                    generation_type=GenerationType.T2I,
                    prompt="",
                    created_at=datetime.now(UTC),
                    error="Grok provider is not configured",
                ),
                status_code=HTTP_503_SERVICE_UNAVAILABLE,
            )

        job = await grok_job_service.get_job(session, job_id)

        if job is None or job.user_id != current_user_id:
            return Response(
                content=GrokJobStatusResponse(
                    job_id=job_id,
                    status=JobStatus.FAILED,
                    name="Not Found",
                    generation_type=GenerationType.T2I,
                    prompt="",
                    created_at=datetime.now(UTC),
                    error=f"Job {job_id} not found",
                ),
                status_code=HTTP_404_NOT_FOUND,
            )

        # Get output URLs if job is complete
        outputs: list[str] = []
        if job.status == JobStatus.COMPLETED.value:
            outputs = await grok_job_service.get_job_outputs(session, job_id)

        # Get billing info
        balance: int | None = None
        if job.token_cost is not None:
            with contextlib.suppress(Exception):
                account = await billing_service.resolve_account_for_user(
                    current_user_id, session=session
                )
                balance = await billing_service.get_balance(account.id, session=session)
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
                tokens_charged=job.token_cost,
                balance_remaining=balance,
            ),
            status_code=HTTP_200_OK,
        )
