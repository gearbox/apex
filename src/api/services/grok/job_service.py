"""Grok job service for orchestrating generation and storage.

Handles the full lifecycle of Grok generation jobs:
- Job creation and tracking in database
- Calling Grok API for generation
- Downloading results and storing in R2
- Status updates and error handling
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING
from uuid import UUID

import httpx
import structlog

from src.api.services.billing import BillingService
from src.api.services.generation.base import ProviderSubmitResult
from src.api.services.grok import (
    GrokAPIError,
    GrokClient,
    GrokImageResult,
    GrokRateLimitError,
    GrokTimeoutError,
    GrokVideoJobStarted,
    GrokVideoResult,
)
from src.api.services.image_thumbnail import make_image_thumbnails
from src.api.services.storage import R2StorageService, StorageType
from src.api.services.thumbnail import extract_video_thumbnail
from src.core.enums import (
    AspectRatio,
    GenerationType,
    JobStatus,
    MediaFormat,
    ModelType,
    Provider,
    VideoPollStatus,
    VideoResolution,
)
from src.core.uid import new_id
from src.db.repositories.job import JobRepository
from src.db.repositories.output import OutputRepository

from .enums import ResponseImageFormat

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from src.db.models import GenerationJob

logger = structlog.get_logger(__name__)


@dataclasses.dataclass(frozen=True)
class VideoPollOutcome:
    """Result of polling xAI for one video job, independent of JobStatus.

    ``poll_video_job_for_worker`` returns this instead of mutating job.status
    directly — the periodic worker drives the actual state transition (and
    any refund) through JobStateTransitionService.
    """

    status: VideoPollStatus
    error_message: str | None = None


class GrokJobError(Exception):
    """Base exception for Grok job service errors."""


class GrokJobNotFoundError(GrokJobError):
    """Raised when a job is not found."""


def _require_job_after_status_update(
    job: GenerationJob | None, job_id: UUID, action: str
) -> GenerationJob:
    if job is None:
        raise GrokJobError(f"Job {job_id} disappeared after {action}")
    return job


class GrokJobService:
    """Service for managing Grok generation jobs.

    Orchestrates:
    - Creating jobs in database
    - Calling Grok API
    - Downloading generated content
    - Storing results in R2
    - Updating job status
    """

    def __init__(
        self,
        grok_client: GrokClient,
        storage: R2StorageService,
        retention_days: int = 7,
    ) -> None:
        """Initialize Grok job service.

        Args:
            grok_client: Grok API client.
            storage: R2 storage service for persisting results.
            retention_days: Days to retain generated content.
        """
        self._grok = grok_client
        self._storage = storage
        self._retention_days = retention_days
        self._http_client: httpx.AsyncClient | None = None

    async def connect(self) -> None:
        """Initialize HTTP client for downloading media."""
        if self._http_client is None:
            self._http_client = httpx.AsyncClient(
                timeout=httpx.Timeout(60.0, connect=10.0),
                follow_redirects=True,
            )

    async def close(self) -> None:
        """Close HTTP client."""
        if self._http_client is not None:
            await self._http_client.aclose()
            self._http_client = None

    @property
    def http_client(self) -> httpx.AsyncClient:
        """Get HTTP client for downloads."""
        if self._http_client is None:
            raise GrokJobError("HTTP client not initialized. Call connect() first.")
        return self._http_client

    # -------------------------------------------------------------------------
    # Image Generation
    # -------------------------------------------------------------------------

    async def create_image_job(
        self,
        session: AsyncSession,
        *,
        user_id: UUID,
        prompt: str,
        model: ModelType,
        generation_type: GenerationType,
        n: int = 1,
        aspect_ratio: AspectRatio = AspectRatio.RATIO_1_1,
        name: str | None = None,
        negative_prompt: str | None = None,
        input_image_url: str | None = None,
        input_image_urls: Sequence[str] | None = None,
        input_image_id: UUID | None = None,
        billing_service: BillingService,
        account_id: UUID,
        token_cost: int,
        product_id: str,
        source_job_id: UUID | None = None,
        source_output_id: UUID | None = None,
    ) -> ProviderSubmitResult:
        """Create and execute an image generation job.

        This method follows the Saga pattern:
        1. Creates job record in database
        2. Deducts tokens pre-flight
        3. Calls Grok API for generation
        4. Downloads generated images
        5. Stores images in R2
        6. Creates output records in database
        7. Updates job status (or refunds tokens on error)

        Args:
            session: Database session.
            user_id: Owner of the job.
            prompt: Generation prompt.
            model: Grok model to use.
            generation_type: T2I or I2I.
            n: Number of images to generate.
            aspect_ratio: Image aspect ratio.
            name: Job name (auto-generated if not provided).
            negative_prompt: Negative prompt (stored but not used by Grok).
            input_image_url: Input image URL for I2I.
            input_image_urls: Input image URLs for Grok multi-reference I2I.
            input_image_id: Database ID of input image.
            billing_service: Service to handle token deductions.
            account_id: Pre-resolved token account ID for the user.
            token_cost: Pre-calculated token cost for this generation.

        Returns:
            ``ProviderSubmitResult`` wrapping the created and processed
            GenerationJob, the post-debit balance, and the pending balance
            event.

        Raises:
            GrokJobError: If job creation or processing fails.
        """
        job_repo = JobRepository(session)
        output_repo = OutputRepository(session)

        # Generate job name from prompt if not provided
        if name is None:
            name = prompt[:50].strip()
            if len(prompt) > 50:
                name += "..."

        # Create job record
        job_id = new_id()
        job: GenerationJob | None = await job_repo.create(
            id=job_id,
            user_id=user_id,
            name=name,
            prompt=prompt,
            generation_type=generation_type,
            status=JobStatus.PENDING,
            provider=Provider.GROK,
            model=model.value,
            aspect_ratio=aspect_ratio.value,
            product_id=product_id,
            source_job_id=source_job_id,
            source_output_id=source_output_id,
            input_image_id=input_image_id,
        )

        # Update with negative prompt if provided
        if negative_prompt and job is not None:
            job.negative_prompt = negative_prompt
            await session.flush()

        try:
            # Update status to running
            job = await job_repo.update_status(
                job_id,
                JobStatus.RUNNING,
                started_at=datetime.now(UTC),
            )

            # --- SAGA: Pre-flight token deduction ---
            reserve_result = await billing_service.check_and_reserve(
                account_id,
                token_cost,
                job_id,
                metadata={
                    "type": "generation",
                    "provider": "grok",
                    "generation_type": generation_type.value,
                    "model": model.value,
                },
                description="Generation charge",
                session=session,
                product_id=product_id,
            )
            if job is not None:
                job.token_cost = token_cost
                job.debit_transaction_id = reserve_result.txn.id
            await session.flush()

            # Call Grok API
            if generation_type == GenerationType.I2I and (
                input_image_url is not None or input_image_urls is not None
            ):
                # Image input references are resolved to provider-readable URLs upstream.
                results = await self._grok.edit_image(
                    prompt=prompt,
                    image_url=input_image_url,
                    image_urls=input_image_urls,
                    model=model,
                    n=n,
                    aspect_ratio=aspect_ratio,
                    image_format=ResponseImageFormat.URL,
                )
            else:
                # Text-to-image
                results = await self._grok.generate_image(
                    prompt=prompt,
                    model=model,
                    n=n,
                    aspect_ratio=aspect_ratio,
                    image_format=ResponseImageFormat.URL,
                )

            # Store revised prompt if available
            if results and results[0].revised_prompt and job is not None:
                job.enhanced_prompt = results[0].revised_prompt
                await session.flush()

            # Download and store each result
            for idx, image_result in enumerate(results):
                await self._store_image_result(
                    session=session,
                    output_repo=output_repo,
                    user_id=user_id,
                    job_id=job_id,
                    result=image_result,
                    output_index=idx,
                    input_image_id=input_image_id,
                    product_id=product_id,
                )

            # Mark job complete
            job = await job_repo.update_status(
                job_id,
                JobStatus.COMPLETED,
                completed_at=datetime.now(UTC),
            )

            logger.info("grok.image_job_completed", job_id=str(job_id), output_count=len(results))
            return ProviderSubmitResult(
                job=_require_job_after_status_update(job, job_id, "completion"),
                balance_after=reserve_result.txn.balance_after,
                balance_event=reserve_result.event,
            )

        except GrokAPIError as e:
            logger.exception("grok.api_error", job_id=str(job_id), error=str(e))
            job = await job_repo.update_status(
                job_id,
                JobStatus.FAILED,
                completed_at=datetime.now(UTC),
            )
            if job is not None:
                job.error_message = str(e)
                await session.flush()

            # --- SAGA: Compensation on API Error ---
            try:
                await billing_service.refund(
                    job_id,
                    description=f"Provider error refund: {getattr(e, 'message', str(e))}",
                    session=session,
                    product_id=product_id,
                )
                await session.flush()
            except Exception as refund_error:
                logger.exception(
                    "grok.compensation_refund_failed", job_id=str(job_id), error=str(refund_error)
                )

            raise GrokJobError(f"Image generation failed: {e}") from e

        except Exception as e:
            logger.exception("grok.unexpected_error", job_id=str(job_id))
            job = await job_repo.update_status(
                job_id,
                JobStatus.FAILED,
                completed_at=datetime.now(UTC),
            )
            if job is not None:
                job.error_message = str(e)
                await session.flush()

            # --- SAGA: Compensation on Unexpected Error ---
            try:
                await billing_service.refund(
                    job_id,
                    description=f"Unexpected error refund: {e!s}",
                    session=session,
                    product_id=product_id,
                )
                await session.flush()
            except Exception as refund_error:
                logger.exception(
                    "grok.compensation_refund_failed", job_id=str(job_id), error=str(refund_error)
                )

            raise GrokJobError(f"Unexpected error: {e}") from e

    async def _store_image_result(
        self,
        *,
        session: AsyncSession,  # noqa: ARG002
        output_repo: OutputRepository,
        user_id: UUID,
        job_id: UUID,
        result: GrokImageResult,
        output_index: int,
        input_image_id: UUID | None,
        product_id: str,
    ) -> None:
        """Download and store an image result in R2."""
        if not result.has_url:
            logger.warning("grok.image_result_no_url", output_index=output_index)
            return

        # Download image from xAI CDN
        if result.url is None:
            raise RuntimeError("result.url is None despite has_url check — invariant violated")
        response = await self.http_client.get(result.url)
        response.raise_for_status()
        image_data = response.content

        # Determine format from content-type
        content_type = response.headers.get("content-type", "image/jpeg")
        if "png" in content_type:
            image_format = MediaFormat.PNG
        elif "webp" in content_type:
            image_format = MediaFormat.WEBP
        else:
            image_format = MediaFormat.JPEG

        # Upload to R2
        output_id = new_id()
        storage_key = self._storage.build_storage_key(
            user_id=user_id,
            file_id=output_id,
            storage_type=StorageType.OUTPUT,
            format=image_format,
            job_id=job_id,
        )

        # Upload directly using the storage key
        async with self._storage._get_client() as client:
            await client.put_object(
                Bucket=self._storage._settings.bucket_name,
                Key=storage_key,
                Body=image_data,
                ContentType=image_format.content_type,
            )

        # Create output record
        expires_at = datetime.now(UTC) + timedelta(days=self._retention_days)
        await output_repo.create(
            id=output_id,
            user_id=user_id,
            job_id=job_id,
            storage_key=storage_key,
            content_type=image_format.content_type,
            size_bytes=len(image_data),
            format=image_format.value,
            output_index=output_index,
            expires_at=expires_at,
            input_image_id=input_image_id,
            product_id=product_id,
        )

        logger.debug("grok.image_output_stored", output_id=str(output_id), job_id=str(job_id))

    # -------------------------------------------------------------------------
    # Video Generation (Async Start)
    # -------------------------------------------------------------------------

    async def start_video_job(
        self,
        session: AsyncSession,
        *,
        user_id: UUID,
        prompt: str,
        model: ModelType = ModelType.GROK_IMAGINE_VIDEO,
        generation_type: GenerationType = GenerationType.T2V,
        duration: int = 5,
        aspect_ratio: AspectRatio = AspectRatio.RATIO_16_9,
        resolution: VideoResolution = VideoResolution.RES_720P,
        name: str | None = None,
        input_image_url: str | None = None,
        input_video_url: str | None = None,
        billing_service: BillingService,
        account_id: UUID,
        token_cost: int,
        product_id: str,
        source_job_id: UUID | None = None,
        source_output_id: UUID | None = None,
    ) -> ProviderSubmitResult:
        """Start an async video generation job.

        Creates the job and initiates generation with Grok API.
        This method follows the Saga pattern for token billing.
        The actual video result must be polled using poll_video_job().

        Args:
            session: Database session.
            user_id: Owner of the job.
            prompt: Generation prompt.
            model: Grok model (must be grok-imagine-video).
            generation_type: T2V or I2V.
            duration: Video duration in seconds (1-15).
            aspect_ratio: Video aspect ratio.
            resolution: Video resolution.
            name: Job name.
            input_image_url: Source image URL for I2V.
            input_image_id: Database ID of input image.
            input_video_url: Source video URL for editing.
            billing_service: Service to handle token deductions.
            account_id: Pre-resolved token account ID for the user.
            token_cost: Pre-calculated token cost for this generation.

        Returns:
            ``ProviderSubmitResult`` wrapping the created GenerationJob with
            status QUEUED, the post-debit balance, and the pending balance
            event.
        """
        job_repo = JobRepository(session)

        # Generate job name
        if name is None:
            name = prompt[:50].strip()
            if len(prompt) > 50:
                name += "..."

        # Create job record
        job_id = new_id()
        job: GenerationJob | None = await job_repo.create(
            id=job_id,
            user_id=user_id,
            name=name,
            prompt=prompt,
            generation_type=generation_type,
            status=JobStatus.PENDING,
            provider=Provider.GROK,
            model=model.value,
            aspect_ratio=aspect_ratio.value,
            product_id=product_id,
            source_job_id=source_job_id,
            source_output_id=source_output_id,
        )

        try:
            # --- SAGA: Pre-flight token deduction ---
            reserve_result = await billing_service.check_and_reserve(
                account_id,
                token_cost,
                job_id,
                metadata={
                    "type": "generation",
                    "provider": "grok",
                    "generation_type": generation_type.value,
                    "model": model.value,
                },
                description="Generation charge",
                session=session,
                product_id=product_id,
            )
            if job is not None:
                job.token_cost = token_cost
                job.debit_transaction_id = reserve_result.txn.id
            await session.flush()

            # Start async video generation
            started: GrokVideoJobStarted = await self._grok.start_video_generation(
                prompt=prompt,
                model=model,
                duration=duration,
                aspect_ratio=aspect_ratio,
                resolution=resolution,
                image_url=input_image_url,
                video_url=input_video_url,
            )

            # Store xAI request ID for polling
            job = await job_repo.update_status(
                job_id,
                JobStatus.QUEUED,
                external_request_id=started.request_id,
                started_at=datetime.now(UTC),
            )

            logger.info(
                "grok.video_job_started", job_id=str(job_id), xai_request_id=started.request_id
            )
            return ProviderSubmitResult(
                job=_require_job_after_status_update(job, job_id, "start"),
                balance_after=reserve_result.txn.balance_after,
                balance_event=reserve_result.event,
            )

        except GrokAPIError as e:
            logger.exception("grok.video_job_start_failed", job_id=str(job_id), error=str(e))
            job = await job_repo.update_status(
                job_id,
                JobStatus.FAILED,
                completed_at=datetime.now(UTC),
            )
            if job is not None:
                job.error_message = str(e)
                await session.flush()

            # --- SAGA: Compensation on API Error ---
            try:
                await billing_service.refund(
                    job_id,
                    description=f"Provider error refund: {getattr(e, 'message', str(e))}",
                    session=session,
                    product_id=product_id,
                )
                await session.flush()
            except Exception as refund_error:
                logger.exception(
                    "grok.compensation_refund_failed", job_id=str(job_id), error=str(refund_error)
                )

            raise GrokJobError(f"Failed to start video generation: {e}") from e

        except Exception as e:
            logger.exception("grok.unexpected_error", job_id=str(job_id))
            job = await job_repo.update_status(
                job_id,
                JobStatus.FAILED,
                completed_at=datetime.now(UTC),
            )
            if job is not None:
                job.error_message = str(e)
                await session.flush()

            # --- SAGA: Compensation on Unexpected Error ---
            try:
                await billing_service.refund(
                    job_id,
                    description=f"Unexpected error refund: {e!s}",
                    session=session,
                    product_id=product_id,
                )
                await session.flush()
            except Exception as refund_error:
                logger.exception(
                    "grok.compensation_refund_failed", job_id=str(job_id), error=str(refund_error)
                )

            raise GrokJobError(f"Unexpected error: {e}") from e

    async def poll_video_job(
        self,
        session: AsyncSession,
        job_id: UUID,
    ) -> GenerationJob | None:
        """Poll a video job for completion (read-through path).

        Used by GrokGenerationProvider.refresh_job to poll-on-read within a
        request. Mutates job.status directly via JobRepository — this path
        predates JobStateTransitionService and returns the job synchronously
        for the caller to serialize into the API response, so it is kept as
        a thin wrapper around ``_poll_video_result`` rather than migrated.
        The periodic worker uses ``poll_video_job_for_worker`` instead, which
        drives every transition through JobStateTransitionService.

        Args:
            session: Database session.
            job_id: Job ID to poll.

        Returns:
            Updated GenerationJob.

        Raises:
            GrokJobNotFoundError: If job doesn't exist.
            GrokJobError: If job is not a video job or polling fails.

        Note:
            Transient xAI errors (rate limit, deadline exceeded) are not
            raised here — they are reported as STILL_RUNNING (job returned
            unchanged) so a status request never 500s on an xAI hiccup.
        """
        job_repo = JobRepository(session)
        job = await job_repo.get(job_id)

        if job is None:
            raise GrokJobNotFoundError(f"Job {job_id} not found")

        # Only poll queued/running jobs
        if job.status not in (JobStatus.QUEUED.value, JobStatus.RUNNING.value):
            return job

        was_queued = job.status == JobStatus.QUEUED.value
        try:
            outcome = await self._poll_video_result(session, job)
        except (GrokRateLimitError, GrokTimeoutError) as e:
            logger.warning("grok.video_poll_transient", job_id=str(job.id), error=str(e))
            return job

        if outcome.status == VideoPollStatus.STILL_RUNNING:
            if was_queued:
                job = await job_repo.update_status(job_id, JobStatus.RUNNING)
            return job

        if outcome.status == VideoPollStatus.COMPLETED:
            return await job_repo.update_status(
                job_id,
                JobStatus.COMPLETED,
                completed_at=datetime.now(UTC),
            )

        # FAILED
        job = await job_repo.update_status(
            job_id,
            JobStatus.FAILED,
            completed_at=datetime.now(UTC),
        )
        if job is not None:
            job.error_message = outcome.error_message
            await session.flush()
        raise GrokJobError(f"Video polling failed: {outcome.error_message}")

    async def poll_video_job_for_worker(
        self,
        session: AsyncSession,
        job_id: UUID,
    ) -> VideoPollOutcome:
        """Poll a video job for the periodic worker.

        Unlike ``poll_video_job``, this never mutates job.status — on
        completion it downloads and persists outputs (flushed, not
        committed, so the caller's later commit is atomic with the status
        transition) but leaves every JobStatus change to the caller via
        JobStateTransitionService.

        Args:
            session: Database session (one per job, per the worker's
                session-per-job pattern).
            job_id: Job ID to poll.

        Returns:
            The observed poll outcome.

        Raises:
            GrokJobNotFoundError: If job doesn't exist.
            GrokJobError: If the job has no xAI request ID.
            GrokRateLimitError: If xAI rate-limited this poll. Not handled
                here by design — the caller (worker) decides how to react
                (skip this tick, no transition; see D1/D2).
            GrokTimeoutError: If this poll hit xAI's deadline. Same handling
                as GrokRateLimitError.
        """
        job_repo = JobRepository(session)
        job = await job_repo.get(job_id)

        if job is None:
            raise GrokJobNotFoundError(f"Job {job_id} not found")

        if job.status not in (JobStatus.QUEUED.value, JobStatus.RUNNING.value):
            return VideoPollOutcome(status=VideoPollStatus.STILL_RUNNING)

        return await self._poll_video_result(session, job)

    async def _poll_video_result(
        self,
        session: AsyncSession,
        job: GenerationJob,
    ) -> VideoPollOutcome:
        """Poll xAI for one job's result; download+store on completion.

        Does not mutate job.status — shared by the read-through
        (``poll_video_job``) and worker (``poll_video_job_for_worker``)
        entry points, which each drive the status transition differently.
        """
        request_id = job.external_request_id
        if not request_id:
            raise GrokJobError(f"Job {job.id} has no xAI request ID")

        output_repo = OutputRepository(session)

        try:
            result: GrokVideoResult | None = await self._grok.get_video_result(request_id)

            if result is None:
                return VideoPollOutcome(status=VideoPollStatus.STILL_RUNNING)

            await self._store_video_result(
                session=session,
                output_repo=output_repo,
                user_id=job.user_id,
                job_id=job.id,
                result=result,
                product_id=job.product_id,
            )
            logger.info("grok.video_job_completed", job_id=str(job.id))
            return VideoPollOutcome(status=VideoPollStatus.COMPLETED)

        except (GrokRateLimitError, GrokTimeoutError):
            # Transient: rate limits and deadline-exceeded must not fail the
            # job. Re-raise; each entry point decides (worker: skip this
            # tick; read-through: report STILL_RUNNING). See D1.
            raise
        except GrokAPIError as e:
            logger.exception("grok.video_job_poll_failed", job_id=str(job.id), error=str(e))
            return VideoPollOutcome(status=VideoPollStatus.FAILED, error_message=str(e))

    async def _store_video_result(
        self,
        *,
        session: AsyncSession,  # noqa: ARG002
        output_repo: OutputRepository,
        user_id: UUID,
        job_id: UUID,
        result: GrokVideoResult,
        product_id: str,
    ) -> None:
        """Download, store a video result in R2, and extract a thumbnail frame."""
        # Download video from xAI CDN
        response = await self.http_client.get(result.url)
        response.raise_for_status()
        video_data = response.content

        # Upload to R2
        output_id = new_id()
        storage_key = self._storage.build_storage_key(
            user_id=user_id,
            file_id=output_id,
            storage_type=StorageType.OUTPUT,
            format=MediaFormat.MP4,
            job_id=job_id,
        )

        # Upload directly using the storage key
        async with self._storage._get_client() as client:
            await client.put_object(
                Bucket=self._storage._settings.bucket_name,
                Key=storage_key,
                Body=video_data,
                ContentType=MediaFormat.MP4.content_type,
            )

        # Create output record
        expires_at = datetime.now(UTC) + timedelta(days=self._retention_days)
        await output_repo.create(
            id=output_id,
            user_id=user_id,
            job_id=job_id,
            storage_key=storage_key,
            content_type=MediaFormat.MP4.content_type,
            size_bytes=len(video_data),
            format=MediaFormat.MP4.value,
            output_index=0,
            expires_at=expires_at,
            input_image_id=None,
            is_thumbnail=False,
            product_id=product_id,
        )

        logger.debug("grok.video_output_stored", output_id=str(output_id), job_id=str(job_id))

        # Extract first frame and generate sm + md WEBP poster variants
        frame_bytes = await extract_video_thumbnail(video_data)
        if frame_bytes:
            thumbnails = await make_image_thumbnails(frame_bytes)
            for generated in thumbnails:
                thumb_id = new_id()
                thumb_key = self._storage.build_storage_key(
                    user_id=user_id,
                    file_id=thumb_id,
                    storage_type=StorageType.OUTPUT,
                    format=MediaFormat.WEBP,
                    job_id=job_id,
                )
                try:
                    async with self._storage._get_client() as client:
                        await client.put_object(
                            Bucket=self._storage._settings.bucket_name,
                            Key=thumb_key,
                            Body=generated.result.data,
                            ContentType=MediaFormat.WEBP.content_type,
                        )
                    await output_repo.create(
                        id=thumb_id,
                        user_id=user_id,
                        job_id=job_id,
                        storage_key=thumb_key,
                        content_type=MediaFormat.WEBP.content_type,
                        size_bytes=len(generated.result.data),
                        format=MediaFormat.WEBP.value,
                        output_index=0,
                        expires_at=expires_at,
                        input_image_id=None,
                        is_thumbnail=True,
                        parent_output_id=output_id,
                        thumbnail_max_edge=generated.spec.max_edge,
                        width=generated.result.width,
                        height=generated.result.height,
                        product_id=product_id,
                    )
                    logger.debug(
                        "grok.thumbnail_stored",
                        thumb_id=str(thumb_id),
                        max_edge=generated.spec.max_edge,
                    )
                except Exception:
                    logger.warning(
                        "grok.thumbnail_skipped",
                        job_id=str(job_id),
                        max_edge=generated.spec.max_edge,
                    )
        else:
            logger.warning("grok.thumbnail_skipped", job_id=str(job_id))

    # -------------------------------------------------------------------------
    # Job Status
    # -------------------------------------------------------------------------

    async def get_job(
        self,
        session: AsyncSession,
        job_id: UUID,
    ) -> GenerationJob | None:
        """Get a job by ID.

        Args:
            session: Database session.
            job_id: Job ID.

        Returns:
            GenerationJob if found, None otherwise.
        """
        return await JobRepository(session).get(job_id)

    async def get_job_outputs(
        self,
        session: AsyncSession,
        job_id: UUID,
    ) -> list[str]:
        """Get presigned URLs for job outputs.

        Args:
            session: Database session.
            job_id: Job ID.

        Returns:
            List of presigned URLs for accessing outputs.
        """
        outputs = await OutputRepository(session).list_by_job(job_id)

        urls = []
        for output in outputs:
            result = await self._storage.get_presigned_url(
                output.storage_key,
                expires_in=3600,
            )
            urls.append(result.presigned_url)

        return urls
