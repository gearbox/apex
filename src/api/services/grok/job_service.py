"""Grok job service for orchestrating generation and storage.

Handles the full lifecycle of Grok generation jobs:
- Job creation and tracking in database
- Calling Grok API for generation
- Downloading results and storing in R2
- Status updates and error handling
"""

from __future__ import annotations

import asyncio
import dataclasses
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, NoReturn
from uuid import uuid4

import httpx
import structlog
from sqlalchemy import func, or_, select, text, update
from sqlalchemy.exc import IntegrityError

from src.api.services.generation.base import ProviderSubmitResult
from src.api.services.generation.provider_billing_policy import (
    DEFAULT_PROVIDER_BILLING_POLICIES,
    ProviderBillingPolicyRegistry,
    apply_provider_billing_policy,
)
from src.api.services.generation.provider_failures import (
    ProviderFailure,
    ProviderFailureKind,
    ProviderModerationRejectedError,
    ProviderSubmissionFailedError,
)
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
from src.api.services.job_state_transition import GenerationOutputData, JobStateTransitionService
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
from src.db.models import GenerationJob, GenerationMaterializationAttempt, GenerationOutput
from src.db.repositories.job import JobRepository
from src.db.repositories.output import OutputRepository

from .enums import ResponseImageFormat

if TYPE_CHECKING:
    from collections.abc import Sequence
    from uuid import UUID

    from sqlalchemy.ext.asyncio import AsyncSession

    from src.api.services.billing import BalanceEvent, BillingService
    from src.api.services.event_bus import EventBus
    from src.api.services.ops_event_bus import OpsEventBus
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
    failure: ProviderFailure | None = None
    result: GrokVideoResult | None = None


@dataclasses.dataclass(frozen=True)
class _MaterializedVideo:
    """R2 objects ready to be attached by the claim-owning transaction."""

    outputs: list[GenerationOutputData]
    storage_keys: list[str]
    attempt_id: UUID | None = None
    product_id: str | None = None


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


def _require_i2i_input_url(
    input_image_url: str | None,
    input_image_urls: Sequence[str] | None,
) -> None:
    if input_image_url is None and input_image_urls is None:
        raise ValueError("I2I generation requires a resolved input image URL")


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
        *,
        billing_service: BillingService | None = None,
        event_bus: EventBus | None = None,
        ops_event_bus: OpsEventBus | None = None,
        max_poll_time: int = 600,
        finalization_lease_seconds: int = 120,
        billing_policy: ProviderBillingPolicyRegistry | None = None,
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
        self._billing = billing_service
        self._event_bus = event_bus
        self._ops_event_bus = ops_event_bus
        self._max_poll_time = max_poll_time
        self._finalization_lease_seconds = finalization_lease_seconds
        self._billing_policy = billing_policy or DEFAULT_PROVIDER_BILLING_POLICIES
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
        aspect_ratio: AspectRatio | None = None,
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
        primary_output_id: UUID | None = None,
        primary_upload_id: UUID | None = None,
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
            aspect_ratio: Image aspect ratio. None for t2i defaults to 1:1;
                None for i2i preserves the source image's aspect.
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
        if primary_upload_id is not None:
            if input_image_id is not None and input_image_id != primary_upload_id:
                raise ValueError("Conflicting primary upload lineage values")
            input_image_id = primary_upload_id

        job_repo = JobRepository(session)
        output_repo = OutputRepository(session)

        # Generate job name from prompt if not provided
        if name is None:
            name = prompt[:50].strip()
            if len(prompt) > 50:
                name += "..."

        # t2i: apply the provider-side default so job.aspect_ratio always
        # reflects the actual output. i2i: forward None as-is — the
        # orchestrator (C3) already guarantees any non-None value here is
        # capability-approved for this model, and NULL uniformly means
        # "followed source aspect" across providers.
        resolved_aspect_ratio = (
            aspect_ratio
            if generation_type == GenerationType.I2I
            else (aspect_ratio or AspectRatio.RATIO_1_1)
        )
        if generation_type == GenerationType.I2I:
            _require_i2i_input_url(input_image_url, input_image_urls)

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
            aspect_ratio=resolved_aspect_ratio.value if resolved_aspect_ratio is not None else None,
            product_id=product_id,
            source_job_id=source_job_id,
            primary_output_id=primary_output_id,
            input_image_id=input_image_id,
        )

        # Update with negative prompt if provided
        if negative_prompt and job is not None:
            job.negative_prompt = negative_prompt
            await session.flush()

        reserve_event = None
        reservation_created = False
        provider_attempted = False
        provider_accepted = False
        try:
            # Update status to running
            job = await job_repo.update_status(
                job_id,
                JobStatus.RUNNING,
                started_at=datetime.now(UTC),
            )

            # --- SAGA: Pre-flight token deduction ---
            try:
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
            except Exception:
                # Billing/domain failures are not provider failures.  The
                # caller rolls back the uncommitted job and preserves the
                # billing exception's HTTP contract.
                await session.rollback()
                raise
            reservation_created = True
            if job is not None:
                job.token_cost = token_cost
                job.debit_transaction_id = reserve_result.txn.id
            reserve_event = reserve_result.event
            await session.flush()

            # Call Grok API
            provider_attempted = True
            if generation_type == GenerationType.I2I:
                # Image input references are resolved to provider-readable URLs upstream.
                results = await self._grok.edit_image(
                    prompt=prompt,
                    image_url=input_image_url,
                    image_urls=input_image_urls,
                    model=model,
                    n=n,
                    aspect_ratio=resolved_aspect_ratio,
                    image_format=ResponseImageFormat.URL,
                )
            else:
                # Text-to-image
                results = await self._grok.generate_image(
                    prompt=prompt,
                    model=model,
                    n=n,
                    aspect_ratio=resolved_aspect_ratio,
                    image_format=ResponseImageFormat.URL,
                )
            provider_accepted = True

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

        except GrokAPIError as exc:
            await self._handle_submission_failure(
                exc,
                job=job,
                job_id=job_id,
                user_id=user_id,
                product_id=product_id,
                reserve_event=reserve_event,
            )

        except Exception as exc:
            if provider_attempted and not provider_accepted and reservation_created:
                self._raise_unexpected_submission_failure(
                    exc,
                    job=job,
                    job_id=job_id,
                    user_id=user_id,
                    product_id=product_id,
                    reserve_event=reserve_event,
                )
            # R2/database/local faults after acceptance remain internal
            # failures.  Do not invent a provider UNKNOWN result or refund a
            # reservation which may not exist.
            raise

    async def _handle_submission_failure(
        self,
        exc: GrokAPIError,
        *,
        job: GenerationJob | None,
        job_id: UUID,
        user_id: UUID,
        product_id: str,
        reserve_event: BalanceEvent | None,
    ) -> NoReturn:
        """Classify and raise; GenerationService is the sole settler.

        Provider adapters never commit a terminal state or compensation.  The
        request-level orchestrator owns the job, ledger, idempotency record,
        and post-commit notifications in one unit of work.
        """
        failure = apply_provider_billing_policy(exc.failure, registry=self._billing_policy)
        self._log_provider_failure(
            failure,
            job_id=job_id,
            user_id=user_id,
            product_id=product_id,
        )
        previous_status = str(job.status) if job is not None else JobStatus.PENDING.value
        error_type = (
            ProviderModerationRejectedError
            if failure.kind is ProviderFailureKind.MODERATION_REJECTED
            else ProviderSubmissionFailedError
        )
        raise error_type(
            failure=failure,
            job_id=job_id,
            previous_status=previous_status,
            balance_event=reserve_event,
        ) from exc

    def _raise_unexpected_submission_failure(
        self,
        exc: Exception,
        *,
        job: GenerationJob | None,
        job_id: UUID,
        user_id: UUID,
        product_id: str,
        reserve_event: BalanceEvent | None,
    ) -> NoReturn:
        """Convert non-provider submission faults into the shared settlement path."""
        failure = ProviderFailure(
            kind=ProviderFailureKind.UNKNOWN,
            provider=Provider.GROK,
            sanitized_message=ProviderFailure.safe_message_for_kind(ProviderFailureKind.UNKNOWN),
            retryable=True,
        )
        self._log_provider_failure(
            failure,
            job_id=job_id,
            user_id=user_id,
            product_id=product_id,
        )
        logger.exception("grok.unexpected_submission_error", job_id=str(job_id))
        previous_status = str(job.status) if job is not None else JobStatus.PENDING.value
        raise ProviderSubmissionFailedError(
            failure=failure,
            job_id=job_id,
            previous_status=previous_status,
            balance_event=reserve_event,
        ) from exc

    @staticmethod
    def _log_provider_failure(
        failure: ProviderFailure,
        *,
        job_id: UUID,
        user_id: UUID,
        product_id: str,
    ) -> None:
        """Log normalized provider diagnostics without raw provider payloads."""
        logger.warning(
            "grok.failure_classified",
            provider=failure.provider.value,
            job_id=str(job_id),
            user_id=str(user_id),
            product_id=product_id,
            provider_job_id=failure.provider_request_id,
            failure_kind=failure.kind.value,
            provider_status=failure.provider_status_code,
            provider_error_code=failure.provider_error_code,
            retryable=failure.retryable,
            billable=failure.billable,
        )

    async def _store_image_result(
        self,
        *,
        session: AsyncSession,
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

        await self._put_output_object(storage_key, image_data, image_format.content_type)

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

        await self._store_output_thumbnails(
            session=session,
            output_repo=output_repo,
            user_id=user_id,
            job_id=job_id,
            parent_output_id=output_id,
            parent_output_index=output_index,
            source_bytes=image_data,
            expires_at=expires_at,
            product_id=product_id,
        )

    async def _store_output_thumbnails(
        self,
        *,
        session: AsyncSession,
        output_repo: OutputRepository,
        user_id: UUID,
        job_id: UUID,
        parent_output_id: UUID,
        parent_output_index: int,
        source_bytes: bytes,
        expires_at: datetime,
        product_id: str,
    ) -> None:
        """Generate and store sm/md WEBP variants for a stored output.

        Best-effort per variant: a thumbnail failure must never fail the parent
        output (the original is already stored and billed). Mirrors the existing
        video-path behaviour.
        """
        thumbnails = await make_image_thumbnails(source_bytes)
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
                await self._put_output_object(
                    thumb_key,
                    generated.result.data,
                    MediaFormat.WEBP.content_type,
                )
                # SAVEPOINT: OutputRepository.create flushes immediately; without the
                # nested transaction a failed INSERT aborts the whole outer transaction
                # (parent output row, job state) and poisons the session for the rest
                # of the request. Rolling back to the savepoint keeps "best-effort"
                # honest: only this variant is lost.
                # Ordering note: the R2 put above is outside the savepoint, so a failed
                # insert orphans one WEBP object in R2 — accepted best-effort trade-off
                # (same D3 philosophy as the retention sweeper).
                async with session.begin_nested():
                    await output_repo.create(
                        id=thumb_id,
                        user_id=user_id,
                        job_id=job_id,
                        storage_key=thumb_key,
                        content_type=MediaFormat.WEBP.content_type,
                        size_bytes=len(generated.result.data),
                        format=MediaFormat.WEBP.value,
                        output_index=parent_output_index,
                        expires_at=expires_at,
                        input_image_id=None,
                        is_thumbnail=True,
                        parent_output_id=parent_output_id,
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
        aspect_ratio: AspectRatio | None = None,
        resolution: VideoResolution = VideoResolution.RES_720P,
        name: str | None = None,
        input_image_url: str | None = None,
        input_video_url: str | None = None,
        billing_service: BillingService,
        account_id: UUID,
        token_cost: int,
        product_id: str,
        source_job_id: UUID | None = None,
        primary_output_id: UUID | None = None,
        primary_upload_id: UUID | None = None,
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
            aspect_ratio: Video aspect ratio. None applies the provider
                default (16:9) — no video reshape-capability semantics
                change in this PR, pure behavior preservation.
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

        resolved_aspect_ratio = aspect_ratio or AspectRatio.RATIO_16_9

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
            aspect_ratio=resolved_aspect_ratio.value,
            product_id=product_id,
            source_job_id=source_job_id,
            primary_output_id=primary_output_id,
            primary_upload_id=primary_upload_id,
        )

        reserve_event = None
        reservation_created = False
        provider_attempted = False
        provider_accepted = False
        try:
            # --- SAGA: Pre-flight token deduction ---
            try:
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
            except Exception:
                await session.rollback()
                raise
            reservation_created = True
            if job is not None:
                job.token_cost = token_cost
                job.debit_transaction_id = reserve_result.txn.id
            reserve_event = reserve_result.event
            await session.flush()

            # Start async video generation
            provider_attempted = True
            started: GrokVideoJobStarted = await self._grok.start_video_generation(
                prompt=prompt,
                model=model,
                duration=duration,
                aspect_ratio=resolved_aspect_ratio,
                resolution=resolution,
                image_url=input_image_url,
                video_url=input_video_url,
            )
            provider_accepted = True

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

        except GrokAPIError as exc:
            await self._handle_submission_failure(
                exc,
                job=job,
                job_id=job_id,
                user_id=user_id,
                product_id=product_id,
                reserve_event=reserve_event,
            )

        except Exception as exc:
            if provider_attempted and not provider_accepted and reservation_created:
                self._raise_unexpected_submission_failure(
                    exc,
                    job=job,
                    job_id=job_id,
                    user_id=user_id,
                    product_id=product_id,
                    reserve_event=reserve_event,
                )
            raise

    async def poll_video_job(
        self,
        session: AsyncSession,
        job_id: UUID,
    ) -> GenerationJob | None:
        """Poll a video job for completion (read-through path).

        Used by GrokGenerationProvider.refresh_job to poll-on-read within a
        request. Terminal states are settled through the same
        ``settle_video_poll_outcome`` entry point as the periodic worker, so
        this path remains correct in deployments with no video worker.

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

        # ``started_at`` records provider acceptance and is write-once across
        # queued→running transitions. It is therefore the authoritative poll
        # TTL basis for both read-through and periodic polling.
        if self._is_video_job_overdue(job):
            return await self.settle_video_poll_outcome(
                session,
                job_id=job_id,
                outcome=self._timeout_outcome(),
                product_id=str(job.product_id),
            )

        try:
            outcome = await self._poll_video_result(job)
        except (GrokRateLimitError, GrokTimeoutError) as exc:
            failure = exc.failure
            logger.warning(
                "grok.video_poll_transient",
                job_id=str(job.id),
                user_id=str(job.user_id),
                product_id=job.product_id,
                provider_job_id=job.external_request_id,
                failure_kind=failure.kind.value,
                provider_status=failure.provider_status_code,
                retryable=failure.retryable,
            )
            return job

        return await self.settle_video_poll_outcome(
            session,
            job_id=job_id,
            outcome=outcome,
            product_id=str(job.product_id),
        )

    async def settle_video_poll_outcome(
        self,
        session: AsyncSession,
        *,
        job_id: UUID,
        outcome: VideoPollOutcome,
        product_id: str,
    ) -> GenerationJob:
        """Apply a video poll result exactly once through shared transitions.

        Both polling callers reach this method. Conditional state updates in
        ``JobStateTransitionService`` make repeated reads and worker/read
        races no-ops after the first terminal settlement, including its refund
        and terminal event.
        """
        transition_service = self._make_transition_service(session)
        if outcome.status == VideoPollStatus.STILL_RUNNING:
            return await transition_service.transition_to_running(job_id)
        if outcome.status == VideoPollStatus.COMPLETED:
            if outcome.result is None:
                # Compatibility for a synthetic outcome in a unit test or a
                # future provider that has already materialized its outputs.
                return await transition_service.transition_to_completed(
                    job_id, outputs=[], product_id=product_id
                )
            return await self._finalize_completed_video(
                session,
                job_id=job_id,
                result=outcome.result,
                product_id=product_id,
            )

        terminal_failure = outcome.failure
        refund = not terminal_failure.billable if terminal_failure is not None else True
        public_error_message = (
            ProviderFailure.public_message_for_failure(terminal_failure)
            if terminal_failure is not None
            else "The AI provider could not complete the request."
        )
        failure_code = terminal_failure.public_code if terminal_failure is not None else None
        job, _ = await transition_service.transition_to_failed(
            job_id,
            public_error_message=public_error_message,
            failure_code=failure_code,
            refund=refund,
            product_id=product_id,
        )
        return job

    def _make_transition_service(self, session: AsyncSession) -> JobStateTransitionService:
        """Create the one settlement primitive used by read-through and workers."""
        from src.api.services.billing import BillingService

        return JobStateTransitionService(
            session=session,
            event_bus=self._event_bus,
            billing_service=self._billing or BillingService(),
            ops_event_bus=self._ops_event_bus,
        )

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

        if self._is_video_job_overdue(job):
            return self._timeout_outcome()

        return await self._poll_video_result(job)

    async def _poll_video_result(self, job: GenerationJob) -> VideoPollOutcome:
        """Poll xAI for one job's result without materializing outputs.

        Does not mutate job.status — shared by the read-through
        (``poll_video_job``) and worker (``poll_video_job_for_worker``)
        entry points, which each drive the status transition differently.
        """
        request_id = job.external_request_id
        if not request_id:
            raise GrokJobError(f"Job {job.id} has no xAI request ID")

        try:
            result: GrokVideoResult | None = await self._grok.get_video_result(request_id)

            if result is None:
                return VideoPollOutcome(status=VideoPollStatus.STILL_RUNNING)

            return VideoPollOutcome(status=VideoPollStatus.COMPLETED, result=result)

        except (GrokRateLimitError, GrokTimeoutError):
            # Transient: rate limits and deadline-exceeded must not fail the
            # job. Re-raise; each entry point decides (worker: skip this
            # tick; read-through: report STILL_RUNNING). See D1.
            raise
        except GrokAPIError as exc:
            failure = apply_provider_billing_policy(exc.failure, registry=self._billing_policy)
            self._log_provider_failure(
                failure,
                job_id=job.id,
                user_id=job.user_id,
                product_id=job.product_id,
            )
            return VideoPollOutcome(
                status=VideoPollStatus.FAILED,
                error_message=failure.sanitized_message,
                failure=failure,
            )

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
        """Legacy helper used by image/video storage unit tests.

        Production polling uses ``_finalize_completed_video`` so that it owns
        a durable claim before calling the materializer below.  Keeping this
        thin wrapper preserves the independently useful storage helper while
        ensuring its rows mirror the claim-owned path.
        """
        materialized = await self._materialize_video_result(
            user_id=user_id,
            job_id=job_id,
            result=result,
            product_id=product_id,
        )
        try:
            for output in sorted(materialized.outputs, key=lambda item: item.is_thumbnail):
                await output_repo.create(
                    id=output.id,
                    user_id=user_id,
                    job_id=job_id,
                    storage_key=output.storage_key,
                    content_type=output.content_type,
                    size_bytes=output.size_bytes,
                    format=output.format,
                    output_index=output.output_index,
                    expires_at=output.expires_at,
                    input_image_id=None,
                    is_thumbnail=output.is_thumbnail,
                    parent_output_id=output.parent_output_id,
                    thumbnail_max_edge=output.thumbnail_max_edge,
                    width=output.width,
                    height=output.height,
                    product_id=product_id,
                )
        except Exception:
            await self._cleanup_materialized_objects(materialized.storage_keys)
            raise

    def _is_video_job_overdue(self, job: GenerationJob) -> bool:
        """TTL runs from provider acceptance, falling back to submission time."""
        basis = job.started_at or job.created_at
        if not isinstance(basis, datetime):
            return False
        return (datetime.now(UTC) - basis).total_seconds() > self._max_poll_time

    @staticmethod
    def _timeout_outcome() -> VideoPollOutcome:
        failure = ProviderFailure(
            kind=ProviderFailureKind.TIMEOUT,
            provider=Provider.GROK,
            sanitized_message=ProviderFailure.safe_message_for_kind(ProviderFailureKind.TIMEOUT),
        )
        return VideoPollOutcome(
            status=VideoPollStatus.FAILED,
            error_message=failure.sanitized_message,
            failure=failure,
        )

    async def _finalize_completed_video(
        self,
        session: AsyncSession,
        *,
        job_id: UUID,
        result: GrokVideoResult,
        product_id: str,
    ) -> GenerationJob:
        """Claim, materialize, and attach one completed Grok video.

        The claim transaction is intentionally separate and committed before
        the CDN/R2 work.  A process crash leaves a short, durable lease that a
        later poller can recover; a losing poller has no R2 write or output
        row to commit.
        """
        claim_token = await self._claim_video_finalization(session, job_id)
        if claim_token is None:
            return await self._refresh_authoritative_job(session, job_id)

        materialized: _MaterializedVideo | None = None
        try:
            materialized = await self._materialize_video_result(
                user_id=(await self._refresh_authoritative_job(session, job_id)).user_id,
                job_id=job_id,
                result=result,
                product_id=product_id,
                session=session,
                claim_token=claim_token,
            )
            transition_service = self._make_transition_service(session)
            (
                settled_job,
                did_transition,
            ) = await transition_service.transition_claimed_video_to_completed(
                job_id,
                claim_token=claim_token,
                outputs=materialized.outputs,
                product_id=product_id,
                materialization_attempt_id=materialized.attempt_id,
            )
            if did_transition:
                logger.info(
                    "grok.video_job_completed",
                    job_id=str(job_id),
                    output_count=len(materialized.outputs),
                )
            else:
                # Another terminal transition won.  Never delete based on a
                # stale in-memory list: first reconcile DB references, then
                # queue only the proven-unreferenced attempt keys.
                await self._reconcile_materialization_attempt(
                    session,
                    job_id=job_id,
                    product_id=product_id,
                    materialized=materialized,
                )
        except asyncio.CancelledError:
            # Cancellation is a BaseException, so it must be handled before
            # the ordinary error path.  Shield cleanup/release so the task is
            # still discoverable even while shutdown is cancelling workers.
            if materialized is not None:
                await asyncio.shield(
                    self._reconcile_materialization_attempt(
                        session,
                        job_id=job_id,
                        product_id=product_id,
                        materialized=materialized,
                    )
                )
            else:
                await asyncio.shield(
                    self._release_video_finalization_claim(session, job_id, claim_token)
                )
            raise
        except IntegrityError:
            await session.rollback()
            if materialized is not None:
                await self._reconcile_materialization_attempt(
                    session,
                    job_id=job_id,
                    product_id=product_id,
                    materialized=materialized,
                )
            else:
                await self._release_video_finalization_claim(session, job_id, claim_token)
            return await self._refresh_authoritative_job(session, job_id)
        except Exception:
            # ``commit()`` can raise after PostgreSQL has committed, and a
            # post-commit refresh can fail too.  Re-read in a new transaction
            # before deciding whether any R2 key is safe to remove.
            await session.rollback()
            if materialized is not None:
                await self._reconcile_materialization_attempt(
                    session,
                    job_id=job_id,
                    product_id=product_id,
                    materialized=materialized,
                )
            else:
                await self._release_video_finalization_claim(session, job_id, claim_token)
            raise
        else:
            return settled_job

    async def _claim_video_finalization(self, session: AsyncSession, job_id: UUID) -> str | None:
        """Durably acquire a recoverable PostgreSQL claim for one video job."""
        token = str(uuid4())
        result = await session.execute(
            update(GenerationJob)
            .where(
                GenerationJob.id == job_id,
                GenerationJob.status.in_([JobStatus.QUEUED.value, JobStatus.RUNNING.value]),
                or_(
                    GenerationJob.finalization_claim_token.is_(None),
                    GenerationJob.finalization_lease_expires_at <= func.now(),
                ),
            )
            .values(
                finalization_claim_token=token,
                # PostgreSQL time is authoritative for both the takeover
                # predicate and expiry.  Host-clock skew cannot create an
                # immediately expired (or unexpectedly long) lease.
                finalization_lease_expires_at=(
                    func.now()
                    + text("(:lease_seconds * interval '1 second')").bindparams(
                        lease_seconds=self._finalization_lease_seconds
                    )
                ),
            )
            .execution_options(synchronize_session=False)
        )
        if result.rowcount == 0:  # type: ignore[attr-defined]
            return None
        await session.commit()
        return token

    async def _release_video_finalization_claim(
        self, session: AsyncSession, job_id: UUID, claim_token: str
    ) -> None:
        """Release a failed materializer's claim so retries need not await TTL."""
        try:
            await session.execute(
                update(GenerationJob)
                .where(
                    GenerationJob.id == job_id,
                    GenerationJob.finalization_claim_token == claim_token,
                )
                .values(finalization_claim_token=None, finalization_lease_expires_at=None)
            )
            await session.commit()
        except Exception:
            await session.rollback()
            logger.warning("grok.video_finalization_claim_release_failed", job_id=str(job_id))

    async def _refresh_authoritative_job(
        self, session: AsyncSession, job_id: UUID
    ) -> GenerationJob:
        job = await session.get(GenerationJob, job_id)
        if job is None:
            raise GrokJobNotFoundError(f"Job {job_id} not found")
        await session.refresh(job)
        return job

    async def _materialize_video_result(
        self,
        *,
        user_id: UUID,
        job_id: UUID,
        result: GrokVideoResult,
        product_id: str | None = None,
        session: AsyncSession | None = None,
        claim_token: str | None = None,
    ) -> _MaterializedVideo:
        """Download/upload artifacts after recording their planned R2 keys.

        The durable attempt is inserted and committed before the first R2
        upload.  It contains every planned key, not just successful uploads,
        because a process can die after R2 accepts ``put_raw`` but before the
        following database acknowledgement reaches this process.
        """
        response = await self.http_client.get(result.url)
        response.raise_for_status()
        video_data = response.content
        expires_at = datetime.now(UTC) + timedelta(days=self._retention_days)
        output_id = new_id()
        storage_key = self._storage.build_storage_key(
            user_id=user_id,
            file_id=output_id,
            storage_type=StorageType.OUTPUT,
            format=MediaFormat.MP4,
            job_id=job_id,
        )
        artifacts: list[tuple[GenerationOutputData, bytes]] = [
            (
                GenerationOutputData(
                    id=output_id,
                    storage_key=storage_key,
                    content_type=MediaFormat.MP4.content_type,
                    size_bytes=len(video_data),
                    format=MediaFormat.MP4.value,
                    output_index=0,
                    expires_at=expires_at,
                ),
                video_data,
            )
        ]

        # Derive all thumbnails before the first upload so their keys are also
        # durable before they can exist in R2. Thumbnail generation is still
        # best-effort; a failed upload merely omits that output record.
        frame_bytes = await extract_video_thumbnail(video_data)
        if frame_bytes:
            for generated in await make_image_thumbnails(frame_bytes):
                thumbnail_id = new_id()
                thumbnail_key = self._storage.build_storage_key(
                    user_id=user_id,
                    file_id=thumbnail_id,
                    storage_type=StorageType.OUTPUT,
                    format=MediaFormat.WEBP,
                    job_id=job_id,
                )
                artifacts.append(
                    (
                        GenerationOutputData(
                            id=thumbnail_id,
                            storage_key=thumbnail_key,
                            content_type=MediaFormat.WEBP.content_type,
                            size_bytes=len(generated.result.data),
                            format=MediaFormat.WEBP.value,
                            output_index=0,
                            expires_at=expires_at,
                            is_thumbnail=True,
                            parent_output_id=output_id,
                            thumbnail_max_edge=generated.spec.max_edge,
                            width=generated.result.width,
                            height=generated.result.height,
                        ),
                        generated.result.data,
                    )
                )
        else:
            logger.warning("grok.thumbnail_skipped", job_id=str(job_id))

        attempt_id: UUID | None = None
        if session is not None and claim_token is not None:
            if product_id is None:
                raise ValueError("product_id is required for durable materialization attempts")
            attempt_id = await self._create_materialization_attempt(
                session,
                job_id=job_id,
                product_id=product_id,
                claim_token=claim_token,
                planned_storage_keys=[artifact.storage_key for artifact, _ in artifacts],
            )

        uploaded_keys: list[str] = []
        persisted_outputs: list[GenerationOutputData] = []
        try:
            for position, (artifact, payload) in enumerate(artifacts):
                try:
                    await self._put_output_object(
                        artifact.storage_key,
                        payload,
                        artifact.content_type,
                    )
                except asyncio.CancelledError:
                    raise
                except Exception:
                    if position == 0:
                        raise
                    logger.warning(
                        "grok.thumbnail_skipped",
                        job_id=str(job_id),
                        max_edge=artifact.thumbnail_max_edge,
                    )
                    continue
                uploaded_keys.append(artifact.storage_key)
                persisted_outputs.append(artifact)
                if attempt_id is not None and session is not None and product_id is not None:
                    await self._record_materialization_upload(
                        session,
                        attempt_id,
                        product_id=product_id,
                        uploaded_storage_keys=uploaded_keys,
                    )
            return _MaterializedVideo(
                outputs=persisted_outputs,
                storage_keys=uploaded_keys,
                attempt_id=attempt_id,
                product_id=product_id,
            )
        except BaseException:
            if attempt_id is not None and session is not None and product_id is not None:
                await asyncio.shield(
                    self._mark_materialization_attempt_cleanup_pending(
                        session,
                        attempt_id=attempt_id,
                        product_id=product_id,
                        error="materialization interrupted before outputs were committed",
                    )
                )
                # No output transition has started on this path, so every
                # uploaded key is provably unreferenced.  Try immediately,
                # but keep the durable attempt pending if R2 is unavailable.
                await asyncio.shield(self._cleanup_materialized_objects(uploaded_keys))
            else:
                await self._cleanup_materialized_objects(uploaded_keys)
            raise

    async def _put_output_object(self, key: str, data: bytes, content_type: str) -> None:
        """Store a generated artifact through R2's public service boundary."""
        await self._storage.put_raw(key, data, content_type=content_type)

    async def _create_materialization_attempt(
        self,
        session: AsyncSession,
        *,
        job_id: UUID,
        product_id: str,
        claim_token: str,
        planned_storage_keys: list[str],
    ) -> UUID:
        """Commit attempt evidence before the first external storage write."""
        attempt_id = new_id()
        session.add(
            GenerationMaterializationAttempt(
                id=attempt_id,
                job_id=job_id,
                product_id=product_id,
                claim_token=claim_token,
                state="planned",
                planned_storage_keys=planned_storage_keys,
                uploaded_storage_keys=[],
            )
        )
        await session.commit()
        return attempt_id

    async def _record_materialization_upload(
        self,
        session: AsyncSession,
        attempt_id: UUID,
        *,
        product_id: str,
        uploaded_storage_keys: list[str],
    ) -> None:
        """Persist upload progress; planned keys cover acknowledgement loss."""
        await session.execute(
            update(GenerationMaterializationAttempt)
            .where(
                GenerationMaterializationAttempt.id == attempt_id,
                GenerationMaterializationAttempt.product_id == product_id,
            )
            .values(
                state="uploading",
                uploaded_storage_keys=list(uploaded_storage_keys),
                updated_at=func.now(),
            )
        )
        await session.commit()

    async def _mark_materialization_attempt_cleanup_pending(
        self,
        session: AsyncSession,
        *,
        attempt_id: UUID,
        product_id: str,
        error: str | None = None,
    ) -> None:
        """Leave durable cleanup work for a restart-safe reconciliation pass."""
        try:
            await session.rollback()
            await session.execute(
                update(GenerationMaterializationAttempt)
                .where(
                    GenerationMaterializationAttempt.id == attempt_id,
                    GenerationMaterializationAttempt.product_id == product_id,
                )
                .values(
                    state="cleanup_pending",
                    reconciliation_error=error,
                    updated_at=func.now(),
                )
            )
            await session.commit()
        except Exception:
            await session.rollback()
            logger.exception(
                "grok.video_materialization_attempt_mark_cleanup_failed",
                attempt_id=str(attempt_id),
            )

    async def _reconcile_materialization_attempt(
        self,
        session: AsyncSession,
        *,
        job_id: UUID,
        product_id: str,
        materialized: _MaterializedVideo,
    ) -> None:
        """Use DB truth before deleting objects after an uncertain transition.

        The caller may have lost a commit acknowledgement or failed while
        refreshing a committed row.  Referenced keys are preserved.  When the
        transaction outcome is not provable, the attempt remains ``ambiguous``
        for the reconciliation worker instead of deleting media optimistically.
        """
        if materialized.attempt_id is None:
            # Compatibility with direct unit-test helpers: without a durable
            # attempt we can still safely delete only after a DB reference
            # query, but cannot promise restart recovery.
            attempt_keys = materialized.storage_keys
        else:
            await session.rollback()
            attempt = await session.scalar(
                select(GenerationMaterializationAttempt).where(
                    GenerationMaterializationAttempt.id == materialized.attempt_id,
                    GenerationMaterializationAttempt.product_id == product_id,
                )
            )
            attempt_keys = list(attempt.planned_storage_keys) if attempt is not None else []

        try:
            job = await self._refresh_authoritative_job(session, job_id)
            referenced_keys = set(
                (
                    await session.scalars(
                        select(GenerationOutput.storage_key).where(
                            GenerationOutput.job_id == job_id,
                            GenerationOutput.product_id == product_id,
                        )
                    )
                ).all()
            )
        except Exception:
            if materialized.attempt_id is not None:
                await self._mark_materialization_attempt_cleanup_pending(
                    session,
                    attempt_id=materialized.attempt_id,
                    product_id=product_id,
                    error="could not establish database commit outcome",
                )
                await session.execute(
                    update(GenerationMaterializationAttempt)
                    .where(
                        GenerationMaterializationAttempt.id == materialized.attempt_id,
                        GenerationMaterializationAttempt.product_id == product_id,
                    )
                    .values(state="ambiguous", updated_at=func.now())
                )
                await session.commit()
            logger.exception("grok.video_materialization_reconcile_unknown", job_id=str(job_id))
            return

        unreferenced = [key for key in attempt_keys if key not in referenced_keys]
        if job.status == JobStatus.COMPLETED.value and not unreferenced:
            if materialized.attempt_id is not None:
                await session.execute(
                    update(GenerationMaterializationAttempt)
                    .where(
                        GenerationMaterializationAttempt.id == materialized.attempt_id,
                        GenerationMaterializationAttempt.product_id == product_id,
                    )
                    .values(state="committed", completed_at=func.now(), updated_at=func.now())
                )
                await session.commit()
            return

        if materialized.attempt_id is not None:
            await self._mark_materialization_attempt_cleanup_pending(
                session,
                attempt_id=materialized.attempt_id,
                product_id=product_id,
                error="attempt has unreferenced artifact keys" if referenced_keys else None,
            )
        # These keys are proven not to be attached to an output row in the
        # authoritative read.  A failure remains durable through the attempt.
        try:
            await self._storage.delete_many(unreferenced)
        except Exception:
            logger.exception(
                "grok.video_materialization_cleanup_failed",
                count=len(unreferenced),
                job_id=str(job_id),
            )
            return

        if materialized.attempt_id is not None:
            await session.execute(
                update(GenerationMaterializationAttempt)
                .where(
                    GenerationMaterializationAttempt.id == materialized.attempt_id,
                    GenerationMaterializationAttempt.product_id == product_id,
                )
                .values(state="cleaned", reconciliation_error=None, updated_at=func.now())
            )
            await session.commit()

    async def _cleanup_materialized_objects(self, storage_keys: list[str]) -> None:
        if not storage_keys:
            return
        try:
            await self._storage.delete_many(storage_keys)
        except Exception:
            # Do not hide an already-known DB/materialization failure.  The
            # retention/orphan worker remains the final safety net.
            logger.exception("grok.video_materialization_cleanup_failed", count=len(storage_keys))

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
