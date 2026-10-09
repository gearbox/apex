"""Wan 3.0 video job service (Alibaba Model Studio).

Starts asynchronous Wan tasks and exposes the polling/settlement entry points
``AsyncVideoJobWorker`` and poll-on-read drive. Completion (claim, download,
R2 materialization, reconciliation) is delegated to the shared
``RemoteVideoMaterializer``; this service only knows how Model Studio reports
progress and failures.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, ClassVar, NoReturn

import structlog

from src.api.services.generation.base import ProviderSubmitResult
from src.api.services.generation.provider_billing_policy import (
    DEFAULT_PROVIDER_BILLING_POLICIES,
    apply_provider_billing_policy,
)
from src.api.services.generation.provider_failures import (
    ProviderFailure,
    ProviderFailureKind,
    ProviderModerationRejectedError,
    ProviderSubmissionFailedError,
)
from src.api.services.model_studio.errors import ModelStudioAPIError, ModelStudioTransientError
from src.api.services.model_studio.failure_classifier import ModelStudioFailureClassifier
from src.api.services.model_studio.types import ModelStudioTask, ModelStudioTaskStatus
from src.api.services.video_jobs.outcome import RemoteVideoResult, VideoPollOutcome
from src.api.services.video_jobs.settlement import settle_video_poll_outcome
from src.api.services.video_jobs.transition import make_transition_service_factory
from src.core.enums import JobStatus, Provider, VideoPollStatus
from src.core.uid import new_id
from src.db.models.billing import TokenTransaction
from src.db.repositories.job import JobRepository

if TYPE_CHECKING:
    from collections.abc import Mapping
    from uuid import UUID

    from sqlalchemy.ext.asyncio import AsyncSession

    from src.api.services.billing import BalanceEvent, BillingService
    from src.api.services.event_bus import EventBus
    from src.api.services.generation.provider_billing_policy import ProviderBillingPolicyRegistry
    from src.api.services.model_studio.client import ModelStudioClient
    from src.api.services.ops_event_bus import OpsEventBus
    from src.api.services.video_jobs.download import VideoDownloadClient
    from src.api.services.video_jobs.materializer import RemoteVideoMaterializer
    from src.core.enums import AspectRatio, GenerationType, ModelType, VideoResolution
    from src.db.models import GenerationJob

logger = structlog.get_logger(__name__)


class WanJobError(Exception):
    """Base exception for Wan job service errors."""


class WanJobNotFoundError(WanJobError):
    """Raised when a job is not found."""


def _require_job(job: GenerationJob | None, job_id: UUID) -> GenerationJob:
    if job is None:
        raise WanJobError(f"Job {job_id} disappeared after start")
    return job


class WanJobService:
    """Starts, polls and settles Wan video jobs."""

    provider: ClassVar[Provider] = Provider.ALIBABA
    transient_poll_errors: ClassVar[tuple[type[Exception], ...]] = (ModelStudioTransientError,)
    # Model Studio reports no progress; we do not invent a percentage.
    running_progress_pct: ClassVar[int | None] = None

    def __init__(
        self,
        client: ModelStudioClient,
        materializer: RemoteVideoMaterializer,
        download_client: VideoDownloadClient,
        *,
        billing_service: BillingService | None = None,
        event_bus: EventBus | None = None,
        ops_event_bus: OpsEventBus | None = None,
        max_poll_time: int = 1800,
        billing_policy: ProviderBillingPolicyRegistry | None = None,
    ) -> None:
        """Wire the service.

        Args:
            client: Model Studio REST client (already ``connect()``-ed by the owner).
            materializer: Shared completion path for finished videos.
            download_client: Owns the credential-free HTTP client the materializer uses.
            billing_service: Ledger service for settlement transitions.
            event_bus: Real-time event bus.
            ops_event_bus: Ops event bus.
            max_poll_time: Seconds from acceptance after which a job is timed out.
            billing_policy: Per-provider charge/refund rules.
        """
        self._client = client
        self._materializer = materializer
        self._download = download_client
        self._billing = billing_service
        self._max_poll_time = max_poll_time
        self._billing_policy = billing_policy or DEFAULT_PROVIDER_BILLING_POLICIES
        self._classifier = ModelStudioFailureClassifier()
        self._transition_factory = make_transition_service_factory(
            billing_service=billing_service, event_bus=event_bus, ops_event_bus=ops_event_bus
        )

    async def connect(self) -> None:
        """Open the download client (the API client is owned by the container)."""
        await self._download.connect()

    async def close(self) -> None:
        """Close the download client."""
        await self._download.close()

    # -------------------------------------------------------------------------
    # Submission
    # -------------------------------------------------------------------------

    async def start_video_job(
        self,
        session: AsyncSession,
        *,
        user_id: UUID,
        prompt: str,
        model: ModelType,
        generation_type: GenerationType,
        payload: Mapping[str, object],
        aspect_ratio: AspectRatio | None,
        name: str | None,
        billing_service: BillingService,
        account_id: UUID,
        token_cost: int,
        billed_seconds: int,
        video_resolution: VideoResolution,
        product_id: str,
        source_job_id: UUID | None = None,
        primary_output_id: UUID | None = None,
        primary_upload_id: UUID | None = None,
    ) -> ProviderSubmitResult:
        """Create the job, reserve tokens and submit the Wan task.

        Saga: job row (PENDING) → ``check_and_reserve`` → ``create_video_task`` →
        QUEUED with the task id. Provider failures are raised as
        ``ProviderSubmissionFailedError`` for the orchestrator to settle.

        Args:
            payload: The prebuilt request body (see ``build_wan_video_payload``).
            aspect_ratio: The ratio the client asked for; ``None`` is stored as
                such rather than fabricating one for ``adaptive``.
            billed_seconds: The duration basis the reservation was priced on.
        """
        job_repo = JobRepository(session)
        if name is None:
            name = prompt[:50].strip()
            if len(prompt) > 50:
                name += "..."

        job_id = new_id()
        job: GenerationJob | None = await job_repo.create(
            id=job_id,
            user_id=user_id,
            name=name,
            prompt=prompt,
            generation_type=generation_type,
            status=JobStatus.PENDING,
            provider=Provider.ALIBABA,
            model=model.value,
            aspect_ratio=aspect_ratio.value if aspect_ratio is not None else None,
            product_id=product_id,
            source_job_id=source_job_id,
            primary_output_id=primary_output_id,
            primary_upload_id=primary_upload_id,
        )

        reserve_event: BalanceEvent | None = None
        reservation_created = False
        provider_attempted = False
        provider_accepted = False
        try:
            try:
                reserve_result = await billing_service.check_and_reserve(
                    account_id,
                    token_cost,
                    job_id,
                    metadata={
                        "type": "generation",
                        "provider": Provider.ALIBABA.value,
                        "generation_type": generation_type.value,
                        "model": model.value,
                        "billed_seconds": billed_seconds,
                        "video_resolution": video_resolution.value,
                    },
                    description="Generation charge",
                    session=session,
                    product_id=product_id,
                )
            except Exception:
                # Billing/domain failures are not provider failures; the caller
                # rolls back the uncommitted job.
                await session.rollback()
                raise
            reservation_created = True
            if job is not None:
                job.token_cost = token_cost
                job.debit_transaction_id = reserve_result.txn.id
            reserve_event = reserve_result.event
            await session.flush()

            provider_attempted = True
            task = await self._client.create_video_task(payload)
            provider_accepted = True

            job = await job_repo.update_status(
                job_id,
                JobStatus.QUEUED,
                external_request_id=task.task_id,
                started_at=datetime.now(UTC),
            )
            job = _require_job(job, job_id)
            logger.info("alibaba.video_job_started", job_id=str(job_id), task_id=task.task_id)
            return ProviderSubmitResult(
                job=job,
                balance_after=reserve_result.txn.balance_after,
                balance_event=reserve_result.event,
            )

        except ModelStudioAPIError as exc:
            self._raise_submission_failure(
                exc.failure,
                cause=exc,
                job=job,
                job_id=job_id,
                user_id=user_id,
                product_id=product_id,
                reserve_event=reserve_event,
            )
        except Exception as exc:
            if provider_attempted and not provider_accepted and reservation_created:
                logger.exception("alibaba.unexpected_submission_error", job_id=str(job_id))
                self._raise_submission_failure(
                    ProviderFailure(
                        kind=ProviderFailureKind.UNKNOWN,
                        provider=Provider.ALIBABA,
                        sanitized_message=ProviderFailure.safe_message_for_kind(
                            ProviderFailureKind.UNKNOWN
                        ),
                        retryable=True,
                    ),
                    cause=exc,
                    job=job,
                    job_id=job_id,
                    user_id=user_id,
                    product_id=product_id,
                    reserve_event=reserve_event,
                )
            raise

    def _raise_submission_failure(
        self,
        raw_failure: ProviderFailure,
        *,
        cause: BaseException,
        job: GenerationJob | None,
        job_id: UUID,
        user_id: UUID,
        product_id: str,
        reserve_event: BalanceEvent | None,
    ) -> NoReturn:
        """Apply billing policy, log, and raise; GenerationService is the sole settler."""
        failure = apply_provider_billing_policy(raw_failure, registry=self._billing_policy)
        self._log_failure(failure, job_id=job_id, user_id=user_id, product_id=product_id)
        error_type = (
            ProviderModerationRejectedError
            if failure.kind is ProviderFailureKind.MODERATION_REJECTED
            else ProviderSubmissionFailedError
        )
        raise error_type(
            failure=failure,
            job_id=job_id,
            previous_status=str(job.status) if job is not None else JobStatus.PENDING.value,
            balance_event=reserve_event,
        ) from cause

    @staticmethod
    def _log_failure(
        failure: ProviderFailure, *, job_id: UUID, user_id: UUID, product_id: str
    ) -> None:
        logger.warning(
            "alibaba.failure_classified",
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

    # -------------------------------------------------------------------------
    # Polling and settlement
    # -------------------------------------------------------------------------

    async def poll_video_job(self, session: AsyncSession, job_id: UUID) -> GenerationJob | None:
        """Poll-on-read: settle a queued/running job through the shared path.

        Transient upstream errors are reported as "unchanged" so a status request
        never 500s on a Model Studio hiccup.
        """
        job = await JobRepository(session).get(job_id)
        if job is None:
            raise WanJobNotFoundError(f"Job {job_id} not found")
        if job.status not in (JobStatus.QUEUED.value, JobStatus.RUNNING.value):
            return job

        product_id = str(job.product_id)
        if self._is_overdue(job):
            return await self.settle_video_poll_outcome(
                session, job_id=job_id, outcome=self._timeout_outcome(job), product_id=product_id
            )
        try:
            outcome = await self._poll_video_result(session, job)
        except ModelStudioTransientError as exc:
            self._log_transient(job, exc.failure)
            return job
        return await self.settle_video_poll_outcome(
            session, job_id=job_id, outcome=outcome, product_id=product_id
        )

    async def poll_video_job_for_worker(
        self, session: AsyncSession, job_id: UUID
    ) -> VideoPollOutcome:
        """Poll one job for the periodic worker without mutating ``job.status``.

        Raises:
            WanJobNotFoundError: The job does not exist.
            ModelStudioTransientError: Rate limit, timeout or upstream 5xx; the worker
                skips this tick without a transition.
        """
        job = await JobRepository(session).get(job_id)
        if job is None:
            raise WanJobNotFoundError(f"Job {job_id} not found")
        if job.status not in (JobStatus.QUEUED.value, JobStatus.RUNNING.value):
            return VideoPollOutcome(status=VideoPollStatus.STILL_RUNNING)
        if self._is_overdue(job):
            return self._timeout_outcome(job)
        return await self._poll_video_result(session, job)

    async def settle_video_poll_outcome(
        self,
        session: AsyncSession,
        *,
        job_id: UUID,
        outcome: VideoPollOutcome,
        product_id: str,
    ) -> GenerationJob:
        """Apply a poll outcome exactly once (see ``settle_video_poll_outcome``)."""
        return await settle_video_poll_outcome(
            session,
            job_id=job_id,
            outcome=outcome,
            product_id=product_id,
            transition_service=self._transition_factory(session),
            materializer=self._materializer,
            settle_failure=self._settle_failed_outcome,
        )

    async def _settle_failed_outcome(
        self,
        session: AsyncSession,
        job_id: UUID,
        outcome: VideoPollOutcome,
        product_id: str,
    ) -> GenerationJob:
        return await self.settle_video_poll_outcome(
            session, job_id=job_id, outcome=outcome, product_id=product_id
        )

    async def _poll_video_result(
        self, session: AsyncSession, job: GenerationJob
    ) -> VideoPollOutcome:
        task_id = job.external_request_id
        if not task_id:
            raise WanJobError(f"Job {job.id} has no Model Studio task id")

        try:
            task = await self._client.get_task(task_id)
        except ModelStudioTransientError:
            raise
        except ModelStudioAPIError as exc:
            return self._failed_outcome(exc.failure, job)

        if task.task_status in (ModelStudioTaskStatus.PENDING, ModelStudioTaskStatus.RUNNING):
            return VideoPollOutcome(status=VideoPollStatus.STILL_RUNNING)

        if task.task_status is ModelStudioTaskStatus.SUCCEEDED:
            if not task.video_url:
                return self._failed_outcome(
                    self._classifier.malformed(
                        provider_request_accepted=True, provider_request_id=task_id
                    ),
                    job,
                )
            await self._observe_billing_basis(session, job, task)
            return VideoPollOutcome(
                status=VideoPollStatus.COMPLETED,
                result=RemoteVideoResult(url=task.video_url, provider_request_id=task_id),
            )

        # FAILED carries a code; CANCELED/UNKNOWN carry none and classify as UNKNOWN.
        return self._failed_outcome(
            self._classifier.classify(
                code=task.code, provider_request_accepted=True, provider_request_id=task_id
            ),
            job,
        )

    def _failed_outcome(self, raw_failure: ProviderFailure, job: GenerationJob) -> VideoPollOutcome:
        failure = apply_provider_billing_policy(raw_failure, registry=self._billing_policy)
        self._log_failure(
            failure, job_id=job.id, user_id=job.user_id, product_id=str(job.product_id)
        )
        return VideoPollOutcome(
            status=VideoPollStatus.FAILED,
            error_message=failure.sanitized_message,
            failure=failure,
        )

    async def _observe_billing_basis(
        self, session: AsyncSession, job: GenerationJob, task: ModelStudioTask
    ) -> None:
        """Log when upstream's billed duration differs from the reserved basis.

        Observation only: V2V edits can come back close to the input length, so
        ``usage.duration`` may differ from what we priced. No extra charge or
        refund is made here; reconciliation needs real data first.
        """
        if task.usage is None or task.usage.duration is None or job.debit_transaction_id is None:
            return
        try:
            debit = await session.get(TokenTransaction, job.debit_transaction_id)
            reserved = debit.metadata_.get("billed_seconds") if debit is not None else None
            if isinstance(reserved, int) and round(task.usage.duration) != reserved:
                logger.warning(
                    "alibaba.billing_basis_mismatch",
                    job_id=str(job.id),
                    task_id=task.task_id,
                    reserved_seconds=reserved,
                    upstream_seconds=round(task.usage.duration),
                )
        except Exception:
            logger.warning("alibaba.billing_basis_check_failed", job_id=str(job.id))

    def _is_overdue(self, job: GenerationJob) -> bool:
        """TTL runs from provider acceptance, falling back to submission time."""
        basis = job.started_at or job.created_at
        if not isinstance(basis, datetime):
            return False
        return (datetime.now(UTC) - basis).total_seconds() > self._max_poll_time

    def _timeout_outcome(self, job: GenerationJob) -> VideoPollOutcome:
        return self._failed_outcome(
            self._classifier.timeout(
                provider_request_accepted=True, provider_request_id=job.external_request_id
            ),
            job,
        )

    @staticmethod
    def _log_transient(job: GenerationJob, failure: ProviderFailure) -> None:
        logger.warning(
            "alibaba.video_poll_transient",
            job_id=str(job.id),
            user_id=str(job.user_id),
            product_id=job.product_id,
            provider_job_id=job.external_request_id,
            failure_kind=failure.kind.value,
            provider_status=failure.provider_status_code,
            retryable=failure.retryable,
        )
