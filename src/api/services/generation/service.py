"""Unified generation orchestrator — routes requests to providers."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

import structlog

from src.api.schemas.jobs import JobCreatedResponse
from src.api.schemas.ops_events import GenerationCreatedOpsPayload, OpsEventType
from src.api.services.generation.provider_failures import ProviderModerationRejectedError
from src.api.services.ops_event_bus import OpsEventBus
from src.core.enums import GenerationType, JobStatus, Provider
from src.core.model_registry import get_model_meta
from src.db.repositories.generation_model import GenerationModelRepository
from src.db.repositories.user import UserRepository
from src.db.repositories.user_image import UserImageRepository

if TYPE_CHECKING:
    from uuid import UUID

    from sqlalchemy.ext.asyncio import AsyncSession

    from src.api.schemas.unified_generation import UnifiedGenerationRequest
    from src.api.services.billing import BalanceEvent, BillingService
    from src.api.services.event_bus import EventBus
    from src.api.services.generation.base import GenerationProvider
    from src.api.services.generation.rate_limiter import ModelRateLimiter
    from src.api.services.pricing import PricingService
    from src.core.product import ProductConfig
    from src.db.models.storage import GenerationJob

logger = structlog.get_logger(__name__)


class GenerationError(Exception):
    """Base error for generation failures."""


class FeatureNotSupportedError(GenerationError):
    """Raised when a provider does not (yet) support a requested capability.

    Maps to HTTP 400 with error code "not_implemented". The message propagates
    to API responses — keep it provider-framed and user-safe.
    """


class ProviderUnavailableError(GenerationError):
    """Raised when the resolved provider is not configured."""


class ModelDisabledError(GenerationError):
    """Raised when the requested model is disabled via generation_models."""


class ModelNotAllowedError(GenerationError):
    """Raised when the model is not available on the current product."""


class AgeVerificationRequiredError(GenerationError):
    """Raised when the model requires age verification and the user is not verified."""


class ProviderResponseError(GenerationError):
    """Raised when a generation backend returned a malformed or unexpected response.

    The user-facing message on this exception MUST NOT name the specific backend
    (ComfyUI, Grok, etc.) — it propagates directly into API responses. Backend
    detail belongs in structured logs only.
    """


class GenerationService:
    """Orchestrates generation requests across providers.

    Responsibilities:
    1. Validate model-generation_type compatibility (via ModelType enum)
    2. Validate required inputs (image_id, video_url)
    3. Validate n <= model.max_concurrent_outputs
    4. Check model is enabled in generation_models table
    4.5. Global per-model rate limit check
    5. Resolve provider from ModelType
    6. Delegate provider-specific validation
    7. Billing saga (reserve -> submit -> refund on error)
    8. Delegate to provider strategy
    """

    def __init__(
        self,
        providers: dict[Provider, GenerationProvider],
        billing_service: BillingService,
        pricing_service: PricingService,
        rate_limiter: ModelRateLimiter,
        event_bus: EventBus | None = None,
        ops_event_bus: OpsEventBus | None = None,
        *,
        retention_days: int = 7,
    ) -> None:
        self._providers = providers
        self._billing = billing_service
        self._pricing = pricing_service
        self._rate_limiter = rate_limiter
        self._event_bus = event_bus
        self._ops_event_bus = (
            ops_event_bus if ops_event_bus is not None else OpsEventBus(enabled=False)
        )
        self._retention_days = retention_days

    @property
    def configured_providers(self) -> frozenset[Provider]:
        """Providers that are fully wired and able to serve requests."""
        return frozenset(self._providers)

    async def generate(
        self,
        request: UnifiedGenerationRequest,
        *,
        user_id: UUID,
        session: AsyncSession,
        product_config: ProductConfig,
    ) -> JobCreatedResponse:
        """Execute the full generation pipeline."""
        # 1. Model-generation_type compatibility (declarative, enum-driven)
        if not request.model.supports_generation_type(request.generation_type):
            raise ValueError(
                f"Model '{request.model.value}' does not support generation type '{request.generation_type.value}'"
            )

        # 1.5. Product model access check
        if not product_config.is_model_allowed(request.model):
            raise ModelNotAllowedError(
                f"Model '{request.model.value}' is not available on {product_config.display_name}"
            )

        # 1.6. Per-model age verification gate (authoritative regardless of product age_gate policy)
        if request.model.requires_age_verification:
            age_user = await UserRepository(session).get_user(user_id)
            if age_user is None or age_user.age_verified_at is None:
                raise AgeVerificationRequiredError(
                    f"Model '{request.model.value}' requires age verification"
                )

        # 1.7 Aspect-ratio capability validation (registry-driven)
        if request.aspect_ratio is not None and not request.generation_type.is_video:
            meta = get_model_meta(request.model)
            if request.generation_type == GenerationType.I2I:
                allowed = meta.image.edit_aspect_ratios if meta.image else ()
                if request.aspect_ratio not in allowed:
                    raise ValueError(
                        f"Model '{request.model.value}' cannot reshape aspect ratio during "
                        f"image editing; omit aspect_ratio to preserve the source aspect"
                        + (f" or use one of: {[a.value for a in allowed]}" if allowed else "")
                    )
            elif request.aspect_ratio not in meta.aspect_ratios:
                raise ValueError(
                    f"Model '{request.model.value}' does not support aspect ratio "
                    f"'{request.aspect_ratio.value}' for t2i; supported: "
                    f"{[a.value for a in meta.aspect_ratios]}"
                )

        # 2. Cross-cutting input validation
        self._validate_inputs(request)

        # 3. Output count cap
        max_n = request.model.max_concurrent_outputs
        if request.n > max_n:
            raise ValueError(
                f"Model '{request.model.value}' supports max {max_n} outputs per request, requested {request.n}"
            )

        # 4. Check model is enabled in DB
        model_repo = GenerationModelRepository(session)
        model_record = await model_repo.get_by_model_key(request.model.value)
        if model_record is None or not model_record.is_enabled:
            raise ModelDisabledError(f"Model '{request.model.value}' is currently disabled")

        # 4.5 Global per-model rate limit check
        await self._rate_limiter.check(request.model)

        # 5. Resolve provider
        provider_key = request.model.provider
        provider = self._providers.get(provider_key)
        if provider is None:
            raise ProviderUnavailableError(f"Provider '{provider_key.value}' is not configured")

        if not provider.supports_image_sizing and (
            request.image_resolution is not None
            or request.width is not None
            or request.height is not None
        ):
            logger.info(
                "generation.image_sizing.ignored_for_grok",
                model=request.model.value,
                image_resolution=request.image_resolution.value
                if request.image_resolution
                else None,
                width=request.width,
                height=request.height,
            )

        # 6. Provider-specific validation
        provider.validate(request)

        # 7. Billing pre-flight
        account = await self._billing.resolve_account_for_user(user_id, session=session)
        token_cost = await self._pricing.quote(
            provider_key.value,
            request.generation_type.value,
            request.model.value,
            n=request.n,
            input_image_count=self._input_image_count(request),
            session=session,
        )
        await self._billing.assert_sufficient_balance(account.id, token_cost, session=session)

        # 8. Submit to provider (with billing saga)
        product_id = product_config.slug

        # Resolve lineage: if source_output_id is provided, look up the source job
        source_job_id: UUID | None = None
        resolved_source_output_id: UUID | None = request.source_output_id
        lineage_output_id: UUID | None = request.source_output_id
        if lineage_output_id is None and request.source_images is not None:
            lineage_output_id = next(
                (
                    image_ref.source_output_id
                    for image_ref in request.source_images
                    if image_ref.source_output_id is not None
                ),
                None,
            )
        if lineage_output_id is not None:
            from src.db.repositories.output import OutputRepository

            source_output = await OutputRepository(session).get(lineage_output_id, user_id=user_id)
            if source_output is None:
                raise ValueError("Source output not found")
            source_job_id = source_output.job_id
            resolved_source_output_id = lineage_output_id

        # Sliding retention: using an upload as generation input resets its
        # expiry window, so actively-used images are never swept mid-use.
        # Rides in the submit transaction — rolled back if submit fails.
        if request.input_image_id is not None:
            touched = await UserImageRepository(session).touch_expiry(
                request.input_image_id,
                user_id=user_id,
                expires_at=datetime.now(UTC) + timedelta(days=self._retention_days),
            )
            if not touched:
                raise ValueError("Input image not found")
            logger.info(
                "generation.input_image_expiry_extended",
                image_id=str(request.input_image_id),
                user_id=str(user_id),
                retention_days=self._retention_days,
            )

        job: GenerationJob | None = None
        reserve_event: BalanceEvent | None = None
        try:
            submit_result = await provider.submit(
                request,
                user_id=user_id,
                session=session,
                billing_service=self._billing,
                account_id=account.id,
                token_cost=token_cost,
                product_id=product_id,
                source_job_id=source_job_id,
                source_output_id=resolved_source_output_id,
            )
            job = submit_result.job
            reserve_event = submit_result.balance_event
            await session.commit()

        except ProviderModerationRejectedError as exc:
            # The Grok job service deliberately left the accepted moderation
            # reservation in this transaction. Commit the failed job + debit
            # together, then publish the corresponding events post-commit.
            await session.commit()
            logger.warning(
                "generation.failed",
                provider=exc.failure.provider.value,
                job_id=str(exc.job_id),
                user_id=str(user_id),
                product_id=product_id,
                failure_kind=exc.failure.kind.value,
                billable=exc.failure.billable,
            )
            await self._publish_moderation_rejection(
                exc,
                user_id=user_id,
                request=request,
            )
            raise
        except GenerationError:
            # Domain errors carry user-safe messages by contract (see
            # ProviderResponseError's docstring) — refund if a debit was
            # already written, then re-raise unchanged so the route surfaces
            # the deliberate message.
            await self._resolve_submit_failure(
                job, session=session, product_id=product_id, reserve_event=reserve_event
            )
            raise
        except Exception as exc:
            logger.exception("generation.submit_failed", model=request.model.value)
            await self._resolve_submit_failure(
                job, session=session, product_id=product_id, reserve_event=reserve_event
            )
            # Any other exception (asyncpg errors, httpx errors carrying tunnel
            # hostnames, raw provider payload fragments, ...) MUST NOT reach
            # the client — only this fixed message may.
            raise GenerationError("Generation failed due to an internal error.") from exc

        # Post-commit, best-effort notifications — a publish failure must never
        # affect the already-committed billing/job state (no refund, no client error).
        if self._event_bus is not None:
            try:
                await self._event_bus.publish_balance(reserve_event)

                from src.api.schemas.events import EventType, JobStatusPayload

                await self._event_bus.publish(
                    user_id=user_id,
                    event_type=EventType.JOB_STATUS_CHANGED,
                    payload=JobStatusPayload(
                        job_id=job.id,
                        status=job.status if isinstance(job.status, str) else job.status.value,
                        previous_status="none",
                        generation_type=request.generation_type.value,
                        provider=request.model.provider.value,
                    ),
                )
            except Exception:
                logger.exception("generation.post_commit_publish_failed", job_id=str(job.id))

        await self._ops_event_bus.publish(
            event_type=OpsEventType.GENERATION_CREATED,
            product_id=product_id,
            payload=GenerationCreatedOpsPayload(
                job_id=job.id,
                user_id=user_id,
                provider=request.model.provider.value,
                generation_type=request.generation_type.value,
            ),
        )

        # 10. Build response
        return JobCreatedResponse(
            job_id=job.id,
            status=JobStatus(job.status),
            name=job.name,
            model=request.model.value,
            generation_type=request.generation_type,
            created_at=job.created_at,
            tokens_charged=token_cost,
            balance_remaining=submit_result.balance_after,
        )

    async def _resolve_submit_failure(
        self,
        job: GenerationJob | None,
        *,
        session: AsyncSession,
        product_id: str,
        reserve_event: BalanceEvent | None = None,
    ) -> None:
        """Resolve the DB transaction after a failed ``provider.submit()`` call.

        ``job is None`` ⇒ no external side effect is durable yet — the job
        row and its debit, if either was written, are still uncommitted in
        this same transaction. Roll back rather than committing an orphaned
        charge for a job that never ran. ``reserve_event`` is never published
        in this branch (nothing committed).

        ``job is not None`` ⇒ the job row (and its debit) were already
        written by the provider. Refund and commit so the ledger nets to
        zero and the failed job stays visible in the user's history. A
        refund failure is treated the same as "nothing durable" — roll back
        rather than leave an uncovered debit. On success, both the original
        reserve's event and the refund's event describe rows that just
        committed together in this one transaction — publish both.
        """
        if job is None:
            await session.rollback()
            return
        try:
            refund_result = await self._billing.refund(
                job.id,
                description="Generation failed — tokens refunded",
                session=session,
                product_id=product_id,
            )
        except Exception:
            logger.exception("generation.refund_failed", job_id=str(job.id))
            await session.rollback()
            return
        await session.commit()
        if self._event_bus is not None:
            await self._event_bus.publish_balance(reserve_event)
            await self._event_bus.publish_balance(refund_result.event)

    async def _publish_moderation_rejection(
        self,
        exc: ProviderModerationRejectedError,
        *,
        user_id: UUID,
        request: UnifiedGenerationRequest,
    ) -> None:
        """Emit safe terminal job and balance events after the moderation commit."""
        if self._event_bus is None:
            return
        try:
            await self._event_bus.publish_balance(exc.balance_event)

            from src.api.schemas.events import EventType, JobStatusPayload

            await self._event_bus.publish(
                user_id=user_id,
                event_type=EventType.JOB_STATUS_CHANGED,
                payload=JobStatusPayload(
                    job_id=exc.job_id,
                    status=JobStatus.FAILED.value,
                    previous_status=JobStatus.RUNNING.value,
                    generation_type=request.generation_type.value,
                    provider=exc.failure.provider.value,
                    failure_code=exc.public_code,
                    error_message=exc.public_message,
                ),
            )
        except Exception:
            logger.exception(
                "generation.moderation_post_commit_publish_failed",
                job_id=str(exc.job_id),
            )

    @staticmethod
    def _validate_inputs(request: UnifiedGenerationRequest) -> None:
        """Cross-cutting validation: generation_type vs required inputs."""
        image_input_sources = [
            request.input_image_id is not None,
            request.source_output_id is not None,
            request.source_images is not None,
        ]
        if sum(image_input_sources) > 1:
            raise ValueError(
                "input_image_id, source_output_id, and source_images are mutually exclusive"
            )
        if request.source_images is not None and request.generation_type != GenerationType.I2I:
            raise ValueError("source_images is only supported for i2i generation")
        if request.generation_type.requires_image_input and not any(image_input_sources):
            raise ValueError(
                f"generation_type '{request.generation_type.value}' requires input_image_id, source_output_id, or source_images"
            )
        if request.generation_type.requires_video_input and request.input_video_url is None:
            raise ValueError(
                f"generation_type '{request.generation_type.value}' requires input_video_url"
            )

    @staticmethod
    def _input_image_count(request: UnifiedGenerationRequest) -> int:
        if request.source_images is not None:
            return len(request.source_images)
        if request.input_image_id is not None or request.source_output_id is not None:
            return 1
        return 0
