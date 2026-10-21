"""Unified generation orchestrator — routes requests to providers."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

import structlog

from src.api.schemas.jobs import JobCreatedResponse
from src.api.schemas.ops_events import GenerationCreatedOpsPayload, OpsEventType
from src.api.services.billing_errors import BillingError
from src.api.services.bundle_index import BundleIndexService, BundleNotFoundError
from src.api.services.generation.generation_modes import resolve_generation_modes
from src.api.services.generation.provider_failures import ProviderSubmissionFailedError
from src.api.services.generation.public_errors import (
    INTERNAL_FAILURE_CODE,
    INTERNAL_FAILURE_MESSAGE,
)
from src.api.services.generation.request_validation import validate_seed
from src.api.services.generation.source_media import (
    ResolvedSourceMedia,
    SourceMediaResolver,
    SourceMediaValidationError,
)
from src.api.services.generation.video_validation import (
    input_video_seconds,
    unsupported_video_parameters,
    validate_source_media,
    validate_video_request,
)
from src.api.services.job_state_transition import JobStateTransitionService
from src.api.services.ops_event_bus import OpsEventBus
from src.api.services.workflow.contract import (
    DIMENSION_REQUEST_PARAMETERS,
    DIMENSION_WRITABLE_PARAMETERS,
    NEGATIVE_PROMPT_PARAMETER,
    REQUEST_PARAMETER_BINDINGS,
)
from src.core.enums import (
    GenerationType,
    JobStatus,
    MediaKind,
    ModelType,
    Provider,
    ProvisioningMode,
)
from src.core.library_ref import LibraryAssetSource
from src.core.model_registry import get_model_meta
from src.db.repositories.generation_job_source import GenerationJobSourceRepository
from src.db.repositories.generation_model import GenerationModelRepository
from src.db.repositories.user import UserRepository
from src.db.repositories.user_image import UserImageRepository

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable
    from uuid import UUID

    from sqlalchemy.ext.asyncio import AsyncSession

    from src.api.schemas.unified_generation import UnifiedGenerationRequest
    from src.api.services.billing import BalanceEvent, BillingService
    from src.api.services.event_bus import EventBus
    from src.api.services.generation.base import GenerationProvider
    from src.api.services.generation.rate_limiter import ModelRateLimiter
    from src.api.services.pricing import PricingService
    from src.api.services.workflow.contract import BundleCapabilities
    from src.core.bundle_config import BundleMapping
    from src.core.generation_config import BundleGenerationConfig
    from src.core.generation_mode import GenerationModes, SourceMediaConstraints
    from src.core.model_registry import VideoMeta
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


class UnsupportedGenerationParameterError(GenerationError):
    """The resolved bundle cannot apply one or more request controls."""

    def __init__(self, parameters: list[str]) -> None:
        self.parameters = tuple(sorted(set(parameters)))
        super().__init__("Unsupported generation parameter(s): " + ", ".join(self.parameters))


class GenerationService:
    """Orchestrates generation requests across providers.

    Responsibilities:
    1. Validate model-generation_type compatibility (via ModelType enum)
    2. Resolve and validate owned media inputs
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
        bundle_index: BundleIndexService | None = None,
        *,
        retention_days: int = 7,
        transition_service_factory: Callable[[AsyncSession], JobStateTransitionService]
        | None = None,
    ) -> None:
        self._providers = providers
        self._billing = billing_service
        self._pricing = pricing_service
        self._rate_limiter = rate_limiter
        self._event_bus = event_bus
        self._ops_event_bus = (
            ops_event_bus if ops_event_bus is not None else OpsEventBus(enabled=False)
        )
        self._bundle_index = bundle_index
        self._retention_days = retention_days
        self._transition_service_factory = (
            transition_service_factory or self._default_transition_service_factory
        )

    @property
    def configured_providers(self) -> frozenset[Provider]:
        """Providers that are fully wired and able to serve requests."""
        return frozenset(self._providers)

    def _make_transition_service(self, session: AsyncSession) -> JobStateTransitionService:
        """Build the state-transition service for this transactional scope."""
        return self._transition_service_factory(session)

    def _default_transition_service_factory(
        self, session: AsyncSession
    ) -> JobStateTransitionService:
        return JobStateTransitionService(
            session=session,
            event_bus=self._event_bus,
            billing_service=self._billing,
            ops_event_bus=self._ops_event_bus,
        )

    def _aisha_mapping(self, model: ModelType) -> BundleMapping | None:
        if (
            self._bundle_index is None
            or model.provider.provisioning_mode is not ProvisioningMode.ON_DEMAND
        ):
            return None
        try:
            return self._bundle_index.resolve_bundle(model.value)
        except BundleNotFoundError:
            return None

    def get_aisha_capabilities(self, model: ModelType) -> BundleCapabilities | None:
        """Return the active bundle capability for an Aisha model, if indexed."""
        mapping = self._aisha_mapping(model)
        return mapping.capabilities if mapping is not None else None

    @staticmethod
    def _validate_bundle_capabilities(
        request: UnifiedGenerationRequest,
        capabilities: BundleCapabilities,
        generation: BundleGenerationConfig,
        modes: GenerationModes,
    ) -> None:
        """Reject overrides a bound workflow cannot write before billing begins."""
        if request.generation_type not in modes:
            raise UnsupportedGenerationParameterError(["generation_type"])
        unsupported: list[str] = []
        if request.negative_prompt is not None and not capabilities.supports_negative_prompt:
            unsupported.append(NEGATIVE_PROMPT_PARAMETER)
        writable = capabilities.writable
        # ``None`` means the client left a value to the bundle.  Sampler values
        # are compared by their scalar wire values, so StrEnums and strings both
        # work without a provider-specific branch.
        generation_defaults = generation.defaults
        for binding in REQUEST_PARAMETER_BINDINGS:
            value = getattr(request, binding.request_attribute)
            default = (
                getattr(generation_defaults, binding.generation_default_attribute)
                if binding.generation_default_attribute is not None
                else binding.request_default
            )
            if value is None or value == default:
                continue
            if binding.writable not in writable:
                unsupported.append(binding.parameter)
        if not DIMENSION_WRITABLE_PARAMETERS.issubset(writable):
            unsupported.extend(
                parameter
                for parameter in DIMENSION_REQUEST_PARAMETERS
                if getattr(request, parameter) is not None
            )
        if unsupported:
            raise UnsupportedGenerationParameterError(unsupported)

    async def generate(
        self,
        request: UnifiedGenerationRequest,
        *,
        user_id: UUID,
        session: AsyncSession,
        product_config: ProductConfig,
        defer_commit: bool = False,
        post_commit_callbacks: list[Callable[[], Awaitable[None]]] | None = None,
    ) -> JobCreatedResponse:
        """Execute the full generation pipeline."""
        # 1. Model-generation_type compatibility (declarative, enum-driven)
        if not request.model.supports_generation_type(request.generation_type):
            raise ValueError(
                f"Model '{request.model.value}' does not support generation type '{request.generation_type.value}'"
            )

        bundle_mapping = self._aisha_mapping(request.model)
        bundle_capabilities = bundle_mapping.capabilities if bundle_mapping is not None else None

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

        modes = resolve_generation_modes(request.model, capabilities=bundle_capabilities)

        # A production service always receives the index. Keeping the dependency
        # optional lets lightweight callers exercise the shared orchestration
        # path without fabricating a bundle graph, while an indexed on-demand
        # model must still have a valid, capability-bearing bundle.
        if (
            self._bundle_index is not None
            and request.model.provider.provisioning_mode is ProvisioningMode.ON_DEMAND
        ):
            if bundle_capabilities is None or bundle_mapping is None:
                raise ModelDisabledError(
                    f"Model '{request.model.value}' has no indexed workflow bundle"
                )
            self._validate_bundle_capabilities(
                request,
                bundle_capabilities,
                self._bundle_index.get_generation_config(
                    bundle_mapping.bundle_name, bundle_mapping.bundle_version
                ),
                modes,
            )

        # This lookup is total: step 1 ensures static registry membership, and
        # bundle validation above rejects any bundle-narrowed type first.
        contract = modes[request.generation_type].source_media

        # 1.7 Aspect-ratio capability validation (registry-driven)
        if (
            request.aspect_ratio is not None
            and request.generation_type.output_kind is not MediaKind.VIDEO
        ):
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

        # 1.75 Seed range (registry-driven, every model), before any billing work.
        validate_seed(request, get_model_meta(request.model))

        # 1.8 Video parameter validation (registry-driven), before any billing work.
        # Image outputs never accept the video-only optional parameters.
        video_meta: VideoMeta | None = None
        if request.generation_type.output_kind is MediaKind.VIDEO:
            model_meta = get_model_meta(request.model)
            video_meta = validate_video_request(request, model_meta)
            unsupported = unsupported_video_parameters(request, video_meta.supported_parameters)
        else:
            unsupported = unsupported_video_parameters(request, frozenset())
        if unsupported:
            raise UnsupportedGenerationParameterError(unsupported)

        # 2. Reject impossible source counts before resolving any asset. This
        # keeps count errors deterministic and avoids needless DB queries.
        self._validate_source_cardinality(request, contract)

        # 2.1 Resolve owned-library media once, before any consumer needs it.
        resolved_source_media: list[ResolvedSourceMedia] = []
        if request.source_media is not None:
            resolved_source_media = await SourceMediaResolver().resolve(
                request.source_media,
                user_id=user_id,
                session=session,
                product_id=product_config.slug,
            )

        # 2.2 Per-item validation requires the ownership-checked rows.
        self._validate_resolved_sources(resolved_source_media, contract)
        if video_meta is not None:
            validate_source_media(resolved_source_media, request, video_meta)

        # 3. Output count cap
        max_n = request.model.max_concurrent_outputs
        if bundle_capabilities is not None:
            max_n = min(max_n, bundle_capabilities.max_batch_size)
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
        is_video = video_meta is not None
        token_cost = await self._pricing.quote(
            provider_key.value,
            request.generation_type.value,
            request.model.value,
            n=request.n,
            source_media_count=self._source_media_count(request),
            billed_seconds=(
                request.duration + input_video_seconds(resolved_source_media, video_meta)
                if video_meta is not None
                else 0
            ),
            video_resolution=request.resolution if is_video else None,
            session=session,
        )
        await self._billing.assert_sufficient_balance(account.id, token_cost, session=session)

        # 8. Submit to provider (with billing saga)
        product_id = product_config.slug

        # Existing primary-lineage columns retain their historic meaning:
        # they point at the first output source, while ordered provenance is
        # written separately below.
        first_output_source = next(
            (
                source
                for source in resolved_source_media
                if source.ref.source is LibraryAssetSource.OUTPUT
            ),
            None,
        )
        source_job_id: UUID | None = (
            first_output_source.job_id if first_output_source is not None else None
        )
        primary_output_id: UUID | None = (
            first_output_source.ref.asset_id if first_output_source is not None else None
        )
        first_upload_source = next(
            (
                source
                for source in resolved_source_media
                if source.ref.source is LibraryAssetSource.UPLOAD
            ),
            None,
        )
        primary_upload_id: UUID | None = (
            first_upload_source.ref.asset_id if first_upload_source is not None else None
        )

        # Sliding retention: using an upload as generation input resets its
        # expiry window, so actively-used images are never swept mid-use.
        # Rides in the submit transaction — rolled back if submit fails.
        upload_ids = [
            source.ref.asset_id
            for source in resolved_source_media
            if source.ref.source is LibraryAssetSource.UPLOAD
        ]
        if upload_ids:
            touched = await UserImageRepository(session).touch_expiry_many(
                upload_ids,
                user_id=user_id,
                expires_at=datetime.now(UTC) + timedelta(days=self._retention_days),
            )
            if touched < len(upload_ids):
                raise SourceMediaValidationError(
                    "One or more source assets are no longer available"
                )
            logger.info(
                "generation.source_media.expiry_extended",
                user_id=str(user_id),
                retention_days=self._retention_days,
                count=len(upload_ids),
                positions=[
                    source.position
                    for source in resolved_source_media
                    if source.ref.source is LibraryAssetSource.UPLOAD
                ],
            )

        job: GenerationJob | None = None
        reserve_event: BalanceEvent | None = None
        receipt_held = False
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
                primary_output_id=primary_output_id,
                primary_upload_id=primary_upload_id,
                source_media=resolved_source_media,
            )
            job = submit_result.job
            reserve_event = submit_result.balance_event
            receipt_held = submit_result.receipt_held
            if resolved_source_media:
                await GenerationJobSourceRepository(session).create_many(
                    job_id=job.id,
                    product_id=product_id,
                    sources=resolved_source_media,
                )
            if not defer_commit:
                await session.commit()

        except ProviderSubmissionFailedError as exc:
            # Provider adapters only classify and preserve the submission
            # context. This single layer settles the terminal job, debit/refund
            # choice, commit, and post-commit events so no outer error handler
            # can roll the normalized failure back.
            transition_service = self._make_transition_service(session)
            settled_job, did_transition = await transition_service.transition_to_failed(
                exc.job_id,
                public_error_message=exc.public_message,
                failure_code=exc.public_code,
                refund=not exc.failure.billable,
                product_id=product_id,
                reserve_event=exc.balance_event,
                commit=not defer_commit,
            )
            if defer_commit and did_transition:
                publication = transition_service.deferred_failure
                if publication is None:
                    raise RuntimeError("Missing deferred failure publication") from None
                if post_commit_callbacks is None:
                    raise RuntimeError(
                        "Deferred generation requires post-commit callbacks"
                    ) from None
                post_commit_callbacks.append(
                    lambda: transition_service.publish_deferred_failure(publication)
                )
            logger.warning(
                "generation.failed",
                provider=exc.failure.provider.value,
                job_id=str(exc.job_id),
                user_id=str(user_id),
                product_id=product_id,
                failure_kind=exc.failure.kind.value,
                billable=exc.failure.billable,
                previous_status=exc.previous_status,
                did_transition=did_transition,
            )
            if exc.failure.billable and did_transition:
                logger.info(
                    "billing.charge_finalized",
                    job_id=str(settled_job.id),
                    product_id=product_id,
                    tokens_reserved=settled_job.token_cost,
                    tokens_finalized=settled_job.token_cost,
                )
            raise
        except BillingError:
            # A locked reservation can fail after the optimistic preflight.
            # It is a billing-domain response, not provider UNKNOWN, and no
            # refund is valid because check_and_reserve did not create a debit.
            await session.rollback()
            raise
        except GenerationError:
            # Domain errors carry user-safe messages by contract (see
            # ProviderResponseError's docstring) — refund if a debit was
            # already written, then re-raise unchanged so the route surfaces
            # the deliberate message.
            await self._resolve_submit_failure(
                job,
                session=session,
                product_id=product_id,
                reserve_event=reserve_event,
                receipt_held=receipt_held,
            )
            raise
        except Exception as exc:
            logger.exception("generation.submit_failed", model=request.model.value)
            await self._resolve_submit_failure(
                job,
                session=session,
                product_id=product_id,
                reserve_event=reserve_event,
                receipt_held=receipt_held,
            )
            # Any other exception (asyncpg errors, httpx errors carrying tunnel
            # hostnames, raw provider payload fragments, ...) MUST NOT reach
            # the client — only this fixed message may.
            raise GenerationError(INTERNAL_FAILURE_MESSAGE) from exc

        async def callback() -> None:
            await self._publish_generation_created(
                job_id=job.id,
                status=str(job.status),
                user_id=user_id,
                product_id=product_id,
                generation_type=request.generation_type.value,
                provider=request.model.provider.value,
                reserve_event=reserve_event,
            )

        if defer_commit:
            if post_commit_callbacks is None:
                raise RuntimeError("Deferred generation requires post-commit callbacks")
            post_commit_callbacks.append(callback)
        else:
            await callback()

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

    async def _publish_generation_created(
        self,
        *,
        job_id: UUID,
        status: str,
        user_id: UUID,
        product_id: str,
        generation_type: str,
        provider: str,
        reserve_event: BalanceEvent | None,
    ) -> None:
        """Emit success events only after the request-level transaction commits."""
        if self._event_bus is not None:
            try:
                await self._event_bus.publish_balance(reserve_event)

                from src.api.schemas.events import EventType, JobStatusPayload

                await self._event_bus.publish(
                    user_id=user_id,
                    event_type=EventType.JOB_STATUS_CHANGED,
                    payload=JobStatusPayload(
                        job_id=job_id,
                        status=status,
                        previous_status="none",
                        generation_type=generation_type,
                        provider=provider,
                    ),
                )
            except Exception:
                logger.exception("generation.post_commit_publish_failed", job_id=str(job_id))

        await self._ops_event_bus.publish(
            event_type=OpsEventType.GENERATION_CREATED,
            product_id=product_id,
            payload=GenerationCreatedOpsPayload(
                job_id=job_id,
                user_id=user_id,
                provider=provider,
                generation_type=generation_type,
            ),
        )

    async def _resolve_submit_failure(
        self,
        job: GenerationJob | None,
        *,
        session: AsyncSession,
        product_id: str,
        receipt_held: bool,
        reserve_event: BalanceEvent | None = None,
    ) -> None:
        """Resolve the DB transaction after a failed step following ``provider.submit()``.

        Three cases, so a failed request never leaves a live job behind (the
        client got a 500, so nothing may keep running for free):

        * ``job is None`` (or the job has no debit) ⇒ nothing durable yet. Roll
          back rather than commit an orphaned charge for a job that never ran.
        * ``receipt_held`` ⇒ the provider accepted an async task and this
          transaction holds its accepted-task receipt. Roll back: the job, its
          debit and the receipt delete all vanish, the receipt is restored, and
          the reconciler cancels or absorbs the task and never charges.
        * otherwise (Aisha, Grok image, or a receipt that failed to record) ⇒ the
          job row is the only record of the work. Fail it (``internal_error``) and
          refund in one transaction, then commit: it stays visible in history as a
          terminal job and is never polled. A job that is already terminal (a
          completed Grok image) cannot transition, so it is rolled back like the
          first case instead of leaving a charged job beside a 500. If the
          transition itself fails, roll back as well.

        ``reserve_event`` is published only for a settlement that committed (the
        transition service publishes it with the refund); nothing is published
        when this method rolls back.
        """
        if job is None or job.debit_transaction_id is None:
            await session.rollback()
            return
        if receipt_held:
            logger.warning(
                "generation.post_submit_failure_receipt_restored",
                job_id=str(job.id),
                provider=str(job.provider),
            )
            await session.rollback()
            return
        job_id = job.id
        transition_service = self._make_transition_service(session)
        try:
            _, did_transition = await transition_service.transition_to_failed(
                job_id,
                public_error_message=INTERNAL_FAILURE_MESSAGE,
                failure_code=INTERNAL_FAILURE_CODE,
                refund=True,
                product_id=product_id,
                reserve_event=reserve_event,
                commit=False,
            )
            if not did_transition:
                await session.rollback()
                return
            publication = transition_service.deferred_failure
            await session.commit()
        except Exception:
            logger.exception("generation.post_submit_failure_settle_failed", job_id=str(job_id))
            await session.rollback()
            return
        if publication is not None:
            try:
                await transition_service.publish_deferred_failure(publication)
            except Exception:
                logger.exception("generation.post_commit_publish_failed", job_id=str(job_id))

    @staticmethod
    def _validate_source_cardinality(
        request: UnifiedGenerationRequest,
        contract: SourceMediaConstraints | None,
    ) -> None:
        """Validate source-media presence and count without resolving assets."""
        refs = request.source_media
        if contract is None:
            if refs:
                raise SourceMediaValidationError(
                    f"generation_type '{request.generation_type.value}' does not accept source_media"
                )
            return
        count = len(refs or [])
        if not contract.min <= count <= contract.max:
            raise SourceMediaValidationError(
                f"generation_type '{request.generation_type.value}' requires between "
                f"{contract.min} and {contract.max} source_media items; got {count}"
            )

    @staticmethod
    def _validate_resolved_sources(
        resolved_source_media: list[ResolvedSourceMedia],
        contract: SourceMediaConstraints | None,
    ) -> None:
        """Validate resolved source media against the one mode's kind limits."""
        if contract is None:
            return

        for source in resolved_source_media:
            accepted = contract.kind_at(source.position)
            if source.media_kind not in accepted:
                allowed = ", ".join(sorted(kind.value for kind in accepted))
                role = (
                    f" (role '{contract.roles[source.position].value}')" if contract.roles else ""
                )
                raise SourceMediaValidationError(
                    f"source_media position {source.position}{role} has media kind "
                    f"'{source.media_kind.value}'; accepted: {allowed}"
                )

    @staticmethod
    def _source_media_count(request: UnifiedGenerationRequest) -> int:
        """Return source-media cardinality for input-aware pricing."""
        return len(request.source_media or [])
