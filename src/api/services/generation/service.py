"""Unified generation orchestrator — routes requests to providers."""

from __future__ import annotations

from uuid import UUID

import structlog
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.schemas.jobs import JobCreatedResponse
from src.api.schemas.unified_generation import UnifiedGenerationRequest
from src.api.services.billing import BillingService
from src.api.services.generation.base import GenerationProvider
from src.api.services.pricing import PricingService
from src.core.enums import JobStatus, Provider
from src.db.repositories.generation_model import GenerationModelRepository

logger = structlog.get_logger(__name__)


class GenerationError(Exception):
    """Base error for generation failures."""


class ProviderUnavailableError(GenerationError):
    """Raised when the resolved provider is not configured."""


class ModelDisabledError(GenerationError):
    """Raised when the requested model is disabled via generation_models."""


class GenerationService:
    """Orchestrates generation requests across providers.

    Responsibilities:
    1. Validate model-generation_type compatibility (via ModelType enum)
    2. Validate required inputs (image_id, video_url)
    3. Validate n <= model.max_concurrent_outputs
    4. Check model is enabled in generation_models table
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
    ) -> None:
        self._providers = providers
        self._billing = billing_service
        self._pricing = pricing_service

    async def generate(
        self,
        request: UnifiedGenerationRequest,
        *,
        user_id: UUID,
        session: AsyncSession,
    ) -> JobCreatedResponse:
        """Execute the full generation pipeline."""
        # 1. Model-generation_type compatibility (declarative, enum-driven)
        if not request.model.supports_generation_type(request.generation_type):
            raise ValueError(
                f"Model '{request.model.value}' does not support generation type '{request.generation_type.value}'"
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

        # 5. Resolve provider
        provider_key = request.model.provider
        provider = self._providers.get(provider_key)
        if provider is None:
            raise ProviderUnavailableError(f"Provider '{provider_key.value}' is not configured")

        # 6. Provider-specific validation
        provider.validate(request)

        # 7. Billing pre-flight
        account = await self._billing.resolve_account_for_user(user_id, session=session)
        token_cost = await self._pricing.get_price(
            provider_key.value,
            request.generation_type.value,
            request.model.value,
            session=session,
        )
        await self._billing.assert_sufficient_balance(account.id, token_cost, session=session)

        # 8. Submit to provider (with billing saga)
        job = None
        try:
            job = await provider.submit(
                request,
                user_id=user_id,
                session=session,
                billing_service=self._billing,
                account_id=account.id,
                token_cost=token_cost,
            )
            await session.commit()

        except Exception as exc:
            # 9. Refund on failure
            if job is not None:
                try:
                    await self._billing.refund(
                        job.id,
                        description=f"Generation failed: {str(exc)[:200]}",
                        session=session,
                    )
                    await session.commit()
                except Exception as refund_err:
                    logger.exception(
                        "generation.refund_failed",
                        job_id=str(job.id) if job else "unknown",
                        error=str(refund_err),
                    )
            raise GenerationError(str(exc)) from exc

        # 10. Build response
        balance = await self._billing.get_balance(account.id, session=session)
        return JobCreatedResponse(
            job_id=job.id,
            status=JobStatus(job.status),
            name=job.name,
            model=request.model.value,
            generation_type=request.generation_type,
            created_at=job.created_at,
            tokens_charged=token_cost,
            balance_remaining=balance,
        )

    @staticmethod
    def _validate_inputs(request: UnifiedGenerationRequest) -> None:
        """Cross-cutting validation: generation_type vs required inputs."""
        if request.generation_type.requires_image_input and request.input_image_id is None:
            raise ValueError(
                f"generation_type '{request.generation_type.value}' requires input_image_id"
            )
        if request.generation_type.requires_video_input and request.input_video_url is None:
            raise ValueError(
                f"generation_type '{request.generation_type.value}' requires input_video_url"
            )
