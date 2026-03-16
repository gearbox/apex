"""Unified generation endpoint — provider-agnostic content creation."""

from __future__ import annotations

from collections.abc import Sequence
from uuid import UUID

import structlog
from litestar import Controller, Response, post
from litestar.di import Provide
from litestar.status_codes import (
    HTTP_201_CREATED,
    HTTP_400_BAD_REQUEST,
    HTTP_403_FORBIDDEN,
    HTTP_429_TOO_MANY_REQUESTS,
    HTTP_503_SERVICE_UNAVAILABLE,
)
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.dependencies.auth import get_current_user_id
from src.api.schemas.errors import ErrorEnvelope
from src.api.schemas.jobs import JobCreatedResponse
from src.api.schemas.unified_generation import UnifiedGenerationRequest
from src.api.security import auth_guard
from src.api.services.generation.rate_limiter import RateLimitExceededError
from src.api.services.generation.service import (
    GenerationError,
    GenerationService,
    ModelDisabledError,
    ModelNotAllowedError,
    ProviderUnavailableError,
)
from src.core.product import ProductConfig

logger = structlog.get_logger(__name__)


class UnifiedGenerationController(Controller):
    """Single endpoint for all generation types and providers."""

    path = "/v1/generate"
    tags: Sequence[str] | None = ["Generation"]
    guards = [auth_guard]
    dependencies = {"current_user_id": Provide(get_current_user_id)}

    @post("/")
    async def generate(
        self,
        current_user_id: UUID,
        data: UnifiedGenerationRequest,
        session: AsyncSession,
        generation_service: GenerationService,
        product_config: ProductConfig,
    ) -> Response[JobCreatedResponse | ErrorEnvelope]:
        """Submit a generation request.

        The backend resolves the correct provider from ``model``, validates
        the request, handles billing, and delegates to the provider.

        Returns a JobCreatedResponse with the job_id for polling via GET /v1/jobs/{job_id}.
        """
        try:
            result = await generation_service.generate(
                data,
                user_id=current_user_id,
                session=session,
                product_config=product_config,
            )
            return Response(content=result, status_code=HTTP_201_CREATED)

        except RateLimitExceededError as exc:
            logger.warning(
                "generation.rate_limited",
                model=data.model.value,
                retry_after=exc.retry_after,
            )
            return Response(
                content=ErrorEnvelope(
                    error="rate_limited",
                    message=str(exc),
                    status_code=HTTP_429_TOO_MANY_REQUESTS,
                    detail={"retry_after": exc.retry_after},
                ),
                status_code=HTTP_429_TOO_MANY_REQUESTS,
                headers={"Retry-After": str(exc.retry_after)},
            )

        except ProviderUnavailableError as exc:
            logger.warning("generation.provider_unavailable", error=str(exc))
            return Response(
                content=ErrorEnvelope(
                    error="service_unavailable",
                    message=str(exc),
                    status_code=HTTP_503_SERVICE_UNAVAILABLE,
                ),
                status_code=HTTP_503_SERVICE_UNAVAILABLE,
            )

        except ModelNotAllowedError as exc:
            logger.info("generation.model_not_allowed", error=str(exc))
            return Response(
                content=ErrorEnvelope(
                    error="model_not_allowed",
                    message=str(exc),
                    status_code=HTTP_403_FORBIDDEN,
                ),
                status_code=HTTP_403_FORBIDDEN,
            )

        except ModelDisabledError as exc:
            logger.info("generation.model_disabled", error=str(exc))
            return Response(
                content=ErrorEnvelope(
                    error="model_disabled",
                    message=str(exc),
                    status_code=HTTP_400_BAD_REQUEST,
                ),
                status_code=HTTP_400_BAD_REQUEST,
            )

        except ValueError as exc:
            logger.info("generation.validation_failed", error=str(exc))
            return Response(
                content=ErrorEnvelope(
                    error="validation_error",
                    message=str(exc),
                    status_code=HTTP_400_BAD_REQUEST,
                ),
                status_code=HTTP_400_BAD_REQUEST,
            )

        except GenerationError as exc:
            logger.error("generation.failed", error=str(exc))
            return Response(
                content=ErrorEnvelope(
                    error="generation_failed",
                    message=str(exc),
                    status_code=HTTP_400_BAD_REQUEST,
                ),
                status_code=HTTP_400_BAD_REQUEST,
            )
