"""Unified generation endpoint — provider-agnostic content creation."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Annotated
from uuid import UUID

import msgspec
import structlog
from litestar import Controller, Response, post
from litestar.di import Provide
from litestar.params import Parameter
from litestar.status_codes import (
    HTTP_201_CREATED,
    HTTP_400_BAD_REQUEST,
    HTTP_403_FORBIDDEN,
    HTTP_409_CONFLICT,
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
    AgeVerificationRequiredError,
    GenerationError,
    GenerationService,
    ModelDisabledError,
    ModelNotAllowedError,
    ProviderUnavailableError,
)
from src.api.services.gpu_session.exceptions import NoActiveSessionError
from src.api.services.idempotency import IdempotencyReplayResult, IdempotencyService
from src.core.product import ProductConfig

logger = structlog.get_logger(__name__)


async def _mark_failed(
    idempotency_service: IdempotencyService, record_id: UUID, session: AsyncSession
) -> None:
    """Roll back any pending work, then record the idempotency key as failed.

    A service-layer error can leave the session poisoned (a later statement
    would raise PendingRollbackError) — rolling back first keeps fail()'s own
    write, and the commit below, from raising in turn and stranding the key
    in 'processing' until its TTL expires.
    """
    await session.rollback()
    await idempotency_service.fail(record_id, session=session)
    await session.commit()


class UnifiedGenerationController(Controller):
    """Single endpoint for all generation types and providers."""

    path = "/v1/generate"
    tags: Sequence[str] | None = ["Generation"]
    guards = [auth_guard]  # noqa: RUF012
    dependencies = {  # noqa: RUF012
        "current_user_id": Provide(get_current_user_id),
    }

    @post("/")
    async def generate(
        self,
        current_user_id: UUID,
        data: UnifiedGenerationRequest,
        session: AsyncSession,
        generation_service: GenerationService,
        product_config: ProductConfig,
        product_id: str,
        idempotency_service: IdempotencyService,
        idempotency_key_header: Annotated[
            str,
            Parameter(
                header="Idempotency-Key",
                max_length=64,
                description="Unique key for request deduplication (max 64 chars). Repeated requests with the same key return the cached response.",
            ),
        ],
    ) -> Response[JobCreatedResponse | ErrorEnvelope]:
        """Submit a generation request (idempotent via Idempotency-Key header).

        The backend resolves the correct provider from ``model``, validates
        the request, handles billing, and delegates to the provider.

        Returns a JobCreatedResponse with the job_id for polling via GET /v1/jobs/{job_id}.
        """
        request_hash = IdempotencyService.hash_request(msgspec.json.encode(data))

        check_result = await idempotency_service.check(
            user_id=current_user_id,
            product_id=product_id,
            idempotency_key=idempotency_key_header,
            operation="generation",
            request_hash=request_hash,
            session=session,
        )
        if isinstance(check_result, IdempotencyReplayResult):
            return Response(
                content=check_result.body,  # type: ignore[arg-type]
                status_code=check_result.status_code,
            )

        record_id = check_result
        try:
            result = await generation_service.generate(
                data,
                user_id=current_user_id,
                session=session,
                product_config=product_config,
            )

            response_body = msgspec.to_builtins(result)
            await idempotency_service.complete(
                record_id,
                resource_id=result.job_id,
                response_status_code=HTTP_201_CREATED,
                response_body=response_body,
                session=session,
            )
            await session.commit()
            return Response(content=result, status_code=HTTP_201_CREATED)

        except NoActiveSessionError as exc:
            await _mark_failed(idempotency_service, record_id, session)
            logger.info("generation.no_active_gpu_session", error=str(exc))
            return Response(
                content=ErrorEnvelope(
                    error="no_active_gpu_session",
                    message=str(exc),
                    status_code=HTTP_409_CONFLICT,
                ),
                status_code=HTTP_409_CONFLICT,
            )

        except RateLimitExceededError as exc:
            await _mark_failed(idempotency_service, record_id, session)
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
            await _mark_failed(idempotency_service, record_id, session)
            logger.warning("generation.provider_unavailable", error=str(exc))
            return Response(
                content=ErrorEnvelope(
                    error="service_unavailable",
                    message=str(exc),
                    status_code=HTTP_503_SERVICE_UNAVAILABLE,
                ),
                status_code=HTTP_503_SERVICE_UNAVAILABLE,
            )

        except AgeVerificationRequiredError as exc:
            await _mark_failed(idempotency_service, record_id, session)
            logger.info("generation.age_verification_required", error=str(exc))
            return Response(
                content=ErrorEnvelope(
                    error="age_verification_required",
                    message=str(exc),
                    status_code=HTTP_403_FORBIDDEN,
                ),
                status_code=HTTP_403_FORBIDDEN,
            )

        except ModelNotAllowedError as exc:
            await _mark_failed(idempotency_service, record_id, session)
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
            await _mark_failed(idempotency_service, record_id, session)
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
            await _mark_failed(idempotency_service, record_id, session)
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
            await _mark_failed(idempotency_service, record_id, session)
            logger.exception("generation.failed", error=str(exc))
            return Response(
                content=ErrorEnvelope(
                    error="generation_failed",
                    message=str(exc),
                    status_code=HTTP_400_BAD_REQUEST,
                ),
                status_code=HTTP_400_BAD_REQUEST,
            )

        except Exception:
            await _mark_failed(idempotency_service, record_id, session)
            raise
