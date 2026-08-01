"""Unified generation endpoint — provider-agnostic content creation."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Annotated
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
    HTTP_422_UNPROCESSABLE_ENTITY,
    HTTP_429_TOO_MANY_REQUESTS,
    HTTP_502_BAD_GATEWAY,
    HTTP_503_SERVICE_UNAVAILABLE,
)
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.dependencies.auth import get_current_user_id
from src.api.schemas.errors import ErrorEnvelope
from src.api.schemas.jobs import JobCreatedResponse
from src.api.schemas.unified_generation import UnifiedGenerationRequest
from src.api.security import auth_guard
from src.api.services.generation.provider_failures import (
    ProviderFailureKind,
    ProviderModerationRejectedError,
    ProviderSubmissionFailedError,
)
from src.api.services.generation.rate_limiter import RateLimitExceededError
from src.api.services.generation.service import (
    AgeVerificationRequiredError,
    FeatureNotSupportedError,
    GenerationError,
    GenerationService,
    ModelDisabledError,
    ModelNotAllowedError,
    ProviderUnavailableError,
)
from src.api.services.gpu_session.exceptions import NoActiveSessionError
from src.api.services.idempotency import IdempotencyReplayResult, IdempotencyService
from src.core.product import ProductConfig

if TYPE_CHECKING:
    from collections.abc import Sequence

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


async def _commit_completed_or_replay(
    idempotency_service: IdempotencyService,
    record_id: UUID,
    session: AsyncSession,
) -> IdempotencyReplayResult | None:
    """Commit a completed outcome, recovering a lost acknowledgement safely.

    The business row and idempotency result are written in one transaction.
    If ``commit`` raises, PostgreSQL may nevertheless have committed it.  A
    rollback followed by an authoritative read distinguishes that case before
    any code can release the key for retry.
    """
    try:
        await session.commit()
    except Exception:
        await session.rollback()
        # Do not trust the connection whose commit acknowledgement was lost.
        # The next ORM read checks out a fresh connection from the pool.
        await session.invalidate()
        replay = await idempotency_service.replay_completed(record_id, session=session)
        if replay is not None:
            logger.warning("generation.idempotency_commit_ack_lost", record_id=str(record_id))
            return replay
        # A still-processing row means the atomic business transaction did
        # not commit. Conditional fail is safe here and cannot downgrade a
        # completed row if a concurrent acknowledgement arrives.
        await idempotency_service.fail(record_id, session=session)
        await session.commit()
        raise
    else:
        return None


async def _run_post_commit_callbacks(
    callbacks: list[Callable[[], Awaitable[None]]],
) -> None:
    """Callbacks are notification side effects, never transaction outcomes."""
    for callback in callbacks:
        try:
            await callback()
        except Exception:
            logger.exception("generation.post_commit_callback_failed")


class UnifiedGenerationController(Controller):
    """Single endpoint for all generation types and providers."""

    path = "/v1/generate"
    tags: Sequence[str] | None = ("Generation",)
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
        post_commit_callbacks: list[Callable[[], Awaitable[None]]] = []
        try:
            result = await generation_service.generate(
                data,
                user_id=current_user_id,
                session=session,
                product_config=product_config,
                defer_commit=True,
                post_commit_callbacks=post_commit_callbacks,
            )

            response_body = msgspec.to_builtins(result)
            await idempotency_service.complete(
                record_id,
                resource_id=result.job_id,
                response_status_code=HTTP_201_CREATED,
                response_body=response_body,
                session=session,
            )
            replay = await _commit_completed_or_replay(idempotency_service, record_id, session)
            if replay is not None:
                return Response(content=replay.body, status_code=replay.status_code)  # type: ignore[arg-type]
            await _run_post_commit_callbacks(post_commit_callbacks)
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

        except FeatureNotSupportedError as exc:
            await _mark_failed(idempotency_service, record_id, session)
            logger.info("generation.not_implemented", error=str(exc))
            return Response(
                content=ErrorEnvelope(
                    error="not_implemented",
                    message=str(exc),
                    status_code=HTTP_400_BAD_REQUEST,
                ),
                status_code=HTTP_400_BAD_REQUEST,
            )

        except ProviderModerationRejectedError as exc:
            error = ErrorEnvelope(
                error=exc.public_code,
                message=exc.public_message,
                status_code=HTTP_422_UNPROCESSABLE_ENTITY,
            )
            # This is a completed, billable request, despite the 422 response.
            # Cache the safe response to prevent an idempotent retry from
            # submitting and charging the same generation again.
            await idempotency_service.complete(
                record_id,
                resource_id=exc.job_id,
                response_status_code=HTTP_422_UNPROCESSABLE_ENTITY,
                response_body=msgspec.to_builtins(error),
                session=session,
            )
            replay = await _commit_completed_or_replay(idempotency_service, record_id, session)
            if replay is not None:
                return Response(content=replay.body, status_code=replay.status_code)  # type: ignore[arg-type]
            await _run_post_commit_callbacks(post_commit_callbacks)
            logger.warning(
                "generation.failed",
                provider=exc.failure.provider.value,
                job_id=str(exc.job_id),
                failure_kind=exc.failure.kind.value,
                billable=exc.failure.billable,
            )
            return Response(content=error, status_code=HTTP_422_UNPROCESSABLE_ENTITY)

        except ProviderSubmissionFailedError as exc:
            # The failed state, debit/refund result, idempotency resource and
            # cached response share this one transaction.  Replaying the key
            # therefore cannot run the provider or create a second debit.
            status_code = {
                ProviderFailureKind.INVALID_REQUEST: HTTP_400_BAD_REQUEST,
                ProviderFailureKind.RATE_LIMITED: HTTP_429_TOO_MANY_REQUESTS,
                ProviderFailureKind.MALFORMED_RESPONSE: HTTP_502_BAD_GATEWAY,
            }.get(exc.failure.kind, HTTP_503_SERVICE_UNAVAILABLE)
            error = ErrorEnvelope(
                error=exc.public_code,
                message=exc.public_message,
                status_code=status_code,
            )
            await idempotency_service.complete(
                record_id,
                resource_id=exc.job_id,
                response_status_code=status_code,
                response_body=msgspec.to_builtins(error),
                session=session,
            )
            replay = await _commit_completed_or_replay(idempotency_service, record_id, session)
            if replay is not None:
                return Response(content=replay.body, status_code=replay.status_code)  # type: ignore[arg-type]
            await _run_post_commit_callbacks(post_commit_callbacks)
            logger.warning(
                "generation.provider_submission_failed",
                provider=exc.failure.provider.value,
                job_id=str(exc.job_id),
                failure_kind=exc.failure.kind.value,
                status_code=status_code,
            )
            return Response(
                content=error,
                status_code=status_code,
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
