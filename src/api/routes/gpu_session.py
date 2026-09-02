"""GPU session HTTP endpoints."""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Any
from uuid import UUID

import structlog
from litestar import Controller, Response, get, post
from litestar.di import Provide
from litestar.exceptions import NotFoundException, PermissionDeniedException
from litestar.params import Body, Parameter
from litestar.status_codes import (
    HTTP_200_OK,
    HTTP_201_CREATED,
    HTTP_402_PAYMENT_REQUIRED,
    HTTP_404_NOT_FOUND,
    HTTP_409_CONFLICT,
    HTTP_503_SERVICE_UNAVAILABLE,
)
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.dependencies.auth import get_current_user_id
from src.api.schemas.errors import ErrorEnvelope
from src.api.schemas.gpu_session import (
    GpuSessionResponse,
    ListSessionsResponse,
    StartSessionRequest,
    StopConfirmationResponse,
    StopSessionRequest,
)
from src.api.security import auth_guard
from src.api.services.billing import BillingService
from src.api.services.billing_errors import InsufficientBalanceError
from src.api.services.gpu_session.exceptions import (
    GpuSessionError,
    InvalidSessionStateError,
    SessionAlreadyExistsError,
    SessionHasInFlightJobsError,
)
from src.api.services.gpu_session.schemas import StopConfirmation
from src.api.services.gpu_session.service import (
    GpuSessionService,
    billable_minutes_for_active_seconds,
)
from src.api.services.vastai.exceptions import NoCapacityError, VastAIError
from src.core.config import Settings
from src.core.enums import GpuSessionStatus, UserRole
from src.db.repositories.gpu_session_operation import GpuSessionOperationRepository
from src.db.repositories.job import JobRepository

if TYPE_CHECKING:
    from collections.abc import Sequence

    from src.db.models.gpu_session import GpuSession
    from src.db.models.gpu_session_operation import GpuSessionOperation

logger = structlog.get_logger(__name__)


async def _bootstrap_operations(
    session: AsyncSession, sessions: Sequence[GpuSession]
) -> dict[UUID, GpuSessionOperation]:
    """Fetch all bootstrap response projections in one query, avoiding list N+1s."""
    operation_ids = {
        s.bootstrap_operation_id for s in sessions if s.bootstrap_operation_id is not None
    }
    return await GpuSessionOperationRepository(session).get_many(operation_ids)


async def _bootstrap_operation(
    session: AsyncSession, gpu_session: GpuSession
) -> GpuSessionOperation | None:
    """Fetch the current bootstrap operation projection for one response."""
    if gpu_session.bootstrap_operation_id is None:
        return None
    return await GpuSessionOperationRepository(session).get(gpu_session.bootstrap_operation_id)


class GpuSessionController(Controller):
    """GPU session lifecycle endpoints."""

    path = "/v1/sessions"
    tags: Sequence[str] | None = ("GPU Sessions",)
    guards = [auth_guard]  # noqa: RUF012
    dependencies = {  # noqa: RUF012
        "current_user_id": Provide(get_current_user_id),
    }

    @post("/", status_code=HTTP_201_CREATED)
    async def start(
        self,
        current_user_id: UUID,
        data: Annotated[StartSessionRequest, Body()],
        session: AsyncSession,
        gpu_session_service: GpuSessionService,
        billing_service: BillingService,
        product_id: str,
    ) -> Response[GpuSessionResponse | ErrorEnvelope]:
        """Start a new GPU session for a model."""
        bundle_override = data.bundle_override
        if bundle_override is not None:
            from src.db.repositories import UserRepository

            repo = UserRepository(session)
            user = await repo.get_active_user(current_user_id)
            role = user.role if user is not None else None
            role_value = role.value if isinstance(role, UserRole) else str(role)
            is_admin = role_value in (UserRole.ADMIN.value, UserRole.SUPERADMIN.value)
            if not is_admin:
                raise PermissionDeniedException("bundle_override is admin-only")

        account = await billing_service.resolve_account_for_user(current_user_id, session=session)

        try:
            gpu_session = await gpu_session_service.start_session(
                user_id=current_user_id,
                product_id=product_id,
                model_type=data.model,
                bundle_override=bundle_override,
                account_id=account.id,
            )
        except SessionAlreadyExistsError as exc:
            return _error(HTTP_409_CONFLICT, "session_already_exists", str(exc))
        except InsufficientBalanceError as exc:
            return _error(HTTP_402_PAYMENT_REQUIRED, "insufficient_balance", str(exc))
        except NoCapacityError as exc:
            return _error(HTTP_503_SERVICE_UNAVAILABLE, "no_gpu_capacity", str(exc))
        except VastAIError:
            logger.exception("gpu_session.start.vastai_error", user_id=str(current_user_id))
            return _error(
                HTTP_503_SERVICE_UNAVAILABLE, "provisioning_failed", "GPU provisioning failed"
            )

        operation = await _bootstrap_operation(session, gpu_session)
        return Response(
            content=GpuSessionResponse.from_model(gpu_session, bootstrap_operation=operation),
            status_code=HTTP_201_CREATED,
        )

    @get("/")
    async def list_sessions(
        self,
        current_user_id: UUID,
        gpu_session_service: GpuSessionService,
        product_id: str,
        session: AsyncSession | None = None,
        include_terminal: Annotated[bool, Parameter(query="include_terminal")] = False,
    ) -> ListSessionsResponse:
        """List GPU sessions for the current user."""
        sessions = await gpu_session_service.list_user_sessions(
            user_id=current_user_id,
            product_id=product_id,
            include_terminal=include_terminal,
        )
        operations = await _bootstrap_operations(session, sessions) if session is not None else {}
        return ListSessionsResponse(
            sessions=[
                GpuSessionResponse.from_model(
                    gpu_session,
                    bootstrap_operation=(
                        operations.get(gpu_session.bootstrap_operation_id)
                        if gpu_session.bootstrap_operation_id is not None
                        else None
                    ),
                )
                for gpu_session in sessions
            ]
        )

    @get("/{session_id:uuid}")
    async def get_session(
        self,
        current_user_id: UUID,
        session_id: UUID,
        gpu_session_service: GpuSessionService,
        product_id: str,
        session: AsyncSession,
    ) -> GpuSessionResponse:
        """Get a specific GPU session."""
        session_row = await gpu_session_service.get_session(
            session_id=session_id,
            user_id=current_user_id,
            product_id=product_id,
        )
        if session_row is None:
            raise NotFoundException(detail=f"Session {session_id} not found")

        in_flight_count = 0
        if session_row.status == GpuSessionStatus.active:
            # Only meaningful for active sessions; non-active sessions can't
            # have in-flight jobs by construction (sweeps already ran).
            job_repo = JobRepository(session)
            in_flight_count = await job_repo.count_in_flight_for_session(session_id)

        operation = await _bootstrap_operation(session, session_row)
        return GpuSessionResponse.from_model(
            session_row,
            bootstrap_operation=operation,
            in_flight_job_count=in_flight_count,
        )

    @post("/{session_id:uuid}/pause")
    async def pause(
        self,
        current_user_id: UUID,
        session_id: UUID,
        gpu_session_service: GpuSessionService,
        product_id: str,
        session: AsyncSession | None = None,
    ) -> Response[GpuSessionResponse | ErrorEnvelope]:
        """Pause an active GPU session."""
        try:
            session_row = await gpu_session_service.pause_session(
                session_id=session_id,
                user_id=current_user_id,
                product_id=product_id,
            )
        except SessionHasInFlightJobsError as exc:
            return Response(
                content=ErrorEnvelope(
                    error="jobs_in_flight",
                    message=str(exc),
                    status_code=HTTP_409_CONFLICT,
                    detail={"in_flight_count": exc.in_flight_count},
                ),
                status_code=HTTP_409_CONFLICT,
            )
        except InvalidSessionStateError as exc:
            return _error(HTTP_409_CONFLICT, "invalid_state", str(exc))
        except GpuSessionError as exc:
            return _error(HTTP_404_NOT_FOUND, "session_not_found", str(exc))

        operation = (
            await _bootstrap_operation(session, session_row) if session is not None else None
        )
        return Response(
            content=GpuSessionResponse.from_model(session_row, bootstrap_operation=operation),
            status_code=HTTP_200_OK,
        )

    @post("/{session_id:uuid}/resume")
    async def resume(
        self,
        current_user_id: UUID,
        session_id: UUID,
        gpu_session_service: GpuSessionService,
        product_id: str,
        session: AsyncSession | None = None,
    ) -> Response[GpuSessionResponse | ErrorEnvelope]:
        """Resume a paused GPU session."""
        try:
            session_row = await gpu_session_service.resume_session(
                session_id=session_id,
                user_id=current_user_id,
                product_id=product_id,
            )
        except InvalidSessionStateError as exc:
            return _error(HTTP_409_CONFLICT, "invalid_state", str(exc))
        except GpuSessionError as exc:
            return _error(HTTP_404_NOT_FOUND, "session_not_found", str(exc))

        operation = (
            await _bootstrap_operation(session, session_row) if session is not None else None
        )
        return Response(
            content=GpuSessionResponse.from_model(session_row, bootstrap_operation=operation),
            status_code=HTTP_200_OK,
        )

    @post("/{session_id:uuid}/stop")
    async def stop(
        self,
        current_user_id: UUID,
        session_id: UUID,
        data: Annotated[StopSessionRequest, Body()],
        gpu_session_service: GpuSessionService,
        product_id: str,
        settings: Settings,
        session: AsyncSession | None = None,
    ) -> Response[GpuSessionResponse | StopConfirmationResponse | ErrorEnvelope]:
        """Two-call stop: first call returns cost confirmation, second call executes."""
        try:
            result = await gpu_session_service.stop_session(
                session_id=session_id,
                user_id=current_user_id,
                product_id=product_id,
                confirmed=data.confirmed,
            )
        except InvalidSessionStateError as exc:
            return _error(HTTP_409_CONFLICT, "invalid_state", str(exc))
        except GpuSessionError as exc:
            return _error(HTTP_404_NOT_FOUND, "session_not_found", str(exc))

        if isinstance(result, StopConfirmation):
            minutes = billable_minutes_for_active_seconds(result.active_duration_seconds)
            estimated_tokens = minutes * settings.gpu_session_tokens_per_minute
            return Response(
                content=StopConfirmationResponse(
                    session_id=result.session_id,
                    model_type=result.model_type,
                    vastai_gpu_name=result.vastai_gpu_name,
                    vastai_cost_per_hour_micros=result.vastai_cost_per_hour_micros,
                    active_duration_seconds=result.active_duration_seconds,
                    paused_duration_seconds=result.paused_duration_seconds,
                    estimated_final_tokens=estimated_tokens,
                    message=result.message,
                ),
                status_code=HTTP_200_OK,
            )
        operation = await _bootstrap_operation(session, result) if session is not None else None
        return Response(
            content=GpuSessionResponse.from_model(result, bootstrap_operation=operation),
            status_code=HTTP_200_OK,
        )


def _error(status_code: int, error: str, message: str) -> Response[Any]:
    return Response(
        content=ErrorEnvelope(error=error, message=message, status_code=status_code),
        status_code=status_code,
    )
