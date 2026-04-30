"""Exceptions raised by GpuSessionService."""

from __future__ import annotations

from uuid import UUID

from src.core.enums import GpuSessionStatus


class GpuSessionError(Exception):
    """Base class for GPU session service errors."""


class SessionAlreadyExistsError(GpuSessionError):
    """A non-terminal session already exists for this (user, product, model_type).

    The uniqueness constraint is enforced at the DB level by the partial
    unique index `ix_gpu_sessions_active_user_model`. This exception wraps
    both the pre-check path and the race-condition IntegrityError path.
    """


class NoActiveSessionError(GpuSessionError):
    """No active GPU session found for the requested (user, product, model_type).

    Raised by generation routing when the user tries to submit a job for an
    Aisha model without having started a session first.
    """


class InvalidSessionStateError(GpuSessionError):
    """The requested operation is not valid for the session's current state.

    Examples: calling pause() on an already-paused session, resume() on an
    active session, stop() on an already-stopped session.
    """

    def __init__(
        self,
        message: str,
        *,
        current_status: GpuSessionStatus | str,
        operation: str,
    ) -> None:
        super().__init__(message)
        # Coerce to plain str so downstream consumers (logging, HTTP responses)
        # get a consistent type regardless of whether callers passed the enum
        # or the raw DB column value.
        self.current_status: str = str(current_status)
        self.operation = operation


class SessionHasInFlightJobsError(InvalidSessionStateError):
    """Pause was requested but the session has in-flight Aisha jobs.

    The frontend should disable the Pause button when in_flight_job_count > 0;
    this error is the backend safety net for a stale UI state.
    """

    def __init__(
        self,
        *,
        session_id: UUID,
        in_flight_count: int,
    ) -> None:
        super().__init__(
            f"Cannot pause session {session_id}: {in_flight_count} job(s) in flight.",
            current_status=GpuSessionStatus.active,
            operation="pause",
        )
        self.in_flight_count = in_flight_count
