"""Exceptions raised by GpuSessionService."""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.core.enums import GpuSessionStatus

if TYPE_CHECKING:
    from uuid import UUID


class GpuSessionError(Exception):
    """Base class for GPU session service errors."""


class SessionAlreadyExistsError(GpuSessionError):
    """A live deployment already exists for this (user, product, model_type).

    Since P2, the uniqueness constraint is enforced at the DB level by the
    partial unique index `ix_gpu_session_deployments_live_user_model` on
    gpu_session_deployments (moved off gpu_sessions — see
    GpuSessionDeployment's module docstring). This exception wraps both the
    pre-check path and the race-condition IntegrityError path.
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


class DeploymentAlreadyLiveError(GpuSessionError):
    """Attach (D36): a live deployment for this model_type already exists.

    Wraps both the pre-check path and the race-condition IntegrityError path,
    mirroring SessionAlreadyExistsError's role for the session-level slot.
    """


class DeploymentNotLiveError(GpuSessionError):
    """Remove: no live deployment exists for the requested model_type."""


class LastDeploymentRequiresForceError(GpuSessionError):
    """Remove: this is the session's last live deployment and force=true was not passed.

    The session keeps running and billing with nothing deployed on it — legal,
    but intentional enough that the API requires the caller to say so explicitly.
    """


class DeploymentHasInFlightJobsError(GpuSessionError):
    """Remove (D37): a job of the model_type being removed is in flight.

    Jobs on other models sharing the same session do not block this removal.
    """

    def __init__(self, *, deployment_id: UUID, in_flight_count: int) -> None:
        super().__init__(
            f"Cannot remove deployment {deployment_id}: {in_flight_count} job(s) in flight "
            "for this model."
        )
        self.in_flight_count = in_flight_count


class RetainBundlesUnresolvableError(GpuSessionError):
    """Remove (D11): a sibling live deployment has no resolvable bundle spec.

    Fail loud rather than ship a short retain list — that would delete weights
    another resident bundle is using.
    """
