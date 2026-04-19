"""Exceptions raised by GpuSessionService."""

from __future__ import annotations


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

    def __init__(self, message: str, *, current_status: str, operation: str) -> None:
        super().__init__(message)
        self.current_status = current_status
        self.operation = operation
