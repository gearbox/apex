"""Real-time event schemas for SSE / future WebSocket delivery."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from uuid import UUID

import msgspec

from src.core.enums import NotificationLevel


class EventType(StrEnum):
    """Server-sent event types."""

    JOB_STATUS_CHANGED = "job.status_changed"
    JOB_PROGRESS = "job.progress"
    BALANCE_UPDATED = "balance.updated"
    SYSTEM_NOTIFICATION = "system.notification"
    GPU_SESSION_STATUS_CHANGED = "gpu_session.status_changed"
    GPU_SESSION_CREDIT_WARNING = "gpu_session.credit_warning"
    GPU_DEPLOYMENT_STATUS_CHANGED = "gpu_session.deployment_status_changed"


# --- Payloads ---


class JobStatusPayload(msgspec.Struct, kw_only=True):
    job_id: UUID
    status: str
    previous_status: str
    generation_type: str
    provider: str
    failure_code: str | None = None
    # Public-safe failure text only; never GenerationJob.error_message.
    error_message: str | None = None


class JobProgressPayload(msgspec.Struct, kw_only=True):
    job_id: UUID
    progress_pct: int  # 0 to 100
    generation_type: str


class BalanceUpdatedPayload(msgspec.Struct, kw_only=True):
    account_id: UUID
    balance: int
    delta: int
    transaction_type: str


class SystemNotificationPayload(msgspec.Struct, kw_only=True):
    level: NotificationLevel
    title: str
    message: str
    expires_at: datetime | None = None


class GpuSessionStatusPayload(msgspec.Struct, kw_only=True):
    """Emitted on every GPU session state transition."""

    session_id: UUID
    status: str
    previous_status: str
    model_type: str
    tunnel_hostname: str | None = None
    error_message: str | None = None
    reason: str | None = None


class GpuDeploymentStatusPayload(msgspec.Struct, kw_only=True):
    """Emitted on every P4 deployment state change: attach, provision progress,
    pending_restart, restart, activation, removal. The frontend's four-step story
    (downloading, waiting to restart, restarting, ready) is built entirely from
    ``status`` + ``pending_restart`` + ``routing_suspended`` + the current
    operation's phase/progress —
    the second step is the one that looks like a hang if the client can't see it."""

    deployment_id: UUID
    session_id: UUID
    model_type: str
    status: str
    pending_restart: bool
    routing_suspended: bool
    operation_id: UUID | None = None
    """The operation currently governing this deployment's progress: its
    provision_operation_id while deploying, else its restart_operation_id
    while pending_restart, else None."""
    operation_phase: str | None = None
    operation_progress: dict[str, object] | None = None
    error_message: str | None = None


class GpuSessionCreditWarningPayload(msgspec.Struct, kw_only=True):
    """Emitted when a GPU session balance falls below warning thresholds."""

    session_id: UUID
    level: NotificationLevel
    minutes_remaining: int
    terminate_at: datetime | None
    balance: int


# --- Envelope ---

# Wire format: payload is stored as pre-encoded JSON bytes (msgspec.Raw) to avoid
# union-decoding ambiguity. Callers encode the concrete payload before wrapping it.


class EventEnvelope(msgspec.Struct, kw_only=True):
    """Wire format for all events. Serialized to JSON over SSE."""

    event_type: EventType
    payload: msgspec.Raw
    timestamp: datetime
    event_id: str  # Monotonic ID for SSE Last-Event-ID reconnection


# --- SSE auth ticket ---


class SSETicketResponse(msgspec.Struct, kw_only=True):
    ticket: str


# --- Admin broadcast request ---


class SystemBroadcastRequest(msgspec.Struct, forbid_unknown_fields=True, kw_only=True):
    level: NotificationLevel
    title: str
    message: str
    expires_at: datetime | None = None
