"""Real-time event schemas for SSE / future WebSocket delivery."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from uuid import UUID

import msgspec


class EventType(StrEnum):
    """Server-sent event types."""

    JOB_STATUS_CHANGED = "job.status_changed"
    JOB_PROGRESS = "job.progress"
    BALANCE_UPDATED = "balance.updated"
    SYSTEM_NOTIFICATION = "system.notification"
    GPU_SESSION_STATUS_CHANGED = "gpu_session.status_changed"


# --- Payloads ---


class JobStatusPayload(msgspec.Struct, kw_only=True):
    job_id: UUID
    status: str
    previous_status: str
    generation_type: str
    provider: str


class JobProgressPayload(msgspec.Struct, kw_only=True):
    job_id: UUID
    progress_pct: int  # 0–100
    generation_type: str


class BalanceUpdatedPayload(msgspec.Struct, kw_only=True):
    account_id: UUID
    balance: int
    delta: int
    transaction_type: str


class SystemNotificationPayload(msgspec.Struct, kw_only=True):
    level: str  # "info" | "warning" | "critical"
    title: str
    message: str
    expires_at: datetime | None = None


class GpuSessionStatusPayload(msgspec.Struct, kw_only=True):
    """Emitted on every GPU session state transition."""

    session_id: UUID
    status: str
    previous_status: str
    model_type: str
    bundle_name: str
    tunnel_hostname: str | None = None
    error_message: str | None = None


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
    level: str  # "info" | "warning" | "critical"
    title: str
    message: str
    expires_at: datetime | None = None
