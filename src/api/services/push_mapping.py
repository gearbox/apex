"""Pure event -> push-notification mapping.

No I/O, no Redis, no webpush — feed it a decoded ``EventEnvelope`` and get
back a ``PushNotificationPayload`` (send it) or ``None`` (skip). Unit-testable
in complete isolation from PushDispatcher/PushService.

Implements the locked v1 mapping:
  - job.status_changed          -> only terminal states (completed/failed)
  - gpu_session.credit_warning  -> all levels
  - system.notification         -> all levels (broadcast)
  - balance.updated             -> only delta > 0 AND a payment/top-up or
                                    admin credit (CREDIT / ADMIN_ADJUSTMENT).
                                    Per-generation debits and refunds never push.
  - everything else (job.progress, gpu_session.status_changed) -> ignored
"""

from __future__ import annotations

import msgspec

from src.api.schemas.events import (
    BalanceUpdatedPayload,
    EventEnvelope,
    EventType,
    GpuSessionCreditWarningPayload,
    JobStatusPayload,
    SystemNotificationPayload,
)
from src.api.schemas.push import PushNotificationPayload
from src.core.enums import JobStatus, TransactionType

_TERMINAL_JOB_STATUSES = frozenset({JobStatus.COMPLETED.value, JobStatus.FAILED.value})

# "payment/top-up or admin credit" per the locked mapping — REFUND is a
# positive delta too but is not a top-up, so it is deliberately excluded.
_POSITIVE_BALANCE_TRANSACTION_TYPES = frozenset(
    {TransactionType.CREDIT.value, TransactionType.ADMIN_ADJUSTMENT.value}
)


def map_event_to_notification(envelope: EventEnvelope) -> PushNotificationPayload | None:
    """Map a decoded EventEnvelope to a push notification, or None to skip."""
    if envelope.event_type is EventType.JOB_STATUS_CHANGED:
        return _map_job_status_changed(envelope.payload)
    if envelope.event_type is EventType.GPU_SESSION_CREDIT_WARNING:
        return _map_credit_warning(envelope.payload)
    if envelope.event_type is EventType.SYSTEM_NOTIFICATION:
        return _map_system_notification(envelope.payload)
    if envelope.event_type is EventType.BALANCE_UPDATED:
        return _map_balance_updated(envelope.payload)
    return None


def _map_job_status_changed(raw: msgspec.Raw) -> PushNotificationPayload | None:
    payload = msgspec.json.decode(raw, type=JobStatusPayload)
    if payload.status not in _TERMINAL_JOB_STATUSES:
        return None
    completed = payload.status == JobStatus.COMPLETED.value
    return PushNotificationPayload(
        title="Generation complete" if completed else "Generation failed",
        body=f"Your {payload.generation_type} generation has {payload.status}.",
        url=f"/gallery/{payload.job_id}",
        tag=f"job-{payload.job_id}",
        category="job",
        level="info" if completed else "warning",
    )


def _map_credit_warning(raw: msgspec.Raw) -> PushNotificationPayload:
    payload = msgspec.json.decode(raw, type=GpuSessionCreditWarningPayload)
    return PushNotificationPayload(
        title="GPU session running low on credit",
        body=f"~{payload.minutes_remaining} minutes remaining at the current usage rate.",
        url="/sessions",
        tag=f"gpu-credit-{payload.session_id}",
        category="gpu_credit",
        level=payload.level.value,
    )


def _map_system_notification(raw: msgspec.Raw) -> PushNotificationPayload:
    payload = msgspec.json.decode(raw, type=SystemNotificationPayload)
    return PushNotificationPayload(
        title=payload.title,
        body=payload.message,
        url="/",
        tag="system-notification",
        category="system",
        level=payload.level.value,
    )


def _map_balance_updated(raw: msgspec.Raw) -> PushNotificationPayload | None:
    payload = msgspec.json.decode(raw, type=BalanceUpdatedPayload)
    if payload.delta <= 0:
        return None
    if payload.transaction_type not in _POSITIVE_BALANCE_TRANSACTION_TYPES:
        return None
    return PushNotificationPayload(
        title="Balance updated",
        body=f"+{payload.delta} tokens added to your balance.",
        url="/billing",
        tag=f"balance-{payload.account_id}",
        category="balance",
        level="info",
    )
