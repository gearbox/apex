"""Pure ops-event -> Telegram-message mapping. Mirrors push_mapping.py.

No I/O — feed it a decoded ``OpsEventEnvelope`` and get back an
``OpsNotification`` (send it) or ``None`` (skip, e.g. an unknown event_type
during a rolling deploy). Unit-testable in complete isolation from
``TelegramDispatcher``/Redis/Telegram.
"""

from __future__ import annotations

import msgspec

from src.api.schemas.ops_events import (
    GenerationCreatedOpsPayload,
    GenerationFailedOpsPayload,
    GpuNodeStartedOpsPayload,
    HealthTransitionOpsPayload,
    OpsEventEnvelope,
    OpsEventType,
    TokenRevocationFailedOpsPayload,
    UserRegisteredOpsPayload,
)
from src.api.services.telegram.html import escape
from src.core.enums import NotificationClass


class OpsNotification(msgspec.Struct, kw_only=True):
    notification_class: NotificationClass
    product_id: str
    text: str  # final HTML message, ready to send as-is


def map_ops_event(envelope: OpsEventEnvelope) -> OpsNotification | None:
    """Map a decoded OpsEventEnvelope to a Telegram notification, or None to skip."""
    if envelope.event_type is OpsEventType.USER_REGISTERED:
        return _map_user_registered(envelope)
    if envelope.event_type is OpsEventType.GENERATION_CREATED:
        return _map_generation_created(envelope)
    if envelope.event_type is OpsEventType.GENERATION_FAILED:
        return _map_generation_failed(envelope)
    if envelope.event_type is OpsEventType.GPU_NODE_STARTED:
        return _map_gpu_node_started(envelope)
    if envelope.event_type is OpsEventType.HEALTH_SUBSYSTEM_DEGRADED:
        return _map_health_transition(
            envelope,
            notification_class=NotificationClass.HEALTH_DEGRADED,
            headline="Health degraded",
            icon="🔻",
        )
    if envelope.event_type is OpsEventType.HEALTH_SUBSYSTEM_RESTORED:
        return _map_health_transition(
            envelope,
            notification_class=NotificationClass.HEALTH_RESTORED,
            headline="Health restored",
            icon="✅",
        )
    if envelope.event_type is OpsEventType.TOKEN_REVOCATION_FAILED:
        return _map_token_revocation_failed(envelope)
    return None


def _tag(product_id: str) -> str:
    return f"[{escape(product_id)}]"


def _map_user_registered(envelope: OpsEventEnvelope) -> OpsNotification:
    payload = msgspec.json.decode(envelope.payload, type=UserRegisteredOpsPayload)
    text = (
        f"{_tag(envelope.product_id)} 🆕 <b>New registration</b>\n"
        f"user <code>{escape(str(payload.user_id))}</code>"
    )
    return OpsNotification(
        notification_class=NotificationClass.USER_REGISTERED,
        product_id=envelope.product_id,
        text=text,
    )


def _map_generation_created(envelope: OpsEventEnvelope) -> OpsNotification:
    payload = msgspec.json.decode(envelope.payload, type=GenerationCreatedOpsPayload)
    text = (
        f"{_tag(envelope.product_id)} 🎨 <b>New generation</b>\n"
        f"job <code>{escape(str(payload.job_id))}</code> · "
        f"{escape(payload.provider)}/{escape(payload.generation_type)}"
    )
    return OpsNotification(
        notification_class=NotificationClass.GENERATION_CREATED,
        product_id=envelope.product_id,
        text=text,
    )


def _map_generation_failed(envelope: OpsEventEnvelope) -> OpsNotification:
    payload = msgspec.json.decode(envelope.payload, type=GenerationFailedOpsPayload)
    text = (
        f"{_tag(envelope.product_id)} ❌ <b>Generation failed</b>\n"
        f"job <code>{escape(str(payload.job_id))}</code> · "
        f"{escape(payload.provider)}/{escape(payload.generation_type)}"
    )
    return OpsNotification(
        notification_class=NotificationClass.GENERATION_FAILED,
        product_id=envelope.product_id,
        text=text,
    )


def _map_gpu_node_started(envelope: OpsEventEnvelope) -> OpsNotification:
    payload = msgspec.json.decode(envelope.payload, type=GpuNodeStartedOpsPayload)
    text = (
        f"{_tag(envelope.product_id)} 🖥 <b>GPU node started</b>\n"
        f"session <code>{escape(str(payload.session_id))}</code> · {escape(payload.model_type)}"
    )
    return OpsNotification(
        notification_class=NotificationClass.GPU_NODE_STARTED,
        product_id=envelope.product_id,
        text=text,
    )


def _map_health_transition(
    envelope: OpsEventEnvelope,
    *,
    notification_class: NotificationClass,
    headline: str,
    icon: str,
) -> OpsNotification:
    payload = msgspec.json.decode(envelope.payload, type=HealthTransitionOpsPayload)
    text = (
        f"{_tag(envelope.product_id)} {icon} <b>{headline}</b>\n"
        f"{escape(payload.subsystem)}: {escape(payload.previous_status)} → "
        f"{escape(payload.current_status)}\n"
        f"overall: <b>{escape(payload.overall_status)}</b>"
    )
    return OpsNotification(
        notification_class=notification_class,
        product_id=envelope.product_id,
        text=text,
    )


def _map_token_revocation_failed(envelope: OpsEventEnvelope) -> OpsNotification:
    payload = msgspec.json.decode(envelope.payload, type=TokenRevocationFailedOpsPayload)
    text = (
        f"{_tag(envelope.product_id)} 🚨 <b>Token revocation failed</b>\n"
        f"user <code>{escape(str(payload.user_id))}</code> · op: <code>{escape(payload.op)}</code>\n"
        f"bulk access-token revocation could not reach Redis — this user's existing "
        f"access tokens and content cookies remain valid until they expire"
    )
    return OpsNotification(
        notification_class=NotificationClass.TOKEN_REVOCATION_FAILED,
        product_id=envelope.product_id,
        text=text,
    )
