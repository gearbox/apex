"""Ops event contract — internal admin-notification events published to Redis.

Separate from ``src.api.schemas.events`` (user-facing SSE/push events): ops
events carry no PII (D10 — IDs and enums only, never email or error text) and
are consumed exclusively by ``TelegramDispatcher``, not by any user-facing
channel. Publishing is explicit at each source call site via ``OpsEventBus``.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Final
from uuid import UUID

import msgspec

# Sentinel product_id for platform-scoped events (health) that are not tied
# to a single product.
PLATFORM_PRODUCT_ID: Final[str] = "platform"


class OpsEventType(StrEnum):
    """Wire event types on the ``ops:events`` Redis channel.

    Publishing a new member here does not, by itself, reach an operator —
    it must also have a ``NotificationClass`` (``src.core.enums``); see that
    enum's docstring for the four wiring steps.
    """

    USER_REGISTERED = "ops.user.registered"
    GENERATION_CREATED = "ops.generation.created"
    GENERATION_FAILED = "ops.generation.failed"
    GPU_NODE_STARTED = "ops.gpu_node.started"
    HEALTH_SUBSYSTEM_DEGRADED = "ops.health.subsystem_degraded"
    HEALTH_SUBSYSTEM_RESTORED = "ops.health.subsystem_restored"
    TOKEN_REVOCATION_FAILED = "ops.auth.token_revocation_failed"  # noqa: S105
    PUSH_SUBSCRIPTIONS_CLEANUP_FAILED = "ops.push.subscriptions_cleanup_failed"


class OpsEventEnvelope(msgspec.Struct, kw_only=True):
    """Wire format for all ops events. Serialized to JSON over Redis Pub/Sub."""

    event_type: OpsEventType
    product_id: str  # PLATFORM_PRODUCT_ID sentinel for health events
    payload: msgspec.Raw
    timestamp: datetime
    event_id: str


# --- Payloads ---
#
# IDs and enums only — no email, no error text, no tunnel hostnames (D10).
# Telegram is a third-party transport; error strings may embed tunnel
# hostnames or provider payload fragments.


class UserRegisteredOpsPayload(msgspec.Struct, kw_only=True):
    user_id: UUID


class GenerationCreatedOpsPayload(msgspec.Struct, kw_only=True):
    job_id: UUID
    user_id: UUID
    provider: str
    generation_type: str


class GenerationFailedOpsPayload(msgspec.Struct, kw_only=True):
    job_id: UUID
    user_id: UUID
    provider: str
    generation_type: str
    failure_code: str | None = None


class GpuNodeStartedOpsPayload(msgspec.Struct, kw_only=True):
    session_id: UUID
    user_id: UUID
    model_type: str


class HealthTransitionOpsPayload(msgspec.Struct, kw_only=True):
    subsystem: str
    previous_status: str
    current_status: str
    overall_status: str


class TokenRevocationFailedOpsPayload(msgspec.Struct, kw_only=True):
    """issue #142 F5 — a bulk access-token revocation write failed against a
    Redis that IS configured (TokenRevocationService.enabled), so this is an
    active security-relevant degradation, not the expected no-op of Redis
    being unset. `op` names the triggering action, e.g. "logout_all",
    "change_password", "deactivate_account", "reset_password",
    "token_reuse_detected", "refresh_race_detected"."""

    user_id: UUID
    op: str


class PushSubscriptionsCleanupFailedOpsPayload(msgspec.Struct, kw_only=True):
    """A bulk-revocation event's push-subscription cleanup (``delete_all_for_user``)
    failed. Independent of `TokenRevocationFailedOpsPayload` — push deletion is a
    plain DB operation with its own failure mode, not gated on Redis (see
    `TokenRevocationService`). `op` names the triggering action, using the same
    vocabulary as `TokenRevocationFailedOpsPayload.op`: "logout_all",
    "change_password", "deactivate_account", "reset_password",
    "token_reuse_detected"."""

    user_id: UUID
    op: str
