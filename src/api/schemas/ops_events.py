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
    """Wire event types on the ``ops:events`` Redis channel."""

    USER_REGISTERED = "ops.user.registered"
    GENERATION_CREATED = "ops.generation.created"
    GENERATION_FAILED = "ops.generation.failed"
    GPU_NODE_STARTED = "ops.gpu_node.started"
    HEALTH_SUBSYSTEM_DEGRADED = "ops.health.subsystem_degraded"
    HEALTH_SUBSYSTEM_RESTORED = "ops.health.subsystem_restored"


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


class GpuNodeStartedOpsPayload(msgspec.Struct, kw_only=True):
    session_id: UUID
    user_id: UUID
    model_type: str


class HealthTransitionOpsPayload(msgspec.Struct, kw_only=True):
    subsystem: str
    previous_status: str
    current_status: str
    overall_status: str
