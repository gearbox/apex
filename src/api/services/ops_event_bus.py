"""Publisher for ops (admin-notification) events — mirrors EventBus in shape.

Dedicated Redis channel (``ops:events``), separate from the user-facing
``EventBus`` channels, so ops semantics never couple to user-facing event
shapes. The sole consumer is ``TelegramDispatcher``.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Final

import msgspec
import structlog

from src.api.schemas.ops_events import OpsEventEnvelope, OpsEventType
from src.core.redis import get_redis_client
from src.core.uid import new_id

logger = structlog.get_logger(__name__)

_encoder = msgspec.json.Encoder()

# TODO(redis-namespacing): unnamespaced like every other Redis key in this
# codebase — leave it to the future namespacing arc rather than inventing a
# one-off scheme here.
OPS_EVENTS_CHANNEL: Final[str] = "ops:events"


class OpsEventBus:
    """Publishes ops events to the ``ops:events`` Redis channel."""

    def __init__(self, *, enabled: bool = True) -> None:
        """Args:
        enabled: When ``False`` (no Redis configured), ``publish`` no-ops
            instead of touching Redis. Keeps the DI type non-optional across
            Redis-less deployments.
        """
        self._enabled = enabled

    async def publish(
        self,
        *,
        event_type: OpsEventType,
        product_id: str,
        payload: object,
    ) -> None:
        """Publish an ops event. Never raises — all call sites are best-effort.

        Callers are post-commit / best-effort paths (see the four publish
        call sites): a Redis blip here must never affect already-committed
        state.
        """
        if not self._enabled:
            logger.debug("ops_event_bus.disabled_skip", event_type=event_type.value)
            return
        try:
            envelope = OpsEventEnvelope(
                event_type=event_type,
                product_id=product_id,
                payload=msgspec.Raw(_encoder.encode(payload)),
                timestamp=datetime.now(UTC),
                event_id=str(new_id()),
            )
            data = _encoder.encode(envelope)

            client = get_redis_client()
            await client.publish(OPS_EVENTS_CHANNEL, data)
            logger.debug(
                "ops_event.published",
                event_type=event_type.value,
                product_id=product_id,
                event_id=envelope.event_id,
            )
        except Exception:
            logger.warning(
                "ops_event.publish_failed",
                event_type=event_type.value,
                product_id=product_id,
            )
