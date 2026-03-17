"""Transport-agnostic event bus backed by Redis Pub/Sub.

To migrate to WebSockets later:
  1. Replace Redis publish with ChannelsPlugin.publish
  2. Replace the SSE subscriber loop with a WebSocket handler
  3. Everything else (schemas, publish call sites) stays identical.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from datetime import UTC, datetime
from uuid import UUID

import msgspec
import structlog
from redis.asyncio.client import PubSub

from src.api.schemas.events import (
    EventEnvelope,
    EventType,
)
from src.core.redis import get_redis_client
from src.core.uid import new_id

logger = structlog.get_logger(__name__)

_encoder = msgspec.json.Encoder()
_decoder = msgspec.json.Decoder(EventEnvelope)

# Union of concrete payload types for type hints at call sites
type EventPayload = (
    object  # any msgspec.Struct subclass — encoded via msgspec.json.encode before wrapping
)


class EventBus:
    """Publishes and subscribes to user-scoped real-time events."""

    # Channel naming convention
    USER_CHANNEL_PREFIX = "user:"
    SYSTEM_CHANNEL = "system:broadcast"

    def user_channel(self, user_id: UUID) -> str:
        return f"{self.USER_CHANNEL_PREFIX}{user_id}"

    async def publish(
        self,
        *,
        user_id: UUID,
        event_type: EventType,
        payload: object,
    ) -> None:
        """Publish an event to a user's channel."""
        envelope = EventEnvelope(
            event_type=event_type,
            payload=msgspec.Raw(_encoder.encode(payload)),
            timestamp=datetime.now(UTC),
            event_id=str(new_id()),
        )
        channel = self.user_channel(user_id)
        data = _encoder.encode(envelope)

        client = get_redis_client()
        await client.publish(channel, data)
        logger.debug(
            "event_bus.published",
            channel=channel,
            event_type=event_type.value,
        )

    async def publish_system(
        self,
        *,
        event_type: EventType,
        payload: object,
    ) -> None:
        """Publish a system-wide event (all connected users)."""
        envelope = EventEnvelope(
            event_type=event_type,
            payload=msgspec.Raw(_encoder.encode(payload)),
            timestamp=datetime.now(UTC),
            event_id=str(new_id()),
        )
        data = _encoder.encode(envelope)

        client = get_redis_client()
        await client.publish(self.SYSTEM_CHANNEL, data)
        logger.debug(
            "event_bus.published_system",
            event_type=event_type.value,
        )

    async def subscribe(
        self,
        user_id: UUID,
    ) -> AsyncGenerator[EventEnvelope]:
        """Subscribe to a user's channel + system broadcast.

        Yields EventEnvelope instances. Caller is responsible for
        converting to SSE format or WebSocket frames.
        """
        client = get_redis_client()
        pubsub: PubSub = client.pubsub()
        user_channel = self.user_channel(user_id)

        try:
            await pubsub.subscribe(user_channel, self.SYSTEM_CHANNEL)
            logger.info(
                "event_bus.subscribed",
                channels=[user_channel, self.SYSTEM_CHANNEL],
            )

            async for message in pubsub.listen():
                if message["type"] != "message":
                    continue
                try:
                    yield _decoder.decode(message["data"])
                except Exception:
                    logger.exception(
                        "event_bus.decode_error",
                        channel=message.get("channel"),
                    )
        finally:
            await pubsub.unsubscribe(user_channel, self.SYSTEM_CHANNEL)
            await pubsub.aclose()  # type: ignore[no-untyped-call]
            logger.info("event_bus.unsubscribed", user_id=str(user_id))
