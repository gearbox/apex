"""Transport-agnostic event bus backed by Redis Pub/Sub.

To migrate to WebSockets later:
  1. Replace Redis publish with ChannelsPlugin.publish
  2. Replace the SSE subscriber loop with a WebSocket handler
  3. Everything else (schemas, publish call sites) stays identical.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from datetime import UTC, datetime
from typing import TYPE_CHECKING
from uuid import UUID

import msgspec
import structlog
from redis.asyncio.client import PubSub

from src.api.schemas.events import (
    BalanceUpdatedPayload,
    EventEnvelope,
    EventType,
)
from src.core.redis import get_redis_client
from src.core.uid import new_id

if TYPE_CHECKING:
    from src.api.services.billing import BalanceEvent

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

    async def publish_balance(self, event: BalanceEvent | None) -> None:
        """Publish a pending ``BalanceEvent`` built by ``BillingService``.

        No-ops on ``None`` (no SSE target, or the caller has no event bus).
        Callers MUST invoke this strictly after committing the transaction
        that wrote the ledger row the event describes — never before, or a
        rolled-back transaction would produce a phantom balance update.

        A publish failure here must never surface as a request error: the
        ledger write already committed, so this is best-effort UX only.
        """
        if event is None:
            return
        payload = BalanceUpdatedPayload(
            account_id=event.account_id,
            balance=event.balance,
            delta=event.delta,
            transaction_type=event.transaction_type,
        )
        try:
            for uid in event.user_ids:
                await self.publish(
                    user_id=uid,
                    event_type=EventType.BALANCE_UPDATED,
                    payload=payload,
                )
        except Exception:
            logger.exception(
                "event_bus.balance_publish_failed",
                account_id=str(event.account_id),
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
        *,
        heartbeat_interval: float,
    ) -> AsyncGenerator[EventEnvelope | None]:
        """Subscribe to a user's channel + system broadcast.

        Yields EventEnvelope on a message, or None on heartbeat timeout.
        Uses get_message() so the read is never cancelled mid-flight —
        avoids the redis-py CancelledError→TimeoutError conversion that
        caused SSE stream drops on idle periods.
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

            while True:
                message = await pubsub.get_message(
                    ignore_subscribe_messages=True, timeout=heartbeat_interval
                )
                if message is None:
                    yield None  # heartbeat tick — no event within the interval
                    continue
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
