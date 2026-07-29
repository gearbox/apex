"""SSE event generator for admin health stream.

Extracted from the controller to keep routes thin. Handles both
Redis Pub/Sub mode (production) and direct polling fallback (no Redis).
"""

from __future__ import annotations

import asyncio
from contextlib import suppress
from typing import TYPE_CHECKING

import msgspec
import structlog
from redis.exceptions import RedisError

from src.api.services.health import HEALTH_STREAM_CHANNEL
from src.core.redis import get_sse_redis_client, get_sse_redis_pool

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from src.api.services.health.service import HealthService
    from src.core.config import Settings

logger = structlog.get_logger()

_HEARTBEAT_INTERVAL_SECONDS = 15


async def health_sse_generator(
    *,
    health_service: HealthService,
    settings: Settings,
) -> AsyncGenerator[dict[str, str]]:
    """Yield SSE-formatted events for the admin health stream.

    With Redis: subscribes to health:stream Pub/Sub channel.
    Without Redis: polls HealthService directly at the configured interval.

    Yields:
        Dicts with "event"+"data" (snapshot) or "comment" (keepalive).
    """
    if settings.redis_url is not None:
        async for event in _redis_stream():
            yield event
    else:
        async for event in _polling_stream(health_service, settings):
            yield event


async def _redis_stream() -> AsyncGenerator[dict[str, str]]:
    """Subscribe to Redis health:stream channel and yield SSE events."""

    # Long-lived per-connected-admin subscription — belongs on the SSE pool,
    # not the shared short-lived-operation pool. See src/core/redis.py.
    client = get_sse_redis_client()
    # Created outside the try: PubSub connects lazily, so this acquires
    # nothing, but it must be in scope for the finally that releases it.
    pubsub = client.pubsub()

    try:
        try:
            await pubsub.subscribe(HEALTH_STREAM_CHANNEL)
        except (RedisError, OSError) as exc:
            logger.exception(
                "health.sse.subscribe_failed",
                error=str(exc),
                error_type=type(exc).__name__,
                sse_pool_max=get_sse_redis_pool().max_connections,
            )
            raise

        while True:
            try:
                # Single timeout via asyncio.wait_for — no nested timeout
                # on get_message. Let get_message block; wait_for enforces
                # the heartbeat deadline.
                message = await asyncio.wait_for(
                    pubsub.get_message(ignore_subscribe_messages=True),
                    timeout=float(_HEARTBEAT_INTERVAL_SECONDS),
                )
                if message is not None and message["type"] == "message":
                    data = message["data"]
                    yield {
                        "event": "health.snapshot",
                        "data": data.decode() if isinstance(data, bytes) else str(data),
                    }
                else:
                    yield {"comment": "keepalive"}
            except asyncio.TimeoutError:  # noqa: UP041
                yield {"comment": "keepalive"}
    except asyncio.CancelledError:
        logger.debug("health.sse.client_disconnected")
    finally:
        # unsubscribe is a courtesy and will itself raise if the connection
        # is already dead — suppressed so it cannot replace the exception
        # actually propagating to the caller. aclose() is the call that
        # returns the connection to the SSE pool.
        with suppress(RedisError, OSError):
            await pubsub.unsubscribe(HEALTH_STREAM_CHANNEL)
        await pubsub.aclose()  # type: ignore[no-untyped-call]


async def _polling_stream(
    health_service: HealthService,
    settings: Settings,
) -> AsyncGenerator[dict[str, str]]:
    """Fallback: poll HealthService directly when Redis is unavailable."""
    while True:
        try:
            data_dict = await health_service.check_all_and_build()
            yield {
                "event": "health.snapshot",
                "data": msgspec.json.encode(data_dict).decode(),
            }
        except Exception:
            logger.exception("health.sse.poll_error")
            yield {"comment": "error"}
        await asyncio.sleep(settings.health_snapshot_interval_seconds)
