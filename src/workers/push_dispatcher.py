"""Background worker: fans out real-time events as Web Push notifications.

Subscribes directly to the same Redis Pub/Sub channels ``EventBus`` publishes
to (``user:*`` via PSUBSCRIBE, plus ``system:broadcast``) and decodes each
``EventEnvelope``. Mapping from event to notification content is pure and
lives in ``src.api.services.push_mapping``; this worker's only job is decode
-> map -> send. It never modifies ``EventBus`` or any existing publish call
site (open/closed).

Delivery guarantee: best-effort, same as SSE (see ``EventBus`` docstring). If
this worker is down, or the Redis connection drops, pushes are simply lost —
no replay, no persistence queue. Acceptable for v1.

Deferred to v2 (deliberately NOT implemented here):
  - Per-category user notification preferences.
  - Suppressing push while the user has an active SSE connection (presence
    tracking) — a user with the tab open still receives a push today.

Only one process should run this loop in a multi-worker deployment. Unlike
the tick-based ``PeriodicWorker`` subclasses elsewhere, this is a long-lived
blocking read loop, so it reuses ``LeaderLease`` directly rather than
subclassing ``PeriodicWorker``: a non-leader process never opens a Redis
Pub/Sub subscription, so it never buffers messages it can't drain.
"""

from __future__ import annotations

import asyncio
from contextlib import suppress
from typing import TYPE_CHECKING
from uuid import UUID

import msgspec
import structlog

from src.api.schemas.events import EventEnvelope
from src.api.services.push_mapping import map_event_to_notification
from src.core.redis import get_redis_client
from src.workers.base import LeaderLease

if TYPE_CHECKING:
    from collections.abc import Callable

    from sqlalchemy.ext.asyncio import AsyncSession

    from src.api.services.push import PushService

logger = structlog.get_logger(__name__)

_decoder = msgspec.json.Decoder(EventEnvelope)

_USER_CHANNEL_PATTERN = "user:*"
_USER_CHANNEL_PREFIX = "user:"
_SYSTEM_CHANNEL = "system:broadcast"
_POLL_TIMEOUT_SECONDS = 5.0
_NOT_LEADER_SLEEP_SECONDS = 5.0
_ERROR_BACKOFF_SECONDS = 5.0
_LEASE_TTL_SECONDS = 90


class PushDispatcher:
    """Consumes EventBus channels and dispatches Web Push notifications."""

    def __init__(
        self,
        *,
        push_service: PushService,
        session_factory: Callable[[], AsyncSession],
        redis_enabled: bool,
    ) -> None:
        self._push_service = push_service
        self._session_factory = session_factory
        self._running = False
        self._task: asyncio.Task[None] | None = None
        self._stop_event = asyncio.Event()
        self._lease = LeaderLease(
            key="worker:push_dispatcher:lease",
            ttl_seconds=_LEASE_TTL_SECONDS,
            redis_enabled=redis_enabled,
        )

    @property
    def is_running(self) -> bool:
        return self._running

    async def start(self) -> None:
        """Start the loop in the background. Idempotent."""
        if self._running:
            return
        self._running = True
        self._stop_event = asyncio.Event()
        self._task = asyncio.create_task(self._run_loop())
        logger.info("push_dispatcher.started")

    async def stop(self) -> None:
        """Signal the loop to stop and drain the in-flight read. Idempotent."""
        if not self._running:
            return
        self._running = False
        self._stop_event.set()

        if self._task is not None:
            task = self._task
            try:
                await asyncio.wait_for(task, timeout=10.0)
            except TimeoutError:
                logger.warning("push_dispatcher.drain_timeout")
                with suppress(asyncio.CancelledError):
                    await task
            self._task = None

        await self._lease.release()
        logger.info("push_dispatcher.stopped")

    async def _interruptible_sleep(self, seconds: float) -> None:
        with suppress(TimeoutError):
            await asyncio.wait_for(self._stop_event.wait(), timeout=seconds)

    async def _run_loop(self) -> None:
        while self._running:
            if not await self._lease.acquire_or_renew():
                await self._interruptible_sleep(_NOT_LEADER_SLEEP_SECONDS)
                continue

            try:
                await self._listen_while_leader()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("push_dispatcher.loop_error")
                await self._interruptible_sleep(_ERROR_BACKOFF_SECONDS)

    async def _listen_while_leader(self) -> None:
        client = get_redis_client()
        pubsub = client.pubsub()
        try:
            await pubsub.psubscribe(_USER_CHANNEL_PATTERN)
            await pubsub.subscribe(_SYSTEM_CHANNEL)
            logger.info("push_dispatcher.subscribed")

            while self._running:
                message = await pubsub.get_message(
                    ignore_subscribe_messages=True, timeout=_POLL_TIMEOUT_SECONDS
                )
                if message is None:
                    if not await self._lease.acquire_or_renew():
                        return  # lost leadership — drop subscription and retry from the top
                    continue
                if message["type"] not in ("message", "pmessage"):
                    continue
                await self._handle_raw_message(channel=message["channel"], data=message["data"])
        finally:
            with suppress(Exception):
                await pubsub.punsubscribe(_USER_CHANNEL_PATTERN)
                await pubsub.unsubscribe(_SYSTEM_CHANNEL)
                await pubsub.aclose()  # type: ignore[no-untyped-call]

    async def _handle_raw_message(self, *, channel: str, data: str) -> None:
        try:
            envelope = _decoder.decode(data)
        except Exception:
            logger.exception("push_dispatcher.decode_error", channel=channel)
            return

        notification = map_event_to_notification(envelope)
        if notification is None:
            return

        try:
            async with self._session_factory() as session, session.begin():
                if channel == _SYSTEM_CHANNEL:
                    await self._push_service.send_broadcast(notification, session=session)
                else:
                    user_id = UUID(channel.removeprefix(_USER_CHANNEL_PREFIX))
                    await self._push_service.send_to_user(user_id, notification, session=session)
        except Exception:
            logger.exception(
                "push_dispatcher.send_failed",
                channel=channel,
                event_type=envelope.event_type.value,
            )
