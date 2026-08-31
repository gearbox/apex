"""Background worker: fans out ops events as Telegram messages to admins.

Structural clone of ``PushDispatcher`` (src/workers/push_dispatcher.py):
subscribes directly to the ``ops:events`` Redis channel OpsEventBus publishes
to, decodes each ``OpsEventEnvelope``, maps it via the pure
``telegram.mapping.map_ops_event``, resolves recipients, throttles, and
sends. Never modifies OpsEventBus or any publish call site (open/closed).

Delivery guarantee: best-effort, same as PushDispatcher (D6) — dispatcher
down = events lost. No outbox, no replay.

Throttling (D7): in-memory, keyed (user_id, notification_class). Valid
because this dispatcher is a leader-leased singleton — only one process
ever drains the subscription at a time, so in-memory state is never split
across processes.
"""

from __future__ import annotations

import asyncio
import time
from contextlib import suppress
from typing import TYPE_CHECKING, NamedTuple

import msgspec
import structlog

from src.api.schemas.ops_events import OpsEventEnvelope
from src.api.services.telegram.mapping import map_ops_event
from src.core.enums import PLATFORM_SCOPED_NOTIFICATION_CLASSES
from src.core.redis import get_operational_redis_client, get_redis_client
from src.db.repositories.admin_notifications import AdminNotificationRepository
from src.workers.base import LeaderLease, lease_key

if TYPE_CHECKING:
    from collections.abc import Callable
    from uuid import UUID

    from sqlalchemy.ext.asyncio import AsyncSession

    from src.api.services.telegram.sender import TelegramSender

logger = structlog.get_logger(__name__)

_decoder = msgspec.json.Decoder(OpsEventEnvelope)

_LEASE_TTL_SECONDS = 90
_POLL_TIMEOUT_SECONDS = 5.0
_NOT_LEADER_SLEEP_SECONDS = 5.0
_ERROR_BACKOFF_SECONDS = 5.0


class _ThrottleState(NamedTuple):
    last_sent_monotonic: float
    suppressed_count: int


class TelegramDispatcher:
    """Consumes the ops:events channel and dispatches Telegram messages."""

    def __init__(
        self,
        *,
        sender: TelegramSender,
        session_factory: Callable[[], AsyncSession],
        redis_enabled: bool,
    ) -> None:
        self._sender = sender
        self._session_factory = session_factory
        self._running = False
        self._task: asyncio.Task[None] | None = None
        self._stop_event = asyncio.Event()
        self._lease = LeaderLease(
            key=lease_key("telegram_dispatcher"),
            ttl_seconds=_LEASE_TTL_SECONDS,
            redis_enabled=redis_enabled,
            client_factory=get_operational_redis_client,
        )
        # (user_id, notification_class) -> throttle state. Correct only under
        # single-leader execution — see module docstring.
        self._throttle: dict[tuple[UUID, str], _ThrottleState] = {}

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
        logger.info("telegram.dispatcher.started")

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
                logger.warning("telegram.dispatcher.drain_timeout")
                with suppress(asyncio.CancelledError):
                    await task
            self._task = None

        await self._lease.release()
        logger.info("telegram.dispatcher.stopped")

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
                logger.exception("telegram.dispatcher.loop_error")
                await self._interruptible_sleep(_ERROR_BACKOFF_SECONDS)

    async def _listen_while_leader(self) -> None:
        from src.api.services.ops_event_bus import OPS_EVENTS_CHANNEL

        client = get_redis_client()
        pubsub = client.pubsub()
        try:
            await pubsub.subscribe(OPS_EVENTS_CHANNEL)
            logger.info("telegram.dispatcher.subscribed")

            while self._running:
                # NOTE: renewal-per-iteration here diverges deliberately from
                # PushDispatcher — see review F2.
                if not await self._lease.acquire_or_renew():
                    return  # lost leadership — drop subscription and retry from the top
                message = await pubsub.get_message(
                    ignore_subscribe_messages=True, timeout=_POLL_TIMEOUT_SECONDS
                )
                if message is None:
                    continue
                if message["type"] != "message":
                    continue
                await self._handle_raw_message(message["data"])
        finally:
            with suppress(Exception):
                await pubsub.unsubscribe(OPS_EVENTS_CHANNEL)
                await pubsub.aclose()  # type: ignore[no-untyped-call]

    async def _handle_raw_message(self, data: str) -> None:
        try:
            envelope = _decoder.decode(data)
        except Exception:
            logger.warning("telegram.dispatcher.decode_failed")
            return

        notification = map_ops_event(envelope)
        if notification is None:
            return

        try:
            async with self._session_factory() as session:
                repo = AdminNotificationRepository(session)
                recipients = await repo.list_recipients_for_class(
                    notification.notification_class.value
                )
        except Exception:
            logger.exception(
                "telegram.dispatcher.recipient_lookup_failed",
                notification_class=notification.notification_class.value,
            )
            return

        is_platform_scoped = notification.notification_class in PLATFORM_SCOPED_NOTIFICATION_CLASSES
        for recipient in recipients:
            if not is_platform_scoped and recipient.product_id != notification.product_id:
                continue
            await self._send_throttled(
                user_id=recipient.user_id,
                chat_id=recipient.chat_id,
                notification_class=notification.notification_class.value,
                min_interval_seconds=recipient.min_interval_seconds,
                text=notification.text,
                event_id=envelope.event_id,
            )

    async def _send_throttled(
        self,
        *,
        user_id: UUID,
        chat_id: int,
        notification_class: str,
        min_interval_seconds: int,
        text: str,
        event_id: str,
    ) -> None:
        key = (user_id, notification_class)
        now = time.monotonic()
        state = self._throttle.get(key)

        if min_interval_seconds > 0 and state is not None:
            elapsed = now - state.last_sent_monotonic
            if elapsed < min_interval_seconds:
                self._throttle[key] = _ThrottleState(
                    last_sent_monotonic=state.last_sent_monotonic,
                    suppressed_count=state.suppressed_count + 1,
                )
                logger.debug(
                    "telegram.throttled",
                    user_id=str(user_id),
                    notification_class=notification_class,
                )
                return

        suppressed = state.suppressed_count if state is not None else 0
        final_text = f"{text}\n<i>(+{suppressed} suppressed)</i>" if suppressed > 0 else text

        try:
            await self._sender.send_message(chat_id=chat_id, text=final_text)
        except Exception as exc:
            logger.warning(
                "telegram.send_failed",
                chat_id=chat_id,
                notification_class=notification_class,
                error=str(exc),
            )
            return

        self._throttle[key] = _ThrottleState(last_sent_monotonic=now, suppressed_count=0)
        logger.info(
            "telegram.sent",
            chat_id=chat_id,
            notification_class=notification_class,
            event_id=event_id,
        )
