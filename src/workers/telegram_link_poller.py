"""Background worker: long-polls Telegram getUpdates to confirm deep-link tokens.

Same skeleton as TelegramDispatcher — leader-leased loop, best-effort. Uses
a DIFFERENT lease key: two concurrent ``getUpdates`` pollers cause Telegram
to return 409 Conflict, so exactly one process may hold this subscription
platform-wide.

Flow: an admin requests a link invite (AdminNotificationService.create_link_token),
opens ``t.me/<bot>?start=<token>`` in Telegram, which sends ``/start <token>``
to the bot. This poller reads that update via long-polling and confirms it.

Tokens are single-use — ``confirm_link_by_token`` clears the token in the
same UPDATE that sets ``chat_id``. A restart may replay a few already-seen
updates (offset tracked only in memory); a replayed, already-consumed token
simply lands in the "invalid" branch — harmless.
"""

from __future__ import annotations

import re
import time
from contextlib import suppress
from datetime import UTC, datetime
from http import HTTPStatus
from typing import TYPE_CHECKING, Final

import structlog

from src.api.services.telegram.sender import TelegramSendError
from src.db.repositories.admin_notifications import AdminNotificationRepository
from src.workers.base import PeriodicWorker

if TYPE_CHECKING:
    from collections.abc import Callable

    from redis.asyncio import Redis
    from sqlalchemy.ext.asyncio import AsyncSession

    from src.api.services.telegram.sender import TelegramSender, TelegramUpdate
    from src.db.models.admin_notifications import AdminTelegramLink

logger = structlog.get_logger(__name__)

_LEASE_KEY_NAME = "telegram_link_poller"

_START_COMMAND_RE = re.compile(r"^/start\s+(\S+)$")

_SUCCESS_REPLY = "✅ Telegram linked to your Apex admin account."
_INVALID_REPLY = "⚠️ Link token is invalid or expired. Generate a new link in the admin panel."

# Telegram allows exactly one active getUpdates long-poll per bot token,
# platform-wide — a second holder (orphaned container, dev machine, prod
# sharing staging's token) gets a 409 on every call until it stops. Back off
# instead of hammering the API, and cap how often we log it.
_CONFLICT_BACKOFF_SECONDS: Final[float] = 60.0
_CONFLICT_RELOG_INTERVAL_SECONDS: Final[float] = 300.0
_CONFLICT_REMEDIATION = (
    "Another process is holding this bot token's getUpdates long-poll — find "
    "and stop it, or rotate TELEGRAM_BOT_TOKEN via BotFather (/revoke) if it "
    "cannot be identified or stopped."
)


class TelegramLinkPoller(PeriodicWorker):
    """Long-polls Telegram getUpdates and confirms /start <token> deep links."""

    def __init__(
        self,
        *,
        sender: TelegramSender,
        session_factory: Callable[[], AsyncSession],
        poll_timeout_seconds: int,
        redis_enabled: bool = False,
        redis_client_factory: Callable[[], Redis],
    ) -> None:
        # interval_seconds is nominal — get_updates() itself long-polls for
        # up to poll_timeout_seconds, so each tick already blocks that long.
        super().__init__(
            name=_LEASE_KEY_NAME,
            interval_seconds=1.0,
            redis_enabled=redis_enabled,
            redis_client_factory=redis_client_factory,
        )
        self._sender = sender
        self._session_factory = session_factory
        self._poll_timeout_seconds = poll_timeout_seconds
        self._next_offset: int | None = None
        self._consecutive_conflicts = 0
        self._last_conflict_logged_at: float | None = None

    async def run_once(self) -> None:
        try:
            updates = await self._sender.get_updates(
                offset=self._next_offset, timeout_seconds=self._poll_timeout_seconds
            )
        except TelegramSendError as exc:
            if exc.status_code != HTTPStatus.CONFLICT:
                raise
            await self._handle_conflict()
            return

        if self._consecutive_conflicts:
            logger.info(
                "telegram.getupdates_conflict_cleared",
                consecutive=self._consecutive_conflicts,
            )
            self._consecutive_conflicts = 0
            self._last_conflict_logged_at = None

        for update in updates:
            self._next_offset = update.update_id + 1
            await self._handle_update(update)

    async def _handle_conflict(self) -> None:
        """Another process holds this bot token's getUpdates poll.

        Never resets ``_next_offset`` — the conflict tells us nothing about
        which updates the other holder has consumed.
        """
        self._consecutive_conflicts += 1
        now = time.monotonic()
        if (
            self._last_conflict_logged_at is None
            or (now - self._last_conflict_logged_at) >= _CONFLICT_RELOG_INTERVAL_SECONDS
        ):
            self._last_conflict_logged_at = now
            logger.error(
                "telegram.getupdates_conflict",
                consecutive=self._consecutive_conflicts,
                remediation=_CONFLICT_REMEDIATION,
            )
        await self._interruptible_sleep(_CONFLICT_BACKOFF_SECONDS)

    async def _handle_update(self, update: TelegramUpdate) -> None:
        message = update.message
        if message is None or message.text is None:
            return

        match = _START_COMMAND_RE.match(message.text)
        if match is None:
            return  # never reply to non-/start messages — avoid echo-spam (see module docstring)

        token = match.group(1)
        chat_id = message.chat.id
        now = datetime.now(UTC)

        replay: AdminTelegramLink | None = None
        try:
            async with self._session_factory() as session:
                repo = AdminNotificationRepository(session)
                link = await repo.confirm_link_by_token(token, chat_id, now)
                if link is None:
                    # Offsets are confirmed only by the next getUpdates call — a
                    # restart right after processing this token can replay it.
                    # An already-linked chat means we already replied once;
                    # a stranger's chat has no link row at all.
                    replay = await repo.get_link_by_chat_id(chat_id)
                await session.commit()
        except Exception:
            logger.exception("telegram.link.confirm_failed")
            return

        if link is not None:
            logger.info("telegram.link.confirmed", user_id=str(link.user_id), chat_id=chat_id)
            reply = _SUCCESS_REPLY
        elif replay is not None:
            logger.info("telegram.link.replay_ignored", chat_id=chat_id)
            return
        else:
            logger.info("telegram.link.token_rejected", chat_id=chat_id)
            reply = _INVALID_REPLY

        with suppress(Exception):
            await self._sender.send_message(chat_id=chat_id, text=reply)
