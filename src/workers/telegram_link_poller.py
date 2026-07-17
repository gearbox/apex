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
from contextlib import suppress
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import structlog

from src.db.repositories.admin_notifications import AdminNotificationRepository
from src.workers.base import PeriodicWorker

if TYPE_CHECKING:
    from collections.abc import Callable

    from sqlalchemy.ext.asyncio import AsyncSession

    from src.api.services.telegram.sender import TelegramSender, TelegramUpdate
    from src.db.models.admin_notifications import AdminTelegramLink

logger = structlog.get_logger(__name__)

# TODO(redis-namespacing): unnamespaced like every other worker lease key.
_LEASE_KEY_NAME = "telegram_link_poller"

_START_COMMAND_RE = re.compile(r"^/start\s+(\S+)$")

_SUCCESS_REPLY = "✅ Telegram linked to your Apex admin account."
_INVALID_REPLY = "⚠️ Link token is invalid or expired. Generate a new link in the admin panel."


class TelegramLinkPoller(PeriodicWorker):
    """Long-polls Telegram getUpdates and confirms /start <token> deep links."""

    def __init__(
        self,
        *,
        sender: TelegramSender,
        session_factory: Callable[[], AsyncSession],
        poll_timeout_seconds: int,
        redis_enabled: bool = False,
    ) -> None:
        # interval_seconds is nominal — get_updates() itself long-polls for
        # up to poll_timeout_seconds, so each tick already blocks that long.
        super().__init__(
            name=_LEASE_KEY_NAME,
            interval_seconds=1.0,
            redis_enabled=redis_enabled,
        )
        self._sender = sender
        self._session_factory = session_factory
        self._poll_timeout_seconds = poll_timeout_seconds
        self._next_offset: int | None = None

    async def run_once(self) -> None:
        updates = await self._sender.get_updates(
            offset=self._next_offset, timeout_seconds=self._poll_timeout_seconds
        )
        for update in updates:
            self._next_offset = update.update_id + 1
            await self._handle_update(update)

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
