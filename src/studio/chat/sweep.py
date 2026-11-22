"""The sweep: fail and refund replies and drafts that nobody finished (P2c-18).

A task that died with its process (a crash, a kill, a deploy that did not drain) leaves its row
``pending`` for ever, holding the user's reservation. The leader's sweep finds every artist message
and memory draft that has been ``pending`` for longer than the reply deadline plus a grace, fails it
(``reply_timeout``) with the guarded update **and refunds it in the same transaction** - one
transaction per row, so one bad row never blocks the others. A row the task finished in the
meantime is simply skipped: the guarded update hits nothing and nothing is refunded.

Takes only row locks (never the chat or artist lock), so it joins no lock cycle. Logs carry ids and
codes only.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

import structlog

from src.db.repositories.studio_chat import (
    StudioChatMessageRepository,
    StudioMemoryDraftRepository,
)
from src.studio.chat.failures import ERR_REPLY_TIMEOUT
from src.studio.chat.refunds import refund_reservation

if TYPE_CHECKING:
    from uuid import UUID

    from src.api.services.billing import BalanceEvent, BillingService
    from src.db.session import DatabaseManager
    from src.studio.chat.ports import ChatEvents

__all__ = ["SWEEP_GRACE_SECONDS", "ChatSweeper"]

logger = structlog.get_logger(__name__)

SWEEP_GRACE_SECONDS: Final = 30
"""Added to the deadline: a row is orphaned only when its own task could not still be running."""
_BATCH: Final = 100
"""Most rows of each kind one sweep takes (the next tick takes the rest)."""


class ChatSweeper:
    """Fails and refunds orphaned replies and drafts."""

    def __init__(
        self,
        *,
        db_manager: DatabaseManager,
        billing: BillingService,
        events: ChatEvents,
        deadline_seconds: int,
    ) -> None:
        self._db = db_manager
        self._billing = billing
        self._events = events
        self._older_than = deadline_seconds + SWEEP_GRACE_SECONDS

    async def sweep(self) -> int:
        """Fail every ``pending`` reply and draft past the deadline plus the grace.

        Returns:
            How many rows this sweep failed (rows the task finished first are not counted).
        """
        async with self._db.session() as session:
            replies = await StudioChatMessageRepository(session).expired_pending(
                older_than_seconds=self._older_than, limit=_BATCH
            )
            drafts = await StudioMemoryDraftRepository(session).expired_pending(
                older_than_seconds=self._older_than, limit=_BATCH
            )
        failed = 0
        for message_id, user_id, product_id in replies:
            failed += await self._fail_reply(message_id, user_id, product_id)
        for draft_id, user_id, product_id in drafts:
            failed += await self._fail_draft(draft_id, user_id, product_id)
        return failed

    async def _fail_reply(self, message_id: UUID, user_id: UUID, product_id: str) -> int:
        balance: BalanceEvent | None = None
        async with self._db.session() as session:
            message = await StudioChatMessageRepository(session).finish_failed(
                message_id, user_id=user_id, product_id=product_id, error_code=ERR_REPLY_TIMEOUT
            )
            if message is not None:
                balance = await refund_reservation(
                    self._billing,
                    session,
                    job_id=message.id,
                    price=message.price_tokens or 0,
                    product_id=product_id,
                    user_id=user_id,
                    reason=ERR_REPLY_TIMEOUT,
                    description="Studio chat reply failed",
                )
        if message is None:  # the task finished it first
            return 0
        logger.warning("studio.chat_reply_swept", message_id=str(message_id))
        await self._events.balance(balance)
        await self._events.message_updated(message)
        return 1

    async def _fail_draft(self, draft_id: UUID, user_id: UUID, product_id: str) -> int:
        balance: BalanceEvent | None = None
        async with self._db.session() as session:
            draft = await StudioMemoryDraftRepository(session).finish_failed(
                draft_id, user_id=user_id, product_id=product_id, error_code=ERR_REPLY_TIMEOUT
            )
            if draft is not None:
                balance = await refund_reservation(
                    self._billing,
                    session,
                    job_id=draft.id,
                    price=draft.price_tokens,
                    product_id=product_id,
                    user_id=user_id,
                    reason=ERR_REPLY_TIMEOUT,
                    description="Studio memory draft failed",
                )
        if draft is None:
            return 0
        logger.warning("studio.memory_draft_swept", draft_id=str(draft_id))
        await self._events.balance(balance)
        await self._events.draft_updated(draft)
        return 1
