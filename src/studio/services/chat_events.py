"""Publishing chat events - strictly after the transaction that wrote them committed.

Implements :class:`~src.studio.chat.ports.ChatEvents` over the per-user event bus. Delivery is
best-effort UX: the row is already durable (the client reads it with ``GET``), so a Redis failure
is logged and swallowed, never surfaced to the request or the reply task.

Payloads: ids, a status and a revision - except ``studio.chat.delta``, which carries the reply text
(user-visible content). A delta goes to the **owner's channel only**; ``EventBus.publish`` logs no
payload, and neither does this module: a failure logs the ids.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import structlog

from src.api.schemas.events import (
    EventType,
    StudioChatDeltaPayload,
    StudioChatMessageUpdatedPayload,
    StudioMemoryDraftUpdatedPayload,
)

if TYPE_CHECKING:
    from uuid import UUID

    from src.api.services.billing import BalanceEvent
    from src.api.services.event_bus import EventBus
    from src.db.models.studio_chat import StudioArtistChatMessage, StudioMemoryDraft

__all__ = ["StudioChatEvents"]

logger = structlog.get_logger(__name__)


class StudioChatEvents:
    """Post-commit publisher of the chat's events and of the balance events of its jobs."""

    def __init__(self, event_bus: EventBus | None) -> None:
        self._bus = event_bus

    async def delta(
        self, *, user_id: UUID, chat_id: UUID, message_id: UUID, seq: int, text: str
    ) -> None:
        """A piece of a reply being written. Never raises."""
        if self._bus is None:
            return
        try:
            await self._bus.publish(
                user_id=user_id,
                event_type=EventType.STUDIO_CHAT_DELTA,
                payload=StudioChatDeltaPayload(
                    chat_id=chat_id, message_id=message_id, seq=seq, text=text
                ),
            )
        except Exception:
            logger.warning("studio.chat.event_publish_failed", message_id=str(message_id))

    async def message_updated(self, message: StudioArtistChatMessage) -> None:
        """The artist message's current status and revision. Never raises."""
        if self._bus is None:
            return
        try:
            await self._bus.publish(
                user_id=message.user_id,
                event_type=EventType.STUDIO_CHAT_MESSAGE_UPDATED,
                payload=StudioChatMessageUpdatedPayload(
                    chat_id=message.chat_id,
                    message_id=message.id,
                    status=message.status,
                    revision=message.revision,
                ),
            )
        except Exception:
            logger.warning("studio.chat.event_publish_failed", message_id=str(message.id))

    async def draft_updated(self, draft: StudioMemoryDraft) -> None:
        """The draft's current status and revision. Never raises."""
        if self._bus is None:
            return
        try:
            await self._bus.publish(
                user_id=draft.user_id,
                event_type=EventType.STUDIO_MEMORY_DRAFT_UPDATED,
                payload=StudioMemoryDraftUpdatedPayload(
                    draft_id=draft.id,
                    chat_id=draft.chat_id,
                    source_message_id=draft.source_message_id,
                    status=draft.status,
                    revision=draft.revision,
                ),
            )
        except Exception:
            logger.warning("studio.chat.event_publish_failed", draft_id=str(draft.id))

    async def balance(self, event: BalanceEvent | None) -> None:
        """Publish a ledger change built by ``BillingService`` (a no-op on ``None``)."""
        if self._bus is not None:
            await self._bus.publish_balance(event)
