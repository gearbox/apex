"""The chat's outbound port: how the asynchronous side tells the owner what happened.

Implemented by ``src/studio/services/chat_events.py`` over the event bus. Every method is
best-effort UX - the row is already durable - and **never raises**. Payloads carry ids, a status
and a revision; only ``delta`` carries text (the reply being written), on the owner's channel
only.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from uuid import UUID

    from src.api.services.billing import BalanceEvent
    from src.db.models.studio_chat import StudioArtistChatMessage, StudioMemoryDraft

__all__ = ["ChatEvents"]


class ChatEvents(Protocol):
    """Post-commit notifications of the chat."""

    async def delta(
        self, *, user_id: UUID, chat_id: UUID, message_id: UUID, seq: int, text: str
    ) -> None:
        """A piece of a reply being written (``studio.chat.delta``)."""
        ...

    async def message_updated(self, message: StudioArtistChatMessage) -> None:
        """An artist message changed status (``studio.chat.message.updated``)."""
        ...

    async def draft_updated(self, draft: StudioMemoryDraft) -> None:
        """A memory draft changed status (``studio.memory_draft.updated``)."""
        ...

    async def balance(self, event: BalanceEvent | None) -> None:
        """A ledger change built by ``BillingService`` (a no-op on ``None``)."""
        ...
