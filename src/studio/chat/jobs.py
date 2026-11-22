"""What the asynchronous side of the chat is handed: plain data, no session, no connection.

A job is assembled **inside the request transaction** (consistent reads) and then travels in memory
to the supervisor's task, which opens short transactions of its own only to finish or fail the
row. Jobs hold message text: they are never logged and never put in an exception.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from uuid import UUID

    from src.llm.port import LlmMessage

__all__ = ["ChatReplyJob", "MemoryDraftJob"]


@dataclass(frozen=True, slots=True)
class ChatReplyJob:
    """One artist reply to write."""

    message_id: UUID
    """The pending artist message; also the ledger's ``job_id`` of the reservation."""
    chat_id: UUID
    user_id: UUID
    product_id: str
    model_key: str
    price_tokens: int
    messages: tuple[LlmMessage, ...]
    """The system message, then the history window, the new user message last."""
    memory_ids_used: tuple[UUID, ...]
    max_output_tokens: int
    deadline_seconds: int


@dataclass(frozen=True, slots=True)
class MemoryDraftJob:
    """One memory draft to write."""

    draft_id: UUID
    """The pending draft; also the ledger's ``job_id`` of the reservation."""
    chat_id: UUID
    source_message_id: UUID
    user_id: UUID
    product_id: str
    model_key: str
    price_tokens: int
    messages: tuple[LlmMessage, ...]
    deadline_seconds: int
