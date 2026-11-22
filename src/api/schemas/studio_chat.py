"""Request / response DTOs of the studio artist chat (``docs/contracts/studio-contract.md`` §13).

Chat text is user data: these structs return it to its owner only, and nothing logs it.
"""

from __future__ import annotations

from datetime import datetime
from typing import Annotated
from uuid import UUID

import msgspec
from msgspec import Meta

from src.core.enums import StudioChatMessageStatus, StudioChatRole, StudioMemoryDraftStatus
from src.studio.constraints import CHAT_MESSAGE_MAX, CHAT_TITLE_MAX

__all__ = [
    "ChatCreate",
    "ChatOut",
    "ChatPatch",
    "MemoryDraftContentOut",
    "MemoryDraftOut",
    "MessageCreate",
    "MessageOut",
    "SendOut",
]

_ChatTitle = Annotated[str, Meta(min_length=1, max_length=CHAT_TITLE_MAX)]
_MessageContent = Annotated[str, Meta(min_length=1, max_length=CHAT_MESSAGE_MAX)]


class ChatCreate(msgspec.Struct, forbid_unknown_fields=True, kw_only=True):
    """A new conversation. The client proposes the title (e.g. the first question)."""

    title: _ChatTitle


class ChatPatch(msgspec.Struct, forbid_unknown_fields=True, kw_only=True):
    """Rename a chat (the only editable field)."""

    title: _ChatTitle


class ChatOut(msgspec.Struct, kw_only=True):
    id: UUID
    artist_id: UUID
    title: str
    message_count: int
    created_at: datetime
    updated_at: datetime
    """The last time a message was sent in the chat (a rename does not change it)."""


class MessageCreate(msgspec.Struct, forbid_unknown_fields=True, kw_only=True):
    """One question to the artist."""

    client_message_id: UUID
    """The client's id for this message: sending it again (a retry) returns the same pair."""
    content: _MessageContent


class MessageOut(msgspec.Struct, kw_only=True):
    """A chat message. ``content`` is ``""`` for an artist message until it is ``complete``."""

    id: UUID
    chat_id: UUID
    position: int
    role: StudioChatRole
    status: StudioChatMessageStatus
    content: str
    truncated: bool
    """The reply was cut at the output limit; it is complete as far as it goes."""
    memory_ids_used: list[UUID]
    """Artist memories the reply was given, in prompt order: ids only, never their text."""
    error_code: str | None
    revision: int
    """Strictly increases with every status change (see ``studio.chat.message.updated``)."""
    created_at: datetime
    finished_at: datetime | None


class SendOut(msgspec.Struct, kw_only=True):
    """The pair a send creates: the user's message and the artist's pending reply."""

    user_message: MessageOut
    artist_message: MessageOut


class MemoryDraftContentOut(msgspec.Struct, kw_only=True):
    """The fields of a draft memory - what the client shows, edits and sends to the memory create."""

    title: str
    body: str
    themes: list[str]
    imagery: list[str]


class MemoryDraftOut(msgspec.Struct, kw_only=True):
    id: UUID
    chat_id: UUID
    source_message_id: UUID
    status: StudioMemoryDraftStatus
    draft: MemoryDraftContentOut | None
    """``None`` until the draft is ``complete``."""
    error_code: str | None
    revision: int
    created_at: datetime
    finished_at: datetime | None
