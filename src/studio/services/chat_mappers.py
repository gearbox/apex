"""ORM row → API DTO mapping for the artist chat."""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.api.schemas.studio_chat import ChatOut, MemoryDraftContentOut, MemoryDraftOut, MessageOut
from src.core.enums import StudioChatMessageStatus, StudioChatRole, StudioMemoryDraftStatus

if TYPE_CHECKING:
    from src.db.models.studio_chat import (
        StudioArtistChat,
        StudioArtistChatMessage,
        StudioMemoryDraft,
    )

__all__ = ["chat_out", "draft_out", "message_out"]


def chat_out(chat: StudioArtistChat) -> ChatOut:
    return ChatOut(
        id=chat.id,
        artist_id=chat.artist_id,
        title=chat.title,
        message_count=chat.message_count,
        created_at=chat.created_at,
        updated_at=chat.updated_at,
    )


def message_out(message: StudioArtistChatMessage) -> MessageOut:
    return MessageOut(
        id=message.id,
        chat_id=message.chat_id,
        position=message.position,
        role=StudioChatRole(message.role),
        status=StudioChatMessageStatus(message.status),
        content=message.content,
        truncated=message.truncated,
        memory_ids_used=list(message.memory_ids_used),
        error_code=message.error_code,
        revision=message.revision,
        created_at=message.created_at,
        finished_at=message.finished_at,
    )


def draft_out(draft: StudioMemoryDraft) -> MemoryDraftOut:
    content = draft.draft
    return MemoryDraftOut(
        id=draft.id,
        chat_id=draft.chat_id,
        source_message_id=draft.source_message_id,
        status=StudioMemoryDraftStatus(draft.status),
        draft=(
            MemoryDraftContentOut(
                title=content["title"],
                body=content["body"],
                themes=list(content["themes"]),
                imagery=list(content["imagery"]),
            )
            if content is not None
            else None
        ),
        error_code=draft.error_code,
        revision=draft.revision,
        created_at=draft.created_at,
        finished_at=draft.finished_at,
    )
