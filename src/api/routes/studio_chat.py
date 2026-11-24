"""Studio artist chat - "interview your artist" (``/v1/studio``, contract §13).

Endpoints:
  GET    /v1/studio/artists/{artist_id}/chats        - the artist's chats, most recently active first
  POST   /v1/studio/artists/{artist_id}/chats        - start a chat (201)
  GET    /v1/studio/chats/{chat_id}                  - one chat
  PATCH  /v1/studio/chats/{chat_id}                  - rename
  DELETE /v1/studio/chats/{chat_id}                  - delete (204; 409 while a reply is pending)
  GET    /v1/studio/chats/{chat_id}/messages         - messages, newest first (``before`` = position)
  POST   /v1/studio/chats/{chat_id}/messages         - send a message: 202, the reply streams over SSE
  GET    /v1/studio/chat-messages/{message_id}       - one message (read the stored reply here)
  POST   /v1/studio/chat-messages/{message_id}/memory-draft - draft a memory from a message (202)
  GET    /v1/studio/memory-drafts/{draft_id}         - one memory draft

Guarded by the studio feature, then by ``STUDIO_CHAT_ENABLED`` (the chat's own switch), then by
authentication; all answer 404 while either switch is off.

The reply is **not** in the POST response: it is written by a task and streamed to the owner over
the existing per-user SSE stream as ``studio.chat.delta`` ... ``studio.chat.message.updated``. A
client that disconnects loses nothing - the reply completes, is charged and is read with ``GET``.
Chat text goes to the configured LLM provider and our Redis (transient) and nowhere else.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated
from uuid import UUID

from litestar import Controller, delete, get, patch, post
from litestar.di import Provide
from litestar.openapi.datastructures import ResponseSpec
from litestar.params import Parameter
from litestar.status_codes import HTTP_201_CREATED, HTTP_202_ACCEPTED, HTTP_204_NO_CONTENT
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.dependencies.auth import get_current_user_id
from src.api.schemas.errors import ErrorEnvelope
from src.api.schemas.pagination import CursorPage
from src.api.schemas.studio_chat import (
    ChatCreate,
    ChatOut,
    ChatPatch,
    MemoryDraftOut,
    MessageCreate,
    MessageOut,
    SendOut,
)
from src.api.security import auth_guard
from src.api.security.studio import studio_chat_guard, studio_feature_guard
from src.studio.services.chat import ChatService
from src.studio.services.chat_mappers import draft_out, message_out
from src.studio.services.memory_draft import MemoryDraftService

if TYPE_CHECKING:
    from collections.abc import Sequence

_ERRORS = {
    400: ResponseSpec(
        data_container=ErrorEnvelope,
        description="idempotency_key_required (memory-draft) or invalid_cursor.",
    ),
    402: ResponseSpec(data_container=ErrorEnvelope, description="Insufficient token balance."),
    404: ResponseSpec(
        data_container=ErrorEnvelope,
        description=(
            "artist_not_found, chat_not_found, message_not_found or memory_draft_not_found "
            "(or the artist chat is disabled)."
        ),
    ),
    409: ResponseSpec(
        data_container=ErrorEnvelope,
        description=(
            "chat_limit_reached, chat_full, chat_reply_in_progress, chat_busy, "
            "memory_draft_in_progress or idempotency_conflict."
        ),
    ),
    422: ResponseSpec(
        data_container=ErrorEnvelope,
        description="document_invalid (a title or message) or message_not_complete.",
    ),
    429: ResponseSpec(data_container=ErrorEnvelope, description="chat_rate_limited."),
}


class StudioChatController(Controller):
    """Artist chats, messages and memory drafts."""

    path = "/v1/studio"
    tags: Sequence[str] | None = ("Studio",)
    guards = [studio_feature_guard, studio_chat_guard, auth_guard]  # noqa: RUF012
    dependencies = {"current_user_id": Provide(get_current_user_id)}  # noqa: RUF012

    # -- chats -------------------------------------------------------------------------------

    @get("/artists/{artist_id:uuid}/chats", responses=_ERRORS)
    async def list_chats(
        self,
        current_user_id: UUID,
        product_id: str,
        artist_id: UUID,
        studio_chat_service: ChatService,
        limit: Annotated[int, Parameter(ge=1, le=50)] = 30,
        cursor: str | None = None,
    ) -> CursorPage[ChatOut]:
        """The artist's chats, most recently active first."""
        return await studio_chat_service.list_page(
            artist_id, current_user_id, product_id, limit=limit, cursor=cursor
        )

    @post("/artists/{artist_id:uuid}/chats", status_code=HTTP_201_CREATED, responses=_ERRORS)
    async def create_chat(
        self,
        current_user_id: UUID,
        product_id: str,
        artist_id: UUID,
        data: ChatCreate,
        studio_chat_service: ChatService,
    ) -> ChatOut:
        """Start a chat with the artist. An artist holds at most 50 chats (409 on the 51st)."""
        return await studio_chat_service.create(artist_id, current_user_id, product_id, data)

    @get("/chats/{chat_id:uuid}", responses=_ERRORS)
    async def get_chat(
        self,
        current_user_id: UUID,
        product_id: str,
        chat_id: UUID,
        studio_chat_service: ChatService,
    ) -> ChatOut:
        """One chat."""
        return await studio_chat_service.get(chat_id, current_user_id, product_id)

    @patch("/chats/{chat_id:uuid}", responses=_ERRORS)
    async def patch_chat(
        self,
        current_user_id: UUID,
        product_id: str,
        chat_id: UUID,
        data: ChatPatch,
        studio_chat_service: ChatService,
    ) -> ChatOut:
        """Rename a chat."""
        return await studio_chat_service.patch(chat_id, current_user_id, product_id, data)

    @delete("/chats/{chat_id:uuid}", status_code=HTTP_204_NO_CONTENT, responses=_ERRORS)
    async def delete_chat(
        self,
        current_user_id: UUID,
        product_id: str,
        chat_id: UUID,
        studio_chat_service: ChatService,
    ) -> None:
        """Delete a chat with its messages and memory drafts (409 while one is pending)."""
        await studio_chat_service.delete(chat_id, current_user_id, product_id)

    # -- messages ----------------------------------------------------------------------------

    @get("/chats/{chat_id:uuid}/messages", responses=_ERRORS)
    async def list_messages(
        self,
        current_user_id: UUID,
        product_id: str,
        chat_id: UUID,
        studio_chat_service: ChatService,
        before: Annotated[int | None, Parameter(ge=1)] = None,
        limit: Annotated[int, Parameter(ge=1, le=50)] = 30,
    ) -> CursorPage[MessageOut]:
        """Messages newest first by ``position``. ``before`` is a position: only older messages
        are returned; a page's ``next_cursor`` is the value to pass as ``before``."""
        return await studio_chat_service.list_messages(
            chat_id, current_user_id, product_id, before=before, limit=limit
        )

    @post("/chats/{chat_id:uuid}/messages", status_code=HTTP_202_ACCEPTED, responses=_ERRORS)
    async def send_message(
        self,
        current_user_id: UUID,
        product_id: str,
        chat_id: UUID,
        data: MessageCreate,
        session: AsyncSession,
        studio_chat_service: ChatService,
    ) -> SendOut:
        """Send a message to the artist (at most 4,000 characters; ``client_message_id`` makes a
        retry idempotent).

        Answers ``202`` with the user's message and the artist's ``pending`` reply; the reply streams
        over SSE (``studio.chat.delta``, then ``studio.chat.message.updated``) and is read with
        ``GET /v1/studio/chat-messages/{id}``. The price is reserved now and refunded when there is
        no reply. The message and the artist's context go to the configured LLM provider.
        """
        write = await studio_chat_service.send(chat_id, current_user_id, product_id, data)
        # Durable first: the events tell the client to fetch rows that must already exist, and the
        # reply task finishes its row in a transaction of its own.
        await session.commit()
        await studio_chat_service.dispatch(write)
        return SendOut(
            user_message=message_out(write.user_message),
            artist_message=message_out(write.artist_message),
        )

    @get("/chat-messages/{message_id:uuid}", responses=_ERRORS)
    async def get_message(
        self,
        current_user_id: UUID,
        product_id: str,
        message_id: UUID,
        studio_chat_service: ChatService,
    ) -> MessageOut:
        """One message: status, revision and - once ``complete`` - the stored text."""
        return await studio_chat_service.get_message(message_id, current_user_id, product_id)

    # -- memory drafts -----------------------------------------------------------------------

    @post(
        "/chat-messages/{message_id:uuid}/memory-draft",
        status_code=HTTP_202_ACCEPTED,
        responses=_ERRORS,
    )
    async def create_memory_draft(
        self,
        current_user_id: UUID,
        product_id: str,
        message_id: UUID,
        session: AsyncSession,
        studio_memory_draft_service: MemoryDraftService,
        idempotency_key_header: Annotated[
            str | None,
            Parameter(
                header="Idempotency-Key",
                description=(
                    "Required (1-128 characters). The same key for the same message returns the "
                    "same draft; the same key for another message is 409."
                ),
            ),
        ] = None,
    ) -> MemoryDraftOut:
        """Turn one complete message (either role) into a draft memory. Nothing is saved: the
        client edits the draft and creates the memory with ``from_draft_id``.

        Answers ``202`` with the ``pending`` draft; ``studio.memory_draft.updated`` announces the
        result and ``GET /v1/studio/memory-drafts/{id}`` reads it. The message text goes to the
        configured LLM provider. Priced like a message; refunded when there is no draft.
        """
        write = await studio_memory_draft_service.create(
            message_id,
            idempotency_key=idempotency_key_header,
            user_id=current_user_id,
            product_id=product_id,
        )
        await session.commit()
        await studio_memory_draft_service.dispatch(write)
        return draft_out(write.draft)

    @get("/memory-drafts/{draft_id:uuid}", responses=_ERRORS)
    async def get_memory_draft(
        self,
        current_user_id: UUID,
        product_id: str,
        draft_id: UUID,
        studio_memory_draft_service: MemoryDraftService,
    ) -> MemoryDraftOut:
        """One memory draft."""
        return await studio_memory_draft_service.get(draft_id, current_user_id, product_id)


__all__ = ["StudioChatController"]
