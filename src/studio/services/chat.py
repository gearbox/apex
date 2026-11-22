"""Artist chats, request side: CRUD, messages, and sending a message (S1-P2c).

A chat is a conversation with one artist ("interview your artist"). This service is request-scoped
and **commit-free**: the route commits, then calls :meth:`ChatService.dispatch`, which publishes the
events and hands the reply to the task supervisor. The reply itself is written by an asyncio task
(``src/studio/chat/supervisor.py``) - never in a request.

**Lock order is artist → chat → message, never the reverse.** ``create`` locks the artist (and counts
the chats under that lock: the 50-per-artist cap is hard); ``send`` and ``delete`` lock the chat; a
per-user advisory lock for the limit checks is taken after the chat lock.

Sending a message (``send``) is one transaction: lock the chat, replay a known ``client_message_id``,
refuse a second reply in flight / a full chat / a user over their limits, assemble the context, write
**both messages** (the user's, ``complete``; the artist's, ``pending``) and the reservation in one
savepoint - an insufficient balance leaves nothing behind - and return the job for the task.

Logs carry ids, lengths and counts only - never a title or a message.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

import msgspec
import structlog
from sqlalchemy.exc import IntegrityError

from src.api.schemas.pagination import CursorPage, decode_cursor, encode_cursor
from src.core.enums import StudioLlmUsageKind
from src.core.uid import new_id
from src.db.integrity import violated_constraint
from src.db.models.studio_chat import CHAT_ONE_PENDING_REPLY, MESSAGE_CLIENT_ID_UNIQUE
from src.db.repositories.studio import StudioArtistRepository, StudioLlmUsageRepository
from src.db.repositories.studio_chat import (
    StudioChatMessageRepository,
    StudioChatRepository,
    StudioMemoryDraftRepository,
)
from src.studio.chat.jobs import ChatReplyJob
from src.studio.chat.prompts.chat_v1 import PROMPT_VERSION
from src.studio.constraints import (
    MAX_CHATS_PER_ARTIST,
    MAX_MESSAGES_PER_CHAT,
    collect_length_issues,
)
from src.studio.errors import (
    ArtistNotFound,
    ChatBusy,
    ChatFull,
    ChatLimitReached,
    ChatNotFound,
    ChatRateLimited,
    ChatReplyInProgress,
    DocumentIssue,
    IdempotencyConflict,
    InvalidCursor,
    MessageNotFound,
    StudioDocumentInvalid,
)
from src.studio.services.chat_mappers import chat_out, message_out
from src.studio.validation import normalize_block, normalize_text

if TYPE_CHECKING:
    from uuid import UUID

    from sqlalchemy.ext.asyncio import AsyncSession

    from src.api.schemas.studio_chat import (
        ChatCreate,
        ChatOut,
        ChatPatch,
        MessageCreate,
        MessageOut,
    )
    from src.api.services.billing import BalanceEvent, BillingService
    from src.core.config import Settings
    from src.db.models.studio_chat import StudioArtistChatMessage
    from src.studio.chat.ports import ChatEvents
    from src.studio.chat.supervisor import ChatTaskSupervisor
    from src.studio.services.chat_context import ArtistChatContextAssembler

__all__ = ["ChatService", "ChatWrite"]

logger = structlog.get_logger(__name__)

_EMPTY: Final = "empty_after_normalization"
_CONSTRAINTS: Final = frozenset({CHAT_ONE_PENDING_REPLY, MESSAGE_CLIENT_ID_UNIQUE})


@dataclass(frozen=True, slots=True)
class ChatWrite:
    """The result of a send, to be dispatched once the transaction committed."""

    user_message: StudioArtistChatMessage
    artist_message: StudioArtistChatMessage
    changed: bool
    """``False`` for an idempotent replay: nothing was written, nothing is published or started."""
    balance_event: BalanceEvent | None = None
    job: ChatReplyJob | None = None
    """The reply to write; ``None`` for a replay."""


class ChatService:
    """Chats and messages of the caller's artists (one request transaction; never commits)."""

    def __init__(
        self,
        session: AsyncSession,
        *,
        billing: BillingService,
        settings: Settings,
        events: ChatEvents,
        assembler: ArtistChatContextAssembler,
        supervisor: ChatTaskSupervisor,
    ) -> None:
        self._session = session
        self._billing = billing
        self._settings = settings
        self._events = events
        self._assembler = assembler
        self._supervisor = supervisor
        self._artists = StudioArtistRepository(session)
        self._chats = StudioChatRepository(session)
        self._messages = StudioChatMessageRepository(session)
        self._drafts = StudioMemoryDraftRepository(session)
        self._usage = StudioLlmUsageRepository(session)

    # -- chats -------------------------------------------------------------------------------

    async def create(
        self, artist_id: UUID, user_id: UUID, product_id: str, data: ChatCreate
    ) -> ChatOut:
        """Start a conversation with an artist.

        The artist is locked first and the chats are counted under that lock, so the cap holds
        under concurrent creates.

        Raises:
            ArtistNotFound: The artist is not the caller's.
            StudioDocumentInvalid: The title is empty after normalization, or NFC pushed it past
                its limit (RD6).
            ChatLimitReached: The artist already has 50 chats.
        """
        artist = await self._artists.lock(artist_id, user_id=user_id, product_id=product_id)
        if artist is None:
            raise ArtistNotFound
        issues: list[DocumentIssue] = []
        title = normalize_text(data.title)
        if not title:
            issues.append(DocumentIssue("title", _EMPTY))
        collect_length_issues(msgspec.structs.replace(data, title=title), issues)
        if issues:
            raise StudioDocumentInvalid(tuple(issues))
        if (
            await self._chats.count_for_artist(
                artist_id, user_id=artist.user_id, product_id=artist.product_id
            )
            >= MAX_CHATS_PER_ARTIST
        ):
            raise ChatLimitReached
        # The locked artist row's owner, not the caller's: what the composite FK checks against.
        chat = await self._chats.insert(
            user_id=artist.user_id, product_id=artist.product_id, artist_id=artist_id, title=title
        )
        logger.info(
            "studio.chat_created",
            chat_id=str(chat.id),
            artist_id=str(artist_id),
            user_id=str(user_id),
        )
        return chat_out(chat)

    async def list_page(
        self,
        artist_id: UUID,
        user_id: UUID,
        product_id: str,
        *,
        limit: int,
        cursor: str | None,
    ) -> CursorPage[ChatOut]:
        """The artist's chats, most recently active first (``updated_at DESC, id DESC``).

        Raises:
            InvalidCursor: ``cursor`` is not one this endpoint issued.
            ArtistNotFound: The artist is not the caller's.
        """
        try:
            cursor_ts, cursor_id = decode_cursor(cursor) if cursor is not None else (None, None)
        except ValueError as exc:
            raise InvalidCursor from exc
        if await self._artists.get(artist_id, user_id=user_id, product_id=product_id) is None:
            raise ArtistNotFound
        rows = await self._chats.list_page(
            artist_id,
            user_id=user_id,
            product_id=product_id,
            limit=limit,
            cursor_ts=cursor_ts,
            cursor_id=cursor_id,
        )
        has_more = len(rows) > limit
        page = list(rows[:limit])
        return CursorPage(
            items=[chat_out(c) for c in page],
            limit=limit,
            has_more=has_more,
            next_cursor=encode_cursor(page[-1].updated_at, page[-1].id)
            if has_more and page
            else None,
        )

    async def get(self, chat_id: UUID, user_id: UUID, product_id: str) -> ChatOut:
        """One chat.

        Raises:
            ChatNotFound: No such chat for the caller.
        """
        chat = await self._chats.get(chat_id, user_id=user_id, product_id=product_id)
        if chat is None:
            raise ChatNotFound
        return chat_out(chat)

    async def patch(
        self, chat_id: UUID, user_id: UUID, product_id: str, data: ChatPatch
    ) -> ChatOut:
        """Rename a chat. The title is the only editable field; the chat keeps its place in the
        list (a rename is not activity).

        Raises:
            ChatNotFound: No such chat for the caller.
            StudioDocumentInvalid: The title is empty after normalization, or too long (RD6).
        """
        issues: list[DocumentIssue] = []
        title = normalize_text(data.title)
        if not title:
            issues.append(DocumentIssue("title", _EMPTY))
        collect_length_issues(msgspec.structs.replace(data, title=title), issues)
        if issues:
            raise StudioDocumentInvalid(tuple(issues))
        chat = await self._chats.update_title(
            chat_id, user_id=user_id, product_id=product_id, title=title
        )
        if chat is None:
            raise ChatNotFound
        return chat_out(chat)

    async def delete(self, chat_id: UUID, user_id: UUID, product_id: str) -> None:
        """Delete a chat; its messages and memory drafts go with it.

        Refused while work is in flight: a pending reply or memory draft has been paid for and a
        task is about to write it.

        Raises:
            ChatNotFound: No such chat for the caller.
            ChatBusy: A reply or a memory draft of this chat is pending.
        """
        chat = await self._chats.lock(chat_id, user_id=user_id, product_id=product_id)
        if chat is None:
            raise ChatNotFound
        if await self._messages.has_pending_reply(
            chat_id, user_id=user_id, product_id=product_id
        ) or await self._drafts.has_pending_for_chat(
            chat_id, user_id=user_id, product_id=product_id
        ):
            raise ChatBusy
        await self._chats.delete(chat_id, user_id=user_id, product_id=product_id)
        logger.info("studio.chat_deleted", chat_id=str(chat_id), user_id=str(user_id))

    # -- messages ----------------------------------------------------------------------------

    async def list_messages(
        self,
        chat_id: UUID,
        user_id: UUID,
        product_id: str,
        *,
        before: int | None,
        limit: int,
    ) -> CursorPage[MessageOut]:
        """The chat's messages, newest first by ``position``.

        ``before`` is the keyset: only messages with a smaller ``position``. The page's
        ``next_cursor`` is the last item's position, to be passed as ``before``.

        Raises:
            ChatNotFound: No such chat for the caller.
        """
        if await self._chats.get(chat_id, user_id=user_id, product_id=product_id) is None:
            raise ChatNotFound
        rows = await self._messages.list_page(
            chat_id, user_id=user_id, product_id=product_id, before=before, limit=limit
        )
        has_more = len(rows) > limit
        page = list(rows[:limit])
        return CursorPage(
            items=[message_out(m) for m in page],
            limit=limit,
            has_more=has_more,
            next_cursor=str(page[-1].position) if has_more and page else None,
        )

    async def get_message(self, message_id: UUID, user_id: UUID, product_id: str) -> MessageOut:
        """One message - how a client reads a reply (the stored text) after ``complete``.

        Raises:
            MessageNotFound: No such message for the caller.
        """
        message = await self._messages.get(message_id, user_id=user_id, product_id=product_id)
        if message is None:
            raise MessageNotFound
        return message_out(message)

    async def send(
        self, chat_id: UUID, user_id: UUID, product_id: str, data: MessageCreate
    ) -> ChatWrite:
        """Send a message: write the pair (the user's, the artist's ``pending``), reserve, and
        assemble the context - one transaction; the reply itself is written by a task.

        Args:
            chat_id: The chat.
            user_id: Owner scope.
            product_id: Product scope.
            data: The decoded body.

        Returns:
            The pair and the job; ``changed`` is ``False`` when ``client_message_id`` was already
            sent (the stored pair is returned, nothing else happens).

        Raises:
            StudioDocumentInvalid: The message is empty after normalization, or NFC pushed it
                past its limit (RD6).
            ChatNotFound: The chat is not the caller's.
            IdempotencyConflict: ``client_message_id`` was used for a different message.
            ChatReplyInProgress: The previous message has not been answered yet.
            ChatFull: The chat holds 200 messages.
            ChatRateLimited: Too many replies and drafts pending, or too many in the last hour.
            InsufficientBalanceError: The reservation was refused (``402``); nothing is written.
        """
        content = self._normalized_content(data)
        chat = await self._chats.lock(chat_id, user_id=user_id, product_id=product_id)
        if chat is None:
            raise ChatNotFound

        replay = await self._replay(chat_id, data.client_message_id, content, user_id, product_id)
        if replay is not None:
            return replay
        if await self._messages.has_pending_reply(chat_id, user_id=user_id, product_id=product_id):
            raise ChatReplyInProgress
        if chat.message_count + 2 > MAX_MESSAGES_PER_CHAT:
            raise ChatFull
        await self._check_limits(user_id, product_id)

        context = await self._assembler.assemble(
            self._session, chat=chat, new_message=content, new_position=chat.next_position
        )
        price = self._settings.studio_chat_message_price
        model_key = self._settings.studio_chat_model_key(product_id)
        try:
            user_message, artist_message, balance_event = await self._write_pair(
                chat_id=chat_id,
                position=chat.next_position,
                client_message_id=data.client_message_id,
                content=content,
                memory_ids=context.memory_ids_used,
                model_key=model_key,
                price=price,
                user_id=user_id,
                product_id=product_id,
            )
        except IntegrityError as exc:
            constraint = violated_constraint(exc, _CONSTRAINTS)
            if constraint == CHAT_ONE_PENDING_REPLY:
                raise ChatReplyInProgress from None
            if constraint == MESSAGE_CLIENT_ID_UNIQUE:
                replay = await self._replay(
                    chat_id, data.client_message_id, content, user_id, product_id
                )
                if replay is not None:
                    return replay
            raise
        logger.info(
            "studio.chat_message_sent",
            chat_id=str(chat_id),
            user_message_id=str(user_message.id),
            artist_message_id=str(artist_message.id),
            user_id=str(user_id),
            content_length=len(content),
            memory_count=len(context.memory_ids_used),
            model_key=model_key,
            price_tokens=price,
        )
        job = ChatReplyJob(
            message_id=artist_message.id,
            chat_id=chat_id,
            user_id=user_id,
            product_id=product_id,
            model_key=model_key,
            price_tokens=price,
            messages=context.messages,
            memory_ids_used=context.memory_ids_used,
            max_output_tokens=self._settings.studio_chat_max_output_tokens,
            deadline_seconds=self._settings.studio_chat_reply_deadline_seconds,
        )
        return ChatWrite(
            user_message=user_message,
            artist_message=artist_message,
            changed=True,
            balance_event=balance_event,
            job=job,
        )

    @staticmethod
    def _normalized_content(data: MessageCreate) -> str:
        """The message as stored: NFC, trimmed, line breaks kept; RD6 re-checked."""
        issues: list[DocumentIssue] = []
        content = normalize_block(data.content)
        if content is None:
            issues.append(DocumentIssue("content", _EMPTY))
            raise StudioDocumentInvalid(tuple(issues))
        collect_length_issues(msgspec.structs.replace(data, content=content), issues)
        if issues:
            raise StudioDocumentInvalid(tuple(issues))
        return content

    async def _replay(
        self,
        chat_id: UUID,
        client_message_id: UUID,
        content: str,
        user_id: UUID,
        product_id: str,
    ) -> ChatWrite | None:
        """The stored pair of an earlier send with this id, or ``None``; other content is 409."""
        existing = await self._messages.get_by_client_id(
            chat_id, client_message_id, user_id=user_id, product_id=product_id
        )
        if existing is None:
            return None
        if existing.content != content:
            raise IdempotencyConflict
        reply = await self._messages.get_at_position(
            chat_id, existing.position + 1, user_id=user_id, product_id=product_id
        )
        if reply is None:  # a user message is always written with its reply: a broken invariant
            raise RuntimeError("chat user message without its artist message")
        return ChatWrite(user_message=existing, artist_message=reply, changed=False)

    async def _check_limits(self, user_id: UUID, product_id: str) -> None:
        """The per-user limits, counted in the database under the user's quota lock.

        The hourly count reads the usage ledger, not the chat rows: deleting a chat must not reset
        it (R2). Rows that no longer count are purged first, under the same lock.
        """
        await self._chats.lock_user_quota(user_id)
        pending = await self._messages.count_pending_replies(
            user_id=user_id, product_id=product_id
        ) + await self._drafts.count_pending(user_id=user_id, product_id=product_id)
        if pending >= self._settings.studio_chat_max_pending_per_user:
            raise ChatRateLimited
        kinds = StudioLlmUsageKind.chat()
        await self._usage.purge_older_than_hour(user_id=user_id, product_id=product_id, kinds=kinds)
        recent = await self._usage.count_last_hour(
            user_id=user_id, product_id=product_id, kinds=kinds
        )
        if recent >= self._settings.studio_chat_jobs_per_user_per_hour:
            raise ChatRateLimited

    async def _write_pair(
        self,
        *,
        chat_id: UUID,
        position: int,
        client_message_id: UUID,
        content: str,
        memory_ids: tuple[UUID, ...],
        model_key: str,
        price: int,
        user_id: UUID,
        product_id: str,
    ) -> tuple[StudioArtistChatMessage, StudioArtistChatMessage, BalanceEvent | None]:
        """Both messages, the counters and the reservation in one savepoint: all or nothing."""
        async with self._session.begin_nested():
            user_message = await self._messages.insert_user_message(
                user_id=user_id,
                product_id=product_id,
                chat_id=chat_id,
                position=position,
                client_message_id=client_message_id,
                content=content,
            )
            reply_id = new_id()
            account_id: UUID | None = None
            if price > 0:
                account_id = (
                    await self._billing.resolve_account_for_user(user_id, session=self._session)
                ).id
            artist_message = await self._messages.insert_pending_reply(
                message_id=reply_id,
                user_id=user_id,
                product_id=product_id,
                chat_id=chat_id,
                position=position + 1,
                price_tokens=price,
                model_key=model_key,
                prompt_version=PROMPT_VERSION,
                memory_ids_used=memory_ids,
                billing_account_id=account_id,
            )
            balance_event: BalanceEvent | None = None
            if price > 0 and account_id is not None:
                reserved = await self._billing.check_and_reserve(
                    account_id,
                    price,
                    reply_id,
                    metadata={"studio_chat_message_id": str(reply_id)},
                    session=self._session,
                    product_id=product_id,
                    user_id=user_id,
                    description="Studio chat message",
                )
                balance_event = reserved.event
            await self._chats.add_messages(chat_id, user_id=user_id, product_id=product_id, count=2)
            # Accepted: it counts towards the hourly limit however the reply ends, and whatever
            # happens to the chat afterwards. A 402 above rolls this savepoint back with the pair.
            await self._usage.record(
                user_id=user_id, product_id=product_id, kind=StudioLlmUsageKind.CHAT_MESSAGE
            )
        return user_message, artist_message, balance_event

    # -- after commit ------------------------------------------------------------------------

    async def dispatch(self, write: ChatWrite) -> None:
        """Publish what ``write`` changed and start the reply. Call strictly after the transaction
        committed: the events tell the client to fetch rows that must already exist, and the task
        finishes the row with its own transaction."""
        if not write.changed or write.job is None:
            return
        await self._events.balance(write.balance_event)
        await self._events.message_updated(write.artist_message)
        self._supervisor.start_reply(write.job)
