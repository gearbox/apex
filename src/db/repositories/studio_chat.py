"""Repositories for the studio artist chat (S1-P2c).

Conventions (the studio's, P1-9): every query filters ``product_id`` **and** ``user_id`` on the row
itself (the composite owner FKs make a mismatch with the parent unrepresentable, the filters make a
foreign id read nothing); writes are Core ``insert/update/delete … RETURNING``; repositories never
``flush()`` or ``commit()``; locking reads use ``populate_existing``.

Lock order: **artist → chat → message** (never the reverse). ``StudioChatRepository.lock`` takes
the chat row only; the artist lock is the caller's, taken first. The per-user quota lock
(``lock_user_quota``) is taken after the chat lock and nothing takes it before one.

Every state change of a reply or a draft is one guarded ``UPDATE … WHERE status = 'pending'
RETURNING`` that bumps ``revision``; ``None`` means the row had already left ``pending`` (the other
side of the task/sweep race won), which is how a row stays *terminal once* - and how a refund is
issued exactly once: only the caller whose update hit the row refunds.
"""

from __future__ import annotations

from datetime import timedelta
from typing import TYPE_CHECKING, Any

from sqlalchemy import delete, func, select, text, update
from sqlalchemy.dialects.postgresql import insert as pg_insert

from src.core.enums import StudioChatMessageStatus, StudioChatRole, StudioMemoryDraftStatus
from src.core.uid import new_id
from src.db.models.studio_chat import (
    StudioArtistChat,
    StudioArtistChatMessage,
    StudioMemoryDraft,
)
from src.db.repositories.studio import _FRESH, _keyset

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from datetime import datetime
    from uuid import UUID

    from sqlalchemy.ext.asyncio import AsyncSession

__all__ = [
    "StudioChatMessageRepository",
    "StudioChatRepository",
    "StudioMemoryDraftRepository",
]

_CHAT_QUOTA_LOCK_NAMESPACE = 0x57_0C_4A
"""Advisory-lock namespace of the per-user chat quota (first key of ``pg_advisory_xact_lock``)."""

_PENDING_MESSAGE = StudioChatMessageStatus.PENDING.value
_PENDING_DRAFT = StudioMemoryDraftStatus.PENDING.value


class StudioChatRepository:
    """Data access for ``studio_artist_chats``."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    @staticmethod
    def _owned(chat_id: UUID, user_id: UUID, product_id: str) -> tuple[Any, ...]:
        return (
            StudioArtistChat.id == chat_id,
            StudioArtistChat.user_id == user_id,
            StudioArtistChat.product_id == product_id,
        )

    async def insert(
        self, *, user_id: UUID, product_id: str, artist_id: UUID, title: str
    ) -> StudioArtistChat:
        """Insert an empty chat. ``user_id`` / ``product_id`` are the **locked artist's**; the
        composite FK refuses a pair that is not the artist's."""
        result = await self._session.execute(
            pg_insert(StudioArtistChat)
            .values(
                id=new_id(),
                user_id=user_id,
                product_id=product_id,
                artist_id=artist_id,
                title=title,
            )
            .returning(StudioArtistChat)
            .execution_options(**_FRESH)
        )
        return result.scalar_one()

    async def get(
        self, chat_id: UUID, *, user_id: UUID, product_id: str
    ) -> StudioArtistChat | None:
        result = await self._session.execute(
            select(StudioArtistChat)
            .where(*self._owned(chat_id, user_id, product_id))
            .execution_options(**_FRESH)
        )
        return result.scalar_one_or_none()

    async def lock(
        self, chat_id: UUID, *, user_id: UUID, product_id: str
    ) -> StudioArtistChat | None:
        """``SELECT … FOR UPDATE`` the chat: serializes sends, deletes and the counters."""
        result = await self._session.execute(
            select(StudioArtistChat)
            .where(*self._owned(chat_id, user_id, product_id))
            .with_for_update()
            .execution_options(**_FRESH)
        )
        return result.scalar_one_or_none()

    async def count_for_artist(self, artist_id: UUID, *, user_id: UUID, product_id: str) -> int:
        result = await self._session.execute(
            select(func.count())
            .select_from(StudioArtistChat)
            .where(
                StudioArtistChat.artist_id == artist_id,
                StudioArtistChat.user_id == user_id,
                StudioArtistChat.product_id == product_id,
            )
        )
        return result.scalar_one()

    async def list_page(
        self,
        artist_id: UUID,
        *,
        user_id: UUID,
        product_id: str,
        limit: int,
        cursor_ts: datetime | None,
        cursor_id: UUID | None,
    ) -> Sequence[StudioArtistChat]:
        """Most recently active first (``updated_at DESC, id DESC``): ``limit + 1`` rows."""
        query = select(StudioArtistChat).where(
            StudioArtistChat.artist_id == artist_id,
            StudioArtistChat.user_id == user_id,
            StudioArtistChat.product_id == product_id,
        )
        result = await self._session.execute(
            _keyset(
                query, StudioArtistChat.updated_at, StudioArtistChat.id, cursor_ts, cursor_id, limit
            ).execution_options(**_FRESH)
        )
        return result.scalars().all()

    async def update_title(
        self, chat_id: UUID, *, user_id: UUID, product_id: str, title: str
    ) -> StudioArtistChat | None:
        result = await self._session.execute(
            update(StudioArtistChat)
            .where(*self._owned(chat_id, user_id, product_id))
            # A rename is not activity: the explicit value keeps the column's ``onupdate`` from firing.
            .values(title=title, updated_at=StudioArtistChat.updated_at)
            .returning(StudioArtistChat)
            .execution_options(**_FRESH, synchronize_session=False)
        )
        return result.scalar_one_or_none()

    async def add_messages(
        self, chat_id: UUID, *, user_id: UUID, product_id: str, count: int
    ) -> StudioArtistChat | None:
        """Account for ``count`` new messages: the counters move and the chat becomes the most
        recently active. The caller holds the chat lock and has checked the cap."""
        result = await self._session.execute(
            update(StudioArtistChat)
            .where(*self._owned(chat_id, user_id, product_id))
            .values(
                message_count=StudioArtistChat.message_count + count,
                next_position=StudioArtistChat.next_position + count,
                updated_at=func.now(),
            )
            .returning(StudioArtistChat)
            .execution_options(**_FRESH, synchronize_session=False)
        )
        return result.scalar_one_or_none()

    async def delete(self, chat_id: UUID, *, user_id: UUID, product_id: str) -> bool:
        """Delete the chat; its messages and drafts cascade. ``False`` when it is not the caller's."""
        result = await self._session.execute(
            delete(StudioArtistChat)
            .where(*self._owned(chat_id, user_id, product_id))
            .returning(StudioArtistChat.id)
            .execution_options(synchronize_session=False)
        )
        return result.first() is not None

    async def lock_user_quota(self, user_id: UUID) -> None:
        """Serialize the per-user limit checks of concurrent sends and draft requests.

        Released at commit. Taken after the chat lock; nothing takes it before one, so it joins no
        cycle. A hash collision only serializes two unrelated users.
        """
        await self._session.execute(
            text("SELECT pg_advisory_xact_lock(:ns, hashtext(:uid))"),
            {"ns": _CHAT_QUOTA_LOCK_NAMESPACE, "uid": str(user_id)},
        )


class StudioChatMessageRepository:
    """Data access for ``studio_artist_chat_messages``."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    @staticmethod
    def _scope(user_id: UUID, product_id: str) -> tuple[Any, ...]:
        return (
            StudioArtistChatMessage.user_id == user_id,
            StudioArtistChatMessage.product_id == product_id,
        )

    # -- create ------------------------------------------------------------------------------

    async def insert_user_message(
        self,
        *,
        user_id: UUID,
        product_id: str,
        chat_id: UUID,
        position: int,
        client_message_id: UUID,
        content: str,
    ) -> StudioArtistChatMessage:
        """Insert a ``complete`` user message (born finished, never changes)."""
        result = await self._session.execute(
            pg_insert(StudioArtistChatMessage)
            .values(
                id=new_id(),
                user_id=user_id,
                product_id=product_id,
                chat_id=chat_id,
                position=position,
                role=StudioChatRole.USER.value,
                status=StudioChatMessageStatus.COMPLETE.value,
                content=content,
                client_message_id=client_message_id,
                finished_at=func.now(),
            )
            .returning(StudioArtistChatMessage)
            .execution_options(**_FRESH)
        )
        return result.scalar_one()

    async def insert_pending_reply(
        self,
        *,
        message_id: UUID,
        user_id: UUID,
        product_id: str,
        chat_id: UUID,
        position: int,
        price_tokens: int,
        model_key: str,
        prompt_version: str,
        memory_ids_used: Sequence[UUID],
        billing_account_id: UUID | None,
    ) -> StudioArtistChatMessage:
        """Insert the ``pending`` artist message that will hold the reply.

        ``message_id`` is chosen by the caller: it is the ledger's ``job_id`` of the reservation.

        Raises:
            sqlalchemy.exc.IntegrityError: The chat already has a pending reply
                (``uq_studio_chat_one_pending_reply``); call inside ``session.begin_nested()``.
        """
        result = await self._session.execute(
            pg_insert(StudioArtistChatMessage)
            .values(
                id=message_id,
                user_id=user_id,
                product_id=product_id,
                chat_id=chat_id,
                position=position,
                role=StudioChatRole.ARTIST.value,
                status=_PENDING_MESSAGE,
                content="",
                model_key=model_key,
                prompt_version=prompt_version,
                memory_ids_used=list(memory_ids_used),
                price_tokens=price_tokens,
                billing_account_id=billing_account_id,
            )
            .returning(StudioArtistChatMessage)
            .execution_options(**_FRESH)
        )
        return result.scalar_one()

    # -- read --------------------------------------------------------------------------------

    async def get(
        self, message_id: UUID, *, user_id: UUID, product_id: str
    ) -> StudioArtistChatMessage | None:
        result = await self._session.execute(
            select(StudioArtistChatMessage)
            .where(StudioArtistChatMessage.id == message_id, *self._scope(user_id, product_id))
            .execution_options(**_FRESH)
        )
        return result.scalar_one_or_none()

    async def get_by_client_id(
        self, chat_id: UUID, client_message_id: UUID, *, user_id: UUID, product_id: str
    ) -> StudioArtistChatMessage | None:
        result = await self._session.execute(
            select(StudioArtistChatMessage)
            .where(
                StudioArtistChatMessage.chat_id == chat_id,
                StudioArtistChatMessage.client_message_id == client_message_id,
                *self._scope(user_id, product_id),
            )
            .execution_options(**_FRESH)
        )
        return result.scalar_one_or_none()

    async def get_at_position(
        self, chat_id: UUID, position: int, *, user_id: UUID, product_id: str
    ) -> StudioArtistChatMessage | None:
        result = await self._session.execute(
            select(StudioArtistChatMessage)
            .where(
                StudioArtistChatMessage.chat_id == chat_id,
                StudioArtistChatMessage.position == position,
                *self._scope(user_id, product_id),
            )
            .execution_options(**_FRESH)
        )
        return result.scalar_one_or_none()

    async def has_pending_reply(self, chat_id: UUID, *, user_id: UUID, product_id: str) -> bool:
        result = await self._session.execute(
            select(StudioArtistChatMessage.id)
            .where(
                StudioArtistChatMessage.chat_id == chat_id,
                StudioArtistChatMessage.role == StudioChatRole.ARTIST.value,
                StudioArtistChatMessage.status == _PENDING_MESSAGE,
                *self._scope(user_id, product_id),
            )
            .limit(1)
        )
        return result.first() is not None

    async def list_page(
        self,
        chat_id: UUID,
        *,
        user_id: UUID,
        product_id: str,
        before: int | None,
        limit: int,
    ) -> Sequence[StudioArtistChatMessage]:
        """Newest first by ``position``; ``before`` is the keyset (``position < before``).
        ``limit + 1`` rows, so ``has_more`` is ``len > limit``."""
        query = select(StudioArtistChatMessage).where(
            StudioArtistChatMessage.chat_id == chat_id, *self._scope(user_id, product_id)
        )
        if before is not None:
            query = query.where(StudioArtistChatMessage.position < before)
        result = await self._session.execute(
            query.order_by(StudioArtistChatMessage.position.desc())
            .limit(limit + 1)
            .execution_options(**_FRESH)
        )
        return result.scalars().all()

    async def complete_history_before(
        self, chat_id: UUID, position: int, *, user_id: UUID, product_id: str, limit: int
    ) -> Sequence[StudioArtistChatMessage]:
        """The chat's ``complete`` messages before ``position``, newest first (failed and pending
        ones are not history: they have no text)."""
        result = await self._session.execute(
            select(StudioArtistChatMessage)
            .where(
                StudioArtistChatMessage.chat_id == chat_id,
                StudioArtistChatMessage.position < position,
                StudioArtistChatMessage.status == StudioChatMessageStatus.COMPLETE.value,
                *self._scope(user_id, product_id),
            )
            .order_by(StudioArtistChatMessage.position.desc())
            .limit(limit)
            .execution_options(**_FRESH)
        )
        return result.scalars().all()

    # -- limits ------------------------------------------------------------------------------

    async def count_pending_replies(self, *, user_id: UUID, product_id: str) -> int:
        result = await self._session.execute(
            select(func.count())
            .select_from(StudioArtistChatMessage)
            .where(
                StudioArtistChatMessage.role == StudioChatRole.ARTIST.value,
                StudioArtistChatMessage.status == _PENDING_MESSAGE,
                *self._scope(user_id, product_id),
            )
        )
        return result.scalar_one()

    async def count_user_messages_in_last_hour(self, *, user_id: UUID, product_id: str) -> int:
        result = await self._session.execute(
            select(func.count())
            .select_from(StudioArtistChatMessage)
            .where(
                StudioArtistChatMessage.role == StudioChatRole.USER.value,
                StudioArtistChatMessage.created_at > func.now() - timedelta(hours=1),
                *self._scope(user_id, product_id),
            )
        )
        return result.scalar_one()

    # -- guarded transitions -----------------------------------------------------------------

    async def finish_complete(
        self,
        message_id: UUID,
        *,
        user_id: UUID,
        product_id: str,
        content: str,
        model_id: str | None,
        prompt_tokens: int,
        completion_tokens: int,
        truncated: bool,
    ) -> StudioArtistChatMessage | None:
        """``pending → complete`` with the reply; ``None`` when the message is no longer pending."""
        result = await self._session.execute(
            update(StudioArtistChatMessage)
            .where(
                StudioArtistChatMessage.id == message_id,
                StudioArtistChatMessage.status == _PENDING_MESSAGE,
                *self._scope(user_id, product_id),
            )
            .values(
                status=StudioChatMessageStatus.COMPLETE.value,
                content=content,
                model_id=model_id,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                truncated=truncated,
                finished_at=func.now(),
                updated_at=func.now(),
                revision=StudioArtistChatMessage.revision + 1,
            )
            .returning(StudioArtistChatMessage)
            .execution_options(**_FRESH, synchronize_session=False)
        )
        return result.scalar_one_or_none()

    async def finish_failed(
        self,
        message_id: UUID,
        *,
        user_id: UUID,
        product_id: str,
        error_code: str,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
    ) -> StudioArtistChatMessage | None:
        """``pending → failed``; ``None`` when the message is no longer pending (the caller then
        does **not** refund: whoever finished it first owns the money)."""
        result = await self._session.execute(
            update(StudioArtistChatMessage)
            .where(
                StudioArtistChatMessage.id == message_id,
                StudioArtistChatMessage.status == _PENDING_MESSAGE,
                *self._scope(user_id, product_id),
            )
            .values(
                status=StudioChatMessageStatus.FAILED.value,
                error_code=error_code,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                finished_at=func.now(),
                updated_at=func.now(),
                revision=StudioArtistChatMessage.revision + 1,
            )
            .returning(StudioArtistChatMessage)
            .execution_options(**_FRESH, synchronize_session=False)
        )
        return result.scalar_one_or_none()

    async def expired_pending(
        self, *, older_than_seconds: int, limit: int
    ) -> Sequence[tuple[UUID, UUID, str]]:
        """``(id, user_id, product_id)`` of artist messages pending for longer than the budget.

        A plain read: the sweep then finishes each row in its own transaction with the guarded
        update, so a row the task finished in the meantime is simply skipped.
        """
        result = await self._session.execute(
            select(
                StudioArtistChatMessage.id,
                StudioArtistChatMessage.user_id,
                StudioArtistChatMessage.product_id,
            )
            .where(
                StudioArtistChatMessage.status == _PENDING_MESSAGE,
                StudioArtistChatMessage.created_at
                < func.now() - timedelta(seconds=older_than_seconds),
            )
            .order_by(StudioArtistChatMessage.created_at)
            .limit(limit)
        )
        return [(row[0], row[1], row[2]) for row in result.all()]


class StudioMemoryDraftRepository:
    """Data access for ``studio_memory_drafts``."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    @staticmethod
    def _scope(user_id: UUID, product_id: str) -> tuple[Any, ...]:
        return (StudioMemoryDraft.user_id == user_id, StudioMemoryDraft.product_id == product_id)

    async def insert(
        self,
        *,
        draft_id: UUID,
        user_id: UUID,
        product_id: str,
        chat_id: UUID,
        source_message_id: UUID,
        idempotency_key: str,
        model_key: str,
        prompt_version: str,
        price_tokens: int,
        billing_account_id: UUID | None,
    ) -> StudioMemoryDraft:
        """Insert a ``pending`` draft; ``draft_id`` is the ledger's ``job_id``.

        Raises:
            sqlalchemy.exc.IntegrityError: The source already has a pending draft
                (``uq_studio_memory_drafts_one_pending``) or the idempotency key is taken
                (``uq_studio_memory_drafts_user_idempotency``); call inside ``begin_nested()``.
        """
        result = await self._session.execute(
            pg_insert(StudioMemoryDraft)
            .values(
                id=draft_id,
                user_id=user_id,
                product_id=product_id,
                chat_id=chat_id,
                source_message_id=source_message_id,
                status=_PENDING_DRAFT,
                idempotency_key=idempotency_key,
                model_key=model_key,
                prompt_version=prompt_version,
                price_tokens=price_tokens,
                billing_account_id=billing_account_id,
            )
            .returning(StudioMemoryDraft)
            .execution_options(**_FRESH)
        )
        return result.scalar_one()

    async def get(
        self, draft_id: UUID, *, user_id: UUID, product_id: str
    ) -> StudioMemoryDraft | None:
        result = await self._session.execute(
            select(StudioMemoryDraft)
            .where(StudioMemoryDraft.id == draft_id, *self._scope(user_id, product_id))
            .execution_options(**_FRESH)
        )
        return result.scalar_one_or_none()

    async def get_by_key(
        self, *, user_id: UUID, product_id: str, idempotency_key: str
    ) -> StudioMemoryDraft | None:
        result = await self._session.execute(
            select(StudioMemoryDraft)
            .where(
                StudioMemoryDraft.idempotency_key == idempotency_key,
                *self._scope(user_id, product_id),
            )
            .execution_options(**_FRESH)
        )
        return result.scalar_one_or_none()

    async def has_pending_for_source(
        self, source_message_id: UUID, *, user_id: UUID, product_id: str
    ) -> bool:
        result = await self._session.execute(
            select(StudioMemoryDraft.id)
            .where(
                StudioMemoryDraft.source_message_id == source_message_id,
                StudioMemoryDraft.status == _PENDING_DRAFT,
                *self._scope(user_id, product_id),
            )
            .limit(1)
        )
        return result.first() is not None

    async def has_pending_for_chat(self, chat_id: UUID, *, user_id: UUID, product_id: str) -> bool:
        result = await self._session.execute(
            select(StudioMemoryDraft.id)
            .where(
                StudioMemoryDraft.chat_id == chat_id,
                StudioMemoryDraft.status == _PENDING_DRAFT,
                *self._scope(user_id, product_id),
            )
            .limit(1)
        )
        return result.first() is not None

    async def count_pending(self, *, user_id: UUID, product_id: str) -> int:
        result = await self._session.execute(
            select(func.count())
            .select_from(StudioMemoryDraft)
            .where(StudioMemoryDraft.status == _PENDING_DRAFT, *self._scope(user_id, product_id))
        )
        return result.scalar_one()

    async def count_in_last_hour(self, *, user_id: UUID, product_id: str) -> int:
        result = await self._session.execute(
            select(func.count())
            .select_from(StudioMemoryDraft)
            .where(
                StudioMemoryDraft.created_at > func.now() - timedelta(hours=1),
                *self._scope(user_id, product_id),
            )
        )
        return result.scalar_one()

    async def finish_complete(
        self,
        draft_id: UUID,
        *,
        user_id: UUID,
        product_id: str,
        draft: Mapping[str, object],
        model_id: str | None,
        prompt_tokens: int,
        completion_tokens: int,
    ) -> StudioMemoryDraft | None:
        """``pending → complete`` with the draft; ``None`` when it is no longer pending."""
        result = await self._session.execute(
            update(StudioMemoryDraft)
            .where(
                StudioMemoryDraft.id == draft_id,
                StudioMemoryDraft.status == _PENDING_DRAFT,
                *self._scope(user_id, product_id),
            )
            .values(
                status=StudioMemoryDraftStatus.COMPLETE.value,
                draft=dict(draft),
                model_id=model_id,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                finished_at=func.now(),
                updated_at=func.now(),
                revision=StudioMemoryDraft.revision + 1,
            )
            .returning(StudioMemoryDraft)
            .execution_options(**_FRESH, synchronize_session=False)
        )
        return result.scalar_one_or_none()

    async def finish_failed(
        self,
        draft_id: UUID,
        *,
        user_id: UUID,
        product_id: str,
        error_code: str,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
    ) -> StudioMemoryDraft | None:
        """``pending → failed``; ``None`` when it is no longer pending (the caller does not refund)."""
        result = await self._session.execute(
            update(StudioMemoryDraft)
            .where(
                StudioMemoryDraft.id == draft_id,
                StudioMemoryDraft.status == _PENDING_DRAFT,
                *self._scope(user_id, product_id),
            )
            .values(
                status=StudioMemoryDraftStatus.FAILED.value,
                error_code=error_code,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                finished_at=func.now(),
                updated_at=func.now(),
                revision=StudioMemoryDraft.revision + 1,
            )
            .returning(StudioMemoryDraft)
            .execution_options(**_FRESH, synchronize_session=False)
        )
        return result.scalar_one_or_none()

    async def expired_pending(
        self, *, older_than_seconds: int, limit: int
    ) -> Sequence[tuple[UUID, UUID, str]]:
        """``(id, user_id, product_id)`` of drafts pending for longer than the budget."""
        result = await self._session.execute(
            select(StudioMemoryDraft.id, StudioMemoryDraft.user_id, StudioMemoryDraft.product_id)
            .where(
                StudioMemoryDraft.status == _PENDING_DRAFT,
                StudioMemoryDraft.created_at < func.now() - timedelta(seconds=older_than_seconds),
            )
            .order_by(StudioMemoryDraft.created_at)
            .limit(limit)
        )
        return [(row[0], row[1], row[2]) for row in result.all()]
