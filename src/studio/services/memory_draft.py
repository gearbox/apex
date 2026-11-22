"""Memory drafts, request side: ask for a draft of one chat message, read it (S1-P2c, C8).

"Turn this into a memory" is an asynchronous, priced LLM job over **one message of a chat**: it
returns a draft the user edits and saves through the normal memory create (``from_draft_id``,
which makes the memory ``origin = chat``). Nothing is written to the artist's memories here
(P2c-6: nothing said in a chat is canon until the user saves it).

The job runs on the chat's task machinery with its own table (P2c-28): reserve in the request, a
task asks the model, then finish or fail + refund, and the sweep fails the orphans. ``studio_runs``
is untouched.

The request is idempotent through a required ``Idempotency-Key`` (``UNIQUE (user_id, key)``): a
retried POST after success must not charge twice. The same key for another source message is
``409 idempotency_conflict``.

Lock order: artist → chat → message; this service locks the message's chat, then the per-user quota.
Logs carry ids, counts and codes only - never a message or a draft.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

import structlog
from sqlalchemy.exc import IntegrityError

from src.core.enums import StudioChatMessageStatus
from src.core.product_registry import get_product_config_by_slug
from src.core.uid import new_id
from src.db.integrity import violated_constraint
from src.db.models.studio_chat import DRAFT_ONE_PENDING_PER_SOURCE, DRAFT_USER_IDEMPOTENCY
from src.db.repositories.studio import StudioArtistRepository
from src.db.repositories.studio_chat import (
    StudioChatMessageRepository,
    StudioChatRepository,
    StudioMemoryDraftRepository,
)
from src.studio.chat.jobs import MemoryDraftJob
from src.studio.chat.prompts.memory_draft_v1 import PROMPT_VERSION, build_draft_messages
from src.studio.errors import (
    ChatNotFound,
    ChatRateLimited,
    IdempotencyConflict,
    IdempotencyKeyRequired,
    MemoryDraftInProgress,
    MemoryDraftNotFound,
    MessageNotComplete,
    MessageNotFound,
)
from src.studio.services.chat_mappers import draft_out
from src.studio.services.mappers import artist_snapshot

if TYPE_CHECKING:
    from uuid import UUID

    from sqlalchemy.ext.asyncio import AsyncSession

    from src.api.schemas.studio_chat import MemoryDraftOut
    from src.api.services.billing import BalanceEvent, BillingService
    from src.core.config import Settings
    from src.db.models.studio_chat import StudioMemoryDraft
    from src.studio.chat.ports import ChatEvents
    from src.studio.chat.supervisor import ChatTaskSupervisor

__all__ = ["DraftWrite", "MemoryDraftService"]

logger = structlog.get_logger(__name__)

_MAX_KEY_LENGTH: Final = 128
_CONSTRAINTS: Final = frozenset({DRAFT_ONE_PENDING_PER_SOURCE, DRAFT_USER_IDEMPOTENCY})


@dataclass(frozen=True, slots=True)
class DraftWrite:
    """The result of a draft request, to be dispatched once the transaction committed."""

    draft: StudioMemoryDraft
    changed: bool
    """``False`` for an idempotent replay: nothing was written, nothing is published or started."""
    balance_event: BalanceEvent | None = None
    job: MemoryDraftJob | None = None


class MemoryDraftService:
    """Request a memory draft of a chat message, and read it (one request transaction)."""

    def __init__(
        self,
        session: AsyncSession,
        *,
        billing: BillingService,
        settings: Settings,
        events: ChatEvents,
        supervisor: ChatTaskSupervisor,
    ) -> None:
        self._session = session
        self._billing = billing
        self._settings = settings
        self._events = events
        self._supervisor = supervisor
        self._artists = StudioArtistRepository(session)
        self._chats = StudioChatRepository(session)
        self._messages = StudioChatMessageRepository(session)
        self._drafts = StudioMemoryDraftRepository(session)

    async def create(
        self,
        message_id: UUID,
        *,
        idempotency_key: str | None,
        user_id: UUID,
        product_id: str,
    ) -> DraftWrite:
        """Accept a draft request (or replay it): check, reserve, insert ``pending``.

        Args:
            message_id: The chat message to turn into a memory (either role).
            idempotency_key: The ``Idempotency-Key`` header value.
            user_id: Owner scope.
            product_id: Product scope.

        Returns:
            The draft and its job; ``changed`` is ``False`` when the key replays an earlier request.

        Raises:
            IdempotencyKeyRequired: No usable key.
            MessageNotFound: The message is not the caller's.
            IdempotencyConflict: The key belongs to a draft of another message.
            MessageNotComplete: The message is still being written, or failed.
            MemoryDraftInProgress: A draft of this message is already being written.
            ChatRateLimited: Too many replies and drafts pending, or too many in the last hour.
            InsufficientBalanceError: The reservation was refused (``402``); nothing is written.
        """
        key = self._valid_key(idempotency_key)
        message = await self._messages.get(message_id, user_id=user_id, product_id=product_id)
        if message is None:
            raise MessageNotFound
        replay = await self._replay(key, message_id, user_id, product_id)
        if replay is not None:
            return replay

        chat = await self._chats.lock(message.chat_id, user_id=user_id, product_id=product_id)
        if chat is None:
            raise MessageNotFound  # the chat was deleted while this request waited
        # The lock may have waited for another request: read again what it could have changed.
        replay = await self._replay(key, message_id, user_id, product_id)
        if replay is not None:
            return replay
        message = await self._messages.get(message_id, user_id=user_id, product_id=product_id)
        if message is None:
            raise MessageNotFound
        if message.status != StudioChatMessageStatus.COMPLETE.value:
            raise MessageNotComplete
        if await self._drafts.has_pending_for_source(
            message_id, user_id=user_id, product_id=product_id
        ):
            raise MemoryDraftInProgress
        await self._check_limits(user_id, product_id)

        artist = await self._artists.get(chat.artist_id, user_id=user_id, product_id=product_id)
        if artist is None:
            raise ChatNotFound
        previous = await self._messages.complete_history_before(
            chat.id, message.position, user_id=user_id, product_id=product_id, limit=1
        )
        context = build_draft_messages(
            artist=artist_snapshot(artist),
            source_role=message.role,
            source_text=message.content,
            previous_role=previous[0].role if previous else None,
            previous_text=previous[0].content if previous else None,
            rating=get_product_config_by_slug(product_id).content_policy.rating,
        )
        model_key = self._settings.studio_memory_draft_model_key
        price = self._settings.studio_memory_draft_price
        try:
            draft, balance_event = await self._insert_and_reserve(
                draft_id=new_id(),
                chat_id=chat.id,
                source_message_id=message_id,
                key=key,
                model_key=model_key,
                price=price,
                user_id=user_id,
                product_id=product_id,
            )
        except IntegrityError as exc:
            constraint = violated_constraint(exc, _CONSTRAINTS)
            if constraint == DRAFT_ONE_PENDING_PER_SOURCE:
                raise MemoryDraftInProgress from None
            if constraint == DRAFT_USER_IDEMPOTENCY:
                replay = await self._replay(key, message_id, user_id, product_id)
                if replay is not None:
                    return replay
            raise
        logger.info(
            "studio.memory_draft_requested",
            draft_id=str(draft.id),
            chat_id=str(chat.id),
            source_message_id=str(message_id),
            user_id=str(user_id),
            model_key=model_key,
            price_tokens=price,
        )
        job = MemoryDraftJob(
            draft_id=draft.id,
            chat_id=chat.id,
            source_message_id=message_id,
            user_id=user_id,
            product_id=product_id,
            model_key=model_key,
            price_tokens=price,
            messages=context,
            deadline_seconds=self._settings.studio_chat_reply_deadline_seconds,
        )
        return DraftWrite(draft=draft, changed=True, balance_event=balance_event, job=job)

    async def _insert_and_reserve(
        self,
        *,
        draft_id: UUID,
        chat_id: UUID,
        source_message_id: UUID,
        key: str,
        model_key: str,
        price: int,
        user_id: UUID,
        product_id: str,
    ) -> tuple[StudioMemoryDraft, BalanceEvent | None]:
        """The draft row and its reservation in one savepoint: both, or neither."""
        async with self._session.begin_nested():
            account_id: UUID | None = None
            if price > 0:
                account_id = (
                    await self._billing.resolve_account_for_user(user_id, session=self._session)
                ).id
            draft = await self._drafts.insert(
                draft_id=draft_id,
                user_id=user_id,
                product_id=product_id,
                chat_id=chat_id,
                source_message_id=source_message_id,
                idempotency_key=key,
                model_key=model_key,
                prompt_version=PROMPT_VERSION,
                price_tokens=price,
                billing_account_id=account_id,
            )
            balance_event: BalanceEvent | None = None
            if price > 0 and account_id is not None:
                reserved = await self._billing.check_and_reserve(
                    account_id,
                    price,
                    draft_id,
                    metadata={"studio_memory_draft_id": str(draft_id)},
                    session=self._session,
                    product_id=product_id,
                    user_id=user_id,
                    description="Studio memory draft",
                )
                balance_event = reserved.event
        return draft, balance_event

    async def _replay(
        self, key: str, message_id: UUID, user_id: UUID, product_id: str
    ) -> DraftWrite | None:
        """The draft an earlier request with this key created, or ``None``; another message is 409."""
        existing = await self._drafts.get_by_key(
            user_id=user_id, product_id=product_id, idempotency_key=key
        )
        if existing is None:
            return None
        if existing.source_message_id != message_id:
            raise IdempotencyConflict
        return DraftWrite(draft=existing, changed=False)

    async def _check_limits(self, user_id: UUID, product_id: str) -> None:
        """The same per-user limits as sending a message (they share the budget)."""
        await self._chats.lock_user_quota(user_id)
        pending = await self._messages.count_pending_replies(
            user_id=user_id, product_id=product_id
        ) + await self._drafts.count_pending(user_id=user_id, product_id=product_id)
        if pending >= self._settings.studio_chat_max_pending_per_user:
            raise ChatRateLimited
        recent = await self._messages.count_user_messages_in_last_hour(
            user_id=user_id, product_id=product_id
        ) + await self._drafts.count_in_last_hour(user_id=user_id, product_id=product_id)
        if recent >= self._settings.studio_chat_jobs_per_user_per_hour:
            raise ChatRateLimited

    @staticmethod
    def _valid_key(raw: str | None) -> str:
        key = (raw or "").strip()
        if not key or len(key) > _MAX_KEY_LENGTH:
            raise IdempotencyKeyRequired
        return key

    async def get(self, draft_id: UUID, user_id: UUID, product_id: str) -> MemoryDraftOut:
        """One draft - how a client reads it after ``studio.memory_draft.updated``.

        Raises:
            MemoryDraftNotFound: No such draft for the caller.
        """
        draft = await self._drafts.get(draft_id, user_id=user_id, product_id=product_id)
        if draft is None:
            raise MemoryDraftNotFound
        return draft_out(draft)

    async def dispatch(self, write: DraftWrite) -> None:
        """Publish what ``write`` changed and start the draft task. Call strictly after the
        transaction committed."""
        if not write.changed or write.job is None:
            return
        await self._events.balance(write.balance_event)
        await self._events.draft_updated(write.draft)
        self._supervisor.start_memory_draft(write.job)
