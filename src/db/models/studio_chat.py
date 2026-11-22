"""Database models of the studio artist chat (S1-P2c, migration 060).

Graph::

    studio_artists ─< studio_artist_chats ─< studio_artist_chat_messages
                              │                         │
                              └──────< studio_memory_drafts >┘   (draft of one message)

Every table carries its own ``user_id`` / ``product_id`` and is tied to its parent by a **composite
owner foreign key** (``(parent_id, user_id, product_id)``), so a row whose owner differs from its
parent's is unrepresentable (P2c-33, the P2b R1 pattern); every repository query filters the row's
own scope as well.

Chat text is user data: nothing here is ever logged, and the API returns it to its owner only.

* A **user message** is born ``complete`` and never changes.
* An **artist message** is created ``pending`` together with the user message it answers (one
  reply in flight per chat is the partial unique index ``uq_studio_chat_one_pending_reply``) and
  ends ``complete`` or ``failed`` through one guarded ``UPDATE … WHERE status = 'pending'`` that
  bumps ``revision`` - terminal once, so the task and the sweep can race and exactly one wins.
* A **memory draft** is the same shape for one short structured call (P2c-28): its own table, the
  same machinery, ``studio_runs`` untouched.

Lock order: artist → chat → message. Cascades: deleting an artist deletes its chats, their
messages and drafts; deleting a chat deletes its messages and drafts (the service refuses while a
reply or a draft is pending).
"""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    DateTime,
    ForeignKey,
    ForeignKeyConstraint,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    text,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import Mapped, mapped_column

from src.core.enums import StudioChatMessageStatus, StudioChatRole, StudioMemoryDraftStatus
from src.core.uid import new_id
from src.db.models.base import Base
from src.db.models.studio import _created_at, _in_list, _updated_at
from src.studio.constraints import CHAT_TITLE_MAX, MAX_MESSAGES_PER_CHAT

__all__ = [
    "CHAT_ONE_PENDING_REPLY",
    "DRAFT_ONE_PENDING_PER_SOURCE",
    "DRAFT_USER_IDEMPOTENCY",
    "MESSAGE_CLIENT_ID_UNIQUE",
    "StudioArtistChat",
    "StudioArtistChatMessage",
    "StudioMemoryDraft",
]

CHAT_ONE_PENDING_REPLY = "uq_studio_chat_one_pending_reply"
"""Partial unique index: at most one ``pending`` artist message per chat. The authority for 409."""
MESSAGE_CLIENT_ID_UNIQUE = "uq_studio_chat_messages_client_id"
"""``UNIQUE (chat_id, client_message_id)``: a resent message is the same message."""
DRAFT_USER_IDEMPOTENCY = "uq_studio_memory_drafts_user_idempotency"
"""``UNIQUE (user_id, idempotency_key)``: a replayed draft request is the same draft."""
DRAFT_ONE_PENDING_PER_SOURCE = "uq_studio_memory_drafts_one_pending"
"""Partial unique index: at most one ``pending`` draft per source message."""


class StudioArtistChat(Base):
    """One conversation with an artist. Mutable only in its title and counters."""

    __tablename__ = "studio_artist_chats"

    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=new_id)
    product_id: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    # Indexed: the users ON DELETE CASCADE finds this table's rows by user_id.
    user_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    artist_id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), nullable=False)
    """The FK is composite (see ``__table_args__``): it also pins ``user_id`` / ``product_id``."""
    title: Mapped[str] = mapped_column(String(CHAT_TITLE_MAX), nullable=False)

    message_count: Mapped[int] = mapped_column(Integer, nullable=False, server_default=text("0"))
    """Messages of both roles, the pending reply included (the ``chat_full`` cap counts it)."""
    next_position: Mapped[int] = mapped_column(Integer, nullable=False, server_default=text("1"))
    """The position of the next message: a counter, never ``max() + 1``."""

    created_at: Mapped[datetime] = _created_at()
    updated_at: Mapped[datetime] = _updated_at()

    __table_args__ = (
        ForeignKeyConstraint(
            ["artist_id", "user_id", "product_id"],
            ["studio_artists.id", "studio_artists.user_id", "studio_artists.product_id"],
            name="fk_studio_artist_chats_artist_owner",
            ondelete="CASCADE",
        ),
        CheckConstraint(
            f"message_count BETWEEN 0 AND {MAX_MESSAGES_PER_CHAT}",
            name="ck_studio_artist_chats_message_count",
        ),
        CheckConstraint("next_position >= 1", name="ck_studio_artist_chats_next_position"),
        # The target of the composite owner FKs of messages and drafts.
        UniqueConstraint("id", "user_id", "product_id", name="uq_studio_artist_chats_id_owner"),
        Index(
            "ix_studio_artist_chats_list",
            "artist_id",
            text("updated_at DESC"),
            text("id DESC"),
        ),
    )

    def __repr__(self) -> str:
        return f"<StudioArtistChat {self.id} artist={self.artist_id}>"


class StudioArtistChatMessage(Base):
    """One message of a chat. Immutable once ``complete`` (P2c-2)."""

    __tablename__ = "studio_artist_chat_messages"

    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=new_id)
    product_id: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    user_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    chat_id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), nullable=False)
    """The FK is composite (see ``__table_args__``): it also pins ``user_id`` / ``product_id``."""
    position: Mapped[int] = mapped_column(Integer, nullable=False)
    role: Mapped[str] = mapped_column(String(20), nullable=False)
    status: Mapped[str] = mapped_column(String(20), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False, server_default=text("''"))
    """User text (at most ``CHAT_MESSAGE_MAX``, checked by the service) or the artist's reply
    (``''`` until it is complete). Never logged."""
    client_message_id: Mapped[UUID | None] = mapped_column(PG_UUID(as_uuid=True), nullable=True)
    """The client's idempotency id of a user message; ``NULL`` for an artist message."""

    model_key: Mapped[str | None] = mapped_column(String(60), nullable=True)
    model_id: Mapped[str | None] = mapped_column(String(120), nullable=True)
    prompt_version: Mapped[str | None] = mapped_column(String(40), nullable=True)
    prompt_tokens: Mapped[int] = mapped_column(Integer, nullable=False, server_default=text("0"))
    completion_tokens: Mapped[int] = mapped_column(
        Integer, nullable=False, server_default=text("0")
    )
    memory_ids_used: Mapped[list[UUID]] = mapped_column(
        ARRAY(PG_UUID(as_uuid=True)), nullable=False, server_default=text("'{}'")
    )
    """The artist memories the reply was given, in prompt order: ids only, never their text."""
    truncated: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default=text("false"))
    """The reply hit the output-token limit (``finish_reason = length``) and was kept."""

    price_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    """What the reply reserved (``NULL`` for a user message, which has no price of its own)."""
    billing_account_id: Mapped[UUID | None] = mapped_column(PG_UUID(as_uuid=True), nullable=True)
    revision: Mapped[int] = mapped_column(Integer, nullable=False, server_default=text("1"))
    error_code: Mapped[str | None] = mapped_column(String(60), nullable=True)

    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = _created_at()
    updated_at: Mapped[datetime] = _updated_at()

    __table_args__ = (
        ForeignKeyConstraint(
            ["chat_id", "user_id", "product_id"],
            [
                "studio_artist_chats.id",
                "studio_artist_chats.user_id",
                "studio_artist_chats.product_id",
            ],
            name="fk_studio_artist_chat_messages_chat_owner",
            ondelete="CASCADE",
        ),
        UniqueConstraint("chat_id", "position", name="uq_studio_chat_messages_position"),
        UniqueConstraint("chat_id", "client_message_id", name=MESSAGE_CLIENT_ID_UNIQUE),
        # The target of the composite owner FK of a memory draft.
        UniqueConstraint(
            "id", "user_id", "product_id", name="uq_studio_artist_chat_messages_id_owner"
        ),
        CheckConstraint(_in_list("role", StudioChatRole), name="ck_studio_chat_messages_role"),
        CheckConstraint(
            _in_list("status", StudioChatMessageStatus), name="ck_studio_chat_messages_status"
        ),
        CheckConstraint(
            f"role <> '{StudioChatRole.USER.value}' "
            f"OR status = '{StudioChatMessageStatus.COMPLETE.value}'",
            name="ck_studio_chat_messages_user_complete",
        ),
        CheckConstraint(
            f"(role = '{StudioChatRole.USER.value}') = (client_message_id IS NOT NULL)",
            name="ck_studio_chat_messages_client_id_user",
        ),
        CheckConstraint(
            f"(role = '{StudioChatRole.ARTIST.value}') = (price_tokens IS NOT NULL)",
            name="ck_studio_chat_messages_price_artist",
        ),
        CheckConstraint(
            "price_tokens IS NULL OR price_tokens >= 0", name="ck_studio_chat_messages_price"
        ),
        CheckConstraint(
            f"(status = '{StudioChatMessageStatus.FAILED.value}') = (error_code IS NOT NULL)",
            name="ck_studio_chat_messages_error_code",
        ),
        CheckConstraint(
            f"(status <> '{StudioChatMessageStatus.PENDING.value}') = (finished_at IS NOT NULL)",
            name="ck_studio_chat_messages_finished",
        ),
        CheckConstraint("position >= 1", name="ck_studio_chat_messages_position"),
        CheckConstraint("revision >= 1", name="ck_studio_chat_messages_revision"),
        Index(
            CHAT_ONE_PENDING_REPLY,
            "chat_id",
            unique=True,
            postgresql_where=text(
                f"role = '{StudioChatRole.ARTIST.value}' "
                f"AND status = '{StudioChatMessageStatus.PENDING.value}'"
            ),
        ),
        Index("ix_studio_chat_messages_chat_position", "chat_id", text("position DESC")),
        # The hourly limit counts a user's recent messages.
        Index("ix_studio_chat_messages_user_created", "user_id", text("created_at DESC")),
        Index(
            "ix_studio_chat_messages_pending",
            "created_at",
            postgresql_where=text(f"status = '{StudioChatMessageStatus.PENDING.value}'"),
        ),
    )

    def __repr__(self) -> str:
        return f"<StudioArtistChatMessage {self.id} chat={self.chat_id} {self.role} {self.status}>"


class StudioMemoryDraft(Base):
    """An LLM job that turns one chat message into a memory draft the user then edits and saves.

    Nothing is written to ``studio_artist_memories`` here: the user saves the (edited) fields
    through the normal memory create with ``from_draft_id`` (P2c-29). ``draft`` holds model output
    derived from chat text: never logged.
    """

    __tablename__ = "studio_memory_drafts"

    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=new_id)
    product_id: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    user_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    chat_id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), nullable=False)
    source_message_id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), nullable=False)
    status: Mapped[str] = mapped_column(String(20), nullable=False)
    idempotency_key: Mapped[str] = mapped_column(String(128), nullable=False)
    draft: Mapped[dict[str, Any] | None] = mapped_column(JSONB(none_as_null=True), nullable=True)
    """``{title, body, themes, imagery}`` once ``complete``; SQL ``NULL`` otherwise."""

    model_key: Mapped[str | None] = mapped_column(String(60), nullable=True)
    model_id: Mapped[str | None] = mapped_column(String(120), nullable=True)
    prompt_version: Mapped[str | None] = mapped_column(String(40), nullable=True)
    prompt_tokens: Mapped[int] = mapped_column(Integer, nullable=False, server_default=text("0"))
    completion_tokens: Mapped[int] = mapped_column(
        Integer, nullable=False, server_default=text("0")
    )

    price_tokens: Mapped[int] = mapped_column(Integer, nullable=False)
    billing_account_id: Mapped[UUID | None] = mapped_column(PG_UUID(as_uuid=True), nullable=True)
    revision: Mapped[int] = mapped_column(Integer, nullable=False, server_default=text("1"))
    error_code: Mapped[str | None] = mapped_column(String(60), nullable=True)

    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = _created_at()
    updated_at: Mapped[datetime] = _updated_at()

    __table_args__ = (
        ForeignKeyConstraint(
            ["source_message_id", "user_id", "product_id"],
            [
                "studio_artist_chat_messages.id",
                "studio_artist_chat_messages.user_id",
                "studio_artist_chat_messages.product_id",
            ],
            name="fk_studio_memory_drafts_message_owner",
            ondelete="CASCADE",
        ),
        ForeignKeyConstraint(
            ["chat_id", "user_id", "product_id"],
            [
                "studio_artist_chats.id",
                "studio_artist_chats.user_id",
                "studio_artist_chats.product_id",
            ],
            name="fk_studio_memory_drafts_chat_owner",
            ondelete="CASCADE",
        ),
        UniqueConstraint("user_id", "idempotency_key", name=DRAFT_USER_IDEMPOTENCY),
        CheckConstraint(
            _in_list("status", StudioMemoryDraftStatus), name="ck_studio_drafts_status"
        ),
        CheckConstraint(
            f"(status = '{StudioMemoryDraftStatus.COMPLETE.value}') = (draft IS NOT NULL)",
            name="ck_studio_drafts_draft",
        ),
        CheckConstraint(
            f"(status = '{StudioMemoryDraftStatus.FAILED.value}') = (error_code IS NOT NULL)",
            name="ck_studio_drafts_error_code",
        ),
        CheckConstraint(
            f"(status <> '{StudioMemoryDraftStatus.PENDING.value}') = (finished_at IS NOT NULL)",
            name="ck_studio_drafts_finished",
        ),
        CheckConstraint("price_tokens >= 0", name="ck_studio_drafts_price"),
        CheckConstraint("revision >= 1", name="ck_studio_drafts_revision"),
        Index(
            DRAFT_ONE_PENDING_PER_SOURCE,
            "source_message_id",
            unique=True,
            postgresql_where=text(f"status = '{StudioMemoryDraftStatus.PENDING.value}'"),
        ),
        Index("ix_studio_memory_drafts_chat", "chat_id"),
        Index(
            "ix_studio_memory_drafts_pending",
            "created_at",
            postgresql_where=text(f"status = '{StudioMemoryDraftStatus.PENDING.value}'"),
        ),
        Index("ix_studio_memory_drafts_user_created", "user_id", text("created_at DESC")),
    )

    def __repr__(self) -> str:
        return f"<StudioMemoryDraft {self.id} message={self.source_message_id} {self.status}>"
