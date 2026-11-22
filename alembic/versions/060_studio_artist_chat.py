"""Studio artist chat: chats, messages and memory drafts.

Revision ID: 060
Revises: 059
Create Date: 2026-11-22 00:00:00.000000

Three tables (see ``src/db/models/studio_chat.py`` for the rationale); every existing table is
untouched (a memory draft is its own table, not a kind of planner run):

* ``studio_artist_chats`` - a conversation with an artist. The composite FK
  ``(artist_id, user_id, product_id) -> studio_artists (id, user_id, product_id)``
  (``ON DELETE CASCADE``; the target ``uq_studio_artists_id_owner`` exists since 059) makes a chat
  whose owner differs from its artist's unrepresentable. ``uq_studio_artist_chats_id_owner`` is the
  target of the composite FKs below.
* ``studio_artist_chat_messages`` - the messages. ``(chat_id, user_id, product_id)`` is a composite
  FK to the chat (``ON DELETE CASCADE``). One reply in flight per chat is the partial unique index
  ``uq_studio_chat_one_pending_reply``; ``UNIQUE (chat_id, client_message_id)`` makes a resent
  message the same message; CHECKs tie the columns to the role and the status (a user message is
  always ``complete`` and has a client id; an artist message has a price; ``failed`` has an error
  code; a terminal row has ``finished_at``). ``uq_studio_artist_chat_messages_id_owner`` is the
  target of the draft's composite FK.
* ``studio_memory_drafts`` - an LLM job that turns one message into a memory draft: composite FKs
  to the message and to the chat (both ``ON DELETE CASCADE``), ``UNIQUE (user_id,
  idempotency_key)``, at most one ``pending`` draft per source message.

The CHECK lists are the StrEnums' wire values, spelled out (a migration never imports ``src``).

Downgrade drops the three tables (and every chat, message and draft in them), in reverse order.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision: str = "060"
down_revision: str | Sequence[str] | None = "059"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_TS = sa.DateTime(timezone=True)
_NOW = sa.text("CURRENT_TIMESTAMP")
_UUID = postgresql.UUID(as_uuid=True)


def upgrade() -> None:
    """Upgrade database schema."""
    # -- chats ---------------------------------------------------------------------------------
    op.create_table(
        "studio_artist_chats",
        sa.Column("id", _UUID, nullable=False),
        sa.Column("product_id", sa.String(32), nullable=False),
        sa.Column("user_id", _UUID, nullable=False),
        sa.Column("artist_id", _UUID, nullable=False),
        sa.Column("title", sa.String(120), nullable=False),
        sa.Column("message_count", sa.Integer(), server_default=sa.text("0"), nullable=False),
        sa.Column("next_position", sa.Integer(), server_default=sa.text("1"), nullable=False),
        sa.Column("created_at", _TS, server_default=_NOW, nullable=False),
        sa.Column("updated_at", _TS, server_default=_NOW, nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(
            ["artist_id", "user_id", "product_id"],
            ["studio_artists.id", "studio_artists.user_id", "studio_artists.product_id"],
            name="fk_studio_artist_chats_artist_owner",
            ondelete="CASCADE",
        ),
        sa.CheckConstraint(
            "message_count BETWEEN 0 AND 200", name="ck_studio_artist_chats_message_count"
        ),
        sa.CheckConstraint("next_position >= 1", name="ck_studio_artist_chats_next_position"),
        sa.UniqueConstraint("id", "user_id", "product_id", name="uq_studio_artist_chats_id_owner"),
    )
    op.create_index("ix_studio_artist_chats_product_id", "studio_artist_chats", ["product_id"])
    op.create_index("ix_studio_artist_chats_user_id", "studio_artist_chats", ["user_id"])
    op.create_index(
        "ix_studio_artist_chats_list",
        "studio_artist_chats",
        ["artist_id", sa.text("updated_at DESC"), sa.text("id DESC")],
    )

    # -- messages ------------------------------------------------------------------------------
    op.create_table(
        "studio_artist_chat_messages",
        sa.Column("id", _UUID, nullable=False),
        sa.Column("product_id", sa.String(32), nullable=False),
        sa.Column("user_id", _UUID, nullable=False),
        sa.Column("chat_id", _UUID, nullable=False),
        sa.Column("position", sa.Integer(), nullable=False),
        sa.Column("role", sa.String(20), nullable=False),
        sa.Column("status", sa.String(20), nullable=False),
        sa.Column("content", sa.Text(), server_default=sa.text("''"), nullable=False),
        sa.Column("client_message_id", _UUID, nullable=True),
        sa.Column("model_key", sa.String(60), nullable=True),
        sa.Column("model_id", sa.String(120), nullable=True),
        sa.Column("prompt_version", sa.String(40), nullable=True),
        sa.Column("prompt_tokens", sa.Integer(), server_default=sa.text("0"), nullable=False),
        sa.Column("completion_tokens", sa.Integer(), server_default=sa.text("0"), nullable=False),
        sa.Column(
            "memory_ids_used",
            postgresql.ARRAY(_UUID),
            server_default=sa.text("'{}'"),
            nullable=False,
        ),
        sa.Column("truncated", sa.Boolean(), server_default=sa.text("false"), nullable=False),
        sa.Column("price_tokens", sa.Integer(), nullable=True),
        sa.Column("billing_account_id", _UUID, nullable=True),
        sa.Column("revision", sa.Integer(), server_default=sa.text("1"), nullable=False),
        sa.Column("error_code", sa.String(60), nullable=True),
        sa.Column("finished_at", _TS, nullable=True),
        sa.Column("created_at", _TS, server_default=_NOW, nullable=False),
        sa.Column("updated_at", _TS, server_default=_NOW, nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(
            ["chat_id", "user_id", "product_id"],
            [
                "studio_artist_chats.id",
                "studio_artist_chats.user_id",
                "studio_artist_chats.product_id",
            ],
            name="fk_studio_artist_chat_messages_chat_owner",
            ondelete="CASCADE",
        ),
        sa.UniqueConstraint("chat_id", "position", name="uq_studio_chat_messages_position"),
        sa.UniqueConstraint(
            "chat_id", "client_message_id", name="uq_studio_chat_messages_client_id"
        ),
        sa.UniqueConstraint(
            "id", "user_id", "product_id", name="uq_studio_artist_chat_messages_id_owner"
        ),
        sa.CheckConstraint("role IN ('user', 'artist')", name="ck_studio_chat_messages_role"),
        sa.CheckConstraint(
            "status IN ('pending', 'complete', 'failed')", name="ck_studio_chat_messages_status"
        ),
        sa.CheckConstraint(
            "role <> 'user' OR status = 'complete'", name="ck_studio_chat_messages_user_complete"
        ),
        sa.CheckConstraint(
            "(role = 'user') = (client_message_id IS NOT NULL)",
            name="ck_studio_chat_messages_client_id_user",
        ),
        sa.CheckConstraint(
            "(role = 'artist') = (price_tokens IS NOT NULL)",
            name="ck_studio_chat_messages_price_artist",
        ),
        sa.CheckConstraint(
            "price_tokens IS NULL OR price_tokens >= 0", name="ck_studio_chat_messages_price"
        ),
        sa.CheckConstraint(
            "(status = 'failed') = (error_code IS NOT NULL)",
            name="ck_studio_chat_messages_error_code",
        ),
        sa.CheckConstraint(
            "(status <> 'pending') = (finished_at IS NOT NULL)",
            name="ck_studio_chat_messages_finished",
        ),
        sa.CheckConstraint("position >= 1", name="ck_studio_chat_messages_position"),
        sa.CheckConstraint("revision >= 1", name="ck_studio_chat_messages_revision"),
    )
    op.create_index(
        "ix_studio_artist_chat_messages_product_id", "studio_artist_chat_messages", ["product_id"]
    )
    op.create_index(
        "uq_studio_chat_one_pending_reply",
        "studio_artist_chat_messages",
        ["chat_id"],
        unique=True,
        postgresql_where=sa.text("role = 'artist' AND status = 'pending'"),
    )
    op.create_index(
        "ix_studio_chat_messages_chat_position",
        "studio_artist_chat_messages",
        ["chat_id", sa.text("position DESC")],
    )
    op.create_index(
        "ix_studio_chat_messages_user_created",
        "studio_artist_chat_messages",
        ["user_id", sa.text("created_at DESC")],
    )
    op.create_index(
        "ix_studio_chat_messages_pending",
        "studio_artist_chat_messages",
        ["created_at"],
        postgresql_where=sa.text("status = 'pending'"),
    )

    # -- memory drafts -------------------------------------------------------------------------
    op.create_table(
        "studio_memory_drafts",
        sa.Column("id", _UUID, nullable=False),
        sa.Column("product_id", sa.String(32), nullable=False),
        sa.Column("user_id", _UUID, nullable=False),
        sa.Column("chat_id", _UUID, nullable=False),
        sa.Column("source_message_id", _UUID, nullable=False),
        sa.Column("status", sa.String(20), nullable=False),
        sa.Column("idempotency_key", sa.String(128), nullable=False),
        sa.Column("draft", postgresql.JSONB(none_as_null=True), nullable=True),
        sa.Column("model_key", sa.String(60), nullable=True),
        sa.Column("model_id", sa.String(120), nullable=True),
        sa.Column("prompt_version", sa.String(40), nullable=True),
        sa.Column("prompt_tokens", sa.Integer(), server_default=sa.text("0"), nullable=False),
        sa.Column("completion_tokens", sa.Integer(), server_default=sa.text("0"), nullable=False),
        sa.Column("price_tokens", sa.Integer(), nullable=False),
        sa.Column("billing_account_id", _UUID, nullable=True),
        sa.Column("revision", sa.Integer(), server_default=sa.text("1"), nullable=False),
        sa.Column("error_code", sa.String(60), nullable=True),
        sa.Column("finished_at", _TS, nullable=True),
        sa.Column("created_at", _TS, server_default=_NOW, nullable=False),
        sa.Column("updated_at", _TS, server_default=_NOW, nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(
            ["source_message_id", "user_id", "product_id"],
            [
                "studio_artist_chat_messages.id",
                "studio_artist_chat_messages.user_id",
                "studio_artist_chat_messages.product_id",
            ],
            name="fk_studio_memory_drafts_message_owner",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["chat_id", "user_id", "product_id"],
            [
                "studio_artist_chats.id",
                "studio_artist_chats.user_id",
                "studio_artist_chats.product_id",
            ],
            name="fk_studio_memory_drafts_chat_owner",
            ondelete="CASCADE",
        ),
        sa.UniqueConstraint(
            "user_id", "idempotency_key", name="uq_studio_memory_drafts_user_idempotency"
        ),
        sa.CheckConstraint(
            "status IN ('pending', 'complete', 'failed')", name="ck_studio_drafts_status"
        ),
        sa.CheckConstraint(
            "(status = 'complete') = (draft IS NOT NULL)", name="ck_studio_drafts_draft"
        ),
        sa.CheckConstraint(
            "(status = 'failed') = (error_code IS NOT NULL)", name="ck_studio_drafts_error_code"
        ),
        sa.CheckConstraint(
            "(status <> 'pending') = (finished_at IS NOT NULL)", name="ck_studio_drafts_finished"
        ),
        sa.CheckConstraint("price_tokens >= 0", name="ck_studio_drafts_price"),
        sa.CheckConstraint("revision >= 1", name="ck_studio_drafts_revision"),
    )
    op.create_index("ix_studio_memory_drafts_product_id", "studio_memory_drafts", ["product_id"])
    op.create_index(
        "uq_studio_memory_drafts_one_pending",
        "studio_memory_drafts",
        ["source_message_id"],
        unique=True,
        postgresql_where=sa.text("status = 'pending'"),
    )
    op.create_index("ix_studio_memory_drafts_chat", "studio_memory_drafts", ["chat_id"])
    op.create_index(
        "ix_studio_memory_drafts_pending",
        "studio_memory_drafts",
        ["created_at"],
        postgresql_where=sa.text("status = 'pending'"),
    )
    op.create_index(
        "ix_studio_memory_drafts_user_created",
        "studio_memory_drafts",
        ["user_id", sa.text("created_at DESC")],
    )


def downgrade() -> None:
    """Downgrade database schema (drops every chat, message and memory draft)."""
    # Reverse order: drafts reference messages and chats; messages reference chats.
    op.drop_table("studio_memory_drafts")
    op.drop_table("studio_artist_chat_messages")
    op.drop_table("studio_artist_chats")
