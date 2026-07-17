"""Add admin_notification_preferences + admin_telegram_links.

Revision ID: 024
Revises: 023
Create Date: 2026-07-17 00:00:00.000000

Backs the Telegram ops notification feature: per-admin subscription to
notification classes (row-presence = subscribed, PUT replaces the full set)
and the Telegram deep-link flow (one-time token -> chat_id, single-use).

Both tables are net-new — no data-preservation concern, so downgrade is a
plain symmetric drop.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision: str = "024"
down_revision: str | None = "023"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade database schema."""
    op.create_table(
        "admin_notification_preferences",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("product_id", sa.String(32), nullable=False),
        sa.Column("notification_class", sa.String(50), nullable=False),
        sa.Column(
            "min_interval_seconds", sa.Integer(), nullable=False, server_default=sa.text("0")
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("CURRENT_TIMESTAMP"),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("CURRENT_TIMESTAMP"),
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.UniqueConstraint(
            "user_id",
            "product_id",
            "notification_class",
            name="uq_admin_notif_pref_user_product_class",
        ),
    )
    op.create_index(
        "ix_admin_notification_preferences_user_id",
        "admin_notification_preferences",
        ["user_id"],
    )
    op.create_index(
        "ix_admin_notif_pref_class",
        "admin_notification_preferences",
        ["notification_class"],
    )

    op.create_table(
        "admin_telegram_links",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("product_id", sa.String(32), nullable=False),
        sa.Column("chat_id", sa.BigInteger(), nullable=True),
        sa.Column("link_token", sa.String(64), nullable=True),
        sa.Column("token_expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("linked_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("CURRENT_TIMESTAMP"),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("CURRENT_TIMESTAMP"),
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.UniqueConstraint("user_id", name="uq_admin_telegram_links_user_id"),
        sa.UniqueConstraint("link_token", name="uq_admin_telegram_links_link_token"),
    )


def downgrade() -> None:
    """Downgrade database schema."""
    op.drop_table("admin_telegram_links")
    op.drop_index("ix_admin_notif_pref_class", table_name="admin_notification_preferences")
    op.drop_index(
        "ix_admin_notification_preferences_user_id", table_name="admin_notification_preferences"
    )
    op.drop_table("admin_notification_preferences")
