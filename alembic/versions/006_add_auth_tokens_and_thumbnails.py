"""Add email verification, password reset tokens, email_verified_at, output thumbnail flag.

Changes:
- users.email_verified_at (nullable datetime) — NULL means unverified
- email_verification_tokens table
- password_reset_tokens table
- generation_outputs.is_thumbnail (bool, default false) — marks extracted frame/poster

Revision ID: 006
Revises: 005
Create Date: 2026-02-26 10:00:00.000000
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

# revision identifiers
revision: str = "006"
down_revision: str | None = "005"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Apply schema changes."""

    # -------------------------------------------------------------------------
    # 1. Add email_verified_at to users
    # -------------------------------------------------------------------------
    op.add_column(
        "users",
        sa.Column(
            "email_verified_at",
            sa.DateTime(timezone=True),
            nullable=True,
        ),
    )

    # -------------------------------------------------------------------------
    # 2. email_verification_tokens
    # -------------------------------------------------------------------------
    op.create_table(
        "email_verification_tokens",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            server_default=sa.text("gen_random_uuid()"),
            primary_key=True,
        ),
        sa.Column(
            "user_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("token_hash", sa.String(64), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("used_at", sa.DateTime(timezone=True), nullable=True),
        sa.UniqueConstraint("token_hash", name="uq_email_verification_tokens_token_hash"),
    )
    op.create_index(
        "ix_email_verification_tokens_token_hash",
        "email_verification_tokens",
        ["token_hash"],
        unique=True,
    )
    op.create_index(
        "ix_email_verification_tokens_user_id",
        "email_verification_tokens",
        ["user_id"],
    )
    op.create_index(
        "ix_email_verification_tokens_cleanup",
        "email_verification_tokens",
        ["expires_at"],
    )
    op.create_index(
        "ix_email_verification_tokens_user_active",
        "email_verification_tokens",
        ["user_id", "used_at"],
    )

    # -------------------------------------------------------------------------
    # 3. password_reset_tokens
    # -------------------------------------------------------------------------
    op.create_table(
        "password_reset_tokens",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            server_default=sa.text("gen_random_uuid()"),
            primary_key=True,
        ),
        sa.Column(
            "user_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("token_hash", sa.String(64), nullable=False),
        sa.Column("ip_address", sa.String(45), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("used_at", sa.DateTime(timezone=True), nullable=True),
        sa.UniqueConstraint("token_hash", name="uq_password_reset_tokens_token_hash"),
    )
    op.create_index(
        "ix_password_reset_tokens_token_hash",
        "password_reset_tokens",
        ["token_hash"],
        unique=True,
    )
    op.create_index(
        "ix_password_reset_tokens_user_id",
        "password_reset_tokens",
        ["user_id"],
    )
    op.create_index(
        "ix_password_reset_tokens_cleanup",
        "password_reset_tokens",
        ["expires_at"],
    )
    op.create_index(
        "ix_password_reset_tokens_user_active",
        "password_reset_tokens",
        ["user_id", "used_at"],
    )

    # -------------------------------------------------------------------------
    # 4. generation_outputs.is_thumbnail
    #    Marks the extracted first-frame poster for video outputs.
    #    Also serves as a flag for the "primary" image in multi-image batches.
    # -------------------------------------------------------------------------
    op.add_column(
        "generation_outputs",
        sa.Column(
            "is_thumbnail",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("false"),
        ),
    )
    op.create_index(
        "ix_generation_outputs_thumbnail",
        "generation_outputs",
        ["job_id", "is_thumbnail"],
    )


def downgrade() -> None:
    """Reverse schema changes."""
    op.drop_index("ix_generation_outputs_thumbnail", table_name="generation_outputs")
    op.drop_column("generation_outputs", "is_thumbnail")

    op.drop_index("ix_password_reset_tokens_user_active", table_name="password_reset_tokens")
    op.drop_index("ix_password_reset_tokens_cleanup", table_name="password_reset_tokens")
    op.drop_index("ix_password_reset_tokens_user_id", table_name="password_reset_tokens")
    op.drop_index("ix_password_reset_tokens_token_hash", table_name="password_reset_tokens")
    op.drop_table("password_reset_tokens")

    op.drop_index(
        "ix_email_verification_tokens_user_active", table_name="email_verification_tokens"
    )
    op.drop_index("ix_email_verification_tokens_cleanup", table_name="email_verification_tokens")
    op.drop_index("ix_email_verification_tokens_user_id", table_name="email_verification_tokens")
    op.drop_index("ix_email_verification_tokens_token_hash", table_name="email_verification_tokens")
    op.drop_table("email_verification_tokens")

    op.drop_column("users", "email_verified_at")
