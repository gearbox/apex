"""Add gpu_sessions table for GPU node lifecycle tracking.

Revision ID: 003
Revises: 002
Create Date: 2026-04-01
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision: str = "003"
down_revision: str | None = "002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "gpu_sessions",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column(
            "user_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("product_id", sa.String(32), nullable=False),
        sa.Column("status", sa.String(20), nullable=False),
        sa.Column("vastai_instance_id", sa.Integer(), nullable=True),
        sa.Column("node_host", sa.String(255), nullable=True),
        sa.Column("node_port", sa.Integer(), nullable=True),
        sa.Column("stale_detected_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "stale_notified",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("false"),
        ),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("CURRENT_TIMESTAMP"),
        ),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("stopped_at", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )

    op.create_index("ix_gpu_sessions_user_id", "gpu_sessions", ["user_id"])
    op.create_index("ix_gpu_sessions_product_id", "gpu_sessions", ["product_id"])
    op.create_index("ix_gpu_sessions_status", "gpu_sessions", ["status"])
    op.create_index(
        "ix_gpu_sessions_status_product",
        "gpu_sessions",
        ["status", "product_id"],
    )
    op.create_index(
        "ix_gpu_sessions_active_stale",
        "gpu_sessions",
        ["status", "stale_detected_at"],
        postgresql_where=sa.text("status IN ('active', 'stale')"),
    )


def downgrade() -> None:
    op.drop_table("gpu_sessions")
