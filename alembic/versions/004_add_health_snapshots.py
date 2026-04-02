"""Add health_snapshots table for persisted health check results.

Revision ID: 004
Revises: 003
Create Date: 2026-04-02
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision: str = "004"
down_revision: str | None = "003"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "health_snapshots",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            nullable=False,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column(
            "checked_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("CURRENT_TIMESTAMP"),
        ),
        sa.Column("overall_status", sa.String(16), nullable=False),
        sa.Column("snapshot_data", postgresql.JSONB(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("CURRENT_TIMESTAMP"),
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_health_snapshots_checked_at",
        "health_snapshots",
        ["checked_at"],
        postgresql_using="btree",
    )


def downgrade() -> None:
    op.drop_index("ix_health_snapshots_checked_at", table_name="health_snapshots")
    op.drop_table("health_snapshots")
