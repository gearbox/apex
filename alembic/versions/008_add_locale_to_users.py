"""Add locale column to users table.

Revision ID: 008
Revises: 007
Create Date: 2026-03-14
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "008"
down_revision: str | None = "007"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "users",
        sa.Column("locale", sa.String(10), nullable=False, server_default="en"),
    )


def downgrade() -> None:
    op.drop_column("users", "locale")
