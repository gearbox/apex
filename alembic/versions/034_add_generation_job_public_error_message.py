"""Add a dedicated public-safe generation-job failure message.

Revision ID: 034
Revises: 033
Create Date: 2026-07-31 00:00:00.000000

Legacy ``error_message`` values may contain internal diagnostics, so this
migration intentionally does not copy them into the new user-facing column.
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "034"
down_revision: str | Sequence[str] | None = "033"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Add the nullable public-safe message without rewriting legacy rows."""
    op.add_column(
        "generation_jobs",
        sa.Column("public_error_message", sa.Text(), nullable=True),
    )


def downgrade() -> None:
    """Remove the public-safe failure message column."""
    op.drop_column("generation_jobs", "public_error_message")
