"""Add a normalized public failure code to generation jobs.

Revision ID: 033
Revises: 032
Create Date: 2026-07-30 00:00:00.000000

``error_message`` remains the safe, displayable message for compatibility.
``failure_code`` lets API and SSE consumers branch on a stable normalized
provider failure without parsing text or storing raw upstream payloads.
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "033"
down_revision: str | None = "032"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Add the nullable code column without rewriting existing job rows."""
    op.add_column(
        "generation_jobs",
        sa.Column("failure_code", sa.String(length=100), nullable=True),
    )


def downgrade() -> None:
    """Remove the normalized failure-code column."""
    op.drop_column("generation_jobs", "failure_code")
