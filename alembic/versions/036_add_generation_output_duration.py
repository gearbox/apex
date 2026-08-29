"""Add nullable duration metadata to generated outputs.

Revision ID: 036
Revises: 035
Create Date: 2026-08-29 00:00:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "036"
down_revision: str | Sequence[str] | None = "035"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Store provider-reported output duration without guessing historical values."""
    op.add_column("generation_outputs", sa.Column("duration_ms", sa.Integer(), nullable=True))


def downgrade() -> None:
    """Remove the nullable duration field."""
    op.drop_column("generation_outputs", "duration_ms")
