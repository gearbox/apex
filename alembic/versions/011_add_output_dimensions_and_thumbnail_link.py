"""Add width/height dimensions and parent_output_id FK to generation_outputs

Revision ID: 011
Revises: 010
Create Date: 2026-06-14 00:00:00.000000

Changes:
- Adds width (INTEGER NULL) — pixel width stored for both full and thumbnail rows
- Adds height (INTEGER NULL) — pixel height stored for both full and thumbnail rows
- Adds parent_output_id (UUID NULL) — self-referential FK with ON DELETE CASCADE;
  thumbnail rows point to their parent full output; index for reverse lookups
"""

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID as PG_UUID

from alembic import op

revision: str = "011"
down_revision: str | None = "010"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade database schema."""
    op.add_column(
        "generation_outputs",
        sa.Column("width", sa.Integer(), nullable=True),
    )
    op.add_column(
        "generation_outputs",
        sa.Column("height", sa.Integer(), nullable=True),
    )
    op.add_column(
        "generation_outputs",
        sa.Column("parent_output_id", PG_UUID(as_uuid=True), nullable=True),
    )
    op.create_index(
        "ix_generation_outputs_parent_output_id",
        "generation_outputs",
        ["parent_output_id"],
    )
    op.create_foreign_key(
        "fk_generation_outputs_parent_output_id",
        "generation_outputs",
        "generation_outputs",
        ["parent_output_id"],
        ["id"],
        ondelete="CASCADE",
    )


def downgrade() -> None:
    """Downgrade database schema."""
    op.drop_constraint(
        "fk_generation_outputs_parent_output_id",
        "generation_outputs",
        type_="foreignkey",
    )
    op.drop_index("ix_generation_outputs_parent_output_id", table_name="generation_outputs")
    op.drop_column("generation_outputs", "parent_output_id")
    op.drop_column("generation_outputs", "height")
    op.drop_column("generation_outputs", "width")
