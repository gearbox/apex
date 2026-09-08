"""Add Apex-owned operation revisions and denormalized event owners.

Revision ID: 043
Revises: 042
Create Date: 2026-09-08 00:00:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision: str = "043"
down_revision: str | Sequence[str] | None = "042"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Make the public operation projection independently publishable."""
    op.add_column(
        "gpu_session_operations",
        sa.Column(
            "revision",
            sa.Integer(),
            nullable=False,
            server_default=sa.text("0"),
            comment=(
                "Apex-owned read-model revision; increments on every durable change to the public "
                "projection. Distinct from last_sequence, which is the node's producer sequence."
            ),
        ),
    )
    op.add_column(
        "gpu_session_operations",
        sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=True),
    )
    op.execute(
        """
        UPDATE gpu_session_operations AS operation
           SET user_id = session.user_id
          FROM gpu_sessions AS session
         WHERE session.id = operation.session_id
        """
    )
    op.alter_column("gpu_session_operations", "user_id", nullable=False)


def downgrade() -> None:
    """Remove the public-projection revision and event owner."""
    op.drop_column("gpu_session_operations", "user_id")
    op.drop_column("gpu_session_operations", "revision")
