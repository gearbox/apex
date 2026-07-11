"""Add vastai_machine_id to gpu_sessions.

Revision ID: 018
Revises: 017
Create Date: 2026-07-11 00:00:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "018"
down_revision: str | None = "017"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade database schema."""
    op.add_column(
        "gpu_sessions",
        sa.Column(
            "vastai_machine_id",
            sa.BigInteger(),
            nullable=True,
            comment=(
                "Vast.ai physical machine id of the selected offer; used for the "
                "broken-node cooldown list"
            ),
        ),
    )


def downgrade() -> None:
    """Downgrade database schema."""
    op.drop_column("gpu_sessions", "vastai_machine_id")
