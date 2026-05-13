"""Add vastai_instance_destroyed_at to gpu_sessions for orphan sweeper.

Revision ID: 007
Revises: 006
Create Date: 2026-05-13
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "007"
down_revision: str | None = "006"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "gpu_sessions",
        sa.Column(
            "vastai_instance_destroyed_at",
            sa.DateTime(timezone=True),
            nullable=True,
            comment=(
                "When destroy_instance returned success for this session's Vast.ai "
                "instance. NULL if destroy was never attempted, or attempted but "
                "failed. Used by the orphan sweeper to identify sessions where the "
                "instance may still be running despite the session being in a "
                "terminal state."
            ),
        ),
    )


def downgrade() -> None:
    op.drop_column("gpu_sessions", "vastai_instance_destroyed_at")
