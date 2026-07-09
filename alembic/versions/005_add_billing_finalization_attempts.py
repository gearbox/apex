"""Add billing_finalization_attempts column to gpu_sessions.

Revision ID: 005
Revises: 004
Create Date: 2026-04-27
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import sqlalchemy as sa

from alembic import op

if TYPE_CHECKING:
    from collections.abc import Sequence

revision: str = "005"
down_revision: str | None = "004"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "gpu_sessions",
        sa.Column(
            "billing_finalization_attempts",
            sa.Integer(),
            nullable=False,
            server_default=sa.text("0"),
            comment=(
                "Number of times the billing reconciler has attempted to finalize "
                "this session. Bumped each sweep when finalization fails. Used to "
                "trigger ops alerts after a quarantine threshold without mutating "
                "billing_finalized_at (the session must remain reconcilable once "
                "the underlying issue is fixed)."
            ),
        ),
    )


def downgrade() -> None:
    op.drop_column("gpu_sessions", "billing_finalization_attempts")
