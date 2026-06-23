"""Add credit_warning_level and credit_warned_at to gpu_sessions

Revision ID: 012
Revises: 011
Create Date: 2026-06-23 00:00:00.000000

Changes:
- Adds credit_warning_level (VARCHAR(10) NULL) — current warning level emitted by
  SessionCreditGuard; NULL means no warning has been issued for this session yet.
- Adds credit_warned_at (TIMESTAMPTZ NULL) — timestamp when credit_warning_level was
  last set; used by the guard for hysteresis / de-escalation logic.
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "012"
down_revision: str | None = "011"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "gpu_sessions",
        sa.Column(
            "credit_warning_level",
            sa.String(10),
            nullable=True,
            comment="Current credit warning level (NotificationLevel value); NULL = not warned",
        ),
    )
    op.add_column(
        "gpu_sessions",
        sa.Column(
            "credit_warned_at",
            sa.DateTime(timezone=True),
            nullable=True,
            comment="When credit_warning_level was last set",
        ),
    )


def downgrade() -> None:
    op.drop_column("gpu_sessions", "credit_warned_at")
    op.drop_column("gpu_sessions", "credit_warning_level")
