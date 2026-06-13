"""Add provisioning callback columns to gpu_sessions

Revision ID: 010
Revises: 009
Create Date: 2026-06-13 00:00:00.000000

Changes:
- Drops callback_token (plaintext was pre-wired placeholder, never used for auth)
- Adds callback_token_hash (VARCHAR 64) — SHA-256 hex digest for constant-time bearer auth
- Adds provisioning_phase (VARCHAR 20) — latest phase from node callback
- Adds provisioning_progress (JSONB) — latest progress blob {ts, message, download?, error?}
- Adds last_progress_at (TIMESTAMPTZ) — stamped on each accepted callback, used for stall detection
"""

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

from alembic import op

revision: str = "010"
down_revision: str | None = "009"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade database schema."""
    op.drop_column("gpu_sessions", "callback_token")

    op.add_column(
        "gpu_sessions",
        sa.Column(
            "callback_token_hash",
            sa.String(64),
            nullable=True,
            comment="SHA-256 hex digest of the bearer token sent by the node. Plaintext never stored.",
        ),
    )
    op.add_column(
        "gpu_sessions",
        sa.Column(
            "provisioning_phase",
            sa.String(20),
            nullable=True,
            comment="Latest phase reported by the node via callback.",
        ),
    )
    op.add_column(
        "gpu_sessions",
        sa.Column(
            "provisioning_progress",
            JSONB(),
            nullable=True,
            comment="Latest progress blob from node callback: {ts, message, download?, error?}.",
        ),
    )
    op.add_column(
        "gpu_sessions",
        sa.Column(
            "last_progress_at",
            sa.DateTime(timezone=True),
            nullable=True,
            comment="Wall-clock time of the last accepted callback. Used for stall detection.",
        ),
    )


def downgrade() -> None:
    """Downgrade database schema."""
    op.drop_column("gpu_sessions", "last_progress_at")
    op.drop_column("gpu_sessions", "provisioning_progress")
    op.drop_column("gpu_sessions", "provisioning_phase")
    op.drop_column("gpu_sessions", "callback_token_hash")

    op.add_column(
        "gpu_sessions",
        sa.Column("callback_token", sa.String(128), nullable=True),
    )
