"""add last_reachable_at, drop node_host and node_port

Revision ID: 009
Revises: 008
Create Date: 2026-06-09 00:00:00.000000

- Adds last_reachable_at (TIMESTAMPTZ nullable) to gpu_sessions.
  Stamped at the active transition and on every healthy reconciler probe.
  Used with GPU_SESSION_STALE_GRACE_SECONDS to prevent single-blip flapping.
- Drops node_host and node_port: these were a pre-tunnel direct-IP design
  that nothing populates. The reconciler now resolves sessions via
  tunnel_hostname (same as generation + job poller).
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "009"
down_revision: str | None = "008"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade database schema."""
    op.add_column(
        "gpu_sessions",
        sa.Column(
            "last_reachable_at",
            sa.DateTime(timezone=True),
            nullable=True,
            comment=(
                "Stamped at active transition and on each healthy reconciler probe. "
                "Used with gpu_session_stale_grace_seconds to prevent single-blip "
                "stale flapping."
            ),
        ),
    )
    op.drop_column("gpu_sessions", "node_host")
    op.drop_column("gpu_sessions", "node_port")


def downgrade() -> None:
    """Downgrade database schema."""
    op.add_column(
        "gpu_sessions",
        sa.Column("node_port", sa.Integer(), nullable=True),
    )
    op.add_column(
        "gpu_sessions",
        sa.Column("node_host", sa.String(length=255), nullable=True),
    )
    op.drop_column("gpu_sessions", "last_reachable_at")
