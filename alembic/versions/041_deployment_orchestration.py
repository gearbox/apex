"""Add restart pointer + batch identity to gpu_session_deployments (P4).

Also adds routing_suspended (round-4 remediation, S5): a node-wide restart takes
ComfyUI down for every deployment on the session, not only the ones being
restarted, so routing must be suspended session-wide while a restart cohort
drains and runs, not just gated on the individual deployment's own status.

Revision ID: 041
Revises: 040
Create Date: 2026-09-05 00:00:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision: str = "041"
down_revision: str | Sequence[str] | None = "040"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Additive only. No backfill — existing deployments are all primaries
    that never needed a restart."""
    op.add_column(
        "gpu_session_deployments",
        sa.Column("restart_operation_id", postgresql.UUID(as_uuid=True), nullable=True),
    )
    op.add_column(
        "gpu_session_deployments",
        sa.Column("batch_id", sa.String(length=64), nullable=True),
    )
    op.add_column(
        "gpu_session_deployments",
        sa.Column("pending_restart_since", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index(
        "ix_gpu_session_deployments_pending_restart",
        "gpu_session_deployments",
        ["session_id"],
        postgresql_where=sa.text("pending_restart"),
    )
    op.add_column(
        "gpu_session_deployments",
        sa.Column(
            "routing_suspended",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("false"),
        ),
    )


def downgrade() -> None:
    """Drop the deployment related columns and their index."""
    op.drop_column("gpu_session_deployments", "routing_suspended")
    op.drop_index(
        "ix_gpu_session_deployments_pending_restart", table_name="gpu_session_deployments"
    )
    op.drop_column("gpu_session_deployments", "batch_id")
    op.drop_column("gpu_session_deployments", "pending_restart_since")
    op.drop_column("gpu_session_deployments", "restart_operation_id")
