"""Add restart pointer + batch identity to gpu_session_deployments (P4).

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
    op.create_index(
        "ix_gpu_session_deployments_pending_restart",
        "gpu_session_deployments",
        ["session_id"],
        postgresql_where=sa.text("pending_restart"),
    )


def downgrade() -> None:
    """Drop the deployment related columns and their index."""
    op.drop_index(
        "ix_gpu_session_deployments_pending_restart", table_name="gpu_session_deployments"
    )
    op.drop_column("gpu_session_deployments", "batch_id")
    op.drop_column("gpu_session_deployments", "restart_operation_id")
