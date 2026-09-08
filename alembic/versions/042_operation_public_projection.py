"""Expose operation-to-deployment projection state (P5).

Revision ID: 042
Revises: 041
Create Date: 2026-09-08 00:00:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision: str = "042"
down_revision: str | Sequence[str] | None = "041"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Add public-operation lookup columns; pre-production, so no backfill."""
    op.add_column(
        "gpu_session_operations",
        sa.Column("deployment_id", postgresql.UUID(as_uuid=True), nullable=True),
    )
    op.create_foreign_key(
        "fk_gpu_session_operations_deployment_id_gpu_session_deployments",
        "gpu_session_operations",
        "gpu_session_deployments",
        ["deployment_id"],
        ["id"],
        ondelete="CASCADE",
    )
    op.add_column(
        "gpu_session_operations",
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("CURRENT_TIMESTAMP"),
        ),
    )
    op.create_index(
        "ix_gpu_session_operations_deployment_created",
        "gpu_session_operations",
        ["deployment_id", sa.text("created_at DESC")],
        postgresql_where=sa.text("deployment_id IS NOT NULL"),
    )


def downgrade() -> None:
    """Remove P5 public-operation lookup columns and index."""
    op.drop_index(
        "ix_gpu_session_operations_deployment_created", table_name="gpu_session_operations"
    )
    op.drop_column("gpu_session_operations", "updated_at")
    op.drop_constraint(
        "fk_gpu_session_operations_deployment_id_gpu_session_deployments",
        "gpu_session_operations",
        type_="foreignkey",
    )
    op.drop_column("gpu_session_operations", "deployment_id")
