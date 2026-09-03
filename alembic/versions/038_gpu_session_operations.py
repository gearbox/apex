"""Add operation-scoped GPU telemetry state.

Revision ID: 038
Revises: 037
Create Date: 2026-09-02 00:00:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision: str = "038"
down_revision: str | Sequence[str] | None = "037"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Replace the v1 session projection with operation-scoped telemetry."""
    op.create_table(
        "gpu_session_operations",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("session_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("product_id", sa.String(length=32), nullable=False),
        sa.Column("command_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("kind", sa.String(length=20), nullable=False),
        sa.Column("status", sa.String(length=20), nullable=False),
        sa.Column("phase", sa.String(length=32), nullable=True),
        sa.Column("target_bundle", sa.String(length=100), nullable=True),
        sa.Column("target_bundle_version", sa.String(length=20), nullable=True),
        sa.Column("target_mode", sa.String(length=16), nullable=True),
        sa.Column("batch_id", sa.String(length=64), nullable=True),
        sa.Column("batch_index", sa.Integer(), nullable=True),
        sa.Column("batch_total", sa.Integer(), nullable=True),
        sa.Column("last_sequence", sa.Integer(), nullable=False, server_default=sa.text("-1")),
        sa.Column("last_event_id", sa.String(length=64), nullable=True),
        sa.Column("progress", postgresql.JSONB(), nullable=True),
        sa.Column("plan", postgresql.JSONB(), nullable=True),
        sa.Column("summary", postgresql.JSONB(), nullable=True),
        sa.Column("message", sa.Text(), nullable=True),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("node_started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_event_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("terminal_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("CURRENT_TIMESTAMP"),
        ),
        sa.ForeignKeyConstraint(["session_id"], ["gpu_sessions.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_gpu_session_operations_session_created",
        "gpu_session_operations",
        ["session_id", sa.text("created_at DESC")],
    )
    op.create_index(
        "ix_gpu_session_operations_batch",
        "gpu_session_operations",
        ["batch_id"],
        postgresql_where=sa.text("batch_id IS NOT NULL"),
    )
    op.add_column(
        "gpu_sessions",
        sa.Column("bootstrap_operation_id", postgresql.UUID(as_uuid=True), nullable=True),
    )
    op.drop_column("gpu_sessions", "provisioning_progress")
    op.drop_column("gpu_sessions", "provisioning_phase")


def downgrade() -> None:
    """Restore the v1 session projection and remove operation state."""
    op.add_column(
        "gpu_sessions",
        sa.Column("provisioning_phase", sa.String(length=20), nullable=True),
    )
    op.add_column(
        "gpu_sessions",
        sa.Column("provisioning_progress", postgresql.JSONB(), nullable=True),
    )
    op.drop_column("gpu_sessions", "bootstrap_operation_id")
    op.drop_table("gpu_session_operations")
