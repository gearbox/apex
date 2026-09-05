"""Create gpu_session_commands — the node agent's claimable queue (P3).

Revision ID: 040
Revises: 039
Create Date: 2026-09-03 00:00:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision: str = "040"
down_revision: str | Sequence[str] | None = "039"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Create gpu_session_commands. No backfill — the table starts empty."""
    op.create_table(
        "gpu_session_commands",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("session_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("product_id", sa.String(length=32), nullable=False),
        sa.Column("operation_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("deployment_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("kind", sa.String(length=20), nullable=False),
        sa.Column("payload", postgresql.JSONB(), nullable=False),
        sa.Column("batch_id", sa.String(length=64), nullable=True),
        sa.Column("batch_index", sa.Integer(), nullable=True),
        sa.Column("batch_total", sa.Integer(), nullable=True),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("agent_id", sa.String(length=128), nullable=True),
        sa.Column("claimed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("deadline_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("terminal_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("CURRENT_TIMESTAMP"),
        ),
        sa.ForeignKeyConstraint(["session_id"], ["gpu_sessions.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("operation_id"),
    )
    op.create_index(
        "ix_gpu_session_commands_one_claimed",
        "gpu_session_commands",
        ["session_id"],
        unique=True,
        postgresql_where=sa.text("status = 'claimed'"),
    )
    op.create_index(
        "ix_gpu_session_commands_queue",
        "gpu_session_commands",
        ["session_id", "created_at", "batch_index"],
        postgresql_where=sa.text("status = 'queued'"),
    )
    op.create_index(
        "ix_gpu_session_commands_deadline",
        "gpu_session_commands",
        ["deadline_at"],
        postgresql_where=sa.text("status = 'claimed'"),
    )
    op.create_index(
        "ix_gpu_session_commands_batch",
        "gpu_session_commands",
        ["batch_id"],
        postgresql_where=sa.text("batch_id IS NOT NULL"),
    )


def downgrade() -> None:
    """Drop gpu_session_commands and its indexes."""
    op.drop_table("gpu_session_commands")
