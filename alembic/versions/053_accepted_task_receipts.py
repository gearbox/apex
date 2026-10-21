"""Accepted-task receipts for async video submissions.

Revision ID: 053
Revises: 052
Create Date: 2026-10-15 00:00:00.000000

A provider accepts a video task inside the still-open request transaction, so a
later failure (or a rollback) leaves a billable upstream task that no row
references. ``accepted_task_receipts`` records the upstream task id on its own
connection; the request deletes the row under its transaction's lock, so the
table holds only orphans (plus in-flight requests' locked rows).

``job_id`` has no foreign key on purpose: the job row may never commit.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision: str = "053"
down_revision: str | Sequence[str] | None = "052"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade database schema."""
    op.create_table(
        "accepted_task_receipts",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("provider", sa.String(length=32), nullable=False),
        sa.Column("external_task_id", sa.String(length=255), nullable=False),
        sa.Column("job_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("product_id", sa.String(length=32), nullable=False),
        sa.Column("model", sa.String(length=64), nullable=False),
        sa.Column("token_cost", sa.Integer(), nullable=False),
        sa.Column(
            "status", sa.String(length=16), server_default=sa.text("'pending'"), nullable=False
        ),
        sa.Column("upstream_status", sa.String(length=32), nullable=True),
        sa.Column(
            "accepted_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.Column("last_checked_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("resolved_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.CheckConstraint(
            "status IN ('pending','cancelled','absorbed')", name="chk_accepted_task_receipt_status"
        ),
        sa.CheckConstraint(
            "(status = 'pending') = (resolved_at IS NULL)",
            name="chk_accepted_task_receipt_resolved_at",
        ),
    )
    op.create_index(
        "uq_accepted_task_receipts_provider_task",
        "accepted_task_receipts",
        ["provider", "external_task_id"],
        unique=True,
    )
    op.create_index(
        "ix_accepted_task_receipts_product_id", "accepted_task_receipts", ["product_id"]
    )
    op.create_index(
        "ix_accepted_task_receipts_pending",
        "accepted_task_receipts",
        ["provider", "accepted_at"],
        postgresql_where=sa.text("status = 'pending'"),
    )


def downgrade() -> None:
    """Downgrade database schema."""
    op.drop_index("ix_accepted_task_receipts_pending", table_name="accepted_task_receipts")
    op.drop_index("ix_accepted_task_receipts_product_id", table_name="accepted_task_receipts")
    op.drop_index("uq_accepted_task_receipts_provider_task", table_name="accepted_task_receipts")
    op.drop_table("accepted_task_receipts")
