"""Add feedback_reports (in-product problem reports).

Revision ID: 049
Revises: 048
Create Date: 2026-09-28 00:00:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision: str = "049"
down_revision: str | Sequence[str] | None = "048"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Create feedback_reports: one row per user-submitted problem report."""
    op.create_table(
        "feedback_reports",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("product_id", sa.String(32), nullable=False),
        # Always set on insert; nullable only so a hard user purge keeps the report.
        sa.Column(
            "user_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("users.id", ondelete="SET NULL"),
            nullable=True,
        ),
        # FeedbackCategory / FeedbackStatus values — plain String, no CHECK on
        # the value set (validated at the API boundary; enums grow freely).
        sa.Column("category", sa.String(32), nullable=False),
        sa.Column("status", sa.String(16), nullable=False, server_default="open"),
        sa.Column("message", sa.Text(), nullable=False),
        sa.Column(
            "job_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("generation_jobs.id", ondelete="SET NULL"),
            nullable=True,
        ),
        # Polymorphic asset reference, no FK: outputs are retention-swept.
        sa.Column("asset_source", sa.String(16), nullable=True),
        sa.Column("asset_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("client_path", sa.String(512), nullable=True),
        sa.Column("app_version", sa.String(64), nullable=True),
        sa.Column("user_agent", sa.String(512), nullable=True),
        sa.Column("admin_note", sa.Text(), nullable=True),
        sa.Column("resolved_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "resolved_by",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("users.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("CURRENT_TIMESTAMP"),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("CURRENT_TIMESTAMP"),
        ),
        sa.CheckConstraint(
            "(asset_source IS NULL) = (asset_id IS NULL)",
            name="chk_feedback_asset_pair",
        ),
        sa.CheckConstraint(
            "(status IN ('resolved', 'dismissed')) = (resolved_at IS NOT NULL)",
            name="chk_feedback_resolved_at_terminal",
        ),
    )
    op.create_index(
        "ix_feedback_reports_product_created",
        "feedback_reports",
        ["product_id", "created_at", "id"],
    )


def downgrade() -> None:
    """Drop feedback_reports."""
    op.drop_index("ix_feedback_reports_product_created", table_name="feedback_reports")
    op.drop_table("feedback_reports")
