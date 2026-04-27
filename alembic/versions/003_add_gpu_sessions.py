"""Add gpu_sessions table for GPU node lifecycle tracking.

Revision ID: 003
Revises: 002
Create Date: 2026-04-01
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision: str = "003"
down_revision: str | None = "002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "gpu_sessions",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column(
            "user_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("product_id", sa.String(32), nullable=False),
        sa.Column("status", sa.String(20), nullable=False),
        sa.Column("vastai_instance_id", sa.Integer(), nullable=True),
        sa.Column("node_host", sa.String(255), nullable=True),
        sa.Column("node_port", sa.Integer(), nullable=True),
        sa.Column("stale_detected_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "stale_notified",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("false"),
        ),
        # Bundle identity columns
        sa.Column(
            "bundle_name",
            sa.String(100),
            nullable=False,
            server_default="",
            comment="ai-bundles bundle name (e.g. wan_2.2_i2v)",
        ),
        sa.Column(
            "bundle_version",
            sa.String(20),
            nullable=True,
            comment="Specific bundle version (e.g. 260105-01). None = 'current' symlink",
        ),
        sa.Column(
            "model_type",
            sa.String(50),
            nullable=False,
            server_default="",
            comment="ModelType slug that triggered this session (e.g. aisha-image)",
        ),
        # Cloudflare tunnel columns
        sa.Column("cf_tunnel_id", sa.String(64), nullable=True),
        sa.Column("cf_dns_record_id", sa.String(64), nullable=True),
        sa.Column("tunnel_hostname", sa.String(255), nullable=True),
        # Vast.ai detail columns
        sa.Column("vastai_offer_id", sa.Integer(), nullable=True),
        sa.Column(
            "vastai_cost_per_hour_micros",
            sa.Integer(),
            nullable=True,
            comment="Vast.ai $/hr in microdollars (1_000_000 = $1.00) at instance creation time",
        ),
        sa.Column("vastai_gpu_name", sa.String(50), nullable=True),
        # Provisioning tracking
        sa.Column(
            "provision_attempt",
            sa.Integer(),
            nullable=False,
            server_default=sa.text("1"),
        ),
        sa.Column(
            "provisioning_started_at",
            sa.DateTime(timezone=True),
            nullable=True,
            comment="Set on first provision attempt (or reset on retry) — used for timeout calculation",
        ),
        # Billing (set on session creation; used for debit/refund at stop time)
        sa.Column(
            "account_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("token_accounts.id", ondelete="RESTRICT"),
            nullable=True,
            comment="Billing account to charge/refund at stop time. Captured at session start.",
        ),
        sa.Column(
            "total_paused_seconds",
            sa.Integer(),
            nullable=False,
            server_default="0",
            comment="Cumulative paused duration; subtracted from billable time at stop.",
        ),
        # Pause/resume tracking
        sa.Column("paused_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("resumed_at", sa.DateTime(timezone=True), nullable=True),
        # Phase 2 callback token
        sa.Column("callback_token", sa.String(128), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("CURRENT_TIMESTAMP"),
        ),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("stopped_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "billing_finalized_at",
            sa.DateTime(timezone=True),
            nullable=True,
            comment=(
                "Set by GpuSessionService._finalize_billing on success. NULL means the "
                "session has not had its overage/refund applied yet (either still "
                "pre-stop, or finalize failed). A phase-2 reconciler worker picks up "
                "rows where status='stopped' AND billing_finalized_at IS NULL."
            ),
        ),
        sa.PrimaryKeyConstraint("id"),
    )

    op.create_index("ix_gpu_sessions_user_id", "gpu_sessions", ["user_id"])
    op.create_index("ix_gpu_sessions_account_id", "gpu_sessions", ["account_id"])
    op.create_index("ix_gpu_sessions_product_id", "gpu_sessions", ["product_id"])
    op.create_index("ix_gpu_sessions_status", "gpu_sessions", ["status"])
    op.create_index(
        "ix_gpu_sessions_status_product",
        "gpu_sessions",
        ["status", "product_id"],
    )
    op.create_index(
        "ix_gpu_sessions_active_stale",
        "gpu_sessions",
        ["status", "stale_detected_at"],
        postgresql_where=sa.text("status IN ('active', 'stale', 'paused', 'resuming')"),
    )

    # New partial unique index: one non-terminal session per (user, product, model_type)
    op.create_index(
        "ix_gpu_sessions_active_user_model",
        "gpu_sessions",
        ["user_id", "product_id", "model_type"],
        unique=True,
        postgresql_where=sa.text("status NOT IN ('stopped', 'failed')"),
    )

    # Remove server defaults used only for the NOT NULL migration
    op.alter_column("gpu_sessions", "bundle_name", server_default=None)
    op.alter_column("gpu_sessions", "model_type", server_default=None)


def downgrade() -> None:
    op.drop_table("gpu_sessions")
