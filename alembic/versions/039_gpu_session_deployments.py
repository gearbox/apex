"""Move model identity and routing uniqueness off gpu_sessions onto gpu_session_deployments.

Revision ID: 039
Revises: 038
Create Date: 2026-09-03 00:00:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision: str = "039"
down_revision: str | Sequence[str] | None = "038"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Create gpu_session_deployments, backfill it, then retire the old slot."""
    op.create_table(
        "gpu_session_deployments",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("session_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("product_id", sa.String(length=32), nullable=False),
        sa.Column("model_type", sa.String(length=32), nullable=False),
        sa.Column("bundle_name", sa.String(length=100), nullable=False),
        sa.Column("bundle_version", sa.String(length=20), nullable=True),
        sa.Column("readiness_marker_node_class", sa.String(length=100), nullable=True),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("pending_restart", sa.Boolean(), nullable=False, server_default=sa.text("false")),
        sa.Column("provision_operation_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("is_primary", sa.Boolean(), nullable=False, server_default=sa.text("false")),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("CURRENT_TIMESTAMP"),
        ),
        sa.Column("activated_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("removed_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(["session_id"], ["gpu_sessions.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_gpu_session_deployments_session",
        "gpu_session_deployments",
        ["session_id"],
    )
    op.create_index(
        "ix_gpu_session_deployments_routing",
        "gpu_session_deployments",
        ["user_id", "product_id", "model_type"],
        postgresql_where=sa.text("status = 'active'"),
    )

    # Backfill one deployment per non-terminal session, derived from the session's
    # own columns. gen_random_uuid() is acceptable here and only here: these are
    # backfill rows whose creation time is already carried by created_at, so
    # UUIDv7's ordering property buys nothing.
    op.execute(
        """
        INSERT INTO gpu_session_deployments
            (id, session_id, user_id, product_id, model_type, bundle_name, bundle_version,
             readiness_marker_node_class, status, is_primary, provision_operation_id,
             created_at, activated_at)
        SELECT gen_random_uuid(), s.id, s.user_id, s.product_id, s.model_type, s.bundle_name,
               s.bundle_version, s.readiness_marker_node_class,
               CASE WHEN s.status = 'active' THEN 'active' ELSE 'deploying' END,
               true, s.bootstrap_operation_id, s.created_at, s.started_at
        FROM gpu_sessions s
        WHERE s.status NOT IN ('stopped', 'failed')
        """
    )

    # The uniqueness index must be created only after the backfill, so the
    # window where neither the old nor the new constraint is watching for a
    # concurrent insert stays inside this one transaction.
    op.create_index(
        "ix_gpu_session_deployments_live_user_model",
        "gpu_session_deployments",
        ["user_id", "product_id", "model_type"],
        unique=True,
        postgresql_where=sa.text("status NOT IN ('removed', 'failed')"),
    )

    op.drop_index("ix_gpu_sessions_active_user_model", table_name="gpu_sessions")
    op.drop_column("gpu_sessions", "readiness_marker_node_class")


def downgrade() -> None:
    """Restore the v1 uniqueness slot and drop gpu_session_deployments.

    Repopulates readiness_marker_node_class from each session's primary
    deployment. Rebuilding ix_gpu_sessions_active_user_model will fail if a
    session ever acquired two live deployments — that cannot happen in P2, and
    failing here is the correct outcome if someone downgrades after P4.
    """
    op.add_column(
        "gpu_sessions",
        sa.Column("readiness_marker_node_class", sa.String(length=128), nullable=True),
    )
    op.execute(
        """
        UPDATE gpu_sessions s
        SET readiness_marker_node_class = d.readiness_marker_node_class
        FROM gpu_session_deployments d
        WHERE d.session_id = s.id AND d.is_primary
        """
    )
    op.create_index(
        "ix_gpu_sessions_active_user_model",
        "gpu_sessions",
        ["user_id", "product_id", "model_type"],
        unique=True,
        postgresql_where=sa.text("status NOT IN ('stopped', 'failed')"),
    )
    op.drop_table("gpu_session_deployments")
