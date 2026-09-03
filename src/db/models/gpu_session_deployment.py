"""GPU session deployment model — model identity and routing uniqueness for one session.

Since P2, a deployment (not the parent GpuSession) is the source of truth for which
model is routable on a session. Every session has exactly one deployment today,
created with the session and torn down with it (see GpuSessionService._set_status
and GpuProvisioningWorker._transition for the lifecycle cascade). P4 will let a
session carry more than one.
"""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from sqlalchemy import Boolean, DateTime, ForeignKey, Index, String, text
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import Mapped, mapped_column

from src.db.models.base import Base


class GpuSessionDeployment(Base):
    """Tracks one model deployment's identity, status, and routing eligibility.

    Key columns for routing:
    - status: only 'active' deployments under an 'active' session are routable
      (see GpuSessionDeploymentRepository.get_routable)
    - user_id / product_id / model_type: denormalized from the session so the
      uniqueness index below can be local to this table

    Key columns for lifecycle:
    - is_primary: the deployment created with the session. A record, not a rule —
      nothing may branch on it in P2 beyond backfill and diagnostics.
    - pending_restart: forward slot for P4; no writer in P2.
    - provision_operation_id: the operation that created (or, on retry, is
      recreating) this deployment.
    """

    __tablename__ = "gpu_session_deployments"

    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True)
    session_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("gpu_sessions.id", ondelete="CASCADE"),
        nullable=False,
        comment="GPU session that owns this deployment.",
    )
    user_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        comment="Denormalized from the session — the uniqueness index needs it locally.",
    )
    product_id: Mapped[str] = mapped_column(String(32), nullable=False)
    model_type: Mapped[str] = mapped_column(String(32), nullable=False)
    bundle_name: Mapped[str] = mapped_column(String(100), nullable=False)
    bundle_version: Mapped[str | None] = mapped_column(
        String(20),
        nullable=True,
        comment="Specific bundle version. NULL means the 'current' symlink was used.",
    )
    readiness_marker_node_class: Mapped[str | None] = mapped_column(
        String(100),
        nullable=True,
        comment=(
            "ComfyUI class name to require in /object_info during readiness probe. "
            "NULL = fall back to 200-OK with ERROR log. Moved off gpu_sessions in P2."
        ),
    )
    status: Mapped[str] = mapped_column(
        String(16),
        nullable=False,
        comment="DeploymentStatus value.",
    )
    pending_restart: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        server_default=text("false"),
        comment="Forward slot for P4's restart flow; no writer in P2.",
    )
    provision_operation_id: Mapped[UUID | None] = mapped_column(
        PG_UUID(as_uuid=True),
        nullable=True,
        comment=(
            "The operation that (re)provisioned this deployment. For the primary "
            "deployment this is the session's bootstrap operation; a retry "
            "repoints it at the new bootstrap operation without forking the row."
        ),
    )
    is_primary: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        server_default=text("false"),
        comment="The deployment created with the session. A record, not a rule — see module docstring.",
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=text("CURRENT_TIMESTAMP"),
    )
    activated_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    removed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    __table_args__ = (
        # Mirrors LIVE_DEPLOYMENT_STATUSES / TERMINAL_DEPLOYMENT_STATUSES in
        # src/core/enums.py — the two must always agree; see the comment there.
        Index(
            "ix_gpu_session_deployments_live_user_model",
            "user_id",
            "product_id",
            "model_type",
            unique=True,
            postgresql_where=text("status NOT IN ('removed', 'failed')"),
        ),
        Index("ix_gpu_session_deployments_session", "session_id"),
        Index(
            "ix_gpu_session_deployments_routing",
            "user_id",
            "product_id",
            "model_type",
            postgresql_where=text("status = 'active'"),
        ),
    )
