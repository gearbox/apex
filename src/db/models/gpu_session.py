"""GPU session model — tracks on-demand Vast.ai node lifecycle.

Covers the full provisioning, tunnel, pause/resume, and billing workflow:
- Bundle identity: which ai-bundles bundle/version was deployed
- Cloudflare tunnel: tunnel ID, DNS record, and hostname for routing
- Vast.ai details: offer ID, GPU name, hourly cost captured at creation
- Provisioning tracking: attempt counter for retry logic
- Pause/resume timestamps: for billing and UX
- Telemetry v2: hashed node auth plus a pointer to the current bootstrap operation
"""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from sqlalchemy import (
    BigInteger,
    Boolean,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    text,
)
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import Mapped, mapped_column

from src.db.models.base import Base


class GpuSession(Base):
    """Tracks the lifecycle of an on-demand Vast.ai GPU node session.

    The health reconciler probes active/stale sessions and transitions
    unreachable ones to 'stale'. The provisioning worker drives
    pending → provisioning → active transitions and handles pause/resume.

    Key columns for health reconciliation:
    - status: only 'active' and 'stale' sessions are probed
    - tunnel_hostname: ComfyUI endpoint resolved via CF tunnel
    - last_reachable_at: stamped at activation and on each healthy probe
    - stale_detected_at: set by reconciler when probe fails past grace window
    - stale_notified: prevents duplicate admin notifications

    Key columns for provisioning:
    - bundle_name / bundle_version / model_type: what was deployed. This is the
      provisioning-time record only — it selects hardware and builds the boot
      env. Since P2 (gpu_session_deployments), the generation path reads model
      identity from the deployment, not from these columns.
    - vastai_offer_id / vastai_gpu_name / vastai_machine_id: Vast.ai node selected
    - provision_attempt: retry counter
    - cf_tunnel_id / cf_dns_record_id / tunnel_hostname: routing

    Since P2, the (user, product, model_type) uniqueness slot and
    readiness_marker_node_class live on GpuSessionDeployment, not here.
    """

    __tablename__ = "gpu_sessions"

    # Primary key — use UUIDv7 via new_id() at creation time
    id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        primary_key=True,
    )

    # Owner
    user_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    product_id: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        index=True,
    )

    # Bundle identity
    bundle_name: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        comment="ai-bundles bundle name (e.g. wan_2.2_i2v)",
    )
    bundle_version: Mapped[str | None] = mapped_column(
        String(20),
        nullable=True,
        comment="Specific bundle version (e.g. 260105-01). None = 'current' symlink",
    )
    model_type: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        comment="ModelType slug that triggered this session (e.g. aisha-image)",
    )

    # Session status
    status: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        index=True,
    )

    # Vast.ai node identity
    vastai_instance_id: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
        comment="Vast.ai instance/machine ID",
    )

    # Cloudflare tunnel
    cf_tunnel_id: Mapped[str | None] = mapped_column(
        String(64),
        nullable=True,
    )
    cf_dns_record_id: Mapped[str | None] = mapped_column(
        String(64),
        nullable=True,
    )
    tunnel_hostname: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
    )

    # Vast.ai details (captured at instance creation)
    vastai_offer_id: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
    )
    vastai_cost_per_hour_micros: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
        comment="Vast.ai $/hr in microdollars (1_000_000 = $1.00) at instance creation time",
    )
    vastai_gpu_name: Mapped[str | None] = mapped_column(
        String(50),
        nullable=True,
    )
    vastai_machine_id: Mapped[int | None] = mapped_column(
        BigInteger,
        nullable=True,
        comment="Vast.ai physical machine id of the selected offer; used for the broken-node cooldown list",
    )

    # Provisioning tracking
    provision_attempt: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        server_default=text("1"),
    )
    provisioning_started_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        comment="Set on first provision attempt (or reset on retry) — used for timeout calculation",
    )

    # Billing (set on session creation; used for debit/refund at stop time)
    account_id: Mapped[UUID | None] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("token_accounts.id", ondelete="RESTRICT"),
        nullable=True,
        index=True,
        comment="Billing account captured at start; used for finalization at stop.",
    )
    total_paused_seconds: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        server_default=text("0"),
        comment="Cumulative paused duration; subtracted from billable time at stop.",
    )

    # Pause/resume tracking
    paused_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    resumed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    # Node callback auth and bootstrap-operation projection
    callback_token_hash: Mapped[str | None] = mapped_column(
        String(64),
        nullable=True,
        comment="SHA-256 hex digest of the bearer token sent by the node. Plaintext never stored.",
    )
    bootstrap_operation_id: Mapped[UUID | None] = mapped_column(
        PG_UUID(as_uuid=True),
        nullable=True,
        comment=(
            "Current bootstrap operation id. No FK: the operation row already references this "
            "session and a reverse FK would make creation circular."
        ),
    )
    last_progress_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        comment="Wall-clock time when the last accepted callback was written. Used for stall detection.",
    )

    # Staleness tracking (health reconciler)
    stale_detected_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        comment="Set by health reconciler when node becomes unreachable",
    )
    stale_notified: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        server_default=text("false"),
        comment="Prevents duplicate admin notifications for stale sessions",
    )
    last_reachable_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        comment=(
            "Stamped at active transition and on each healthy reconciler probe. "
            "Used with gpu_session_stale_grace_seconds to prevent single-blip "
            "stale flapping."
        ),
    )

    # Error tracking
    error_message: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=text("CURRENT_TIMESTAMP"),
    )
    started_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        comment="When the node became ready (ComfyUI health check passed)",
    )
    stopped_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    billing_finalized_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        comment=(
            "Set by _finalize_billing on success. NULL after stop = a "
            "phase-2 reconciler worker should retry."
        ),
    )
    vastai_instance_destroyed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        comment=(
            "When destroy_instance returned success for this session's Vast.ai "
            "instance. NULL if destroy was never attempted, or attempted but "
            "failed. Used by the orphan sweeper to identify sessions where the "
            "instance may still be running despite the session being in a "
            "terminal state."
        ),
    )

    billing_finalization_attempts: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        server_default=text("0"),
        comment=(
            "Number of times the billing reconciler has attempted to finalize "
            "this session. Bumped each sweep when finalization fails. Used to "
            "trigger ops alerts after a quarantine threshold without mutating "
            "billing_finalized_at (the session must remain reconcilable once "
            "the underlying issue is fixed)."
        ),
    )

    # Credit guard warning state (set by SessionCreditGuard)
    credit_warning_level: Mapped[str | None] = mapped_column(
        String(10),
        nullable=True,
        comment="Current credit warning level (NotificationLevel value); NULL = not warned",
    )
    credit_warned_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        comment="When credit_warning_level was last set",
    )

    __table_args__ = (
        Index("ix_gpu_sessions_status_product", "status", "product_id"),
        Index(
            "ix_gpu_sessions_active_stale",
            "status",
            "stale_detected_at",
            postgresql_where=text("status IN ('active', 'stale', 'paused', 'resuming')"),
        ),
    )
