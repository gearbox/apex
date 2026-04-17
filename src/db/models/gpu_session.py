"""GPU session model — tracks on-demand Vast.ai node lifecycle.

Covers the full provisioning, tunnel, pause/resume, and billing workflow:
- Bundle identity: which ai-bundles bundle/version was deployed
- Cloudflare tunnel: tunnel ID, DNS record, and hostname for routing
- Vast.ai details: offer ID, GPU name, hourly cost captured at creation
- Provisioning tracking: attempt counter for retry logic
- Pause/resume timestamps: for billing and UX
- Phase 2 callback token: pre-wired for GPU → Apex callbacks
"""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from sqlalchemy import (
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

    The health reconciler probes active/stale/paused/resuming sessions and
    transitions unreachable ones to 'stale'. The provisioning worker drives
    pending → provisioning → active transitions and handles pause/resume.

    Key columns for health reconciliation:
    - status: 'active', 'stale', 'paused', 'resuming' sessions are probed
    - node_host / node_port: ComfyUI endpoint to probe
    - stale_detected_at: set by reconciler when probe fails
    - stale_notified: prevents duplicate admin notifications

    Key columns for provisioning:
    - bundle_name / bundle_version / model_type: what was deployed
    - vastai_offer_id / vastai_gpu_name: Vast.ai node selected
    - provision_attempt: retry counter
    - cf_tunnel_id / cf_dns_record_id / tunnel_hostname: routing
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
    node_host: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
        comment="GPU node hostname or IP",
    )
    node_port: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
        comment="ComfyUI port on the GPU node",
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

    # Provisioning tracking
    provision_attempt: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        server_default=text("1"),
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

    # Phase 2 callback
    callback_token: Mapped[str | None] = mapped_column(
        String(128),
        nullable=True,
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

    __table_args__ = (
        Index("ix_gpu_sessions_status_product", "status", "product_id"),
        Index(
            "ix_gpu_sessions_active_stale",
            "status",
            "stale_detected_at",
            postgresql_where=text("status IN ('active', 'stale', 'paused', 'resuming')"),
        ),
        Index(
            "ix_gpu_sessions_active_user_model",
            "user_id",
            "product_id",
            "model_type",
            unique=True,
            postgresql_where=text("status NOT IN ('stopped', 'failed')"),
        ),
    )
