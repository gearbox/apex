"""GPU session model — tracks on-demand Vast.ai node lifecycle."""

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

    The health reconciler probes active sessions and transitions
    unreachable ones to 'stale'. Future billing and recovery workers
    also operate on this table.

    Key columns for health reconciliation:
    - status: only 'active' and 'stale' sessions are probed
    - node_host / node_port: ComfyUI endpoint to probe
    - stale_detected_at: set by reconciler when probe fails
    - stale_notified: prevents duplicate admin notifications
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
            postgresql_where=text("status IN ('active', 'stale')"),
        ),
    )
