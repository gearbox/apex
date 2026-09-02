"""GPU session operation model — latest durable state for one node operation."""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from sqlalchemy import DateTime, ForeignKey, Index, Integer, String, Text, text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import Mapped, mapped_column

from src.db.models.base import Base


class GpuSessionOperation(Base):
    """Latest received state for one operation emitted by a GPU session node."""

    __tablename__ = "gpu_session_operations"

    # Apex owns operation ids and generates UUIDv7 values before provisioning.
    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True)
    session_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("gpu_sessions.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        comment="GPU session that owns this operation.",
    )
    product_id: Mapped[str] = mapped_column(
        String(32), nullable=False, comment="Denormalized product scope from the owning session."
    )
    command_id: Mapped[UUID | None] = mapped_column(
        PG_UUID(as_uuid=True),
        nullable=True,
        comment="Future command-queue id; intentionally has no FK until P3.",
    )
    kind: Mapped[str] = mapped_column(String(20), nullable=False, comment="OperationKind value.")
    status: Mapped[str] = mapped_column(
        String(20), nullable=False, comment="OperationStatus value."
    )
    phase: Mapped[str | None] = mapped_column(
        String(32),
        nullable=True,
        comment="Latest ProvisioningPhase value; NULL for terminal events.",
    )

    target_bundle: Mapped[str | None] = mapped_column(String(100), nullable=True)
    target_bundle_version: Mapped[str | None] = mapped_column(String(20), nullable=True)
    target_mode: Mapped[str | None] = mapped_column(String(16), nullable=True)
    batch_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    batch_index: Mapped[int | None] = mapped_column(Integer, nullable=True)
    batch_total: Mapped[int | None] = mapped_column(Integer, nullable=True)

    last_sequence: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        server_default=text("-1"),
        comment="Highest applied event sequence; -1 means no event has been applied yet.",
    )
    last_event_id: Mapped[str | None] = mapped_column(
        String(64), nullable=True, comment="Opaque producer event id; not necessarily a UUID."
    )
    progress: Mapped[dict[str, Any] | None] = mapped_column(
        JSONB, nullable=True, comment="Latest generic progress object from the node."
    )
    plan: Mapped[dict[str, Any] | None] = mapped_column(
        JSONB, nullable=True, comment="Opaque execution plan from the operation start event."
    )
    summary: Mapped[dict[str, Any] | None] = mapped_column(
        JSONB, nullable=True, comment="Opaque timing summary from the first terminal event."
    )
    message: Mapped[str | None] = mapped_column(
        Text, nullable=True, comment="Latest node diagnostic."
    )
    error: Mapped[str | None] = mapped_column(
        Text, nullable=True, comment="Latest node-reported error."
    )
    node_started_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        comment="Operation start timestamp reported by the node.",
    )
    last_event_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        comment="Timestamp reported by the latest applied event.",
    )
    terminal_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        comment="Set once when the first terminal event is applied.",
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=text("CURRENT_TIMESTAMP")
    )

    __table_args__ = (
        Index("ix_gpu_session_operations_session_created", "session_id", created_at.desc()),
        Index(
            "ix_gpu_session_operations_batch",
            "batch_id",
            postgresql_where=text("batch_id IS NOT NULL"),
        ),
    )
