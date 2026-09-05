"""GPU session command model — one queue entry claimable by a node's provisioning agent.

A command is the durable, agent-facing counterpart of a gpu_session_operations row:
GpuSessionCommandService.enqueue()/enqueue_batch() write both atomically (the
operation is P1's telemetry sink, the command is what the agent actually claims and
executes). See GpuSessionCommandRepository for the claim statement and cascade rules.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from sqlalchemy import DateTime, ForeignKey, Index, Integer, String, Text, text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import Mapped, mapped_column

from src.db.models.base import Base


class GpuSessionCommand(Base):
    """One command in a GPU session's claim queue."""

    __tablename__ = "gpu_session_commands"

    # Apex owns command ids and generates UUIDv7 values via new_id() at enqueue time.
    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True)
    session_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("gpu_sessions.id", ondelete="CASCADE"),
        nullable=False,
        comment="GPU session whose agent may claim this command.",
    )
    product_id: Mapped[str] = mapped_column(
        String(32), nullable=False, comment="Denormalized product scope from the owning session."
    )
    operation_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        nullable=False,
        unique=True,
        comment="The gpu_session_operations row created atomically with this command.",
    )
    deployment_id: Mapped[UUID | None] = mapped_column(
        PG_UUID(as_uuid=True),
        nullable=True,
        comment="Forward slot for P4's attach/remove/restart routes; no writer in P3.",
    )
    kind: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        comment="OperationKind value; never session_bootstrap — see command_payload.py.",
    )
    payload: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        nullable=False,
        comment="Exactly the wire-format 'payload' object built by command_payload.py.",
    )
    batch_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    batch_index: Mapped[int | None] = mapped_column(Integer, nullable=True)
    batch_total: Mapped[int | None] = mapped_column(Integer, nullable=True)

    status: Mapped[str] = mapped_column(String(16), nullable=False, comment="CommandStatus value.")
    agent_id: Mapped[str | None] = mapped_column(
        String(128), nullable=True, comment="'{session_id}:{hostname}' of the claiming agent."
    )
    claimed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    deadline_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        comment="Stamped at claim time from the per-kind timeout setting (D26).",
    )
    terminal_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        comment="Set once when the command reaches a TERMINAL_COMMAND_STATUSES status.",
    )
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=text("CURRENT_TIMESTAMP")
    )

    __table_args__ = (
        Index(
            "ix_gpu_session_commands_one_claimed",
            "session_id",
            unique=True,
            postgresql_where=text("status = 'claimed'"),
        ),
        Index(
            "ix_gpu_session_commands_queue",
            "session_id",
            "created_at",
            "batch_index",
            postgresql_where=text("status = 'queued'"),
        ),
        Index(
            "ix_gpu_session_commands_deadline",
            "deadline_at",
            postgresql_where=text("status = 'claimed'"),
        ),
        Index(
            "ix_gpu_session_commands_batch",
            "batch_id",
            postgresql_where=text("batch_id IS NOT NULL"),
        ),
    )
