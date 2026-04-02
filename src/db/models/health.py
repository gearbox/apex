"""Health snapshot model — persisted health check results for dashboards."""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from sqlalchemy import DateTime, Index, String, text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import Mapped, mapped_column

from src.db.models.base import Base


class HealthSnapshot(Base):
    """Persisted health check snapshot.

    Each row represents one complete health check cycle — all components
    checked, results aggregated, JSONB blob stored. The admin panel
    queries these for uptime/latency charts.

    No product_id column — the snapshot is global, but snapshot_data
    JSONB contains per-product provider breakdown internally.
    """

    __tablename__ = "health_snapshots"

    id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        primary_key=True,
        server_default=text("gen_random_uuid()"),
    )
    checked_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=text("CURRENT_TIMESTAMP"),
    )
    overall_status: Mapped[str] = mapped_column(
        String(16),
        nullable=False,
    )
    snapshot_data: Mapped[dict] = mapped_column(  # type: ignore[type-arg]
        JSONB,
        nullable=False,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=text("CURRENT_TIMESTAMP"),
    )

    __table_args__ = (
        Index(
            "ix_health_snapshots_checked_at",
            "checked_at",
            postgresql_using="btree",
        ),
    )
