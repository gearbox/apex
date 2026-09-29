"""In-product problem report (feedback) model."""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from sqlalchemy import CheckConstraint, DateTime, ForeignKey, Index, String, Text, text
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import Mapped, mapped_column

from src.core.uid import new_id
from src.db.models.base import Base

__all__ = ["FeedbackReport"]


class FeedbackReport(Base):
    """A user-submitted problem report — the single source of truth (D1).

    ``category``/``status`` hold ``FeedbackCategory``/``FeedbackStatus``
    values as plain strings, validated at the msgspec boundary (no CHECK on
    the value sets, so the enums can grow without a migration).

    ``asset_source`` + ``asset_id`` are a polymorphic reference with no FK:
    outputs are retention-swept, so the reference may dangle. ``user_id`` is
    always set on insert; it is nullable only so a hard user purge keeps the
    report (``ON DELETE SET NULL``).
    """

    __tablename__ = "feedback_reports"

    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=new_id)
    product_id: Mapped[str] = mapped_column(String(32), nullable=False)
    user_id: Mapped[UUID | None] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
    )
    category: Mapped[str] = mapped_column(String(32), nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False, server_default="open")
    message: Mapped[str] = mapped_column(Text, nullable=False)
    job_id: Mapped[UUID | None] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("generation_jobs.id", ondelete="SET NULL"),
        nullable=True,
    )
    asset_source: Mapped[str | None] = mapped_column(String(16), nullable=True)
    asset_id: Mapped[UUID | None] = mapped_column(PG_UUID(as_uuid=True), nullable=True)
    client_path: Mapped[str | None] = mapped_column(String(512), nullable=True)
    app_version: Mapped[str | None] = mapped_column(String(64), nullable=True)
    user_agent: Mapped[str | None] = mapped_column(String(512), nullable=True)
    admin_note: Mapped[str | None] = mapped_column(Text, nullable=True)
    resolved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    resolved_by: Mapped[UUID | None] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=text("CURRENT_TIMESTAMP"),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=text("CURRENT_TIMESTAMP"),
        onupdate=text("clock_timestamp()"),
    )

    __table_args__ = (
        CheckConstraint(
            "(asset_source IS NULL) = (asset_id IS NULL)",
            name="chk_feedback_asset_pair",
        ),
        CheckConstraint(
            "(status IN ('resolved', 'dismissed')) = (resolved_at IS NOT NULL)",
            name="chk_feedback_resolved_at_terminal",
        ),
        Index("ix_feedback_reports_product_created", "product_id", "created_at", "id"),
    )
    # Server-generated created_at/updated_at are fetched via RETURNING at
    # flush, so a row staged by a commit-free service is fully readable after
    # the route's commit (expire_on_commit=False) without a lazy load.
    __mapper_args__ = {"eager_defaults": True}  # noqa: RUF012

    def __repr__(self) -> str:
        return f"<FeedbackReport {self.id} status={self.status} category={self.category}>"
