"""Receipt of an upstream video task that a job row may never reference."""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from sqlalchemy import CheckConstraint, DateTime, ForeignKey, Index, Integer, String, text
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import Mapped, mapped_column

# Runtime import required: SQLAlchemy resolves Mapped[] annotations at runtime.
from src.core.enums import AcceptedTaskReceiptStatus
from src.core.uid import new_id
from src.db.models.base import Base


class AcceptedTaskReceipt(Base):
    """An upstream task id recorded the moment a provider accepts it.

    Written on a dedicated connection and committed *before* the request
    transaction ends; the request then ``DELETE``s the row under its own
    transaction, which holds the row lock until commit or rollback. A request
    that commits therefore removes the receipt atomically with its job, and a
    request that rolls back leaves an unlocked ``pending`` row — an orphan by
    construction. ``job_id`` deliberately has no foreign key: the job row may
    never commit.
    """

    __tablename__ = "accepted_task_receipts"

    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=new_id)
    provider: Mapped[str] = mapped_column(String(32), nullable=False)
    external_task_id: Mapped[str] = mapped_column(String(255), nullable=False)
    job_id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), nullable=False)
    user_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    product_id: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    model: Mapped[str] = mapped_column(String(64), nullable=False)
    token_cost: Mapped[int] = mapped_column(Integer, nullable=False)
    status: Mapped[AcceptedTaskReceiptStatus] = mapped_column(
        String(16),
        nullable=False,
        default=AcceptedTaskReceiptStatus.PENDING,
        server_default=text("'pending'"),
    )
    upstream_status: Mapped[str | None] = mapped_column(String(32), nullable=True)
    accepted_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=text("CURRENT_TIMESTAMP")
    )
    last_checked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    resolved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    __table_args__ = (
        Index(
            "uq_accepted_task_receipts_provider_task",
            "provider",
            "external_task_id",
            unique=True,
        ),
        CheckConstraint(
            "status IN ('pending','cancelled','absorbed')", name="chk_accepted_task_receipt_status"
        ),
        CheckConstraint(
            "(status = 'pending') = (resolved_at IS NULL)",
            name="chk_accepted_task_receipt_resolved_at",
        ),
        Index(
            "ix_accepted_task_receipts_pending",
            "provider",
            "accepted_at",
            postgresql_where=text("status = 'pending'"),
        ),
    )
