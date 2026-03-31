"""Idempotency key model for request deduplication."""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from sqlalchemy import DateTime, ForeignKey, Index, SmallInteger, String, UniqueConstraint, text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import Mapped, mapped_column

from src.db.models.base import Base


class IdempotencyKey(Base):
    """Tracks client-supplied idempotency keys to prevent duplicate mutations.

    Scoped to (user_id, product_id, idempotency_key).
    Stores cached response for replay on retry.
    """

    __tablename__ = "idempotency_keys"

    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True)
    user_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
    )
    product_id: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    idempotency_key: Mapped[str] = mapped_column(String(64), nullable=False)
    operation: Mapped[str] = mapped_column(String(30), nullable=False)
    status: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        server_default=text("'processing'"),
    )
    resource_id: Mapped[UUID | None] = mapped_column(PG_UUID(as_uuid=True), nullable=True)
    response_status_code: Mapped[int | None] = mapped_column(SmallInteger, nullable=True)
    response_body: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    request_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=text("CURRENT_TIMESTAMP"),
    )
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    __table_args__ = (
        UniqueConstraint(
            "user_id",
            "product_id",
            "idempotency_key",
            name="uq_idempotency_user_product_key",
        ),
        Index("ix_idempotency_keys_expires_at", "expires_at"),
    )

    def __repr__(self) -> str:
        return f"<IdempotencyKey {self.idempotency_key} op={self.operation} status={self.status}>"
