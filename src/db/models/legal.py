"""Append-only legal acceptance / consent-withdrawal events."""

from __future__ import annotations

from datetime import date, datetime
from uuid import UUID

from sqlalchemy import (
    BigInteger,
    CheckConstraint,
    Date,
    DateTime,
    ForeignKey,
    Identity,
    Index,
    String,
    UniqueConstraint,
    text,
)
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import Mapped, mapped_column

# Runtime import required: SQLAlchemy resolves Mapped[] annotations at runtime.
from src.core.enums import LegalAcceptanceSource, LegalAction, LegalDocumentType  # noqa: TC001
from src.db.models.base import Base


class LegalAcceptance(Base):
    """One acceptance (or consent withdrawal) of one legal document version.

    Insert-only: rows are never updated or deleted by application code (they
    cascade only with a hard user deletion). ``content_sha256`` is the hash of
    the exact text the user accepted — proof of *what* was accepted, not just
    which version label.

    Event order is ``seq``, never a timestamp. Every insert for a user runs
    after taking that user's ledger advisory lock
    (``LegalAcceptanceRepository.lock_user_ledger``) and identity values are
    drawn at INSERT, so ``seq`` order is exactly the serialized write order.
    ``created_at`` is ``clock_timestamp()`` — when the row was written — and
    is evidence only: a transaction-start timestamp would misorder a
    long-running transaction that took the lock late.
    """

    __tablename__ = "legal_acceptances"

    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True)
    seq: Mapped[int] = mapped_column(BigInteger, Identity(always=True), nullable=False)
    user_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    product_id: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    doc_type: Mapped[LegalDocumentType] = mapped_column(String(32), nullable=False)
    action: Mapped[LegalAction] = mapped_column(String(32), nullable=False)
    version: Mapped[date] = mapped_column(Date, nullable=False)
    content_sha256: Mapped[str | None] = mapped_column(String(64), nullable=True)
    source: Mapped[LegalAcceptanceSource] = mapped_column(String(32), nullable=False)
    ip_address: Mapped[str | None] = mapped_column(String(64), nullable=True)
    user_agent: Mapped[str | None] = mapped_column(String(512), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=text("clock_timestamp()")
    )

    __table_args__ = (
        UniqueConstraint("seq", name="uq_legal_acceptances_seq"),
        Index(
            "ix_legal_acceptances_user_product_doc_seq",
            "user_id",
            "product_id",
            "doc_type",
            "seq",
        ),
        CheckConstraint(
            "action <> 'accept' OR content_sha256 IS NOT NULL",
            name="ck_legal_acceptances_accept_has_hash",
        ),
    )
