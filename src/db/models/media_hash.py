"""Durable PDQ ledger rows, intentionally independent of expiring content."""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from sqlalchemy import (
    BigInteger,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    SmallInteger,
    String,
    text,
)
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import Mapped, mapped_column

from src.db.models.base import Base
from src.db.types import PdqBit256


class MediaHash(Base):
    """A durable, versioned PDQ sample for an original upload or output."""

    __tablename__ = "media_hashes"

    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True)
    product_id: Mapped[str] = mapped_column(String(32), nullable=False)
    user_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    job_id: Mapped[UUID | None] = mapped_column(
        PG_UUID(as_uuid=True), ForeignKey("generation_jobs.id", ondelete="SET NULL"), nullable=True
    )
    source_kind: Mapped[str] = mapped_column(String(16), nullable=False)
    source_id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), nullable=False)
    source_media_type: Mapped[str] = mapped_column(String(8), nullable=False)
    hash_profile: Mapped[str] = mapped_column(String(128), nullable=False)
    sampling_profile: Mapped[str] = mapped_column(String(64), nullable=False)
    sample_index: Mapped[int] = mapped_column(Integer, nullable=False)
    frame_timestamp_ms: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    pdq: Mapped[bytes] = mapped_column(PdqBit256(), nullable=False)
    pdq_quality: Mapped[int] = mapped_column(SmallInteger, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=text("CURRENT_TIMESTAMP")
    )

    __table_args__ = (
        CheckConstraint("source_kind IN ('output', 'upload')", name="ck_media_hashes_source_kind"),
        CheckConstraint(
            "source_media_type IN ('image', 'video')", name="ck_media_hashes_media_type"
        ),
        CheckConstraint("sample_index >= 0", name="ck_media_hashes_sample_index"),
        CheckConstraint(
            "frame_timestamp_ms IS NULL OR frame_timestamp_ms >= 0",
            name="ck_media_hashes_timestamp",
        ),
        CheckConstraint(
            "(source_media_type = 'image' AND sample_index = 0 AND frame_timestamp_ms IS NULL) "
            "OR (source_media_type = 'video' AND frame_timestamp_ms IS NOT NULL)",
            name="ck_media_hashes_sample_shape",
        ),
        CheckConstraint("pdq_quality BETWEEN 0 AND 100", name="ck_media_hashes_quality"),
        Index("ix_media_hashes_product_id", "product_id"),
        Index("ix_media_hashes_user_id", "user_id"),
        Index("ix_media_hashes_job_id", "job_id"),
        Index(
            "uq_media_hashes_source_sample",
            "product_id",
            "source_kind",
            "source_id",
            "hash_profile",
            "sampling_profile",
            "sample_index",
            unique=True,
        ),
    )
