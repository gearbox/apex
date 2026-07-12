"""Frame extraction job model — tracks video preview/extract background jobs.

A job is created at request time (owning exactly one source video: either a
GenerationOutput or a UserImage upload) and claimed by FrameExtractionWorker
via FOR UPDATE SKIP LOCKED. Deliberately not GenerationJob — provider/billing
coupling there would force nullable-column sprawl for a free, non-billed
operation. See docs/contracts/video-frame-extraction.md for the result JSON
contracts (§4 of the implementation prompt).
"""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from sqlalchemy import CheckConstraint, DateTime, ForeignKey, Index, String, Text, text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import Mapped, mapped_column

from src.db.models.base import Base


class FrameExtractionJob(Base):
    """Background job for video frame preview/extraction (see D5).

    ``params``/``result`` store keys and IDs only — never presigned URLs
    (those expire; presigning happens per read in FrameExtractionService).
    """

    __tablename__ = "frame_extraction_jobs"

    id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        primary_key=True,
    )
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

    kind: Mapped[str] = mapped_column(String(20), nullable=False)
    status: Mapped[str] = mapped_column(String(20), nullable=False, index=True)

    # Exactly one of these two is set at creation time (unlike UserImage
    # lineage, where both may degrade to NULL after source deletion) — a
    # queued job cascades away with its source rather than failing at run time.
    source_output_id: Mapped[UUID | None] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("generation_outputs.id", ondelete="CASCADE"),
        nullable=True,
    )
    source_upload_id: Mapped[UUID | None] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("user_images.id", ondelete="CASCADE"),
        nullable=True,
    )

    # params is a JSONB column typed as dict[str, Any];
    # when kind="preview", preview: {"frame_count": N};
    # when kind="extract"extract: {"timestamps_ms": [...]}
    params: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    # See docs/contracts/video-frame-extraction.md for the shape by kind/status.
    result: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=text("CURRENT_TIMESTAMP"),
    )
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    __table_args__ = (
        Index("ix_frame_extraction_jobs_claim", "status", "created_at"),
        CheckConstraint(
            "(source_output_id IS NOT NULL) != (source_upload_id IS NOT NULL)",
            name="ck_frame_extraction_jobs_exactly_one_source",
        ),
    )

    def __repr__(self) -> str:
        return f"<FrameExtractionJob {self.id} kind={self.kind} status={self.status}>"
