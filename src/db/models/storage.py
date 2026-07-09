"""Database models for storage tracking.

These models track file metadata in PostgreSQL alongside the actual
file storage in R2. This enables:
- Efficient queries for user files without listing R2
- Retention policy enforcement via scheduled cleanup
- Association between uploads, jobs, and outputs
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING
from uuid import UUID

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    text,
)
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import Mapped, backref, mapped_column, relationship

from src.core.enums import GenerationType, JobStatus, Provider
from src.db.models.base import Base

if TYPE_CHECKING:
    from .gpu_session import GpuSession
    from .user import User


class UserImage(Base):
    """Tracks user-uploaded images in R2 storage.

    Each record represents a single uploaded image file that can be
    used as input for image-to-image generation.
    """

    __tablename__ = "user_images"

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

    # Storage location
    storage_key: Mapped[str] = mapped_column(
        String(512),
        nullable=False,
        unique=True,
    )

    # File metadata
    original_filename: Mapped[str] = mapped_column(String(255), nullable=False)
    content_type: Mapped[str] = mapped_column(String(100), nullable=False)
    size_bytes: Mapped[int] = mapped_column(Integer, nullable=False)
    format: Mapped[str] = mapped_column(String(10), nullable=False)  # png, jpeg, webp

    # Thumbnail support (mirrors GenerationOutput self-referential pattern)
    is_thumbnail: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        server_default=text("false"),
    )
    parent_image_id: Mapped[UUID | None] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("user_images.id", ondelete="CASCADE"),
        nullable=True,
        index=True,
    )
    thumbnail_max_edge: Mapped[int | None] = mapped_column(Integer, nullable=True)
    width: Mapped[int | None] = mapped_column(Integer, nullable=True)
    height: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=text("CURRENT_TIMESTAMP"),
    )
    expires_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        index=True,
    )

    # Relationships
    user: Mapped[User] = relationship(
        "User",
        back_populates="user_images",
    )
    generation_outputs: Mapped[list[GenerationOutput]] = relationship(
        "GenerationOutput",
        back_populates="input_image",
        foreign_keys="GenerationOutput.input_image_id",
    )
    derivatives: Mapped[list[UserImage]] = relationship(
        "UserImage",
        backref=backref("parent", remote_side="UserImage.id"),
        cascade="all, delete-orphan",
        single_parent=True,
        foreign_keys="UserImage.parent_image_id",
    )

    __table_args__ = (
        Index("ix_user_images_user_created", "user_id", "created_at"),
        Index("ix_user_images_parent_image_id", "parent_image_id"),
    )

    def __repr__(self) -> str:
        return f"<UserImage {self.id} user={self.user_id} key={self.storage_key}>"


class GenerationJob(Base):
    """Tracks generation jobs and their outputs.

    Links the generation request to its input images and output files.
    """

    __tablename__ = "generation_jobs"

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

    # Provider and model tracking
    provider: Mapped[str] = mapped_column(
        String(20),
        default=Provider.AISHA.value,
        index=True,
        nullable=False,
    )
    model: Mapped[str | None] = mapped_column(
        String(50),
        nullable=True,
        index=True,
    )
    # Job metadata
    name: Mapped[str] = mapped_column(String(255), nullable=False, default="Untitled Job")
    status: Mapped[JobStatus] = mapped_column(
        String(20),
        default=JobStatus.PENDING.value,
        index=True,
        nullable=False,
    )
    worker_id: Mapped[str | None] = mapped_column(String(100), nullable=True)
    retry_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    is_deleted: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        server_default=text("false"),
    )
    generation_type: Mapped[GenerationType] = mapped_column(String(20), index=True, nullable=False)

    # Generation parameters
    aspect_ratio: Mapped[str | None] = mapped_column(String(10), nullable=True)

    # Prompts
    prompt: Mapped[str] = mapped_column(Text, nullable=False)
    enhanced_prompt: Mapped[str | None] = mapped_column(Text, nullable=True)
    negative_prompt: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Analytics (populated on completion or by background job)
    theme_detected: Mapped[str | None] = mapped_column(String(100), index=True, nullable=True)
    theme_confidence: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    is_nsfw: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    is_minor_suspected: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    # Billing
    token_cost: Mapped[int | None] = mapped_column(Integer, nullable=True)
    debit_transaction_id: Mapped[UUID | None] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("token_transactions.id"),
        nullable=True,
    )

    # --- Lineage: remix tracking ---
    source_job_id: Mapped[UUID | None] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("generation_jobs.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )

    source_output_id: Mapped[UUID | None] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey(
            "generation_outputs.id",
            name="fk_generation_jobs_source_output",
            ondelete="SET NULL",
            use_alter=True,
        ),
        nullable=True,
    )

    input_image_id: Mapped[UUID | None] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("user_images.id", ondelete="SET NULL"),
        nullable=True,
    )

    # GPU session link (Aisha jobs only; NULL for Grok jobs)
    # CHECK constraint `ck_generation_jobs_aisha_has_session` in migration 006
    # enforces (provider != 'aisha') OR (gpu_session_id IS NOT NULL).
    gpu_session_id: Mapped[UUID | None] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("gpu_sessions.id", ondelete="SET NULL"),
        nullable=True,
        index=False,  # partial index defined in migration 006
    )

    # External provider tracking
    # For Grok: stores request_id (for video polling)
    # For Aisha: stores ComfyUI prompt_id
    external_request_id: Mapped[str | None] = mapped_column(String(64), nullable=True)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=text("CURRENT_TIMESTAMP"),
    )
    started_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    # Relationships
    user: Mapped[User] = relationship(
        "User",
        back_populates="generation_jobs",
    )
    gpu_session: Mapped[GpuSession | None] = relationship(
        "GpuSession",
        foreign_keys="GenerationJob.gpu_session_id",
        lazy="raise_on_sql",  # force explicit eager loading — prevents async lazy-load bugs
        uselist=False,
    )
    outputs: Mapped[list[GenerationOutput]] = relationship(
        "GenerationOutput",
        back_populates="job",
        cascade="all, delete-orphan",
        foreign_keys="GenerationOutput.job_id",
    )
    source_job: Mapped[GenerationJob | None] = relationship(
        "GenerationJob",
        remote_side="GenerationJob.id",
        foreign_keys="GenerationJob.source_job_id",
        uselist=False,
    )
    source_output: Mapped[GenerationOutput | None] = relationship(
        "GenerationOutput",
        foreign_keys="GenerationJob.source_output_id",
        uselist=False,
        overlaps="job",
    )
    input_image: Mapped[UserImage | None] = relationship(
        "UserImage",
        foreign_keys="GenerationJob.input_image_id",
        uselist=False,
        overlaps="generation_outputs",
    )

    __table_args__ = (
        Index(
            "ix_generation_jobs_gpu_session_id_status",
            "gpu_session_id",
            "status",
            postgresql_where=text("status IN ('queued', 'running')"),
        ),
        Index("ix_generation_jobs_user_status", "user_id", "status"),
        Index("ix_generation_jobs_user_created", "user_id", "created_at"),
        Index("ix_generation_jobs_provider_status", "provider", "status"),
        Index("ix_generation_jobs_gallery", "user_id", "product_id", "status", "created_at"),
        Index(
            "ix_generation_jobs_deleted",
            "user_id",
            "is_deleted",
            postgresql_where=text("is_deleted = TRUE"),
        ),
    )

    def __repr__(self) -> str:
        return f"<GenerationJob {self.id} user={self.user_id} status={self.status}>"


class GenerationOutput(Base):
    """Tracks generated output images in R2 storage.

    Each generation job can produce multiple output images.
    Outputs are linked to their source job and optionally to input images.
    """

    __tablename__ = "generation_outputs"

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
    job_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("generation_jobs.id", ondelete="CASCADE"),
        nullable=False,
    )

    product_id: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        index=True,
    )

    # Optional link to input image (for i2i)
    input_image_id: Mapped[UUID | None] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("user_images.id", ondelete="SET NULL"),
        nullable=True,
    )

    # Storage location
    storage_key: Mapped[str] = mapped_column(
        String(512),
        nullable=False,
        unique=True,
    )

    # File metadata
    content_type: Mapped[str] = mapped_column(String(100), nullable=False)
    size_bytes: Mapped[int] = mapped_column(Integer, nullable=False)
    format: Mapped[str] = mapped_column(String(10), nullable=False)

    # Thumbnail flag — True for derived previews (image thumbnails, video poster frames)
    is_thumbnail: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        server_default=text("false"),
    )

    # Size discriminator for thumbnail rows. NULL on full rows.
    # Values match ThumbnailSpec.max_edge (e.g. 150=sm, 512=md).
    thumbnail_max_edge: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # Output index (for batch generation)
    output_index: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    # Dimensions (nullable; populated for both full and thumbnail rows)
    width: Mapped[int | None] = mapped_column(Integer, nullable=True)
    height: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # Self-referential FK: thumbnail → full output (ON DELETE CASCADE)
    parent_output_id: Mapped[UUID | None] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("generation_outputs.id", ondelete="CASCADE"),
        nullable=True,
        index=True,
    )

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=text("CURRENT_TIMESTAMP"),
    )
    expires_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        index=True,
    )

    # Relationships
    user: Mapped[User] = relationship(
        "User",
        back_populates="generation_outputs",
    )
    job: Mapped[GenerationJob] = relationship(
        "GenerationJob",
        foreign_keys="GenerationOutput.job_id",
        back_populates="outputs",
    )
    input_image: Mapped[UserImage | None] = relationship(
        "UserImage",
        back_populates="generation_outputs",
        foreign_keys=[input_image_id],
    )
    # Self-referential: a full output has many derivatives (thumbnails/previews).
    derivatives: Mapped[list[GenerationOutput]] = relationship(
        "GenerationOutput",
        backref=backref("parent", remote_side="GenerationOutput.id"),
        cascade="all, delete-orphan",
        single_parent=True,
        foreign_keys="GenerationOutput.parent_output_id",
    )

    __table_args__ = (
        Index("ix_generation_outputs_job", "job_id"),
        Index("ix_generation_outputs_user_created", "user_id", "created_at"),
        Index("ix_generation_outputs_cleanup", "expires_at"),
        Index("ix_generation_outputs_thumbnail", "job_id", "is_thumbnail"),
    )

    def __repr__(self) -> str:
        return f"<GenerationOutput {self.id} job={self.job_id} index={self.output_index}>"
