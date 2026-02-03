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
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from src.core.enums import GenerationType, JobStatus

if TYPE_CHECKING:
    from .user import User


class Base(DeclarativeBase):
    """Base class for all database models."""

    pass


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

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=text("CURRENT_TIMESTAMP"),
    )
    expires_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        index=True,  # For cleanup queries
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

    __table_args__ = (
        Index("ix_user_images_user_created", "user_id", "created_at"),
        Index("ix_user_images_cleanup", "expires_at"),
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
    generation_type: Mapped[GenerationType] = mapped_column(String(20), index=True, nullable=False)

    # Prompts
    prompt: Mapped[str] = mapped_column(Text, nullable=False)
    enhanced_prompt: Mapped[str | None] = mapped_column(Text, nullable=True)
    negative_prompt: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Analytics (populated on completion or by background job)
    theme_detected: Mapped[str | None] = mapped_column(String(100), index=True, nullable=True)
    theme_confidence: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    is_nsfw: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    is_minor_suspected: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    # ComfyUI tracking
    comfyui_prompt_id: Mapped[str | None] = mapped_column(String(64), nullable=True)

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
    outputs: Mapped[list[GenerationOutput]] = relationship(
        "GenerationOutput",
        back_populates="job",
        cascade="all, delete-orphan",
    )

    __table_args__ = (
        Index("ix_generation_jobs_user_status", "user_id", "status"),
        Index("ix_generation_jobs_user_created", "user_id", "created_at"),
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

    # Output index (for batch generation)
    output_index: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

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
        back_populates="outputs",
    )
    input_image: Mapped[UserImage | None] = relationship(
        "UserImage",
        back_populates="generation_outputs",
        foreign_keys=[input_image_id],
    )

    __table_args__ = (
        Index("ix_generation_outputs_job", "job_id"),
        Index("ix_generation_outputs_user_created", "user_id", "created_at"),
        Index("ix_generation_outputs_cleanup", "expires_at"),
    )

    def __repr__(self) -> str:
        return f"<GenerationOutput {self.id} job={self.job_id} index={self.output_index}>"
