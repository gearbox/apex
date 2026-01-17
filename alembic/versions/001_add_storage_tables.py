"""Add storage tables for user content.

Revision ID: 001_add_storage_tables
Revises:
Create Date: 2026-01-18

"""

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "001_add_storage_tables"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Create storage tables."""
    # Create user_images table
    op.create_table(
        "user_images",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("storage_key", sa.String(512), nullable=False),
        sa.Column("original_filename", sa.String(255), nullable=False),
        sa.Column("content_type", sa.String(100), nullable=False),
        sa.Column("size_bytes", sa.Integer(), nullable=False),
        sa.Column("format", sa.String(10), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("storage_key"),
    )
    op.create_index("ix_user_images_user_id", "user_images", ["user_id"])
    op.create_index("ix_user_images_expires_at", "user_images", ["expires_at"])
    op.create_index("ix_user_images_user_created", "user_images", ["user_id", "created_at"])
    op.create_index("ix_user_images_cleanup", "user_images", ["expires_at"])

    # Create generation_jobs table
    op.create_table(
        "generation_jobs",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("prompt", sa.String(4096), nullable=False),
        sa.Column("status", sa.String(20), nullable=False, default="pending"),
        sa.Column("comfyui_prompt_id", sa.String(64), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_generation_jobs_user_id", "generation_jobs", ["user_id"])
    op.create_index("ix_generation_jobs_user_status", "generation_jobs", ["user_id", "status"])
    op.create_index("ix_generation_jobs_user_created", "generation_jobs", ["user_id", "created_at"])

    # Create generation_outputs table
    op.create_table(
        "generation_outputs",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("job_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("input_image_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("storage_key", sa.String(512), nullable=False),
        sa.Column("content_type", sa.String(100), nullable=False),
        sa.Column("size_bytes", sa.Integer(), nullable=False),
        sa.Column("format", sa.String(10), nullable=False),
        sa.Column("output_index", sa.Integer(), nullable=False, default=0),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["job_id"],
            ["generation_jobs.id"],
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["input_image_id"],
            ["user_images.id"],
            ondelete="SET NULL",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("storage_key"),
    )
    op.create_index("ix_generation_outputs_user_id", "generation_outputs", ["user_id"])
    op.create_index("ix_generation_outputs_job", "generation_outputs", ["job_id"])
    op.create_index(
        "ix_generation_outputs_user_created",
        "generation_outputs",
        ["user_id", "created_at"],
    )
    op.create_index("ix_generation_outputs_cleanup", "generation_outputs", ["expires_at"])


def downgrade() -> None:
    """Drop storage tables."""
    op.drop_table("generation_outputs")
    op.drop_table("generation_jobs")
    op.drop_table("user_images")
