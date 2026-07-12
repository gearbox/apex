"""Add video frame extraction support: user_images lineage + frame_extraction_jobs.

Revision ID: 020
Revises: 019
Create Date: 2026-07-12 00:00:00.000000

Changes:
- user_images: add source_output_id / source_upload_id (nullable FKs, ON DELETE
  SET NULL — deleting the source video must not delete extracted frames),
  source_timestamp_ms, duration_ms; CHECK at most one source FK is set.
- frame_extraction_jobs: new table backing the preview/extract background job
  queue (FrameExtractionWorker claims via FOR UPDATE SKIP LOCKED). CHECK
  exactly one source FK is set (unlike user_images lineage, a job's source
  must exist at creation time — the row cascades away with its source instead
  of failing at run time).
"""

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import UUID as PG_UUID

from alembic import op

revision: str = "020"
down_revision: str | None = "019"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade database schema."""
    # -------------------------------------------------------------------
    # user_images: frame-extraction lineage columns
    # -------------------------------------------------------------------
    op.add_column(
        "user_images",
        sa.Column("source_output_id", PG_UUID(as_uuid=True), nullable=True),
    )
    op.add_column(
        "user_images",
        sa.Column("source_upload_id", PG_UUID(as_uuid=True), nullable=True),
    )
    op.add_column(
        "user_images",
        sa.Column("source_timestamp_ms", sa.Integer(), nullable=True),
    )
    op.add_column(
        "user_images",
        sa.Column("duration_ms", sa.Integer(), nullable=True),
    )

    op.create_index(
        "ix_user_images_source_output_id",
        "user_images",
        ["source_output_id"],
    )
    op.create_index(
        "ix_user_images_source_upload_id",
        "user_images",
        ["source_upload_id"],
    )

    op.create_foreign_key(
        "fk_user_images_source_output",
        "user_images",
        "generation_outputs",
        ["source_output_id"],
        ["id"],
        ondelete="SET NULL",
    )
    op.create_foreign_key(
        "fk_user_images_source_upload_id",
        "user_images",
        "user_images",
        ["source_upload_id"],
        ["id"],
        ondelete="SET NULL",
    )

    op.create_check_constraint(
        "ck_user_images_single_frame_source",
        "user_images",
        "NOT (source_output_id IS NOT NULL AND source_upload_id IS NOT NULL)",
    )

    # -------------------------------------------------------------------
    # frame_extraction_jobs
    # -------------------------------------------------------------------
    op.create_table(
        "frame_extraction_jobs",
        sa.Column("id", PG_UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "user_id",
            PG_UUID(as_uuid=True),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("product_id", sa.String(32), nullable=False),
        sa.Column("kind", sa.String(20), nullable=False),
        sa.Column("status", sa.String(20), nullable=False),
        sa.Column(
            "source_output_id",
            PG_UUID(as_uuid=True),
            sa.ForeignKey("generation_outputs.id", ondelete="CASCADE"),
            nullable=True,
        ),
        sa.Column(
            "source_upload_id",
            PG_UUID(as_uuid=True),
            sa.ForeignKey("user_images.id", ondelete="CASCADE"),
            nullable=True,
        ),
        sa.Column("params", JSONB, nullable=False),
        sa.Column("result", JSONB, nullable=True),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("CURRENT_TIMESTAMP"),
        ),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint(
            "(source_output_id IS NOT NULL) != (source_upload_id IS NOT NULL)",
            name="ck_frame_extraction_jobs_exactly_one_source",
        ),
    )

    op.create_index(
        "ix_frame_extraction_jobs_user_id",
        "frame_extraction_jobs",
        ["user_id"],
    )
    op.create_index(
        "ix_frame_extraction_jobs_product_id",
        "frame_extraction_jobs",
        ["product_id"],
    )
    op.create_index(
        "ix_frame_extraction_jobs_status",
        "frame_extraction_jobs",
        ["status"],
    )
    op.create_index(
        "ix_frame_extraction_jobs_claim",
        "frame_extraction_jobs",
        ["status", "created_at"],
    )


def downgrade() -> None:
    """Downgrade database schema."""
    op.drop_table("frame_extraction_jobs")

    op.drop_constraint(
        "ck_user_images_single_frame_source",
        "user_images",
        type_="check",
    )
    op.drop_constraint(
        "fk_user_images_source_upload_id",
        "user_images",
        type_="foreignkey",
    )
    op.drop_constraint(
        "fk_user_images_source_output",
        "user_images",
        type_="foreignkey",
    )
    op.drop_index("ix_user_images_source_upload_id", table_name="user_images")
    op.drop_index("ix_user_images_source_output_id", table_name="user_images")
    op.drop_column("user_images", "duration_ms")
    op.drop_column("user_images", "source_timestamp_ms")
    op.drop_column("user_images", "source_upload_id")
    op.drop_column("user_images", "source_output_id")
