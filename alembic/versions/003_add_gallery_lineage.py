"""Add gallery lineage columns to generation_jobs."""

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision: str = "003"
down_revision: str | None = "002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "generation_jobs",
        sa.Column("source_job_id", postgresql.UUID(as_uuid=True), nullable=True),
    )
    op.add_column(
        "generation_jobs",
        sa.Column("source_output_id", postgresql.UUID(as_uuid=True), nullable=True),
    )
    op.add_column(
        "generation_jobs",
        sa.Column("input_image_id", postgresql.UUID(as_uuid=True), nullable=True),
    )

    op.create_foreign_key(
        "fk_generation_jobs_source_job",
        "generation_jobs",
        "generation_jobs",
        ["source_job_id"],
        ["id"],
        ondelete="SET NULL",
    )
    op.create_foreign_key(
        "fk_generation_jobs_source_output",
        "generation_jobs",
        "generation_outputs",
        ["source_output_id"],
        ["id"],
        ondelete="SET NULL",
    )
    op.create_foreign_key(
        "fk_generation_jobs_input_image",
        "generation_jobs",
        "user_images",
        ["input_image_id"],
        ["id"],
        ondelete="SET NULL",
    )

    op.create_index(
        "ix_generation_jobs_source_job_id",
        "generation_jobs",
        ["source_job_id"],
    )
    op.create_index(
        "ix_generation_jobs_gallery",
        "generation_jobs",
        ["user_id", "product_id", "status", "created_at"],
    )


def downgrade() -> None:
    op.drop_index("ix_generation_jobs_gallery", table_name="generation_jobs")
    op.drop_index("ix_generation_jobs_source_job_id", table_name="generation_jobs")
    op.drop_constraint("fk_generation_jobs_input_image", "generation_jobs", type_="foreignkey")
    op.drop_constraint("fk_generation_jobs_source_output", "generation_jobs", type_="foreignkey")
    op.drop_constraint("fk_generation_jobs_source_job", "generation_jobs", type_="foreignkey")
    op.drop_column("generation_jobs", "input_image_id")
    op.drop_column("generation_jobs", "source_output_id")
    op.drop_column("generation_jobs", "source_job_id")
