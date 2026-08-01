"""Add durable Grok video finalization claims and output uniqueness guards.

Revision ID: 035
Revises: 034
Create Date: 2026-08-01 00:00:00.000000

The cleanup statements retain the oldest valid output/thumbnail before adding
the partial unique indexes.  They repair duplicate metadata created by older
concurrent pollers; physical objects no longer referenced by metadata remain
eligible for the normal R2 retention/orphan cleanup process.
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "035"
down_revision: str | Sequence[str] | None = "034"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Add recoverable claims and constrain output materialization to once."""
    op.add_column(
        "generation_jobs",
        sa.Column("finalization_claim_token", sa.String(length=36), nullable=True),
    )
    op.add_column(
        "generation_jobs",
        sa.Column("finalization_lease_expires_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index(
        "ix_generation_jobs_finalization_lease",
        "generation_jobs",
        ["finalization_lease_expires_at"],
        postgresql_where=sa.text("finalization_claim_token IS NOT NULL"),
    )

    # A thumbnail belongs to a particular full output/size bucket.  Delete
    # children first so the full-output cleanup does not leave dangling rows.
    op.execute(
        """
        WITH ranked AS (
            SELECT id,
                   row_number() OVER (
                       PARTITION BY parent_output_id, thumbnail_max_edge
                       ORDER BY created_at ASC, id ASC
                   ) AS duplicate_rank
            FROM generation_outputs
            WHERE is_thumbnail = TRUE
        )
        DELETE FROM generation_outputs AS output
        USING ranked
        WHERE output.id = ranked.id AND ranked.duplicate_rank > 1
        """
    )
    op.execute(
        """
        WITH ranked AS (
            SELECT id,
                   row_number() OVER (
                       PARTITION BY job_id, output_index
                       ORDER BY created_at ASC, id ASC
                   ) AS duplicate_rank
            FROM generation_outputs
            WHERE is_thumbnail = FALSE
        )
        DELETE FROM generation_outputs AS output
        USING ranked
        WHERE output.id = ranked.id AND ranked.duplicate_rank > 1
        """
    )
    op.create_index(
        "uq_generation_outputs_full_job_index",
        "generation_outputs",
        ["job_id", "output_index"],
        unique=True,
        postgresql_where=sa.text("is_thumbnail = FALSE"),
    )
    op.create_index(
        "uq_generation_outputs_thumbnail_parent_bucket",
        "generation_outputs",
        ["parent_output_id", "thumbnail_max_edge"],
        unique=True,
        postgresql_where=sa.text("is_thumbnail = TRUE"),
    )


def downgrade() -> None:
    """Remove claim columns and uniqueness guards (duplicate cleanup is irreversible)."""
    op.drop_index("uq_generation_outputs_thumbnail_parent_bucket", table_name="generation_outputs")
    op.drop_index("uq_generation_outputs_full_job_index", table_name="generation_outputs")
    op.drop_index("ix_generation_jobs_finalization_lease", table_name="generation_jobs")
    op.drop_column("generation_jobs", "finalization_lease_expires_at")
    op.drop_column("generation_jobs", "finalization_claim_token")
