"""Drop frame_extraction_jobs (server-side frame extraction removed).

Revision ID: 051
Revises: 050
Create Date: 2026-10-07 00:00:00.000000

Frame extraction now runs client-side. The ``user_images`` lineage columns,
FKs, indexes and ``ck_user_images_single_frame_source`` added by migration 020
are deliberately kept: client-captured frames write them.

``downgrade()`` recreates the table exactly as migration 020 defined it.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import UUID as PG_UUID

from alembic import op

revision: str = "051"
down_revision: str | Sequence[str] | None = "050"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade database schema."""
    # Indexes, FKs and the check constraint go with the table.
    op.drop_table("frame_extraction_jobs")


def downgrade() -> None:
    """Downgrade database schema."""
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
            sa.ForeignKey(
                "generation_outputs.id",
                name="fk_frame_extraction_jobs_source_output_id",
                ondelete="CASCADE",
            ),
            nullable=True,
        ),
        sa.Column(
            "source_upload_id",
            PG_UUID(as_uuid=True),
            sa.ForeignKey(
                "user_images.id",
                name="fk_frame_extraction_jobs_source_upload_id",
                ondelete="CASCADE",
            ),
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
