"""Add durable, versioned PDQ media hash ledger.

Revision ID: 046
Revises: 045
Create Date: 2026-09-22 00:00:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision: str = "046"
down_revision: str | Sequence[str] | None = "045"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Create hash rows independent from expiring upload/output rows."""
    op.create_table(
        "media_hashes",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("product_id", sa.String(length=32), nullable=False),
        sa.Column(
            "user_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "job_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("generation_jobs.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("source_kind", sa.String(length=16), nullable=False),
        sa.Column("source_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("source_media_type", sa.String(length=8), nullable=False),
        sa.Column("hash_profile", sa.String(length=128), nullable=False),
        sa.Column("sampling_profile", sa.String(length=64), nullable=False),
        sa.Column("sample_index", sa.Integer(), nullable=False),
        sa.Column("frame_timestamp_ms", sa.BigInteger(), nullable=True),
        sa.Column("pdq", postgresql.BIT(length=256), nullable=False),
        sa.Column("pdq_quality", sa.SmallInteger(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("CURRENT_TIMESTAMP"),
        ),
        sa.CheckConstraint(
            "source_kind IN ('output', 'upload')", name="ck_media_hashes_source_kind"
        ),
        sa.CheckConstraint(
            "source_media_type IN ('image', 'video')", name="ck_media_hashes_media_type"
        ),
        sa.CheckConstraint("sample_index >= 0", name="ck_media_hashes_sample_index"),
        sa.CheckConstraint(
            "frame_timestamp_ms IS NULL OR frame_timestamp_ms >= 0",
            name="ck_media_hashes_timestamp",
        ),
        sa.CheckConstraint(
            "(source_media_type = 'image' AND sample_index = 0 AND frame_timestamp_ms IS NULL) "
            "OR (source_media_type = 'video' AND frame_timestamp_ms IS NOT NULL)",
            name="ck_media_hashes_sample_shape",
        ),
        sa.CheckConstraint("pdq_quality BETWEEN 0 AND 100", name="ck_media_hashes_quality"),
    )
    op.create_index("ix_media_hashes_product_id", "media_hashes", ["product_id"])
    op.create_index("ix_media_hashes_user_id", "media_hashes", ["user_id"])
    op.create_index("ix_media_hashes_job_id", "media_hashes", ["job_id"])
    op.create_index(
        "uq_media_hashes_source_sample",
        "media_hashes",
        [
            "product_id",
            "source_kind",
            "source_id",
            "hash_profile",
            "sampling_profile",
            "sample_index",
        ],
        unique=True,
    )


def downgrade() -> None:
    """Remove only the additive ledger table."""
    op.drop_table("media_hashes")
