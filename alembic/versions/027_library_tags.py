"""Add library_tags and library_asset_tags.

Revision ID: 027
Revises: 026
Create Date: 2026-07-21 00:00:00.000000

Backs the Library Phase 3 "Tags" feature: normalized, per-user tags with
many-to-many asset tagging (unlike projects, which are one-per-asset).
T2: ``library_asset_tags`` carries denormalized ``user_id``/``product_id``
plus a CHECK on ``asset_type`` so scoped queries never need to join
``library_tags`` — same polymorphic rationale as ``library_asset_metadata``.
T3: tag names are unique per (product_id, user_id, lower(name)), enforced
via a functional index mirroring ``uq_library_projects_owner_name``.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision: str = "027"
down_revision: str | None = "026"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade database schema."""
    op.create_table(
        "library_tags",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("product_id", sa.String(32), nullable=False),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("name", sa.String(50), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("CURRENT_TIMESTAMP"),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("CURRENT_TIMESTAMP"),
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
    )
    op.create_index(
        "uq_library_tags_owner_name",
        "library_tags",
        ["product_id", "user_id", sa.text("lower(name)")],
        unique=True,
    )

    op.create_table(
        "library_asset_tags",
        sa.Column("tag_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("asset_type", sa.String(10), nullable=False),
        sa.Column("asset_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("product_id", sa.String(32), nullable=False),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("CURRENT_TIMESTAMP"),
        ),
        sa.PrimaryKeyConstraint("tag_id", "asset_type", "asset_id"),
        sa.ForeignKeyConstraint(["tag_id"], ["library_tags.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.CheckConstraint(
            "asset_type IN ('upload', 'output')",
            name="ck_library_asset_tags_asset_type",
        ),
    )
    op.create_index(
        "ix_library_asset_tags_asset",
        "library_asset_tags",
        ["asset_type", "asset_id"],
    )
    op.create_index(
        "ix_library_asset_tags_owner_tag",
        "library_asset_tags",
        ["user_id", "product_id", "tag_id"],
    )


def downgrade() -> None:
    """Downgrade database schema."""
    op.drop_index("ix_library_asset_tags_owner_tag", table_name="library_asset_tags")
    op.drop_index("ix_library_asset_tags_asset", table_name="library_asset_tags")
    op.drop_table("library_asset_tags")

    op.drop_index("uq_library_tags_owner_name", table_name="library_tags")
    op.drop_table("library_tags")
