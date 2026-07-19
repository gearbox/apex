"""Add library_asset_metadata.

Revision ID: 025
Revises: 024
Create Date: 2026-07-19 00:00:00.000000

Backs the Library read model (Phase 1): per-asset metadata (favorite,
display title) that isn't naturally owned by either user_images or
generation_outputs, since a single library asset can be either.  Rows are
created lazily on first mutation (favorite/rename) — no eager backfill.
No project_id (D8): scoping is (product_id, user_id, asset_type, asset_id).
"""

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision: str = "025"
down_revision: str | None = "024"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade database schema."""
    op.create_table(
        "library_asset_metadata",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("product_id", sa.String(32), nullable=False),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("asset_type", sa.String(10), nullable=False),
        sa.Column("asset_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("display_title", sa.String(255), nullable=True),
        sa.Column("is_favorite", sa.Boolean(), nullable=False, server_default=sa.text("false")),
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
        sa.UniqueConstraint(
            "product_id",
            "user_id",
            "asset_type",
            "asset_id",
            name="uq_library_asset_metadata_asset",
        ),
        sa.CheckConstraint(
            "asset_type IN ('upload', 'output')",
            name="ck_library_asset_metadata_asset_type",
        ),
    )
    op.create_index(
        "ix_library_asset_metadata_favorites",
        "library_asset_metadata",
        ["user_id", "product_id"],
        postgresql_where=sa.text("is_favorite"),
    )


def downgrade() -> None:
    """Downgrade database schema."""
    op.drop_index("ix_library_asset_metadata_favorites", table_name="library_asset_metadata")
    op.drop_table("library_asset_metadata")
