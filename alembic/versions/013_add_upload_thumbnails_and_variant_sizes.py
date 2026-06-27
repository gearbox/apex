"""Add upload thumbnail support and thumbnail_max_edge size discriminator.

Revision ID: 013
Revises: 012
Create Date: 2026-06-27 00:00:00.000000

Changes:
- generation_outputs: add thumbnail_max_edge INT NULL (size discriminator for sm/md variants)
- user_images: add is_thumbnail BOOL NOT NULL DEFAULT false, parent_image_id UUID NULL (self-FK
  ON DELETE CASCADE), thumbnail_max_edge INT NULL, width INT NULL, height INT NULL
- Data fix (not content backfill): existing output thumbnail rows get thumbnail_max_edge=512
  so they correctly label as 'md' in the new variant system
"""

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID as PG_UUID

from alembic import op

revision: str = "013"
down_revision: str | None = "012"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade database schema."""
    # -----------------------------------------------------------------------
    # generation_outputs: add thumbnail_max_edge size discriminator
    # -----------------------------------------------------------------------
    op.add_column(
        "generation_outputs",
        sa.Column("thumbnail_max_edge", sa.Integer(), nullable=True),
    )

    # Label existing thumbnail rows as md (512) so they resolve correctly.
    op.execute(
        "UPDATE generation_outputs "
        "SET thumbnail_max_edge = 512 "
        "WHERE is_thumbnail = true AND thumbnail_max_edge IS NULL"
    )

    # -----------------------------------------------------------------------
    # user_images: thumbnail columns + self-referential FK
    # -----------------------------------------------------------------------
    op.add_column(
        "user_images",
        sa.Column(
            "is_thumbnail",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("false"),
        ),
    )
    op.add_column(
        "user_images",
        sa.Column("parent_image_id", PG_UUID(as_uuid=True), nullable=True),
    )
    op.add_column(
        "user_images",
        sa.Column("thumbnail_max_edge", sa.Integer(), nullable=True),
    )
    op.add_column(
        "user_images",
        sa.Column("width", sa.Integer(), nullable=True),
    )
    op.add_column(
        "user_images",
        sa.Column("height", sa.Integer(), nullable=True),
    )

    op.create_index(
        "ix_user_images_parent_image_id",
        "user_images",
        ["parent_image_id"],
    )

    op.create_foreign_key(
        "fk_user_images_parent_image_id",
        "user_images",
        "user_images",
        ["parent_image_id"],
        ["id"],
        ondelete="CASCADE",
    )


def downgrade() -> None:
    """Downgrade database schema."""
    op.drop_constraint(
        "fk_user_images_parent_image_id",
        "user_images",
        type_="foreignkey",
    )
    op.drop_index("ix_user_images_parent_image_id", table_name="user_images")
    op.drop_column("user_images", "height")
    op.drop_column("user_images", "width")
    op.drop_column("user_images", "thumbnail_max_edge")
    op.drop_column("user_images", "parent_image_id")
    op.drop_column("user_images", "is_thumbnail")
    op.drop_column("generation_outputs", "thumbnail_max_edge")
