"""Add library_projects and library_asset_metadata.project_id.

Revision ID: 026
Revises: 025
Create Date: 2026-07-20 00:00:00.000000

Backs the Library Phase 2 "Projects" feature: user-created groupings for
library assets. P1: one project per asset (nullable FK on the metadata
row) — many-to-many membership is deferred. P2: deleting a project sets
``library_asset_metadata.project_id`` to NULL; assets themselves are
never touched by a project deletion. P7: project names are unique per
(product_id, user_id) case-insensitively, enforced via a functional
index on ``lower(name)``.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision: str = "026"
down_revision: str | None = "025"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade database schema."""
    op.create_table(
        "library_projects",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("product_id", sa.String(32), nullable=False),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("name", sa.String(100), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
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
        "uq_library_projects_owner_name",
        "library_projects",
        ["product_id", "user_id", sa.text("lower(name)")],
        unique=True,
    )

    op.add_column(
        "library_asset_metadata",
        sa.Column("project_id", postgresql.UUID(as_uuid=True), nullable=True),
    )
    op.create_foreign_key(
        "fk_library_asset_metadata_project_id",
        "library_asset_metadata",
        "library_projects",
        ["project_id"],
        ["id"],
        ondelete="SET NULL",
    )
    op.create_index(
        "ix_library_asset_metadata_project",
        "library_asset_metadata",
        ["project_id"],
    )


def downgrade() -> None:
    """Downgrade database schema."""
    op.drop_index("ix_library_asset_metadata_project", table_name="library_asset_metadata")
    op.drop_constraint(
        "fk_library_asset_metadata_project_id", "library_asset_metadata", type_="foreignkey"
    )
    op.drop_column("library_asset_metadata", "project_id")

    op.drop_index("uq_library_projects_owner_name", table_name="library_projects")
    op.drop_table("library_projects")
