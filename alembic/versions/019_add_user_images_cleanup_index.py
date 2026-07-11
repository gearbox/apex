"""Add ix_user_images_cleanup index on user_images.expires_at.

Revision ID: 019
Revises: 018
Create Date: 2026-07-11 00:00:00.000000

user_images had no index on expires_at (generation_outputs already has
ix_generation_outputs_cleanup). Adds one for symmetry, used by the content
retention sweeper's get_expired() query.
"""

from collections.abc import Sequence

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "019"
down_revision: str | None = "018"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade database schema."""
    op.create_index("ix_user_images_cleanup", "user_images", ["expires_at"])


def downgrade() -> None:
    """Downgrade database schema."""
    op.drop_index("ix_user_images_cleanup", table_name="user_images")
