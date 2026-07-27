"""Add user_images.display_filename (split out of 030).

Revision ID: 031
Revises: 030
Create Date: 2026-07-27 00:00:00.000000

Kept separate from 030 deliberately: 030 disables the token_transactions
immutability trigger, which takes ACCESS EXCLUSIVE on that table and holds
it until the migration commits. Bundling an unrelated add_column into the
same transaction extends that lock over DDL that has nothing to do with the
ledger. Split so each revision holds exactly the locks it needs.

display_filename is a nullable, display/search-only copy of the
client-supplied upload filename, NFC-normalized and control-char stripped at
write time. Historical rows are intentionally left NULL — there is no safe
way to recover the original name from original_filename once it has been
overwritten with the canonical {uuid}.{ext} system name by the
application-side change in this same release.
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "031"
down_revision: str | None = "030"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Add user_images.display_filename."""
    op.add_column(
        "user_images",
        sa.Column("display_filename", sa.String(255), nullable=True),
    )


def downgrade() -> None:
    """Drop user_images.display_filename."""
    op.drop_column("user_images", "display_filename")
