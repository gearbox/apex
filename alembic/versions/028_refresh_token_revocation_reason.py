"""Add revoked_reason to refresh_tokens.

Revision ID: 028
Revises: 027
Create Date: 2026-07-26 00:00:00.000000

Issue #142 (round 3, B2): records *why* a refresh token was revoked so
AuthService.refresh_tokens can distinguish a benign lost race against a bulk
revocation (logout-all/password-change/deactivation/password-reset racing a
concurrent refresh) from genuine reuse-detection theft. Nullable, no
default, no backfill — adding a nullable column with no default is a
metadata-only operation in PostgreSQL 11+ (no table rewrite, no lock held
beyond the catalog update), safe on a hot table. Legacy rows keep
revoked_reason=NULL, which the application layer treats as "not
bulk_revocation" and therefore still routes through theft detection — no
weakening of existing behavior for pre-migration data.
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "028"
down_revision: str | None = "027"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade database schema."""
    op.add_column(
        "refresh_tokens",
        sa.Column("revoked_reason", sa.String(32), nullable=True),
    )


def downgrade() -> None:
    """Downgrade database schema."""
    op.drop_column("refresh_tokens", "revoked_reason")
