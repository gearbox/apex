"""Add is_suppressed — admin deny-list flag for provider-side zombie currencies.

Revision ID: 023
Revises: 022
Create Date: 2026-07-17 00:00:00.000000

NowPayments confirmed a data bug on their side: `merchant/coins` can report
tickers they have effectively delisted/killed. This column is a superadmin-
authored deny-list, orthogonal to the provider-truthful `is_available` flag —
effective public offering is `is_available AND NOT is_suppressed`. The
catalog sync (`PaymentCurrencyRepository.sync_catalog`) never reads or
writes this column, so admin suppressions survive every refresh.

Metadata-only add with a server default — no table rewrite, no backfill.
Downgrade drops the column, losing any suppressions recorded since upgrade —
acceptable, since suppression is a runtime admin override, not source data.
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "023"
down_revision: str | None = "022"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade database schema."""
    op.add_column(
        "payment_currencies",
        sa.Column("is_suppressed", sa.Boolean(), nullable=False, server_default=sa.text("false")),
    )


def downgrade() -> None:
    """Downgrade database schema."""
    op.drop_column("payment_currencies", "is_suppressed")
