"""Widen payments.currency to accommodate customer-chosen NowPayments tickers.

Revision ID: 021
Revises: 020
Create Date: 2026-07-16 00:00:00.000000

Changes:
- payments.currency: String(10) -> String(20). Some NowPayments tickers
  (e.g. "BABYDOGEBSC") exceed 10 characters once the invoice is settled in a
  customer-chosen currency. Widening a VARCHAR is metadata-only in
  PostgreSQL (no table rewrite).

Downgrade narrows back to String(10) without a data-destroying UPDATE —
PostgreSQL will reject the narrowing if any row already has a currency value
longer than 10 characters. That fail-loud behavior is intentional: silently
truncating a real settled-currency value would corrupt reconciliation data.
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "021"
down_revision: str | None = "020"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade database schema."""
    op.alter_column(
        "payments",
        "currency",
        type_=sa.String(20),
        existing_type=sa.String(10),
        existing_nullable=False,
        existing_server_default=sa.text("'USD'"),
    )


def downgrade() -> None:
    """Downgrade database schema."""
    op.alter_column(
        "payments",
        "currency",
        type_=sa.String(10),
        existing_type=sa.String(20),
        existing_nullable=False,
        existing_server_default=sa.text("'USD'"),
    )
