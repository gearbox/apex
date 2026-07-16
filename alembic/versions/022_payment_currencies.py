"""Add payment_currencies — DB-cached provider currency catalog.

Revision ID: 022
Revises: 021
Create Date: 2026-07-16 00:00:00.000000

The table is a cache synced from a payment provider's discovery endpoints
(e.g. NowPayments merchant/coins + full-currencies), never hand-edited.
Rows are never deleted by application code — a refresh flips
`is_available` on tickers missing from the latest successful sync, so a
plain symmetric `drop_table` downgrade is safe here (no data-preservation
concern, unlike 021's currency-widening guard).
"""

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision: str = "022"
down_revision: str | None = "021"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade database schema."""
    op.create_table(
        "payment_currencies",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("product_id", sa.String(32), nullable=False),
        sa.Column("provider", sa.String(30), nullable=False),
        sa.Column("ticker", sa.String(20), nullable=False),
        sa.Column("is_available", sa.Boolean(), nullable=False, server_default=sa.text("true")),
        sa.Column("name", sa.String(100), nullable=True),
        sa.Column("network", sa.String(30), nullable=True),
        sa.Column("logo_source_url", sa.String(500), nullable=True),
        sa.Column("logo_key", sa.String(200), nullable=True),
        sa.Column("logo_synced_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "first_seen_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("CURRENT_TIMESTAMP"),
        ),
        sa.Column("last_seen_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("product_id", "provider", "ticker", name="uq_payment_currencies"),
    )


def downgrade() -> None:
    """Downgrade database schema."""
    op.drop_table("payment_currencies")
