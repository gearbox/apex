"""Add payment provider state and allow non-user audit targets.

Revision ID: 017
Revises: 016
Create Date: 2026-07-10 00:00:00.000000

Downgrading deletes audit rows whose target_user_id is NULL before restoring
the NOT NULL constraint. This intentionally destroys provider-toggle audit
history because those entries cannot be represented by the older schema.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision: str = "017"
down_revision: str | None = "016"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade database schema."""
    op.create_table(
        "payment_provider_state",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("product_id", sa.String(32), nullable=False),
        sa.Column("provider", sa.String(30), nullable=False),
        sa.Column("is_enabled", sa.Boolean(), nullable=False, server_default=sa.text("true")),
        sa.Column("display_order", sa.Integer(), nullable=False, server_default=sa.text("0")),
        sa.Column(
            "updated_by",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("users.id"),
            nullable=True,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("CURRENT_TIMESTAMP"),
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("product_id", "provider", name="uq_payment_provider_state"),
    )
    op.create_index(
        "ix_payment_provider_state_product_id",
        "payment_provider_state",
        ["product_id"],
    )
    op.alter_column(
        "admin_audit_log",
        "target_user_id",
        existing_type=postgresql.UUID(as_uuid=True),
        nullable=True,
    )


def downgrade() -> None:
    """Downgrade schema, deleting provider audit rows with NULL targets."""
    op.execute(sa.text("DELETE FROM admin_audit_log WHERE target_user_id IS NULL"))
    op.alter_column(
        "admin_audit_log",
        "target_user_id",
        existing_type=postgresql.UUID(as_uuid=True),
        nullable=False,
    )
    op.drop_index(
        "ix_payment_provider_state_product_id",
        table_name="payment_provider_state",
    )
    op.drop_table("payment_provider_state")
