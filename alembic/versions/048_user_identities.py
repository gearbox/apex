"""Add user_identities (OAuth subjects) and make users.password_hash nullable.

Revision ID: 048
Revises: 047
Create Date: 2026-09-27 00:00:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision: str = "048"
down_revision: str | Sequence[str] | None = "047"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Allow password-less (OAuth-only) users and create the identity link table."""
    op.alter_column("users", "password_hash", existing_type=sa.String(255), nullable=True)
    op.create_table(
        "user_identities",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "user_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("product_id", sa.String(32), nullable=False),
        # OAuthProvider value — plain String, not a PG enum.
        sa.Column("provider", sa.String(32), nullable=False),
        sa.Column("subject", sa.String(255), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("CURRENT_TIMESTAMP"),
        ),
        sa.Column("last_login_at", sa.DateTime(timezone=True), nullable=True),
        sa.UniqueConstraint(
            "product_id",
            "provider",
            "subject",
            name="uq_user_identities_product_provider_subject",
        ),
        sa.UniqueConstraint("user_id", "provider", name="uq_user_identities_user_provider"),
    )
    op.create_index("ix_user_identities_product_id", "user_identities", ["product_id"])


def downgrade() -> None:
    """Drop identities; refuse if any password-less user exists.

    Never fabricates a password hash: an OAuth-only account has no credential
    to restore, so the operator must resolve those rows explicitly first.
    """
    bind = op.get_bind()
    null_count = bind.execute(
        sa.text("SELECT count(*) FROM users WHERE password_hash IS NULL")
    ).scalar_one()
    if null_count:
        raise RuntimeError(
            f"Cannot downgrade 048: {null_count} user(s) have password_hash IS NULL "
            "(OAuth-only accounts). Resolve them before downgrading."
        )
    op.drop_index("ix_user_identities_product_id", table_name="user_identities")
    op.drop_table("user_identities")
    op.alter_column("users", "password_hash", existing_type=sa.String(255), nullable=False)
