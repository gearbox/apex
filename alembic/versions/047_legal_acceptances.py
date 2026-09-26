"""Add append-only legal acceptance ledger.

Revision ID: 047
Revises: 046
Create Date: 2026-09-25 00:00:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision: str = "047"
down_revision: str | Sequence[str] | None = "046"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Create legal_acceptances: one row per accept/withdraw event."""
    op.create_table(
        "legal_acceptances",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        # Event order: drawn at INSERT, and every insert for a user happens
        # under that user's ledger advisory lock, so seq order == write order.
        sa.Column("seq", sa.BigInteger(), sa.Identity(always=True), nullable=False),
        sa.Column(
            "user_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("product_id", sa.String(length=32), nullable=False),
        sa.Column("doc_type", sa.String(length=32), nullable=False),
        sa.Column("action", sa.String(length=32), nullable=False),
        sa.Column("version", sa.Date(), nullable=False),
        sa.Column("content_sha256", sa.String(length=64), nullable=True),
        sa.Column("source", sa.String(length=32), nullable=False),
        sa.Column("ip_address", sa.String(length=64), nullable=True),
        sa.Column("user_agent", sa.String(length=512), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            # Insertion time (evidence only) — not transaction start, and
            # never used for ordering.
            server_default=sa.text("clock_timestamp()"),
        ),
        sa.UniqueConstraint("seq", name="uq_legal_acceptances_seq"),
        sa.CheckConstraint(
            "action <> 'accept' OR content_sha256 IS NOT NULL",
            name="ck_legal_acceptances_accept_has_hash",
        ),
    )
    op.create_index("ix_legal_acceptances_product_id", "legal_acceptances", ["product_id"])
    op.create_index(
        "ix_legal_acceptances_user_product_doc_seq",
        "legal_acceptances",
        ["user_id", "product_id", "doc_type", "seq"],
    )


def downgrade() -> None:
    """Drop the legal acceptance ledger."""
    op.drop_table("legal_acceptances")
