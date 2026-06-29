"""Add composite keyset index for audit log cursor pagination.

Revision ID: 014
Revises: 013
Create Date: 2026-06-29 00:00:00.000000

Replace the standalone ``ix_audit_created`` index with a composite
``ix_audit_product_created_id (product_id, created_at, id)``.

Rationale: every audit query is product-scoped, so the composite leading on
``product_id`` serves both the ``ORDER BY created_at DESC, id DESC`` page
query and the keyset seek predicate
``(created_at, id) < (cursor_ts, cursor_id)``.  The standalone
``created_at`` index has no remaining query that benefits from it —
the composite covers all real access patterns — and is therefore dropped.
"""

from collections.abc import Sequence

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "014"
down_revision: str | None = "013"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade database schema."""
    op.create_index(
        "ix_audit_product_created_id",
        "admin_audit_log",
        ["product_id", "created_at", "id"],
    )
    op.drop_index("ix_audit_created", table_name="admin_audit_log")


def downgrade() -> None:
    """Downgrade database schema."""
    op.create_index("ix_audit_created", "admin_audit_log", ["created_at"])
    op.drop_index("ix_audit_product_created_id", table_name="admin_audit_log")
