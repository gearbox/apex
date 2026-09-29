"""Set NULL audit targets when a non-actor user is hard-deleted.

Revision ID: 050
Revises: 049
Create Date: 2026-09-29 00:00:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.engine.interfaces import ReflectedForeignKeyConstraint

from alembic import op

revision: str = "050"
down_revision: str | Sequence[str] | None = "049"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _target_user_foreign_key() -> ReflectedForeignKeyConstraint:
    """Return the sole FK on ``admin_audit_log.target_user_id`` or fail loudly."""
    foreign_keys = sa.inspect(op.get_bind()).get_foreign_keys("admin_audit_log")
    matches = [fk for fk in foreign_keys if fk["constrained_columns"] == ["target_user_id"]]
    if len(matches) != 1:
        raise RuntimeError(
            "Expected exactly one foreign key on admin_audit_log.target_user_id; "
            f"found {len(matches)}"
        )
    return matches[0]


def upgrade() -> None:
    """Replace the legacy target FK with a named ``ON DELETE SET NULL`` FK."""
    foreign_key = _target_user_foreign_key()
    name = foreign_key["name"]
    if not isinstance(name, str):
        raise TypeError("admin_audit_log.target_user_id foreign key has no name")
    op.drop_constraint(name, "admin_audit_log", type_="foreignkey")
    op.create_foreign_key(
        "fk_admin_audit_log_target_user_id",
        "admin_audit_log",
        "users",
        ["target_user_id"],
        ["id"],
        ondelete="SET NULL",
    )


def downgrade() -> None:
    """Restore the restrictive target FK, retaining the explicit constraint name."""
    op.drop_constraint("fk_admin_audit_log_target_user_id", "admin_audit_log", type_="foreignkey")
    op.create_foreign_key(
        "fk_admin_audit_log_target_user_id",
        "admin_audit_log",
        "users",
        ["target_user_id"],
        ["id"],
    )
