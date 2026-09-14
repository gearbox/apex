"""Add consecutive_contract_failures counter for GPU session probe fail-fast.

Revision ID: 044
Revises: 043
Create Date: 2026-09-13 00:00:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "044"
down_revision: str | Sequence[str] | None = "043"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Track consecutive ProbeOutcome.contract_failed results per session."""
    op.add_column(
        "gpu_sessions",
        sa.Column(
            "consecutive_contract_failures",
            sa.Integer(),
            nullable=False,
            server_default=sa.text("0"),
            comment=(
                "Consecutive ProbeOutcome.contract_failed results during the initial "
                "provisioning path. Reset to 0 by any non-contract_failed probe outcome."
            ),
        ),
    )


def downgrade() -> None:
    """Drop the probe fail-fast counter."""
    op.drop_column("gpu_sessions", "consecutive_contract_failures")
