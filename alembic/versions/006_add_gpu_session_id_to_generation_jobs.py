"""Add gpu_session_id FK + check constraint to generation_jobs.

Revision ID: 006
Revises: 005
Create Date: 2026-04-29
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import sqlalchemy as sa

from alembic import op

if TYPE_CHECKING:
    from collections.abc import Sequence

revision: str = "006"
down_revision: str | None = "005"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "generation_jobs",
        sa.Column("gpu_session_id", sa.UUID(), nullable=True),
    )
    op.create_foreign_key(
        "fk_generation_jobs_gpu_session_id",
        "generation_jobs",
        "gpu_sessions",
        ["gpu_session_id"],
        ["id"],
        ondelete="SET NULL",
    )
    op.create_index(
        "ix_generation_jobs_gpu_session_id_status",
        "generation_jobs",
        ["gpu_session_id", "status"],
        postgresql_where=sa.text("status IN ('queued', 'running')"),
    )
    op.create_check_constraint(
        "ck_generation_jobs_aisha_has_session",
        "generation_jobs",
        "(provider != 'aisha') OR (gpu_session_id IS NOT NULL)",
    )


def downgrade() -> None:
    op.drop_constraint(
        "ck_generation_jobs_aisha_has_session",
        "generation_jobs",
        type_="check",
    )
    op.drop_index(
        "ix_generation_jobs_gpu_session_id_status",
        table_name="generation_jobs",
    )
    op.drop_constraint(
        "fk_generation_jobs_gpu_session_id",
        "generation_jobs",
        type_="foreignkey",
    )
    op.drop_column("generation_jobs", "gpu_session_id")
