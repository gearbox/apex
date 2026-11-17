"""Studio lyrics runs: what a run works on (``studio_runs.inputs``).

Revision ID: 058
Revises: 057
Create Date: 2026-11-15 00:00:00.000000

One nullable column. ``inputs`` records what a run works on - the plan (and, for a revision, the
lyrics) and the language - resolved **when the run was accepted** (P2a-40), so what the run does
never depends on where the track's current pointers are when a worker claims it. It is ``NULL`` for
``plan`` runs and for every run that existed before this revision.

There is deliberately no foreign key: ``ON DELETE`` would either block deleting a version for good
or erase the record. "A version an active run takes as input cannot be deleted" is a service rule
(``409 version_in_use_by_run``), checked under the track lock. Downgrade drops the column (and the
record of what each run worked on).
"""

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision: str = "058"
down_revision: str | Sequence[str] | None = "057"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade database schema."""
    op.add_column("studio_runs", sa.Column("inputs", postgresql.JSONB(), nullable=True))


def downgrade() -> None:
    """Downgrade database schema."""
    op.drop_column("studio_runs", "inputs")
