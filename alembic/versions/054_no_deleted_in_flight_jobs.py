"""Forbid soft-deleting an in-flight generation job.

Revision ID: 054
Revises: 053
Create Date: 2026-10-21 00:00:00.000000

``DELETE /v1/jobs/{id}`` used to set ``is_deleted`` on a job in any status. A
deleted ``pending``/``queued``/``running`` job is invisible to every lifecycle
query (the pollers, poll-on-read, the Aisha sweeps), so it was never completed,
refunded or timed out while its upstream work kept running and billing.

The route now refuses (409 ``job_in_progress``); this migration makes the state
unrepresentable:

1. un-delete the existing deleted in-flight rows, so the normal lifecycle settles
   them (output delivered, or refunded on timeout);
2. add ``ck_generation_jobs_not_deleted_in_flight``, ``NOT VALID`` first and then
   ``VALIDATE``d, so the table is not held under ``ACCESS EXCLUSIVE`` for a scan.

Downgrade drops the constraint only. The un-delete is **not** reversed: the
rows it touched are indistinguishable from jobs that were never deleted.
"""

import logging
from collections.abc import Sequence

from alembic import op

revision: str = "054"
down_revision: str | Sequence[str] | None = "053"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

logger = logging.getLogger("alembic.runtime.migration")

_CONSTRAINT = "ck_generation_jobs_not_deleted_in_flight"
_IN_FLIGHT = "('pending','queued','running')"


def upgrade() -> None:
    """Upgrade database schema."""
    result = op.get_bind().exec_driver_sql(
        f"UPDATE generation_jobs SET is_deleted = false "
        f"WHERE is_deleted AND status IN {_IN_FLIGHT}"
    )
    logger.info("054: un-deleted %s in-flight generation_jobs rows", result.rowcount)

    op.execute(
        f"ALTER TABLE generation_jobs ADD CONSTRAINT {_CONSTRAINT} "
        f"CHECK (NOT (is_deleted AND status IN {_IN_FLIGHT})) NOT VALID"
    )
    op.execute(f"ALTER TABLE generation_jobs VALIDATE CONSTRAINT {_CONSTRAINT}")


def downgrade() -> None:
    """Downgrade database schema (the un-delete is not reversed)."""
    op.execute(f"ALTER TABLE generation_jobs DROP CONSTRAINT {_CONSTRAINT}")
