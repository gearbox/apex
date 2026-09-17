"""Replace billing reconciler quarantine-exclusion with per-session backoff (X1).

Round-3's T7 added a `billing_finalization_attempts < quarantine_threshold`
predicate to both reconciliation queries so a chronically failing session
would stop being re-selected (and re-ERRORed) every sweep. That traded a
noisy-but-correct loop for a silent-and-permanent one: nothing ever resets
`billing_finalization_attempts`, so a transient outage lasting
`quarantine_threshold` sweeps permanently drops the session from
reconciliation — contradicting the documented contract that
`billing_finalized_at` stays NULL so the worker keeps retrying once the
underlying issue is fixed.

This migration adds `billing_reconciliation_next_attempt_at`: the queries now
filter `next_attempt_at IS NULL OR next_attempt_at <= now()` instead of
excluding by attempt count, so a row stays retryable but paced by exponential
backoff — it stops flooding the alert channel and stops head-of-line-blocking
healthy candidates without ever becoming permanently unreachable.

Deployment note (Y3, round-6): this nullable column is deliberately not
backfilled. Existing rows at or above the old quarantine threshold are eligible
on the first post-deploy sweep, where their oldest-first ordering can fill a
small ``billing_reconciler_max_per_sweep`` before they receive fresh backoff
timestamps. Count that one-time burst before deploying:

    SELECT count(*) FROM gpu_sessions
    WHERE billing_finalization_attempts >= 10
      AND ((status = 'stopped' AND billing_finalized_at IS NULL)
        OR (status = 'failed' AND started_at IS NULL AND account_id IS NOT NULL));

If the count is large relative to the sweep limit, stage a one-off timestamp
backfill operationally rather than changing this migration's intended NULL =
eligible-immediately semantics.

Revision ID: 045
Revises: 044
Create Date: 2026-09-16 00:00:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "045"
down_revision: str | Sequence[str] | None = "044"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Add the backoff timestamp column and its two supporting partial indexes.

    EXPLAIN (ANALYZE, BUFFERS), measured against Postgres 16 with 50,000 seeded
    gpu_sessions rows (~11,978 'stopped'+unfinalized, ~4,038 'failed' pre-active
    with account_id set) and ~23,000 token_transactions rows:

    list_pending_billing_finalization — Index Scan using
    ix_gpu_sessions_billing_finalization_pending, Execution Time: 0.153 ms,
    Rows Removed by Filter: 10 (the backoff OR-condition is evaluated inline
    as an index filter, not a separate index condition — no sequential scan).

    list_pending_refund_reconciliation — Index Scan using
    ix_gpu_sessions_refund_reconciliation_pending feeding two Nested Loop
    (Semi/Anti) Joins against ix_token_transactions_job_id for the has-debit/
    no-refund EXISTS checks, Execution Time: 0.698 ms — no sequential scan on
    either table.

    Both plans stay index-driven at this scale; revisit only if gpu_sessions
    grows by orders of magnitude beyond a few sessions per user.
    """
    op.add_column(
        "gpu_sessions",
        sa.Column(
            "billing_reconciliation_next_attempt_at",
            sa.DateTime(timezone=True),
            nullable=True,
            comment=(
                "Earliest time the billing reconciler may re-select this "
                "session. NULL means eligible immediately. Set on each "
                "failed attempt to now() + exponential backoff."
            ),
        ),
    )
    op.create_index(
        "ix_gpu_sessions_billing_finalization_pending",
        "gpu_sessions",
        ["stopped_at"],
        postgresql_where=sa.text("status = 'stopped' AND billing_finalized_at IS NULL"),
    )
    op.create_index(
        "ix_gpu_sessions_refund_reconciliation_pending",
        "gpu_sessions",
        ["stopped_at"],
        postgresql_where=sa.text(
            "status = 'failed' AND started_at IS NULL AND account_id IS NOT NULL"
        ),
    )


def downgrade() -> None:
    """Drop the backoff column and its supporting indexes."""
    op.drop_index("ix_gpu_sessions_refund_reconciliation_pending", table_name="gpu_sessions")
    op.drop_index("ix_gpu_sessions_billing_finalization_pending", table_name="gpu_sessions")
    op.drop_column("gpu_sessions", "billing_reconciliation_next_attempt_at")
