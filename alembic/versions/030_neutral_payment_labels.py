"""Neutralize payment gateway names in user-facing billing descriptions.

Revision ID: 030
Revises: 029
Create Date: 2026-07-27 00:00:00.000000

Issue: the payment gateway name (``nowpayments`` / ``stripe``) was written
verbatim into ``token_transactions.description`` and returned unmodified by
``GET /v1/billing/transactions`` — an authenticated end-user endpoint. Users
only need the crypto/card distinction; the gateway identity remains
authoritative via ``payments.payment_provider`` (admin-only surface).

This backfills historical credit rows by joining ``payments.payment_provider``
(never by string-sniffing the description) so the neutralization is
reversible: ``downgrade()`` restores the original text from the same join.
The equality predicates on ``upgrade()``/``downgrade()`` make both directions
idempotent — a second run matches nothing.

``token_transactions`` has carried a ``BEFORE UPDATE OR DELETE`` trigger
(``enforce_token_transactions_immutable``, migration 001) since the very
first schema baseline — it unconditionally raises on any UPDATE/DELETE,
application-level or otherwise. The backfill UPDATE above would fail against
every environment without first disabling that trigger for the duration of
the statement. This is a one-time, migration-scoped, transactional exception
to the append-only invariant — not a relaxation of it: the trigger is
re-enabled before this migration commits, so no application code path gains
any new ability to mutate a ledger row.

``user_images.display_filename`` (D-B4/B4a of the companion filename-safety
fix) is added by the following revision (031), deliberately kept out of this
one — see that migration's docstring for why.

ROLLOUT ORDER — READ BEFORE DEPLOYING
-------------------------------------
The backfill matches the *old* description format, so it only fixes rows
that already exist when it runs. Any top-up settled by the old application
code after this migration completes is written with the gateway name and is
never revisited — the backfill has already passed it.

Deploy the application image and this migration together, or run the
migration last. If they do drift apart, re-running just the data statement
closes the window and is safe at any time (the equality predicates make it
idempotent):

    psql "$DATABASE_URL" -c "ALTER TABLE token_transactions DISABLE TRIGGER enforce_token_transactions_immutable"
    psql "$DATABASE_URL" -f <NEUTRALIZE_DESCRIPTIONS_SQL>
    psql "$DATABASE_URL" -c "ALTER TABLE token_transactions ENABLE TRIGGER enforce_token_transactions_immutable"

PRIVILEGES: `ALTER TABLE ... DISABLE TRIGGER` requires ownership of
token_transactions (or superuser). Verify in staging *and* prod that the
role in DATABASE_URL owns the table — it is not necessarily the role that
ran migration 001:

    SELECT tableowner FROM pg_tables WHERE tablename = 'token_transactions';
    SELECT current_user;
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "030"
down_revision: str | None = "029"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

# Module-level so integration tests can re-run just the data backfill (idempotence)
# without re-invoking the column-add DDL in upgrade()/downgrade().
NEUTRALIZE_DESCRIPTIONS_SQL = """
    UPDATE token_transactions AS t
    SET description = 'Token purchase via ' || m.kind || ' payment'
                      || CASE WHEN t.description LIKE '%(partial)' THEN ' (partial)' ELSE '' END,
        metadata    = t.metadata || jsonb_build_object('payment_method', m.kind)
    FROM payments AS p
    JOIN (VALUES ('nowpayments', 'crypto'), ('stripe', 'card')) AS m(provider, kind)
      ON m.provider = p.payment_provider
    WHERE t.payment_id = p.id
      AND t.transaction_type = 'credit'
      AND (
            t.description = 'Token purchase via ' || p.payment_provider
         OR t.description = 'Token purchase via ' || p.payment_provider || ' (partial)'
      )
"""

RESTORE_DESCRIPTIONS_SQL = """
    UPDATE token_transactions AS t
    SET description = 'Token purchase via ' || p.payment_provider
                      || CASE WHEN t.description LIKE '%(partial)' THEN ' (partial)' ELSE '' END,
        metadata    = t.metadata - 'payment_method'
    FROM payments AS p
    JOIN (VALUES ('nowpayments', 'crypto'), ('stripe', 'card')) AS m(provider, kind)
      ON m.provider = p.payment_provider
    WHERE t.payment_id = p.id
      AND t.transaction_type = 'credit'
      AND (
            t.description = 'Token purchase via ' || m.kind || ' payment'
         OR t.description = 'Token purchase via ' || m.kind || ' payment (partial)'
      )
"""

# token_transactions.enforce_token_transactions_immutable (migration 001) blocks
# every UPDATE/DELETE unconditionally. Disabling by trigger *name* (not
# `DISABLE TRIGGER ALL`) scopes the exception to exactly this one guard.
_TRIGGER_NAME = "enforce_token_transactions_immutable"
DISABLE_IMMUTABILITY_TRIGGER_SQL = f"ALTER TABLE token_transactions DISABLE TRIGGER {_TRIGGER_NAME}"
ENABLE_IMMUTABILITY_TRIGGER_SQL = f"ALTER TABLE token_transactions ENABLE TRIGGER {_TRIGGER_NAME}"

# The trigger toggle below takes ACCESS EXCLUSIVE on token_transactions and
# Postgres holds it until this migration commits — every ledger read and
# write blocks for the duration. Fail fast rather than queueing behind (or
# stalling) live traffic: a timeout aborts the migration cleanly and the
# whole transaction rolls back, leaving the trigger enabled.
_LOCK_TIMEOUT_SQL = "SET LOCAL lock_timeout = '5s'"
_STATEMENT_TIMEOUT_SQL = "SET LOCAL statement_timeout = '60s'"


def upgrade() -> None:
    """Neutralize historical top-up descriptions."""
    op.execute(sa.text(_LOCK_TIMEOUT_SQL))
    op.execute(sa.text(_STATEMENT_TIMEOUT_SQL))
    op.execute(sa.text(DISABLE_IMMUTABILITY_TRIGGER_SQL))
    op.execute(sa.text(NEUTRALIZE_DESCRIPTIONS_SQL))
    op.execute(sa.text(ENABLE_IMMUTABILITY_TRIGGER_SQL))


def downgrade() -> None:
    """Restore gateway-named descriptions."""
    op.execute(sa.text(_LOCK_TIMEOUT_SQL))
    op.execute(sa.text(_STATEMENT_TIMEOUT_SQL))
    op.execute(sa.text(DISABLE_IMMUTABILITY_TRIGGER_SQL))
    op.execute(sa.text(RESTORE_DESCRIPTIONS_SQL))
    op.execute(sa.text(ENABLE_IMMUTABILITY_TRIGGER_SQL))
