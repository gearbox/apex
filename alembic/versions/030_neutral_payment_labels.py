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

Also adds ``user_images.display_filename`` (D-B4/B4a of the companion
filename-safety fix): a nullable, display/search-only copy of the
client-supplied upload filename, NFC-normalized and control-char stripped at
write time. Historical rows are intentionally left NULL — there is no safe
way to recover the original name from ``original_filename`` once it has
been overwritten with the canonical ``{uuid}.{ext}`` system name by the
application-side change in this same release; backfilling would either
require re-deriving from the now-removed R2 ``original-filename`` object
metadata (also being dropped) or fabricating data. NULL correctly represents
"no display name recorded" and the column is never used for anything but
display/search.
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


def upgrade() -> None:
    """Neutralize historical top-up descriptions and add display_filename."""
    op.execute(sa.text(DISABLE_IMMUTABILITY_TRIGGER_SQL))
    op.execute(sa.text(NEUTRALIZE_DESCRIPTIONS_SQL))
    op.execute(sa.text(ENABLE_IMMUTABILITY_TRIGGER_SQL))

    op.add_column(
        "user_images",
        sa.Column("display_filename", sa.String(255), nullable=True),
    )


def downgrade() -> None:
    """Restore gateway-named descriptions and drop display_filename."""
    op.drop_column("user_images", "display_filename")
    op.execute(sa.text(DISABLE_IMMUTABILITY_TRIGGER_SQL))
    op.execute(sa.text(RESTORE_DESCRIPTIONS_SQL))
    op.execute(sa.text(ENABLE_IMMUTABILITY_TRIGGER_SQL))
