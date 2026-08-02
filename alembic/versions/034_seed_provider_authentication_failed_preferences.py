"""Seed provider_authentication.failed preference rows for already-linked admins.

Revision ID: 034
Revises: 033
Create Date: 2026-08-02 00:00:00.000000

Follows the pattern established by ``029_seed_token_revocation_failed_...``:
``AdminNotificationPreference`` uses row-presence as subscription (no
``enabled`` flag), so ``PROVIDER_AUTHENTICATION_FAILED`` starts with zero
subscriber rows and zero recipients no matter how correctly it's mapped
upstream (telegram/mapping.py, PLATFORM_SCOPED_NOTIFICATION_CLASSES).

This seed was originally appended to migration ``033`` at
``min_interval_seconds=0``, matching ``029``'s convention. That was wrong on
two counts:

1. Alembic never re-runs an already-applied revision, so appending the seed
   to ``033`` means any environment already at ``033`` silently never gets
   these preference rows and therefore never receives the alert — no error,
   just zero recipients. Seeding belongs in its own revision, exactly as
   ``029`` established, so it always runs regardless of what revision an
   environment was already at when this was written.
2. ``min_interval_seconds=0`` disables throttling entirely
   (``telegram_dispatcher.py``'s ``_send_throttled`` only throttles when
   ``min_interval_seconds > 0``). ``token_revocation.failed`` (029's seed)
   is a rare, low-volume event, so zero throttle is harmless there.
   ``provider_authentication.failed`` fires once per *failed generation*,
   and its own catalog description states the condition: all generations
   for that provider are failing. During a rejected-key incident that is
   one Telegram message per request — against a per-chat rate limit around
   one message per second, degrading delivery for every other notification
   class too. 300 seconds means the first alert fires immediately
   (``state is None`` on the first event) and then at most once every five
   minutes for the duration of the incident. Admins can still tune this
   through the preferences API. Do not copy 300 back into a future seed
   migration as a new default — it is specific to a class whose event rate
   tracks total generation volume, not a general convention.

Idempotent — ``ON CONFLICT DO NOTHING`` against the existing
``uq_admin_notif_pref_user_product_class`` unique constraint (migration
024), safe to re-run.
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "034"
down_revision: str | Sequence[str] | None = "033"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_NOTIFICATION_CLASS = "provider_authentication.failed"
_MIN_INTERVAL_SECONDS = 300


def upgrade() -> None:
    """Insert a provider_authentication.failed preference row for every linked admin."""
    op.execute(
        sa.text("""
            INSERT INTO admin_notification_preferences
                (id, user_id, product_id, notification_class, min_interval_seconds)
            SELECT gen_random_uuid(), atl.user_id, atl.product_id,
                   :notification_class, :min_interval_seconds
            FROM admin_telegram_links atl
            ON CONFLICT (user_id, product_id, notification_class) DO NOTHING
        """).bindparams(
            notification_class=_NOTIFICATION_CLASS,
            min_interval_seconds=_MIN_INTERVAL_SECONDS,
        )
    )


def downgrade() -> None:
    """Remove only the preference rows for this notification class."""
    op.execute(
        sa.text("""
            DELETE FROM admin_notification_preferences
            WHERE notification_class = :notification_class
        """).bindparams(notification_class=_NOTIFICATION_CLASS)
    )
