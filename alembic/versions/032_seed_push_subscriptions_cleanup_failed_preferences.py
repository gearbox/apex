"""Seed push_subscriptions.cleanup_failed preference rows for already-linked admins.

Revision ID: 032
Revises: 031
Create Date: 2026-07-27 00:00:00.000000

Same gap as migration 029 (``token_revocation.failed``): a newly added
``NotificationClass`` starts with zero subscriber rows and zero recipients no
matter how correctly it's mapped and scoped upstream, since
``AdminNotificationPreference`` rows are only ever created by
``AdminNotificationService.replace_preferences`` (``PUT
/v1/admin/notifications/preferences``).

Unlike ``token_revocation.failed``, this class has **no** second,
preference-independent channel — there is no health checker surfacing push-
cleanup failures the way ``TokenRevocationChecker`` does for revocation
failures — so seeding is the only way an existing install gets any coverage
without an admin manually discovering and selecting this class first.

Seeds one ``push_subscriptions.cleanup_failed`` preference row for every
admin who already has an ``AdminTelegramLink`` row — linked or mid-flow,
since delivery-time filtering (``AdminNotificationRepository.
list_recipients_for_class``) already requires a non-null ``chat_id``, so a
seeded row for an unconfirmed link is harmless. ``product_id`` is taken from
the link row, the same source ``AdminNotificationRepository.
upsert_link_token`` uses when creating it. ``min_interval_seconds=0`` matches
the API's own default (``NotificationPreferenceItem.min_interval_seconds``).

This buys immediate coverage, not durable subscription: the first time a
seeded admin submits a full-set ``PUT /v1/admin/notifications/preferences``
without listing ``push_subscriptions.cleanup_failed``, the replace-
preferences delete-and-insert removes the seeded row. See the release note
in docs/BACKEND_API_REFERENCE.md.

Idempotent — ``ON CONFLICT DO NOTHING`` against the existing
``uq_admin_notif_pref_user_product_class`` unique constraint (migration
024), safe to re-run. IDs are generated with ``gen_random_uuid()`` (built
into PostgreSQL 13+, no extension required) rather than the application's
UUIDv7 scheme (``src/core/uid.py``) — acceptable for this one-off,
low-cardinality data seed; every subsequent row for this class goes through
the normal application path and gets a UUIDv7 id as usual.
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "032"
down_revision: str | None = "031"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_NOTIFICATION_CLASS = "push_subscriptions.cleanup_failed"


def upgrade() -> None:
    """Insert a push_subscriptions.cleanup_failed preference row for every linked admin."""
    op.execute(
        sa.text("""
            INSERT INTO admin_notification_preferences
                (id, user_id, product_id, notification_class, min_interval_seconds)
            SELECT gen_random_uuid(), atl.user_id, atl.product_id, :notification_class, 0
            FROM admin_telegram_links atl
            ON CONFLICT (user_id, product_id, notification_class) DO NOTHING
        """).bindparams(notification_class=_NOTIFICATION_CLASS)
    )


def downgrade() -> None:
    """Remove only the preference rows for this notification class."""
    op.execute(
        sa.text("""
            DELETE FROM admin_notification_preferences
            WHERE notification_class = :notification_class
        """).bindparams(notification_class=_NOTIFICATION_CLASS)
    )
