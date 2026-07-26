"""Seed token_revocation.failed preference rows for already-linked admins.

Revision ID: 029
Revises: 028
Create Date: 2026-07-26 00:00:00.000000

Issue #142 N4: ``AdminNotificationPreference`` uses row-presence as
subscription (no ``enabled`` flag) and rows are created only by
``AdminNotificationService.replace_preferences`` (``PUT
/v1/admin/notifications/preferences``). A newly added ``NotificationClass``
therefore starts with zero subscriber rows and zero recipients, no matter
how correctly it's mapped and scoped upstream (telegram/mapping.py,
PLATFORM_SCOPED_NOTIFICATION_CLASSES).

This seeds one ``token_revocation.failed`` preference row for every admin
who already has an ``AdminTelegramLink`` row — linked or mid-flow, since
delivery-time filtering (``AdminNotificationRepository.
list_recipients_for_class``) already requires a non-null ``chat_id``, so a
seeded row for an unconfirmed link is harmless. ``product_id`` is taken from
the link row, the same source ``AdminNotificationRepository.
upsert_link_token`` uses when creating it. ``min_interval_seconds=0`` matches
the API's own default (``NotificationPreferenceItem.min_interval_seconds``).

This buys immediate coverage, not durable subscription: the first time a
seeded admin submits a full-set ``PUT /v1/admin/notifications/preferences``
without listing ``token_revocation.failed``, the replace-preferences
delete-and-insert removes the seeded row. See the release note in
docs/BACKEND_API_REFERENCE.md — admins need to know this class exists and
must stay selected. The `GET /v1/admin/health` `TokenRevocationChecker` is
a second, preference-independent channel for the same degradation, which is
why seed-plus-release-note (rather than making this class bypass
preferences entirely) is judged sufficient here.

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

revision: str = "029"
down_revision: str | None = "028"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_NOTIFICATION_CLASS = "token_revocation.failed"


def upgrade() -> None:
    """Insert a token_revocation.failed preference row for every linked admin."""
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
