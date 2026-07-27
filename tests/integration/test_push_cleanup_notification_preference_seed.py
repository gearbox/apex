"""Integration test for migration 032: seeds push_subscriptions.cleanup_failed
preferences for admins who already have a Telegram link (M1 — mirrors migration
029's seed for token_revocation.failed).

Runs the migration's upgrade()/downgrade() directly against a scratch
connection bound to an Operations context, rather than via `alembic upgrade`
subprocess calls — the whole test body sits inside one transaction that is
rolled back at the end, so it doesn't disturb the session-scoped db_engine
fixture (already migrated to head, including revision 032) shared by the
rest of the integration suite. Calling upgrade() twice in the same
transaction exercises the ON CONFLICT DO NOTHING idempotency directly,
which a plain `alembic upgrade` round trip cannot: alembic's own version
tracking would just no-op the second call rather than re-running the SQL.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import uuid4

from alembic.operations import Operations
from alembic.runtime.migration import MigrationContext
from sqlalchemy import text

if TYPE_CHECKING:
    from types import ModuleType

    from sqlalchemy import Connection
    from sqlalchemy.ext.asyncio import AsyncEngine

_MIGRATION_PATH = (
    Path(__file__).resolve().parents[2]
    / "alembic"
    / "versions"
    / "032_seed_push_subscriptions_cleanup_failed_preferences.py"
)


def _load_migration() -> ModuleType:
    spec = importlib.util.spec_from_file_location("revision_032", _MIGRATION_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


async def test_seed_creates_one_row_per_linked_admin_idempotently_with_working_downgrade(
    db_engine: AsyncEngine,
) -> None:
    migration = _load_migration()
    user_id = uuid4()
    link_id = uuid4()

    async with db_engine.connect() as conn:
        tx = await conn.begin()
        try:
            await conn.execute(
                text(
                    "INSERT INTO users "
                    "(id, email, password_hash, role, is_active, subscription_tier, "
                    "locale, product_id) "
                    "VALUES (:id, :email, 'x', 'admin', true, 'free', 'en', 'vex')"
                ),
                {"id": user_id, "email": f"admin-{user_id}@example.com"},
            )
            await conn.execute(
                text(
                    "INSERT INTO admin_telegram_links (id, user_id, product_id, chat_id) "
                    "VALUES (:id, :user_id, 'vex', 555)"
                ),
                {"id": link_id, "user_id": user_id},
            )

            def _run_upgrade_twice(sync_conn: Connection) -> None:
                ctx = MigrationContext.configure(sync_conn)
                with Operations.context(ctx):
                    migration.upgrade()
                    migration.upgrade()

            await conn.run_sync(_run_upgrade_twice)

            count_query = text(
                "SELECT count(*) FROM admin_notification_preferences "
                "WHERE user_id = :user_id "
                "AND notification_class = 'push_subscriptions.cleanup_failed'"
            )
            result = await conn.execute(count_query, {"user_id": user_id})
            assert result.scalar_one() == 1

            def _run_downgrade(sync_conn: Connection) -> None:
                ctx = MigrationContext.configure(sync_conn)
                with Operations.context(ctx):
                    migration.downgrade()

            await conn.run_sync(_run_downgrade)

            result = await conn.execute(count_query, {"user_id": user_id})
            assert result.scalar_one() == 0
        finally:
            await tx.rollback()


async def test_downgrade_removes_only_rows_for_this_class(db_engine: AsyncEngine) -> None:
    migration = _load_migration()
    user_id = uuid4()
    link_id = uuid4()

    async with db_engine.connect() as conn:
        tx = await conn.begin()
        try:
            await conn.execute(
                text(
                    "INSERT INTO users "
                    "(id, email, password_hash, role, is_active, subscription_tier, "
                    "locale, product_id) "
                    "VALUES (:id, :email, 'x', 'admin', true, 'free', 'en', 'vex')"
                ),
                {"id": user_id, "email": f"admin-{user_id}@example.com"},
            )
            await conn.execute(
                text(
                    "INSERT INTO admin_telegram_links (id, user_id, product_id, chat_id) "
                    "VALUES (:id, :user_id, 'vex', 555)"
                ),
                {"id": link_id, "user_id": user_id},
            )
            # A preference for an unrelated class must survive the downgrade.
            await conn.execute(
                text(
                    "INSERT INTO admin_notification_preferences "
                    "(id, user_id, product_id, notification_class, min_interval_seconds) "
                    "VALUES (gen_random_uuid(), :user_id, 'vex', 'user.registered', 0)"
                ),
                {"user_id": user_id},
            )

            def _run_upgrade(sync_conn: Connection) -> None:
                ctx = MigrationContext.configure(sync_conn)
                with Operations.context(ctx):
                    migration.upgrade()

            await conn.run_sync(_run_upgrade)

            def _run_downgrade(sync_conn: Connection) -> None:
                ctx = MigrationContext.configure(sync_conn)
                with Operations.context(ctx):
                    migration.downgrade()

            await conn.run_sync(_run_downgrade)

            result = await conn.execute(
                text(
                    "SELECT notification_class FROM admin_notification_preferences "
                    "WHERE user_id = :user_id"
                ),
                {"user_id": user_id},
            )
            assert [row[0] for row in result.all()] == ["user.registered"]
        finally:
            await tx.rollback()
