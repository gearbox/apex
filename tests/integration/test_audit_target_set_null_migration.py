"""Migration 050: audit targets survive a user hard-delete as NULL."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import TYPE_CHECKING, Any

from alembic.operations import Operations
from alembic.runtime.migration import MigrationContext
from sqlalchemy import inspect

if TYPE_CHECKING:
    from types import ModuleType

    from sqlalchemy import Connection
    from sqlalchemy.ext.asyncio import AsyncEngine


_MIGRATION_PATH = (
    Path(__file__).resolve().parents[2] / "alembic" / "versions" / "050_audit_target_set_null.py"
)


def _load_migration() -> ModuleType:
    spec = importlib.util.spec_from_file_location("revision_050", _MIGRATION_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _target_fk(connection: Connection) -> dict[str, Any]:
    matches = [
        fk
        for fk in inspect(connection).get_foreign_keys("admin_audit_log")
        if fk["constrained_columns"] == ["target_user_id"]
    ]
    assert len(matches) == 1
    return dict(matches[0])


async def test_migration_050_round_trips_named_set_null_target_fk(db_engine: AsyncEngine) -> None:
    """The upgrade reflects the old FK; downgrade preserves its explicit name."""
    migration = _load_migration()
    assert migration.down_revision == "049"

    async with db_engine.connect() as connection:
        transaction = await connection.begin()
        try:
            at_head = await connection.run_sync(_target_fk)
            assert at_head["name"] == "fk_admin_audit_log_target_user_id"
            assert at_head["options"].get("ondelete") == "SET NULL"

            def downgrade(sync_connection: Connection) -> None:
                context = MigrationContext.configure(sync_connection)
                with Operations.context(context):
                    migration.downgrade()

            def upgrade(sync_connection: Connection) -> None:
                context = MigrationContext.configure(sync_connection)
                with Operations.context(context):
                    migration.upgrade()

            await connection.run_sync(downgrade)
            at_049 = await connection.run_sync(_target_fk)
            assert at_049["name"] == "fk_admin_audit_log_target_user_id"
            assert at_049["options"].get("ondelete") is None

            await connection.run_sync(upgrade)
            at_head_again = await connection.run_sync(_target_fk)
            assert at_head_again["name"] == "fk_admin_audit_log_target_user_id"
            assert at_head_again["options"].get("ondelete") == "SET NULL"
        finally:
            await transaction.rollback()
