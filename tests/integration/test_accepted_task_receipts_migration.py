"""Integration coverage for migration 053 (accepted_task_receipts).

The session schema fixture is already at head, so this drives revision 053's
``upgrade()``/``downgrade()`` directly inside a rolled-back transaction.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from alembic.operations import Operations
from alembic.runtime.migration import MigrationContext
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError

if TYPE_CHECKING:
    from types import ModuleType

    from sqlalchemy import Connection
    from sqlalchemy.ext.asyncio import AsyncEngine

_MIGRATION_PATH = (
    Path(__file__).resolve().parents[2] / "alembic" / "versions" / "053_accepted_task_receipts.py"
)


def _load_migration() -> ModuleType:
    spec = importlib.util.spec_from_file_location("revision_053", _MIGRATION_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run(connection: Connection, action: str) -> None:
    migration = _load_migration()
    context = MigrationContext.configure(connection)
    with Operations.context(context):
        getattr(migration, action)()


async def _table_exists(connection: object) -> bool:
    result = await connection.execute(  # type: ignore[attr-defined]
        text("SELECT to_regclass('public.accepted_task_receipts') IS NOT NULL")
    )
    return bool(result.scalar_one())


async def test_downgrade_drops_the_table_and_upgrade_recreates_it(db_engine: AsyncEngine) -> None:
    async with db_engine.connect() as connection:
        transaction = await connection.begin()
        try:
            assert await _table_exists(connection)
            await connection.run_sync(_run, "downgrade")
            assert not await _table_exists(connection)
            await connection.run_sync(_run, "upgrade")
            assert await _table_exists(connection)
            indexes = await connection.execute(
                text(
                    "SELECT indexname FROM pg_indexes "
                    "WHERE tablename = 'accepted_task_receipts' ORDER BY indexname"
                )
            )
            assert [row[0] for row in indexes.all()] == [
                "accepted_task_receipts_pkey",
                "ix_accepted_task_receipts_pending",
                "ix_accepted_task_receipts_product_id",
                "uq_accepted_task_receipts_provider_task",
            ]
        finally:
            await transaction.rollback()


@pytest.mark.parametrize(
    ("status", "resolved", "valid"),
    [
        ("pending", False, True),
        ("cancelled", True, True),
        ("absorbed", True, True),
        ("pending", True, False),
        ("cancelled", False, False),
        ("bogus", True, False),
    ],
)
async def test_status_and_resolved_at_constraints(
    db_engine: AsyncEngine, status: str, resolved: bool, valid: bool
) -> None:
    async with db_engine.connect() as connection:
        transaction = await connection.begin()
        try:
            user_id = (
                await connection.execute(
                    text(
                        "INSERT INTO users (id, email, password_hash, product_id, is_active) "
                        "VALUES (gen_random_uuid(), 'receipt-' || gen_random_uuid() || '@e.com', "
                        "'x', 'vex', true) RETURNING id"
                    )
                )
            ).scalar_one()
            insert = text(
                "INSERT INTO accepted_task_receipts "
                "(id, provider, external_task_id, job_id, user_id, product_id, model, "
                "token_cost, status, resolved_at) "
                "VALUES (gen_random_uuid(), 'alibaba', 't', gen_random_uuid(), :user_id, "
                "'vex', 'wan3.0-video', 1, :status, "
                "CASE WHEN :resolved THEN CURRENT_TIMESTAMP ELSE NULL END)"
            )
            if valid:
                await connection.execute(
                    insert, {"user_id": user_id, "status": status, "resolved": resolved}
                )
            else:
                with pytest.raises(IntegrityError):
                    await connection.execute(
                        insert, {"user_id": user_id, "status": status, "resolved": resolved}
                    )
        finally:
            await transaction.rollback()
