"""Integration coverage for migration 052 (per-second pricing + wan3.0-video seed).

The session schema fixture is already at head, so this drives revision 052's
``upgrade()``/``downgrade()`` directly inside a rolled-back transaction.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import uuid4

import pytest
from alembic.operations import Operations
from alembic.runtime.migration import MigrationContext
from sqlalchemy import text

if TYPE_CHECKING:
    from types import ModuleType

    from sqlalchemy import Connection
    from sqlalchemy.ext.asyncio import AsyncEngine

_MIGRATION_PATH = (
    Path(__file__).resolve().parents[2] / "alembic" / "versions" / "052_wan3_model_studio.py"
)
_SYSTEM_USER_ID = "00000000-0000-0000-0000-000000000001"


def _load_migration() -> ModuleType:
    spec = importlib.util.spec_from_file_location("revision_052", _MIGRATION_PATH)
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


async def test_c12_round_trip_seeds_a_disabled_model_and_no_prices(
    db_engine: AsyncEngine,
) -> None:
    async with db_engine.connect() as connection:
        transaction = await connection.begin()
        try:
            await connection.run_sync(_run, "downgrade")
            await connection.run_sync(_run, "upgrade")

            model = await connection.execute(
                text(
                    "SELECT id::text, provider, name, is_enabled FROM generation_models "
                    "WHERE model_key = 'wan3.0-video'"
                )
            )
            assert model.all() == [
                ("00000000-0000-0000-0001-000000000007", "alibaba", "Wan 3.0", False)
            ]
            prices = await connection.execute(
                text("SELECT count(*) FROM pricing_catalog WHERE model = 'wan3.0-video'")
            )
            assert prices.scalar_one() == 0
            columns = await connection.execute(
                text(
                    "SELECT column_name, is_nullable, column_default FROM information_schema.columns "
                    "WHERE table_name = 'pricing_catalog' "
                    "AND column_name IN ('per_second_token_cost', 'video_resolution') "
                    "ORDER BY column_name"
                )
            )
            assert [(c, n) for c, n, _ in columns.all()] == [
                ("per_second_token_cost", "NO"),
                ("video_resolution", "YES"),
            ]
        finally:
            await transaction.rollback()


async def test_c12_upgrade_replay_is_safe(db_engine: AsyncEngine) -> None:
    """The seed insert is ON CONFLICT DO NOTHING; the schema half starts from 051."""
    async with db_engine.connect() as connection:
        transaction = await connection.begin()
        try:
            await connection.run_sync(_run, "downgrade")
            await connection.run_sync(_run, "upgrade")
            count = await connection.execute(
                text("SELECT count(*) FROM generation_models WHERE model_key = 'wan3.0-video'")
            )
            assert count.scalar_one() == 1
        finally:
            await transaction.rollback()


@pytest.mark.parametrize(("per_second", "resolution"), [(5, None), (0, "720p")])
async def test_c12_downgrade_refuses_when_new_columns_are_in_use(
    db_engine: AsyncEngine, per_second: int, resolution: str | None
) -> None:
    async with db_engine.connect() as connection:
        transaction = await connection.begin()
        try:
            await connection.execute(
                text(
                    "INSERT INTO pricing_catalog "
                    "(id, provider, generation_type, model, token_cost, per_second_token_cost, "
                    "video_resolution, created_by) "
                    "VALUES (:id, 'refusal-test', 't2v', NULL, 1, :per_second, :resolution, :admin)"
                ),
                {
                    "id": uuid4(),
                    "per_second": per_second,
                    "resolution": resolution,
                    "admin": _SYSTEM_USER_ID,
                },
            )

            with pytest.raises(Exception, match="Cannot downgrade 052"):
                await connection.run_sync(_run, "downgrade")
        finally:
            await transaction.rollback()
