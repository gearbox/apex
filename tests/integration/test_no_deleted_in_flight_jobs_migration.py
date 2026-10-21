"""Integration coverage for migration 054 (no deleted in-flight jobs).

The session schema fixture is already at head, so this drives revision 054's
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
    from sqlalchemy.ext.asyncio import AsyncConnection, AsyncEngine

_MIGRATION_PATH = (
    Path(__file__).resolve().parents[2]
    / "alembic"
    / "versions"
    / "054_no_deleted_in_flight_jobs.py"
)
_CONSTRAINT = "ck_generation_jobs_not_deleted_in_flight"


def _load_migration() -> ModuleType:
    spec = importlib.util.spec_from_file_location("revision_054", _MIGRATION_PATH)
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


async def _constraint_exists(connection: AsyncConnection) -> bool:
    result = await connection.execute(
        text("SELECT count(*) FROM pg_constraint WHERE conname = :name"), {"name": _CONSTRAINT}
    )
    return result.scalar_one() == 1


async def _seed_user(connection: AsyncConnection) -> object:
    return (
        await connection.execute(
            text(
                "INSERT INTO users (id, email, password_hash, product_id, is_active) "
                "VALUES (gen_random_uuid(), 'mig054-' || gen_random_uuid() || '@e.com', "
                "'x', 'vex', true) RETURNING id"
            )
        )
    ).scalar_one()


_INSERT_JOB = text(
    "INSERT INTO generation_jobs (id, user_id, product_id, provider, name, status, "
    "generation_type, prompt, is_deleted) "
    "VALUES (gen_random_uuid(), :user_id, 'vex', 'grok', 'job', :status, 't2v', 'p', :deleted) "
    "RETURNING id"
)


async def test_downgrade_drops_the_constraint_and_upgrade_restores_it(
    db_engine: AsyncEngine,
) -> None:
    async with db_engine.connect() as connection:
        transaction = await connection.begin()
        try:
            assert await _constraint_exists(connection)
            await connection.run_sync(_run, "downgrade")
            assert not await _constraint_exists(connection)
            await connection.run_sync(_run, "upgrade")
            assert await _constraint_exists(connection)
            validated = await connection.execute(
                text("SELECT convalidated FROM pg_constraint WHERE conname = :name"),
                {"name": _CONSTRAINT},
            )
            assert validated.scalar_one() is True
        finally:
            await transaction.rollback()


async def test_upgrade_undeletes_in_flight_rows_and_keeps_finished_ones_deleted(
    db_engine: AsyncEngine,
) -> None:
    async with db_engine.connect() as connection:
        transaction = await connection.begin()
        try:
            await connection.run_sync(_run, "downgrade")
            user_id = await _seed_user(connection)
            for status in ("pending", "queued", "running", "completed", "failed"):
                await connection.execute(
                    _INSERT_JOB, {"user_id": user_id, "status": status, "deleted": True}
                )

            await connection.run_sync(_run, "upgrade")

            rows = await connection.execute(
                text("SELECT status, is_deleted FROM generation_jobs WHERE user_id = :u"),
                {"u": user_id},
            )
            assert dict(rows.all()) == {
                "pending": False,
                "queued": False,
                "running": False,
                "completed": True,
                "failed": True,
            }
        finally:
            await transaction.rollback()


@pytest.mark.parametrize(
    ("status", "valid"),
    [
        ("pending", False),
        ("queued", False),
        ("running", False),
        ("completed", True),
        ("failed", True),
        ("cancelled", True),
        ("moderated", True),
    ],
)
async def test_the_constraint_rejects_only_deleted_in_flight_rows(
    db_engine: AsyncEngine, status: str, valid: bool
) -> None:
    async with db_engine.connect() as connection:
        transaction = await connection.begin()
        try:
            user_id = await _seed_user(connection)
            params = {"user_id": user_id, "status": status, "deleted": True}
            if valid:
                await connection.execute(_INSERT_JOB, params)
            else:
                with pytest.raises(IntegrityError, match=_CONSTRAINT):
                    await connection.execute(_INSERT_JOB, params)
        finally:
            await transaction.rollback()


async def test_updating_a_running_row_to_deleted_raises(db_engine: AsyncEngine) -> None:
    async with db_engine.connect() as connection:
        transaction = await connection.begin()
        try:
            user_id = await _seed_user(connection)
            job_id = (
                await connection.execute(
                    _INSERT_JOB, {"user_id": user_id, "status": "running", "deleted": False}
                )
            ).scalar_one()
            with pytest.raises(IntegrityError, match=_CONSTRAINT):
                await connection.execute(
                    text("UPDATE generation_jobs SET is_deleted = true WHERE id = :id"),
                    {"id": job_id},
                )
        finally:
            await transaction.rollback()
