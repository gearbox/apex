"""Round-trip migration 036 against an existing generation output."""

from __future__ import annotations

import importlib.util
from datetime import UTC, datetime, timedelta
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
    / "036_add_generation_output_duration.py"
)


def _load_migration() -> ModuleType:
    spec = importlib.util.spec_from_file_location("revision_036", _MIGRATION_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


async def test_duration_migration_round_trips_populated_outputs(
    db_engine: AsyncEngine,
) -> None:
    """Migration 036 can drop and restore its nullable column with data present."""
    migration = _load_migration()
    user_id = uuid4()
    job_id = uuid4()
    output_id = uuid4()
    now = datetime.now(UTC)

    async with db_engine.connect() as connection:
        transaction = await connection.begin()
        try:
            await connection.execute(
                text(
                    "INSERT INTO users (id, email, password_hash, product_id) "
                    "VALUES (:id, :email, 'migration-test-hash', 'vex')"
                ),
                {"id": user_id, "email": f"duration-{user_id.hex}@example.com"},
            )
            await connection.execute(
                text(
                    "INSERT INTO generation_jobs "
                    "(id, user_id, name, status, generation_type, provider, prompt, product_id) "
                    "VALUES (:id, :user_id, 'duration migration', 'completed', 't2i', 'grok', "
                    "'migration fixture', 'vex')"
                ),
                {"id": job_id, "user_id": user_id},
            )
            await connection.execute(
                text(
                    "INSERT INTO generation_outputs "
                    "(id, user_id, job_id, storage_key, content_type, size_bytes, format, "
                    "output_index, product_id, duration_ms, expires_at) "
                    "VALUES (:id, :user_id, :job_id, :storage_key, 'video/mp4', 1, 'mp4', "
                    "0, 'vex', 1234, :expires_at)"
                ),
                {
                    "id": output_id,
                    "user_id": user_id,
                    "job_id": job_id,
                    "storage_key": f"migration/{output_id}.mp4",
                    "expires_at": now + timedelta(days=7),
                },
            )

            def _round_trip(sync_connection: Connection) -> None:
                context = MigrationContext.configure(sync_connection)
                with Operations.context(context):
                    migration.downgrade()
                    migration.upgrade()

            await connection.run_sync(_round_trip)
            result = await connection.execute(
                text("SELECT duration_ms FROM generation_outputs WHERE id = :output_id"),
                {"output_id": output_id},
            )
            assert result.scalar_one() is None
        finally:
            await transaction.rollback()
