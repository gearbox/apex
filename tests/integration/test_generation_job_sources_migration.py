"""Integration coverage for migration 035's legacy-source backfill."""

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
    Path(__file__).resolve().parents[2] / "alembic" / "versions" / "035_generation_job_sources.py"
)


def _load_migration() -> ModuleType:
    spec = importlib.util.spec_from_file_location("revision_035", _MIGRATION_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


async def test_legacy_source_backfill_is_idempotent(db_engine: AsyncEngine) -> None:
    """Running the data backfill twice must retain one row per job position."""
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
                {"id": user_id, "email": f"migration-{user_id.hex}@example.com"},
            )
            await connection.execute(
                text(
                    "INSERT INTO generation_jobs "
                    "(id, user_id, name, status, generation_type, provider, prompt, product_id) "
                    "VALUES (:id, :user_id, 'migration source', 'completed', 't2i', 'grok', "
                    "'migration fixture', 'vex')"
                ),
                {"id": job_id, "user_id": user_id},
            )
            await connection.execute(
                text(
                    "INSERT INTO generation_outputs "
                    "(id, user_id, job_id, storage_key, content_type, size_bytes, format, "
                    "output_index, product_id, expires_at) "
                    "VALUES (:id, :user_id, :job_id, :storage_key, 'image/png', 1, 'png', "
                    "0, 'vex', :expires_at)"
                ),
                {
                    "id": output_id,
                    "user_id": user_id,
                    "job_id": job_id,
                    "storage_key": f"migration/{output_id}.png",
                    "expires_at": now + timedelta(days=7),
                },
            )
            await connection.execute(
                text("UPDATE generation_jobs SET source_output_id = :output_id WHERE id = :job_id"),
                {"output_id": output_id, "job_id": job_id},
            )

            def _run_backfill(sync_connection: Connection) -> None:
                context = MigrationContext.configure(sync_connection)
                with Operations.context(context):
                    migration._backfill()
                    migration._backfill()

            await connection.run_sync(_run_backfill)
            result = await connection.execute(
                text(
                    "SELECT count(*) FROM generation_job_sources "
                    "WHERE job_id = :job_id AND position = 0"
                ),
                {"job_id": job_id},
            )
            assert result.scalar_one() == 1
        finally:
            await transaction.rollback()
