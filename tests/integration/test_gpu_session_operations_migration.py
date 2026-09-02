"""Round-trip migration 038's telemetry-operation schema changes."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from alembic.config import Config
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

from alembic import command

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncEngine


async def _schema_snapshot(database_url: str) -> tuple[set[str], set[str], set[str]]:
    engine = create_async_engine(database_url)
    try:
        async with engine.connect() as connection:
            session_columns = set(
                (
                    await connection.execute(
                        text(
                            "SELECT column_name FROM information_schema.columns "
                            "WHERE table_name = 'gpu_sessions'"
                        )
                    )
                ).scalars()
            )
            operation_columns = set(
                (
                    await connection.execute(
                        text(
                            "SELECT column_name FROM information_schema.columns "
                            "WHERE table_name = 'gpu_session_operations'"
                        )
                    )
                ).scalars()
            )
            operation_indexes = set(
                (
                    await connection.execute(
                        text(
                            "SELECT indexname FROM pg_indexes "
                            "WHERE tablename = 'gpu_session_operations'"
                        )
                    )
                ).scalars()
            )
            return session_columns, operation_columns, operation_indexes
    finally:
        await engine.dispose()


def test_revision_038_downgrade_upgrade_round_trip(
    test_database_url: str,
    db_engine: AsyncEngine,
    monkeypatch,
) -> None:  # type: ignore[no-untyped-def]
    """Downgrade restores v1 columns; upgrade restores v2 columns and indexes."""
    assert db_engine is not None  # The session fixture has upgraded the schema to head.
    monkeypatch.setenv("DATABASE_URL", test_database_url)
    monkeypatch.setenv("DEBUG", "false")
    config = Config("alembic.ini")

    try:
        command.downgrade(config, "037")
        session_columns, operation_columns, _operation_indexes = asyncio.run(
            _schema_snapshot(test_database_url)
        )
        assert {"provisioning_phase", "provisioning_progress"} <= session_columns
        assert "bootstrap_operation_id" not in session_columns
        assert not operation_columns

        command.upgrade(config, "038")
        session_columns, operation_columns, operation_indexes = asyncio.run(
            _schema_snapshot(test_database_url)
        )
        assert "bootstrap_operation_id" in session_columns
        assert not {"provisioning_phase", "provisioning_progress"} & session_columns
        assert {
            "id",
            "session_id",
            "command_id",
            "last_sequence",
            "last_event_id",
            "terminal_at",
        } <= operation_columns
        assert {
            "ix_gpu_session_operations_session_created",
            "ix_gpu_session_operations_batch",
        } <= operation_indexes
    finally:
        # Integration tests share the migrated schema, so always restore head.
        command.upgrade(config, "head")
