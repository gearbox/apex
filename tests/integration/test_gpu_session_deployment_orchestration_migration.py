"""Integration test for migration 041's round-trip (invariant 16: 041 -> 040 -> 041).

Additive only, no backfill — the round trip only has to prove the three new columns
and the new partial index appear/disappear cleanly across 041 <-> 040 <-> 041.

Synchronous test function, same pattern as test_gpu_session_commands_migration.py:
alembic's ``command.upgrade``/``command.downgrade`` call ``asyncio.run()``
internally, so they can't run inside an ``async def`` test under pytest-asyncio.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from alembic.config import Config
from sqlalchemy import text

from alembic import command

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncEngine


def test_migration_041_round_trips(
    test_database_url: str,
    db_engine: AsyncEngine,
    monkeypatch,  # type: ignore[no-untyped-def]
) -> None:
    assert db_engine is not None  # The session fixture has upgraded the schema to head.
    monkeypatch.setenv("DATABASE_URL", test_database_url)
    monkeypatch.setenv("DEBUG", "false")
    config = Config("alembic.ini")

    async def _columns_and_indexes() -> tuple[set[str], set[str]]:
        async with db_engine.connect() as conn:
            columns = set(
                (
                    await conn.execute(
                        text(
                            "SELECT column_name FROM information_schema.columns "
                            "WHERE table_name = 'gpu_session_deployments'"
                        )
                    )
                ).scalars()
            )
            indexes = set(
                (
                    await conn.execute(
                        text(
                            "SELECT indexname FROM pg_indexes "
                            "WHERE tablename = 'gpu_session_deployments'"
                        )
                    )
                ).scalars()
            )
            return columns, indexes

    try:
        # Start from 040 (no P4 orchestration columns).
        command.downgrade(config, "040")
        columns, indexes = asyncio.run(_columns_and_indexes())
        assert "restart_operation_id" not in columns
        assert "batch_id" not in columns
        assert "pending_restart_since" not in columns
        assert "routing_suspended" not in columns
        assert "ix_gpu_session_deployments_pending_restart" not in indexes

        # --- invariant 16: round-trip ------------------------------------
        command.upgrade(config, "041")
        columns, indexes = asyncio.run(_columns_and_indexes())
        assert "restart_operation_id" in columns
        assert "batch_id" in columns
        assert "pending_restart_since" in columns
        assert "routing_suspended" in columns
        assert "ix_gpu_session_deployments_pending_restart" in indexes

        command.downgrade(config, "040")
        columns, indexes = asyncio.run(_columns_and_indexes())
        assert "restart_operation_id" not in columns
        assert "batch_id" not in columns
        assert "pending_restart_since" not in columns
        assert "routing_suspended" not in columns

        command.upgrade(config, "041")
        columns, indexes = asyncio.run(_columns_and_indexes())
        assert "restart_operation_id" in columns
        assert "batch_id" in columns
        assert "pending_restart_since" in columns
        assert "routing_suspended" in columns
        assert "ix_gpu_session_deployments_pending_restart" in indexes
    finally:
        # Integration tests share the migrated schema, so always restore head.
        command.upgrade(config, "head")
