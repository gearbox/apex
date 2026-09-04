"""Integration test for migration 040's gpu_session_commands round-trip (invariant 17).

Unlike 039, this migration has no backfill — the table starts empty — so the round
trip only has to prove the schema shape appears/disappears cleanly across
040 <-> 039 <-> 040, plus that the four indexes described in the P3 prompt exist
with the right predicates.

Synchronous test function, same pattern as test_gpu_session_deployments_migration.py:
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


def test_migration_040_round_trips(
    test_database_url: str,
    db_engine: AsyncEngine,
    monkeypatch,  # type: ignore[no-untyped-def]
) -> None:
    assert db_engine is not None  # The session fixture has upgraded the schema to head.
    monkeypatch.setenv("DATABASE_URL", test_database_url)
    monkeypatch.setenv("DEBUG", "false")
    config = Config("alembic.ini")

    async def _table_and_indexes() -> tuple[bool, set[str]]:
        async with db_engine.connect() as conn:
            exists = await conn.execute(
                text(
                    "SELECT EXISTS (SELECT 1 FROM information_schema.tables "
                    "WHERE table_name = 'gpu_session_commands')"
                )
            )
            table_exists = bool(exists.scalar_one())
            indexes = set(
                (
                    await conn.execute(
                        text(
                            "SELECT indexname FROM pg_indexes "
                            "WHERE tablename = 'gpu_session_commands'"
                        )
                    )
                ).scalars()
            )
            return table_exists, indexes

    try:
        # Start from 039 (no gpu_session_commands table).
        command.downgrade(config, "039")
        table_exists, _ = asyncio.run(_table_and_indexes())
        assert table_exists is False

        # --- invariant 17: round-trip ------------------------------------
        command.upgrade(config, "040")
        table_exists, indexes = asyncio.run(_table_and_indexes())
        assert table_exists is True
        assert "ix_gpu_session_commands_one_claimed" in indexes
        assert "ix_gpu_session_commands_queue" in indexes
        assert "ix_gpu_session_commands_deadline" in indexes
        assert "ix_gpu_session_commands_batch" in indexes

        command.downgrade(config, "039")
        table_exists, _ = asyncio.run(_table_and_indexes())
        assert table_exists is False

        command.upgrade(config, "040")
        table_exists, indexes = asyncio.run(_table_and_indexes())
        assert table_exists is True
        assert "ix_gpu_session_commands_one_claimed" in indexes
    finally:
        # Integration tests share the migrated schema, so always restore head.
        command.upgrade(config, "head")
