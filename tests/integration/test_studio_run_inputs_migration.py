"""Migration 058: ``studio_runs.inputs`` upgrades, downgrades and changes nothing else (invariant 21)."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import TYPE_CHECKING

from alembic.operations import Operations
from alembic.runtime.migration import MigrationContext
from sqlalchemy import text

from src.studio.run_requests import LyricsRunRequest, PlanRunRequest
from tests.studio_lyrics_support import PLAN

if TYPE_CHECKING:
    from types import ModuleType
    from uuid import UUID

    from sqlalchemy import Connection
    from sqlalchemy.ext.asyncio import AsyncConnection, AsyncEngine

    from tests.studio_runs_support import RunWorld

_PATH = Path(__file__).resolve().parents[2] / "alembic" / "versions" / "058_studio_run_inputs.py"


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("revision_058", _PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run(connection: Connection, action: str) -> None:
    with Operations.context(MigrationContext.configure(connection)):
        getattr(_load(), action)()


async def _column(connection: AsyncConnection) -> tuple[str, str] | None:
    row = (
        await connection.execute(
            text(
                "SELECT data_type, is_nullable FROM information_schema.columns "
                "WHERE table_name = 'studio_runs' AND column_name = 'inputs'"
            )
        )
    ).first()
    return (row[0], row[1]) if row else None


async def _columns(connection: AsyncConnection) -> list[str]:
    rows = await connection.execute(
        text(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_name = 'studio_runs' ORDER BY ordinal_position"
        )
    )
    return [row[0] for row in rows]


def test_revision_chain() -> None:
    migration = _load()
    assert migration.revision == "058"
    assert migration.down_revision == "057"


async def test_the_column_is_a_nullable_jsonb(db_engine: AsyncEngine) -> None:
    async with db_engine.connect() as connection:
        assert await _column(connection) == ("jsonb", "YES")


async def test_downgrade_drops_exactly_that_column_and_upgrade_restores_it(
    db_engine: AsyncEngine,
) -> None:
    async with db_engine.connect() as connection:
        transaction = await connection.begin()
        before = await _columns(connection)

        await connection.run_sync(_run, "downgrade")
        assert await _column(connection) is None
        assert await _columns(connection) == [name for name in before if name != "inputs"]

        await connection.run_sync(_run, "upgrade")
        assert await _column(connection) == ("jsonb", "YES")
        assert sorted(await _columns(connection)) == sorted(before)
        await transaction.rollback()


async def test_existing_rows_have_no_inputs_after_the_upgrade(
    run_world: RunWorld, db_engine: AsyncEngine
) -> None:
    """A run that predates the column (or a ``plan`` run) reads ``inputs IS NULL``."""
    actor = await run_world.add_actor()
    track_id: UUID = await run_world.new_track(actor, "T", PLAN)
    plan_run = await run_world.submit(actor, track_id, request=PlanRunRequest(brief="x"))
    async with run_world.factory() as session:
        await run_world.service(session).cancel(
            plan_run.run.id, user_id=actor.user_id, product_id="vex"
        )
        await session.commit()
    lyrics_run = await run_world.submit(actor, track_id, request=LyricsRunRequest(language="en"))
    assert (await run_world.run(plan_run.run.id)).inputs is None  # a plan run never has inputs
    assert (await run_world.run(lyrics_run.run.id)).inputs is not None

    async with db_engine.connect() as connection:
        transaction = await connection.begin()
        await connection.run_sync(_run, "downgrade")
        await connection.run_sync(_run, "upgrade")
        rows = (await connection.execute(text("SELECT id, inputs FROM studio_runs"))).all()
        by_id = {row[0]: row[1] for row in rows}
        assert by_id[plan_run.run.id] is None
        assert by_id[lyrics_run.run.id] is None  # the downgrade dropped what the column held
        await transaction.rollback()
