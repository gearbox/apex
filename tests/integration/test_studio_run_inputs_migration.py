"""Migration 058: ``studio_runs.inputs`` and its two CHECK constraints (invariants 21, 24 and 26)."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from alembic.operations import Operations
from alembic.runtime.migration import MigrationContext
from sqlalchemy import text

from src.llm.port import LlmUnavailable
from src.studio.run_requests import LyricsRunRequest, PlanRunRequest
from tests.llm_support import FakeLlm
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


_CONSTRAINTS = {"ck_studio_runs_inputs_by_kind", "ck_studio_runs_result_by_kind"}


async def _by_kind_constraints(connection: AsyncConnection) -> set[str]:
    rows = await connection.execute(
        text(
            "SELECT conname FROM pg_constraint "
            "WHERE conrelid = 'studio_runs'::regclass AND conname LIKE 'ck_studio_runs_%_by_kind'"
        )
    )
    return {row[0] for row in rows}


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
        assert await _by_kind_constraints(connection) == _CONSTRAINTS

        await connection.run_sync(_run, "downgrade")
        assert await _column(connection) is None
        assert await _columns(connection) == [name for name in before if name != "inputs"]
        assert await _by_kind_constraints(connection) == set()

        await connection.run_sync(_run, "upgrade")
        assert await _column(connection) == ("jsonb", "YES")
        assert sorted(await _columns(connection)) == sorted(before)
        assert await _by_kind_constraints(connection) == _CONSTRAINTS
        await transaction.rollback()


async def test_with_only_plan_runs_the_downgrade_round_trips_and_every_row_has_sql_null_inputs(
    run_world: RunWorld, db_engine: AsyncEngine
) -> None:
    """A ``plan`` run has ``inputs IS NULL`` - in SQL, before and after a downgrade and upgrade."""
    actor = await run_world.add_actor()
    track_id: UUID = await run_world.new_track(actor, "T", PLAN)
    queued = await run_world.submit(actor, track_id, request=PlanRunRequest(brief="x"))
    async with run_world.factory() as session:
        await run_world.service(session).cancel(
            queued.run.id, user_id=actor.user_id, product_id="vex"
        )
        await session.commit()
    failed = await run_world.submit(actor, track_id, request=PlanRunRequest(brief="y"))
    await run_world.drive(FakeLlm(LlmUnavailable("down")))
    assert (await run_world.run(failed.run.id)).status == "failed"

    async with db_engine.connect() as connection:
        transaction = await connection.begin()

        async def sql_nulls() -> dict[UUID, tuple[bool, bool]]:
            rows = await connection.execute(
                text("SELECT id, inputs IS NULL, error_detail IS NULL FROM studio_runs")
            )
            return {row[0]: (row[1], row[2]) for row in rows}

        assert await sql_nulls() == {queued.run.id: (True, True), failed.run.id: (True, True)}
        await connection.run_sync(_run, "downgrade")
        await connection.run_sync(_run, "upgrade")
        assert await sql_nulls() == {queued.run.id: (True, True), failed.run.id: (True, True)}
        assert await _by_kind_constraints(connection) == _CONSTRAINTS
        await transaction.rollback()


async def test_downgrade_refuses_while_a_lyrics_run_exists_and_changes_nothing_invariant_26(
    run_world: RunWorld, db_engine: AsyncEngine
) -> None:
    """The 057 schema cannot represent a lyrics run once ``inputs`` is gone: fail loud (cf. 056)."""
    actor = await run_world.add_actor()
    track_id: UUID = await run_world.new_track(actor, "T", PLAN)
    lyrics_run = await run_world.submit(actor, track_id, request=LyricsRunRequest(language="en"))
    assert (await run_world.run(lyrics_run.run.id)).inputs is not None

    async with db_engine.connect() as connection:
        transaction = await connection.begin()
        before = await _columns(connection)

        with pytest.raises(RuntimeError, match=r"cannot downgrade to 057: 1 studio run\(s\)"):
            await connection.run_sync(_run, "downgrade")

        assert await _columns(connection) == before  # nothing changed ...
        assert await _by_kind_constraints(connection) == _CONSTRAINTS
        rows = (await connection.execute(text("SELECT id, kind FROM studio_runs"))).all()
        assert [(row[0], row[1]) for row in rows] == [(lyrics_run.run.id, "lyrics")]
        await transaction.rollback()


async def test_upgrade_turns_a_stored_json_null_error_detail_into_sql_null(
    run_world: RunWorld, db_engine: AsyncEngine
) -> None:
    """A failure with no detail was stored as the JSON value ``null`` before the model said not to."""
    actor = await run_world.add_actor()
    track_id: UUID = await run_world.new_track(actor, "T", PLAN)
    write = await run_world.submit(actor, track_id, request=PlanRunRequest(brief="x"))
    await run_world.drive(FakeLlm(LlmUnavailable("down")))

    async with db_engine.connect() as connection:
        transaction = await connection.begin()
        await connection.execute(
            text("UPDATE studio_runs SET error_detail = 'null'::jsonb WHERE id = :id"),
            {"id": write.run.id},
        )
        stored = text("SELECT error_detail IS NULL, jsonb_typeof(error_detail) FROM studio_runs")
        assert (await connection.execute(stored)).one() == (False, "null")

        await connection.run_sync(_run, "downgrade")
        await connection.run_sync(_run, "upgrade")
        assert (await connection.execute(stored)).one() == (True, None)
        await transaction.rollback()


async def test_a_database_that_ran_the_earlier_form_of_058_can_downgrade_and_upgrade(
    db_engine: AsyncEngine,
) -> None:
    """That form had the column and no constraints; the downgrade must not trip over their absence."""
    async with db_engine.connect() as connection:
        transaction = await connection.begin()
        for name in _CONSTRAINTS:
            await connection.execute(text(f"ALTER TABLE studio_runs DROP CONSTRAINT {name}"))
        assert await _by_kind_constraints(connection) == set()

        await connection.run_sync(_run, "downgrade")
        assert await _column(connection) is None
        await connection.run_sync(_run, "upgrade")
        assert await _by_kind_constraints(connection) == _CONSTRAINTS
        await transaction.rollback()
