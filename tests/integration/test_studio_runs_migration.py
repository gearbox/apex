"""Migration 057: ``studio_runs`` upgrades, downgrades and carries the constraints the design needs."""

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
    from sqlalchemy.ext.asyncio import AsyncConnection, AsyncEngine

_PATH = Path(__file__).resolve().parents[2] / "alembic" / "versions" / "057_studio_runs.py"


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("revision_057", _PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run(connection: Connection, action: str) -> None:
    with Operations.context(MigrationContext.configure(connection)):
        getattr(_load(), action)()


async def _exists(connection: AsyncConnection) -> bool:
    return bool(
        (
            await connection.execute(text("SELECT to_regclass('public.studio_runs') IS NOT NULL"))
        ).scalar_one()
    )


async def _constraints(connection: AsyncConnection) -> dict[str, tuple[str, str, str]]:
    rows = await connection.execute(
        text(
            "SELECT conname, contype::text, pg_get_constraintdef(oid), confdeltype::text "
            "FROM pg_constraint WHERE conrelid = 'studio_runs'::regclass"
        )
    )
    return {row[0]: (row[1], row[2], row[3]) for row in rows}


async def _indexes(connection: AsyncConnection) -> dict[str, str]:
    rows = await connection.execute(
        text("SELECT indexname, indexdef FROM pg_indexes WHERE tablename = 'studio_runs'")
    )
    return {row[0]: row[1] for row in rows}


def test_revision_chain() -> None:
    migration = _load()
    assert migration.revision == "057"
    assert migration.down_revision == "056"


async def test_upgrade_downgrade_upgrade_round_trip(db_engine: AsyncEngine) -> None:
    async with db_engine.connect() as connection:
        transaction = await connection.begin()
        assert await _exists(connection)

        await connection.run_sync(_run, "downgrade")
        assert not await _exists(connection)

        await connection.run_sync(_run, "upgrade")
        assert await _exists(connection)
        await transaction.rollback()


async def test_the_one_active_run_per_track_index_is_partial_and_unique(
    db_engine: AsyncEngine,
) -> None:
    async with db_engine.connect() as connection:
        definition = (await _indexes(connection))["uq_studio_runs_one_active_per_track"]
        assert "CREATE UNIQUE INDEX" in definition
        assert "(track_id)" in definition
        assert "status" in definition
        assert "'queued'" in definition
        assert "'running'" in definition
        assert "'failed'" not in definition


async def test_the_claim_and_sweep_indexes_are_partial(db_engine: AsyncEngine) -> None:
    async with db_engine.connect() as connection:
        indexes = await _indexes(connection)
    assert "(created_at)" in indexes["ix_studio_runs_queued"]
    assert "'queued'" in indexes["ix_studio_runs_queued"]
    assert "(lease_expires_at)" in indexes["ix_studio_runs_running_lease"]
    assert "'running'" in indexes["ix_studio_runs_running_lease"]
    assert "created_at DESC" in indexes["ix_studio_runs_user_created"]
    assert "id DESC" in indexes["ix_studio_runs_track_created"]
    assert "ix_studio_runs_product_id" in indexes


async def test_constraints_match_the_design(db_engine: AsyncEngine) -> None:
    async with db_engine.connect() as connection:
        constraints = await _constraints(connection)

    assert constraints["uq_studio_runs_user_idempotency"][1] == "UNIQUE (user_id, idempotency_key)"
    # enum CHECKs are generated from the same StrEnums the application uses
    assert "'revise_lyrics'" in constraints["ck_studio_runs_kind"][1]
    for status in ("queued", "running", "succeeded", "failed", "cancelled"):
        assert f"'{status}'" in constraints["ck_studio_runs_status"][1]
    for stage in ("context", "drafting", "validating", "revising", "saving"):
        assert f"'{stage}'" in constraints["ck_studio_runs_stage"][1]
    assert "price_tokens >= 0" in constraints["ck_studio_runs_price"][1]
    assert "finished_at IS NOT NULL" in constraints["ck_studio_runs_finished"][1]
    assert "error_code IS NOT NULL" in constraints["ck_studio_runs_failed_code"][1]


async def test_foreign_key_actions(db_engine: AsyncEngine) -> None:
    async with db_engine.connect() as connection:
        constraints = await _constraints(connection)
    foreign = {
        definition.split(" REFERENCES ")[0]: action
        for kind, definition, action in constraints.values()
        if kind == "f"
    }
    assert foreign["FOREIGN KEY (track_id)"] == "c"  # CASCADE
    assert foreign["FOREIGN KEY (user_id)"] == "c"
    assert foreign["FOREIGN KEY (result_plan_version_id)"] == "n"  # SET NULL
    assert foreign["FOREIGN KEY (result_lyrics_version_id)"] == "n"


async def test_a_run_row_needs_its_owner_and_track(db_engine: AsyncEngine) -> None:
    from sqlalchemy.exc import IntegrityError

    async with db_engine.connect() as connection:
        transaction = await connection.begin()
        with pytest.raises(IntegrityError):
            await connection.execute(
                text(
                    "INSERT INTO studio_runs (id, product_id, user_id, track_id, kind, status, "
                    "idempotency_key, request_hash, request, model_key, price_tokens) VALUES "
                    "(:id, 'vex', :u, :t, 'plan', 'queued', 'k', 'h', '{}', 'm', 0)"
                ),
                {"id": uuid4(), "u": uuid4(), "t": uuid4()},
            )
        await transaction.rollback()
