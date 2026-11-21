"""Migration 059: the ``studio_artist_memories`` table (invariant 18).

The upgrade creates it with its CHECKs, its cascading FK and its two indexes; the downgrade drops
exactly that; ``alembic check`` (no drift between the model and the migrations) is run on a fresh
database in CI and by hand in the PR checklist. The migration never imports application code, so a
later change to the model cannot rewrite what it did.
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
from sqlalchemy.exc import IntegrityError

if TYPE_CHECKING:
    from types import ModuleType

    from sqlalchemy import Connection
    from sqlalchemy.ext.asyncio import AsyncConnection, AsyncEngine

_PATH = (
    Path(__file__).resolve().parents[2] / "alembic" / "versions" / "059_studio_artist_memories.py"
)
_TABLE = "studio_artist_memories"
_CHECKS = {
    "ck_studio_artist_memories_salience",
    "ck_studio_artist_memories_status",
    "ck_studio_artist_memories_origin",
}
_INDEXES = {"ix_studio_artist_memories_list", "ix_studio_artist_memories_selection"}


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("revision_059", _PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run(connection: Connection, action: str) -> None:
    with Operations.context(MigrationContext.configure(connection)):
        getattr(_load(), action)()


async def _exists(connection: AsyncConnection) -> bool:
    return (
        await connection.execute(text("SELECT to_regclass(:name) IS NOT NULL"), {"name": _TABLE})
    ).scalar_one()


async def _scalars(connection: AsyncConnection, sql: str) -> set[str]:
    return {row[0] for row in await connection.execute(text(sql), {"table": _TABLE})}


async def _checks(connection: AsyncConnection) -> set[str]:
    return await _scalars(
        connection,
        "SELECT conname FROM pg_constraint WHERE conrelid = to_regclass(:table) AND contype = 'c'",
    )


async def _indexes(connection: AsyncConnection) -> set[str]:
    return {
        name
        for name in await _scalars(
            connection, "SELECT indexname FROM pg_indexes WHERE tablename = :table"
        )
        if name.startswith("ix_")
    }


def test_revision_chain() -> None:
    migration = _load()
    assert migration.revision == "059"
    assert migration.down_revision == "058"


async def test_the_table_has_the_checks_the_indexes_and_a_cascading_foreign_key(
    db_engine: AsyncEngine,
) -> None:
    async with db_engine.connect() as connection:
        assert await _exists(connection)
        assert await _checks(connection) == _CHECKS
        assert await _indexes(connection) == _INDEXES
        on_delete = (
            await connection.execute(
                text(
                    "SELECT confdeltype::text FROM pg_constraint "
                    "WHERE conrelid = to_regclass(:table) AND contype = 'f'"
                ),
                {"table": _TABLE},
            )
        ).scalar_one()
        assert on_delete == "c"  # ON DELETE CASCADE: ownership is inherited through the artist
        columns = await _scalars(
            connection,
            "SELECT column_name FROM information_schema.columns WHERE table_name = :table",
        )
        # scoping is inherited through the artist (P2-M1): no user / product column of its own
        assert columns == {
            "id", "artist_id", "title", "body", "occurred_on", "themes", "imagery",
            "salience", "status", "origin", "created_at", "updated_at",
        }  # fmt: skip


async def test_the_selection_index_is_partial_and_in_the_selection_order(
    db_engine: AsyncEngine,
) -> None:
    async with db_engine.connect() as connection:
        definition = (
            await connection.execute(
                text(
                    "SELECT indexdef FROM pg_indexes "
                    "WHERE indexname = 'ix_studio_artist_memories_selection'"
                )
            )
        ).scalar_one()
    assert "salience DESC" in definition
    assert "occurred_on DESC NULLS LAST" in definition
    assert "WHERE ((status)::text = 'active'::text)" in definition


async def test_downgrade_drops_exactly_the_table_and_upgrade_restores_it(
    db_engine: AsyncEngine,
) -> None:
    async with db_engine.connect() as connection:
        transaction = await connection.begin()

        await connection.run_sync(_run, "downgrade")
        assert not await _exists(connection)

        await connection.run_sync(_run, "upgrade")
        assert await _exists(connection)
        assert await _checks(connection) == _CHECKS
        assert await _indexes(connection) == _INDEXES
        await transaction.rollback()


@pytest.mark.parametrize(
    ("column", "value"),
    [("salience", 0), ("salience", 6), ("status", "deleted"), ("origin", "import")],
)
async def test_the_checks_hold_in_the_database(
    db_engine: AsyncEngine, column: str, value: object
) -> None:
    """The CHECKs are the backstop behind the service: a bad value cannot be written at all."""
    async with db_engine.connect() as connection:
        transaction = await connection.begin()
        user_id, artist_id = uuid4(), uuid4()
        await connection.execute(
            text(
                "INSERT INTO users (id, email, password_hash, product_id) "
                "VALUES (:id, :email, 'x', 'vex')"
            ),
            {"id": user_id, "email": f"mig-{uuid4().hex}@example.com"},
        )
        await connection.execute(
            text(
                "INSERT INTO studio_artists (id, user_id, product_id, name) "
                "VALUES (:id, :user, 'vex', 'A')"
            ),
            {"id": artist_id, "user": user_id},
        )
        values = {"salience": 3, "status": "active", "origin": "user", column: value}
        with pytest.raises(IntegrityError):
            await connection.execute(
                text(
                    "INSERT INTO studio_artist_memories (id, artist_id, title, body, salience, "
                    "status, origin) VALUES (:id, :artist, 't', 'b', :salience, :status, :origin)"
                ),
                {"id": uuid4(), "artist": artist_id, **values},
            )
        await transaction.rollback()
