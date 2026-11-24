"""Migration 059: the ``studio_artist_memories`` table (invariant 18).

The upgrade creates ``uq_studio_artists_id_owner`` and then the table with its own ``user_id`` /
``product_id``, its CHECKs, its composite owner FK and its indexes; the downgrade drops the table
**and then** the unique constraint. After ``downgrade 058 && upgrade head`` the models and the
migrations do not drift (the ``alembic check`` of the PR checklist, run here through the same
comparison). The migration never imports application code, so a later change to the model cannot
rewrite what it did.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import uuid4

import pytest
from alembic.autogenerate import compare_metadata
from alembic.operations import Operations
from alembic.runtime.migration import MigrationContext
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError

from src.db.models.base import Base

if TYPE_CHECKING:
    from types import ModuleType

    from sqlalchemy import Connection
    from sqlalchemy.ext.asyncio import AsyncConnection, AsyncEngine

_PATH = (
    Path(__file__).resolve().parents[2] / "alembic" / "versions" / "059_studio_artist_memories.py"
)
_PATH_060 = (
    Path(__file__).resolve().parents[2] / "alembic" / "versions" / "060_studio_artist_chat.py"
)
_TABLE = "studio_artist_memories"
_CHECKS = {
    "ck_studio_artist_memories_salience",
    "ck_studio_artist_memories_status",
    "ck_studio_artist_memories_origin",
}
_INDEXES = {
    "ix_studio_artist_memories_list",
    "ix_studio_artist_memories_selection",
    "ix_studio_artist_memories_product_id",
    "ix_studio_artist_memories_user_id",
}
_OWNER_FK = "fk_studio_artist_memories_artist_owner"
_OWNER_UNIQUE = "uq_studio_artists_id_owner"


def _load(path: Path = _PATH, name: str = "revision_059") -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run(connection: Connection, action: str) -> None:
    """059's ``upgrade`` / ``downgrade``, with 060 undone first and redone last: 060's chats
    reference 059's unique key on the artists, so 059 cannot be undone underneath them."""
    with Operations.context(MigrationContext.configure(connection)):
        chat = _load(_PATH_060, "revision_060")
        if action == "downgrade":
            chat.downgrade()
        getattr(_load(), action)()
        if action == "upgrade":
            chat.upgrade()


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


async def _constraint_exists(connection: AsyncConnection, name: str) -> bool:
    return (
        await connection.execute(
            text("SELECT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = :name)"),
            {"name": name},
        )
    ).scalar_one()


async def _index_definition(connection: AsyncConnection, name: str) -> str:
    return (
        await connection.execute(
            text("SELECT indexdef FROM pg_indexes WHERE indexname = :name"), {"name": name}
        )
    ).scalar_one()


def test_revision_chain() -> None:
    migration = _load()
    assert migration.revision == "059"
    assert migration.down_revision == "058"


async def test_the_table_has_the_checks_the_indexes_and_the_cascading_foreign_keys(
    db_engine: AsyncEngine,
) -> None:
    async with db_engine.connect() as connection:
        assert await _exists(connection)
        assert await _checks(connection) == _CHECKS
        assert await _indexes(connection) == _INDEXES
        foreign_keys = {
            row[0]: (row[1], row[2])
            for row in await connection.execute(
                text(
                    "SELECT conname, confdeltype::text, pg_get_constraintdef(oid) "
                    "FROM pg_constraint WHERE conrelid = to_regclass(:table) AND contype = 'f'"
                ),
                {"table": _TABLE},
            )
        }
        # two FKs, both ON DELETE CASCADE: the user, and the artist *with* the owner pair
        assert {on_delete for on_delete, _ in foreign_keys.values()} == {"c"}
        assert _OWNER_FK in foreign_keys
        definition = foreign_keys[_OWNER_FK][1]
        assert "FOREIGN KEY (artist_id, user_id, product_id)" in definition
        assert "REFERENCES studio_artists(id, user_id, product_id)" in definition
        assert await _constraint_exists(connection, _OWNER_UNIQUE)
        columns = await _scalars(
            connection,
            "SELECT column_name FROM information_schema.columns WHERE table_name = :table",
        )
        assert columns == {
            "id", "product_id", "user_id", "artist_id", "title", "body", "occurred_on", "themes",
            "imagery", "salience", "status", "origin", "created_at", "updated_at",
        }  # fmt: skip


async def test_the_list_index_is_in_diary_order(db_engine: AsyncEngine) -> None:
    async with db_engine.connect() as connection:
        definition = await _index_definition(connection, "ix_studio_artist_memories_list")
    assert definition.endswith(
        "(artist_id, status, occurred_on DESC NULLS LAST, created_at DESC, id DESC)"
    )


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


async def test_downgrade_drops_the_table_and_the_unique_constraint_and_upgrade_restores_them(
    db_engine: AsyncEngine,
) -> None:
    async with db_engine.connect() as connection:
        transaction = await connection.begin()

        await connection.run_sync(_run, "downgrade")
        assert not await _exists(connection)
        assert not await _constraint_exists(connection, _OWNER_UNIQUE)

        await connection.run_sync(_run, "upgrade")
        assert await _exists(connection)
        assert await _constraint_exists(connection, _OWNER_UNIQUE)
        assert await _checks(connection) == _CHECKS
        assert await _indexes(connection) == _INDEXES
        await transaction.rollback()


def _drift(connection: Connection) -> list[str]:
    """What ``alembic check`` would report for the two tables this migration touches."""
    diffs = compare_metadata(MigrationContext.configure(connection), Base.metadata)
    flat = [str(item) for diff in diffs for item in (diff if isinstance(diff, list) else [diff])]
    return [item for item in flat if "studio_artist" in item]


async def test_after_downgrade_and_upgrade_the_models_and_the_migration_do_not_drift(
    db_engine: AsyncEngine,
) -> None:
    async with db_engine.connect() as connection:
        transaction = await connection.begin()
        await connection.run_sync(_run, "downgrade")
        await connection.run_sync(_run, "upgrade")
        assert await connection.run_sync(_drift) == []
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
                    "INSERT INTO studio_artist_memories (id, user_id, product_id, artist_id, title, "
                    "body, salience, status, origin) VALUES "
                    "(:id, :user, 'vex', :artist, 't', 'b', :salience, :status, :origin)"
                ),
                {"id": uuid4(), "user": user_id, "artist": artist_id, **values},
            )
        await transaction.rollback()
