"""Migration 061: the library asset holds and the studio cast (S1-P3).

The upgrade creates ``library_asset_holds`` (CHECK, two ``NO ACTION`` asset foreign keys, the two
partial unique indexes that double as the ``EXISTS`` lookups), the four cast tables (composite
owner FKs, ``hold_id NOT NULL UNIQUE``, a deferrable position key) and the artist's performer
column. The downgrade drops all of it, in reverse order, and nothing else; ``upgrade → downgrade →
upgrade`` works, and the models and the migration do not drift (the ``alembic check`` of the PR
checklist, through the same comparison). The migration never imports application code.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import TYPE_CHECKING

from alembic.autogenerate import compare_metadata
from alembic.operations import Operations
from alembic.runtime.migration import MigrationContext
from sqlalchemy import text

from src.db.models.base import Base

if TYPE_CHECKING:
    from types import ModuleType

    from sqlalchemy import Connection
    from sqlalchemy.ext.asyncio import AsyncConnection, AsyncEngine

_PATH = (
    Path(__file__).resolve().parents[2] / "alembic" / "versions" / "061_studio_cast_asset_holds.py"
)
_NEW_TABLES = (
    "library_asset_holds",
    "studio_cast_members",
    "studio_cast_references",
    "studio_cast_looks",
    "studio_cast_look_references",
)


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("revision_061", _PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run(connection: Connection, action: str) -> None:
    with Operations.context(MigrationContext.configure(connection)):
        getattr(_load(), action)()


async def _exists(connection: AsyncConnection, table: str) -> bool:
    return (
        await connection.execute(text("SELECT to_regclass(:name) IS NOT NULL"), {"name": table})
    ).scalar_one()


async def _constraints(
    connection: AsyncConnection, table: str, kind: str
) -> dict[str, tuple[str, bool, bool, str]]:
    """``name -> (confdeltype, deferrable, initially deferred, definition)``."""
    rows = await connection.execute(
        text(
            "SELECT conname, confdeltype::text, condeferrable, condeferred, "
            "pg_get_constraintdef(oid) FROM pg_constraint "
            "WHERE conrelid = to_regclass(:table) AND contype::text = :kind"
        ),
        {"table": table, "kind": kind},
    )
    return {row[0]: (row[1], row[2], row[3], row[4]) for row in rows}


async def _index(connection: AsyncConnection, name: str) -> str:
    return (
        await connection.execute(
            text("SELECT indexdef FROM pg_indexes WHERE indexname = :name"), {"name": name}
        )
    ).scalar_one()


def test_revision_chain() -> None:
    migration = _load()
    assert migration.revision == "061"
    assert migration.down_revision == "060"


async def test_the_holds_table_has_its_check_foreign_keys_and_partial_uniques(
    db_engine: AsyncEngine,
) -> None:
    async with db_engine.connect() as connection:
        checks = await _constraints(connection, "library_asset_holds", "c")
        foreign_keys = await _constraints(connection, "library_asset_holds", "f")
        upload_unique = await _index(connection, "uq_library_asset_holds_upload_holder")
        output_unique = await _index(connection, "uq_library_asset_holds_output_holder")
        holder = await _index(connection, "ix_library_asset_holds_holder")
        product = await _index(connection, "ix_library_asset_holds_product_id")

    assert set(checks) == {"ck_library_asset_holds_one_asset"}
    assert "num_nonnulls(upload_id, output_id) = 1" in checks["ck_library_asset_holds_one_asset"][3]
    # NO ACTION ('a'), never RESTRICT ('r'): a user hard-delete removes holds and assets in one
    # cascading statement, and only NO ACTION is checked at the end of that statement.
    assert foreign_keys["fk_library_asset_holds_upload_id"][0] == "a"
    assert foreign_keys["fk_library_asset_holds_output_id"][0] == "a"
    assert "REFERENCES user_images(id)" in foreign_keys["fk_library_asset_holds_upload_id"][3]
    assert (
        "REFERENCES generation_outputs(id)" in foreign_keys["fk_library_asset_holds_output_id"][3]
    )
    user_fks = [name for name, fk in foreign_keys.items() if "REFERENCES users(id)" in fk[3]]
    assert [foreign_keys[name][0] for name in user_fks] == ["c"]
    assert upload_unique.startswith("CREATE UNIQUE INDEX")
    assert "(upload_id, holder_kind, holder_id)" in upload_unique
    assert "WHERE (upload_id IS NOT NULL)" in upload_unique
    assert output_unique.startswith("CREATE UNIQUE INDEX")
    assert "(output_id, holder_kind, holder_id)" in output_unique
    assert "WHERE (output_id IS NOT NULL)" in output_unique
    assert holder.endswith("(holder_kind, holder_id)")
    assert product.endswith("(product_id)")


async def test_the_cast_tables_have_composite_owner_keys_and_their_uniques(
    db_engine: AsyncEngine,
) -> None:
    async with db_engine.connect() as connection:
        members = await _constraints(connection, "studio_cast_members", "u")
        references_fk = await _constraints(connection, "studio_cast_references", "f")
        references_unique = await _constraints(connection, "studio_cast_references", "u")
        references_check = await _constraints(connection, "studio_cast_references", "c")
        looks_fk = await _constraints(connection, "studio_cast_looks", "f")
        links = await _constraints(connection, "studio_cast_look_references", "f")
        links_pk = await _constraints(connection, "studio_cast_look_references", "p")
        member_name = await _index(connection, "uq_studio_cast_members_owner_name")
        label = await _index(connection, "uq_studio_cast_references_member_label")
        look_name = await _index(connection, "uq_studio_cast_looks_member_name")
        member_list = await _index(connection, "ix_studio_cast_members_owner_created")
        link_index = await _index(connection, "ix_studio_cast_look_references_reference")

    assert "uq_studio_cast_members_id_owner" in members
    assert "UNIQUE (id, user_id, product_id)" in members["uq_studio_cast_members_id_owner"][3]
    owner_fk = references_fk["fk_studio_cast_references_member_owner"]
    assert owner_fk[0] == "c"
    assert "FOREIGN KEY (cast_member_id, user_id, product_id)" in owner_fk[3]
    assert "REFERENCES studio_cast_members(id, user_id, product_id)" in owner_fk[3]
    assert references_fk["fk_studio_cast_references_hold"][0] == "a"  # NO ACTION
    assert (
        "REFERENCES library_asset_holds(id)" in references_fk["fk_studio_cast_references_hold"][3]
    )
    assert "UNIQUE (hold_id)" in references_unique["uq_studio_cast_references_hold_id"][3]
    # a reorder rewrites many rows: the position key is checked at commit
    _, deferrable, deferred, definition = references_unique["uq_studio_cast_references_position"]
    assert (deferrable, deferred) == (True, True)
    assert 'UNIQUE (cast_member_id, "position")' in definition or (
        "UNIQUE (cast_member_id, position)" in definition
    )
    assert "ck_studio_cast_references_position" in references_check
    assert looks_fk["fk_studio_cast_looks_member_owner"][0] == "c"
    assert {fk[0] for fk in links.values()} == {"c"}
    assert "PRIMARY KEY (look_id, reference_id)" in next(iter(links_pk.values()))[3]
    assert member_name.startswith("CREATE UNIQUE INDEX")
    assert "(user_id, product_id, lower((name)::text))" in member_name
    assert label.startswith("CREATE UNIQUE INDEX")
    assert "(cast_member_id, lower((label)::text))" in label
    assert look_name.startswith("CREATE UNIQUE INDEX")
    assert "(cast_member_id, lower((name)::text))" in look_name
    assert member_list.endswith("(user_id, product_id, created_at, id)")
    assert link_index.endswith("(reference_id)")


async def test_the_artist_performer_link_is_a_no_action_composite_key_with_a_partial_index(
    db_engine: AsyncEngine,
) -> None:
    async with db_engine.connect() as connection:
        foreign_keys = await _constraints(connection, "studio_artists", "f")
        index = await _index(connection, "ix_studio_artists_performer")
        nullable = (
            await connection.execute(
                text(
                    "SELECT is_nullable FROM information_schema.columns "
                    "WHERE table_name = 'studio_artists' "
                    "AND column_name = 'performer_cast_member_id'"
                )
            )
        ).scalar_one()

    performer = foreign_keys["fk_studio_artists_performer"]
    assert performer[0] == "a"  # NO ACTION: deleting a performer is refused, not unlinked
    assert "FOREIGN KEY (performer_cast_member_id, user_id, product_id)" in performer[3]
    assert "REFERENCES studio_cast_members(id, user_id, product_id)" in performer[3]
    assert "WHERE (performer_cast_member_id IS NOT NULL)" in index
    assert nullable == "YES"


async def test_downgrade_drops_what_the_revision_made_and_nothing_else_and_upgrade_restores_it(
    db_engine: AsyncEngine,
) -> None:
    async with db_engine.connect() as connection:
        transaction = await connection.begin()

        await connection.run_sync(_run, "downgrade")
        for table in _NEW_TABLES:
            assert not await _exists(connection, table), table
        columns = (
            (
                await connection.execute(
                    text(
                        "SELECT column_name FROM information_schema.columns "
                        "WHERE table_name = 'studio_artists'"
                    )
                )
            )
            .scalars()
            .all()
        )
        assert "performer_cast_member_id" not in columns
        # what the earlier revisions made is untouched
        for table in ("studio_artists", "studio_artist_chats", "user_images", "generation_outputs"):
            assert await _exists(connection, table), table

        await connection.run_sync(_run, "upgrade")
        for table in _NEW_TABLES:
            assert await _exists(connection, table), table
        assert "fk_studio_artists_performer" in await _constraints(
            connection, "studio_artists", "f"
        )
        await transaction.rollback()


def _drift(connection: Connection) -> list[str]:
    """What ``alembic check`` would report for the tables this migration touches."""
    diffs = compare_metadata(MigrationContext.configure(connection), Base.metadata)
    flat = [str(item) for diff in diffs for item in (diff if isinstance(diff, list) else [diff])]
    return [
        item
        for item in flat
        if "cast" in item or "library_asset_holds" in item or "performer" in item
    ]


async def test_models_and_migration_do_not_drift(db_engine: AsyncEngine) -> None:
    async with db_engine.connect() as connection:
        assert await connection.run_sync(_drift) == []


async def test_downgrade_then_upgrade_leaves_no_drift(db_engine: AsyncEngine) -> None:
    async with db_engine.connect() as connection:
        transaction = await connection.begin()
        await connection.run_sync(_run, "downgrade")
        await connection.run_sync(_run, "upgrade")
        assert await connection.run_sync(_drift) == []
        await transaction.rollback()
