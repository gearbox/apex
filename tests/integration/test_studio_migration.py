"""Migration 055: upgrade / downgrade / upgrade, and the constraints the design depends on."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from alembic.operations import Operations
from alembic.runtime.migration import MigrationContext
from sqlalchemy import text

if TYPE_CHECKING:
    from types import ModuleType

    from sqlalchemy import Connection
    from sqlalchemy.ext.asyncio import AsyncConnection, AsyncEngine

_MIGRATION_PATH = (
    Path(__file__).resolve().parents[2] / "alembic" / "versions" / "055_studio_foundation.py"
)
_TABLES = (
    "studio_artists",
    "studio_project_profiles",
    "studio_tracks",
    "studio_document_versions",
)


def _load_migration() -> ModuleType:
    spec = importlib.util.spec_from_file_location("revision_055", _MIGRATION_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run(connection: Connection, action: str) -> None:
    migration = _load_migration()
    with Operations.context(MigrationContext.configure(connection)):
        getattr(migration, action)()


async def _tables(connection: AsyncConnection) -> set[str]:
    rows = await connection.execute(
        text("SELECT tablename FROM pg_tables WHERE tablename LIKE 'studio\\_%'")
    )
    return {row[0] for row in rows}


async def _constraints(connection: AsyncConnection) -> dict[str, tuple[str, str, str]]:
    """name -> (type, definition, delete-action) for every studio constraint."""
    rows = await connection.execute(
        text(
            "SELECT conname, contype::text, pg_get_constraintdef(oid), confdeltype::text "
            "FROM pg_constraint WHERE conrelid::regclass::text LIKE 'studio\\_%'"
        )
    )
    return {row[0]: (row[1], row[2], row[3]) for row in rows}


def test_revision_chain() -> None:
    migration = _load_migration()
    assert migration.revision == "055"
    assert migration.down_revision == "054"


async def test_upgrade_downgrade_upgrade_round_trip(db_engine: AsyncEngine) -> None:
    async with db_engine.connect() as connection:
        transaction = await connection.begin()
        assert await _tables(connection) == set(_TABLES)

        await connection.run_sync(_run, "downgrade")
        assert await _tables(connection) == set()

        await connection.run_sync(_run, "upgrade")
        assert await _tables(connection) == set(_TABLES)
        await transaction.rollback()


async def test_constraints_and_indexes_match_the_design(db_engine: AsyncEngine) -> None:
    async with db_engine.connect() as connection:
        constraints = await _constraints(connection)

        # deferrable unique for position swaps
        _, definition, _ = constraints["uq_studio_tracks_project_position"]
        assert "DEFERRABLE INITIALLY DEFERRED" in definition

        # idempotency key: NULL parents must collide
        nulls_not_distinct = (
            await connection.execute(
                text(
                    "SELECT i.indnullsnotdistinct FROM pg_index i "
                    "JOIN pg_class c ON c.oid = i.indexrelid "
                    "WHERE c.relname = 'uq_studio_doc_versions_sibling_content'"
                )
            )
        ).scalar_one()
        assert nulls_not_distinct is True

        # same-track integrity is composite
        for name, columns in {
            "fk_studio_doc_versions_parent": ("parent_version_id", "track_id", "kind"),
            "fk_studio_doc_versions_based_on_plan": ("based_on_plan_version_id", "track_id"),
            "fk_studio_tracks_current_plan_version": ("current_plan_version_id", "id"),
            "fk_studio_tracks_current_lyrics_version": ("current_lyrics_version_id", "id"),
        }.items():
            kind, definition, action = constraints[name]
            assert kind == "f"
            assert definition.startswith(f"FOREIGN KEY ({', '.join(columns)})"), definition
            assert action == "a"  # NO ACTION

        # ... and its targets exist
        assert "UNIQUE (id, track_id)" in constraints["uq_studio_doc_versions_id_track"][1]
        assert (
            "UNIQUE (id, track_id, kind)" in constraints["uq_studio_doc_versions_id_track_kind"][1]
        )
        assert (
            "UNIQUE (track_id, kind, version_number)"
            in constraints["uq_studio_doc_versions_number"][1]
        )

        # NO ACTION (not RESTRICT / CASCADE) on the backstops
        by_definition = {d: a for _, d, a in constraints.values()}
        assert by_definition["FOREIGN KEY (artist_id) REFERENCES studio_artists(id)"] == "a"
        assert (
            by_definition[
                "FOREIGN KEY (library_project_id) REFERENCES studio_project_profiles(library_project_id)"
            ]
            == "a"
        )
        # cascades
        assert (
            by_definition[
                "FOREIGN KEY (library_project_id) REFERENCES library_projects(id) ON DELETE CASCADE"
            ]
            == "c"
        )
        assert (
            by_definition["FOREIGN KEY (track_id) REFERENCES studio_tracks(id) ON DELETE CASCADE"]
            == "c"
        )

        for check in (
            "ck_studio_project_profiles_release_kind",
            "ck_studio_tracks_position",
            "ck_studio_tracks_lyrics_need_plan",
            "ck_studio_doc_versions_kind",
            "ck_studio_doc_versions_origin",
            "ck_studio_doc_versions_number",
            "ck_studio_doc_versions_schema_version",
            "ck_studio_doc_versions_based_on_iff_lyrics",
        ):
            assert constraints[check][0] == "c"


async def test_check_values_are_generated_from_the_enums(db_engine: AsyncEngine) -> None:
    from src.core.enums import StudioDocumentKind, StudioDocumentOrigin, StudioReleaseKind

    async with db_engine.connect() as connection:
        constraints = await _constraints(connection)
    for name, enum in (
        ("ck_studio_project_profiles_release_kind", StudioReleaseKind),
        ("ck_studio_doc_versions_kind", StudioDocumentKind),
        ("ck_studio_doc_versions_origin", StudioDocumentOrigin),
    ):
        definition = constraints[name][1]
        for member in enum:
            assert f"'{member.value}'" in definition, (name, member)


@pytest.mark.parametrize(
    "index",
    [
        "uq_studio_artists_owner_name",
        "ix_studio_artists_product_id",
        "ix_studio_project_profiles_artist_id",
        "ix_studio_project_profiles_product_id",
        "ix_studio_tracks_product_id",
        "ix_studio_document_versions_product_id",
        "ix_studio_doc_versions_based_on",
        "ix_studio_doc_versions_list",
    ],
)
async def test_indexes_exist(db_engine: AsyncEngine, index: str) -> None:
    async with db_engine.connect() as connection:
        found = (
            await connection.execute(
                text("SELECT indexdef FROM pg_indexes WHERE indexname = :n"), {"n": index}
            )
        ).scalar_one_or_none()
    assert found is not None
    if index == "uq_studio_artists_owner_name":
        assert "lower((name)::text)" in found
        assert "UNIQUE" in found
    if index == "ix_studio_doc_versions_list":
        assert "created_at DESC, id DESC" in found
