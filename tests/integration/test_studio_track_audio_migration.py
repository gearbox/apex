"""Migration 062: library audio profiles and studio track audio (S2a, W5 / W11.8).

The upgrade creates ``library_audio_profiles`` (CHECKs, cascading asset keys, the two partial
uniques), ``studio_track_audio`` (composite owner / version keys, ``hold_id NOT NULL UNIQUE``) and
two track columns with the composite current-audio pointer. The downgrade drops all of it - and
the holds the audio versions took - and nothing else; ``upgrade → downgrade → upgrade`` works, and
the models and the migration do not drift (the ``alembic check`` of the PR checklist, through the
same comparison). The migration never imports application code.
"""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import uuid4

import pytest
from alembic.autogenerate import compare_metadata
from alembic.operations import Operations
from alembic.runtime.migration import MigrationContext
from sqlalchemy import text

from src.db.models.base import Base

if TYPE_CHECKING:
    from types import ModuleType

    from sqlalchemy import Connection
    from sqlalchemy.ext.asyncio import AsyncConnection, AsyncEngine

_PATH = Path(__file__).resolve().parents[2] / "alembic" / "versions" / "062_studio_track_audio.py"
_NEXT = _PATH.with_name("063_studio_audio_timelines.py")
_NEW_TABLES = ("library_audio_profiles", "studio_track_audio")


def _load(path: Path = _PATH, name: str = "revision_062") -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run(connection: Connection, action: str) -> None:
    with Operations.context(MigrationContext.configure(connection)):
        getattr(_load(), action)()


def _run_next(connection: Connection, action: str) -> None:
    """Revision 063 (timelines): it references the audio versions and widens the profiles, so it
    goes first on the way down and last on the way up."""
    with Operations.context(MigrationContext.configure(connection)):
        getattr(_load(_NEXT, "revision_063"), action)()


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
    assert migration.revision == "062"
    assert migration.down_revision == "061"


def test_the_migration_never_imports_application_code() -> None:
    tree = ast.parse(_PATH.read_text())
    imported = {
        node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module
    } | {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    assert not {name for name in imported if name == "src" or name.startswith("src.")}


async def test_the_profile_table_has_its_checks_cascading_keys_and_partial_uniques(
    db_engine: AsyncEngine,
) -> None:
    async with db_engine.connect() as connection:
        checks = await _constraints(connection, "library_audio_profiles", "c")
        foreign_keys = await _constraints(connection, "library_audio_profiles", "f")
        upload_unique = await _index(connection, "uq_library_audio_profiles_upload")
        output_unique = await _index(connection, "uq_library_audio_profiles_output")

    assert set(checks) == {
        "ck_library_audio_profiles_one_asset",
        "ck_library_audio_profiles_peaks_size",
        "ck_library_audio_profiles_channels",
        # added by 063 (the detail peaks level; its own migration test checks them)
        "ck_library_audio_profiles_detail_peaks_pair",
        "ck_library_audio_profiles_detail_peaks_size",
    }
    assert (
        "num_nonnulls(upload_id, output_id) = 1" in checks["ck_library_audio_profiles_one_asset"][3]
    )
    assert "octet_length(peaks) = 2048" in checks["ck_library_audio_profiles_peaks_size"][3]
    assert "channels >= 1" in checks["ck_library_audio_profiles_channels"][3]
    assert "channels <= 8" in checks["ck_library_audio_profiles_channels"][3]
    # the profile has no life of its own: both asset keys cascade (a delete, the sweep, a user)
    by_target = {fk[3].split("REFERENCES ")[1].split("(")[0]: fk[0] for fk in foreign_keys.values()}
    assert by_target == {"user_images": "c", "generation_outputs": "c"}
    assert upload_unique.startswith("CREATE UNIQUE INDEX")
    assert "(upload_id)" in upload_unique
    assert "WHERE (upload_id IS NOT NULL)" in upload_unique
    assert output_unique.startswith("CREATE UNIQUE INDEX")
    assert "WHERE (output_id IS NOT NULL)" in output_unique


async def test_the_audio_table_has_composite_keys_and_a_unique_hold(
    db_engine: AsyncEngine,
) -> None:
    async with db_engine.connect() as connection:
        foreign_keys = await _constraints(connection, "studio_track_audio", "f")
        uniques = await _constraints(connection, "studio_track_audio", "u")
        checks = await _constraints(connection, "studio_track_audio", "c")
        plan_index = await _index(connection, "ix_studio_track_audio_plan_version")
        lyrics_index = await _index(connection, "ix_studio_track_audio_lyrics_version")
        track_uniques = await _constraints(connection, "studio_tracks", "u")
        columns = (
            (
                await connection.execute(
                    text(
                        "SELECT column_name FROM information_schema.columns "
                        "WHERE table_name = 'studio_track_audio'"
                    )
                )
            )
            .scalars()
            .all()
        )

    owner = foreign_keys["fk_studio_track_audio_track_owner"]
    assert owner[0] == "c"
    assert "FOREIGN KEY (track_id, user_id, product_id)" in owner[3]
    assert "REFERENCES studio_tracks(id, user_id, product_id)" in owner[3]
    assert "UNIQUE (id, user_id, product_id)" in track_uniques["uq_studio_tracks_id_owner"][3]
    for name, column in (
        ("fk_studio_track_audio_plan_version", "plan_version_id"),
        ("fk_studio_track_audio_lyrics_version", "lyrics_version_id"),
    ):
        assert foreign_keys[name][0] == "a"  # NO ACTION: deleting a realized version is refused
        assert f"FOREIGN KEY ({column}, track_id)" in foreign_keys[name][3]
        assert "REFERENCES studio_document_versions(id, track_id)" in foreign_keys[name][3]
    assert foreign_keys["fk_studio_track_audio_hold"][0] == "a"  # NO ACTION
    assert "REFERENCES library_asset_holds(id)" in foreign_keys["fk_studio_track_audio_hold"][3]
    user_fks = [name for name, fk in foreign_keys.items() if "REFERENCES users(id)" in fk[3]]
    assert [foreign_keys[name][0] for name in user_fks] == ["c"]
    assert "UNIQUE (hold_id)" in uniques["studio_track_audio_hold_id_key"][3]
    assert "UNIQUE (track_id, version_number)" in uniques["uq_studio_track_audio_number"][3]
    assert "UNIQUE (id, track_id)" in uniques["uq_studio_track_audio_id_track"][3]
    assert set(checks) == {
        "ck_studio_track_audio_number",
        "ck_studio_track_audio_duration",
        "ck_studio_track_audio_lyrics_need_plan",
        "ck_studio_track_audio_origin",
    }
    lyrics_plan_check = checks["ck_studio_track_audio_lyrics_need_plan"][3]
    assert "lyrics_version_id IS NULL" in lyrics_plan_check
    assert "plan_version_id IS NOT NULL" in lyrics_plan_check
    assert " OR " in lyrics_plan_check
    assert "origin" in checks["ck_studio_track_audio_origin"][3]
    assert "'upload'" in checks["ck_studio_track_audio_origin"][3]
    assert "WHERE (plan_version_id IS NOT NULL)" in plan_index
    assert "WHERE (lyrics_version_id IS NOT NULL)" in lyrics_index
    # the asset lives on the hold: there is no copy of it on the version
    assert not {"asset_ref", "asset_id", "upload_id", "output_id"} & set(columns)


async def test_the_track_has_a_counter_and_a_composite_no_action_pointer(
    db_engine: AsyncEngine,
) -> None:
    async with db_engine.connect() as connection:
        foreign_keys = await _constraints(connection, "studio_tracks", "f")
        columns = {
            row[0]: (row[1], row[2], row[3])
            for row in await connection.execute(
                text(
                    "SELECT column_name, is_nullable, column_default, data_type "
                    "FROM information_schema.columns WHERE table_name = 'studio_tracks'"
                )
            )
        }

    pointer = foreign_keys["fk_studio_tracks_current_audio_version"]
    assert pointer[0] == "a"  # NO ACTION, like the plan / lyrics pointers
    assert "FOREIGN KEY (current_audio_version_id, id)" in pointer[3]
    assert "REFERENCES studio_track_audio(id, track_id)" in pointer[3]
    assert columns["current_audio_version_id"][0] == "YES"
    assert columns["next_audio_version_number"][0] == "NO"
    assert columns["next_audio_version_number"][1] == "1"


async def test_downgrade_drops_what_the_revision_made_and_nothing_else_and_upgrade_restores_it(
    db_engine: AsyncEngine,
) -> None:
    async with db_engine.connect() as connection:
        transaction = await connection.begin()

        await connection.run_sync(_run_next, "downgrade")
        await connection.run_sync(_run, "downgrade")
        for table in _NEW_TABLES:
            assert not await _exists(connection, table), table
        track_columns = set(
            (
                await connection.execute(
                    text(
                        "SELECT column_name FROM information_schema.columns "
                        "WHERE table_name = 'studio_tracks'"
                    )
                )
            )
            .scalars()
            .all()
        )
        assert not {"current_audio_version_id", "next_audio_version_number"} & track_columns
        assert "uq_studio_tracks_id_owner" not in await _constraints(
            connection, "studio_tracks", "u"
        )
        # what the earlier revisions made is untouched
        for table in (
            "library_asset_holds",
            "studio_cast_members",
            "studio_tracks",
            "user_images",
            "generation_outputs",
        ):
            assert await _exists(connection, table), table

        await connection.run_sync(_run, "upgrade")
        for table in _NEW_TABLES:
            assert await _exists(connection, table), table
        assert "fk_studio_tracks_current_audio_version" in await _constraints(
            connection, "studio_tracks", "f"
        )
        await connection.run_sync(_run_next, "upgrade")
        await transaction.rollback()


async def test_downgrade_refuses_while_audio_uploads_remain(
    db_engine: AsyncEngine,
) -> None:
    async with db_engine.connect() as connection:
        transaction = await connection.begin()
        user_id, audio_id = uuid4(), uuid4()
        await connection.execute(
            text(
                "INSERT INTO users (id, email, password_hash, product_id) "
                "VALUES (:id, :email, 'x', 'vex')"
            ),
            {"id": user_id, "email": f"mig-{uuid4().hex}@example.com"},
        )
        await connection.execute(
            text(
                "INSERT INTO user_images (id, user_id, product_id, storage_key, "
                "original_filename, content_type, size_bytes, format, expires_at) "
                "VALUES (:id, :user_id, 'vex', :key, :name, 'audio/flac', 10, 'flac', now())"
            ),
            {
                "id": audio_id,
                "user_id": user_id,
                "key": f"k/{audio_id}",
                "name": f"{audio_id}.flac",
            },
        )

        await connection.run_sync(_run_next, "downgrade")
        with pytest.raises(
            RuntimeError, match=r"1 audio upload.*delete those rows first"
        ) as raised:
            await connection.run_sync(_run, "downgrade")

        assert "Deleting them removes users' files" in str(raised.value)
        assert await _exists(connection, "studio_track_audio")
        await transaction.rollback()


async def test_downgrade_succeeds_when_no_audio_assets_remain(db_engine: AsyncEngine) -> None:
    async with db_engine.connect() as connection:
        transaction = await connection.begin()
        await connection.run_sync(_run_next, "downgrade")
        await connection.run_sync(_run, "downgrade")
        for table in _NEW_TABLES:
            assert not await _exists(connection, table), table
        await transaction.rollback()


def _drift(connection: Connection) -> list[str]:
    """What ``alembic check`` would report for the tables this migration touches."""
    diffs = compare_metadata(MigrationContext.configure(connection), Base.metadata)
    flat = [str(item) for diff in diffs for item in (diff if isinstance(diff, list) else [diff])]
    return [
        item
        for item in flat
        if "audio" in item or "studio_tracks" in item or "uq_studio_tracks_id_owner" in item
    ]


async def test_models_and_migration_do_not_drift(db_engine: AsyncEngine) -> None:
    async with db_engine.connect() as connection:
        assert await connection.run_sync(_drift) == []


async def test_downgrade_then_upgrade_leaves_no_drift(db_engine: AsyncEngine) -> None:
    async with db_engine.connect() as connection:
        transaction = await connection.begin()
        await connection.run_sync(_run_next, "downgrade")
        await connection.run_sync(_run, "downgrade")
        await connection.run_sync(_run, "upgrade")
        await connection.run_sync(_run_next, "upgrade")
        assert await connection.run_sync(_drift) == []
        await transaction.rollback()
