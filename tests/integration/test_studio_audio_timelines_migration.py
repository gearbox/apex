"""Migration 063: realized timelines and detail waveform peaks (S3a, W2 / W9.10).

``upgrade → downgrade -1 → upgrade`` on a **fresh database** (created for this module and dropped
afterwards), the constraints the design depends on (looked up in ``pg_constraint`` by name), and
no drift between the models and the migration. The migration never imports application code.
"""

from __future__ import annotations

import ast
import os
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import uuid4

import pytest
from sqlalchemy import make_url, text
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.pool import NullPool

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from sqlalchemy.ext.asyncio import AsyncConnection, AsyncEngine

_ROOT = Path(__file__).resolve().parents[2]
_PATH = _ROOT / "alembic" / "versions" / "063_studio_audio_timelines.py"

_TIMELINE_CONSTRAINTS = {
    # name -> (contype, confdeltype or None)
    "studio_audio_timelines_pkey": ("p", None),
    "studio_audio_timelines_user_id_fkey": ("f", "c"),
    "fk_studio_audio_timelines_track_owner": ("f", "c"),
    "fk_studio_audio_timelines_audio": ("f", "c"),
    "fk_studio_audio_timelines_parent": ("f", "a"),
    "uq_studio_audio_timelines_id_audio": ("u", None),
    "uq_studio_audio_timelines_number": ("u", None),
    "uq_studio_audio_timelines_content": ("u", None),
    "ck_studio_audio_timelines_number": ("c", None),
    "ck_studio_audio_timelines_origin": ("c", None),
}


def _alembic(database_url: str, *arguments: str) -> None:
    """Run the real Alembic command line against ``database_url``."""
    result = subprocess.run(  # noqa: S603 - this interpreter, fixed arguments
        [sys.executable, "-m", "alembic", *arguments],
        capture_output=True,
        text=True,
        check=False,
        cwd=_ROOT,
        env={**os.environ, "DATABASE_URL": database_url, "DEBUG": "false"},
    )
    assert result.returncode == 0, (
        f"alembic {' '.join(arguments)} failed:\n{result.stdout}\n{result.stderr}"
    )


@pytest.fixture
async def fresh_database_url(db_engine: AsyncEngine, test_database_url: str) -> AsyncGenerator[str]:
    """An empty database on the test server, dropped again afterwards."""
    name = f"apex_mig063_{uuid4().hex[:16]}"
    admin = db_engine.execution_options(isolation_level="AUTOCOMMIT")
    async with admin.connect() as connection:
        await connection.execute(text(f'CREATE DATABASE "{name}"'))
    try:
        yield make_url(test_database_url).set(database=name).render_as_string(hide_password=False)
    finally:
        async with admin.connect() as connection:
            await connection.execute(text(f'DROP DATABASE "{name}" WITH (FORCE)'))


async def _constraints(connection: AsyncConnection, table: str) -> dict[str, tuple[str, str, str]]:
    """``name -> (contype, confdeltype, definition)`` of one table, from ``pg_constraint``."""
    rows = await connection.execute(
        text(
            "SELECT conname, contype::text, confdeltype::text, pg_get_constraintdef(oid) "
            "FROM pg_constraint WHERE conrelid = to_regclass(:table) AND contype <> 'n'"
        ),
        {"table": table},
    )
    return {row[0]: (row[1], row[2], row[3]) for row in rows}


async def _columns(
    connection: AsyncConnection, table: str
) -> dict[str, tuple[str, str, str | None]]:
    """``column -> (data_type, is_nullable, default)``."""
    rows = await connection.execute(
        text(
            "SELECT column_name, data_type, is_nullable, column_default "
            "FROM information_schema.columns WHERE table_name = :table"
        ),
        {"table": table},
    )
    return {row[0]: (row[1], row[2], row[3]) for row in rows}


async def _index(connection: AsyncConnection, name: str) -> str | None:
    return (
        await connection.execute(
            text("SELECT indexdef FROM pg_indexes WHERE indexname = :name"), {"name": name}
        )
    ).scalar_one_or_none()


async def _assert_upgraded(url: str) -> None:
    engine = create_async_engine(url, poolclass=NullPool)
    try:
        async with engine.connect() as connection:
            timelines = await _constraints(connection, "studio_audio_timelines")
            audio = await _constraints(connection, "studio_track_audio")
            profiles = await _constraints(connection, "library_audio_profiles")
            timeline_columns = await _columns(connection, "studio_audio_timelines")
            audio_columns = await _columns(connection, "studio_track_audio")
            profile_columns = await _columns(connection, "library_audio_profiles")
            list_index = await _index(connection, "ix_studio_audio_timelines_list")
            other_indexes = [
                await _index(connection, f"ix_studio_audio_timelines_{column}")
                for column in ("product_id", "user_id", "track_id")
            ]
            storage = (
                await connection.execute(
                    text(
                        "SELECT attstorage::text FROM pg_attribute "
                        "WHERE attrelid = to_regclass('library_audio_profiles') "
                        "AND attname = 'detail_peaks'"
                    )
                )
            ).scalar_one()
    finally:
        await engine.dispose()

    assert set(timelines) == set(_TIMELINE_CONSTRAINTS)
    for name, (kind, on_delete) in _TIMELINE_CONSTRAINTS.items():
        assert timelines[name][0] == kind, name
        if on_delete is not None:
            assert timelines[name][1] == on_delete, name
    definitions = {name: value[2] for name, value in timelines.items()}
    assert (
        "FOREIGN KEY (track_id, user_id, product_id) REFERENCES studio_tracks(id, user_id, "
        "product_id)" in definitions["fk_studio_audio_timelines_track_owner"]
    )
    assert (
        "FOREIGN KEY (audio_version_id, track_id) REFERENCES studio_track_audio(id, track_id)"
        in definitions["fk_studio_audio_timelines_audio"]
    )
    assert (
        "FOREIGN KEY (parent_id, audio_version_id) REFERENCES studio_audio_timelines(id, "
        "audio_version_id)" in definitions["fk_studio_audio_timelines_parent"]
    )
    assert definitions["uq_studio_audio_timelines_id_audio"] == "UNIQUE (id, audio_version_id)"
    assert (
        definitions["uq_studio_audio_timelines_number"]
        == "UNIQUE (audio_version_id, version_number)"
    )
    assert definitions["uq_studio_audio_timelines_content"] == (
        "UNIQUE NULLS NOT DISTINCT (audio_version_id, parent_id, content_hash)"
    )
    assert "version_number >= 1" in definitions["ck_studio_audio_timelines_number"]
    assert "'user'" in definitions["ck_studio_audio_timelines_origin"]
    assert "'aligner'" in definitions["ck_studio_audio_timelines_origin"]
    assert "'import'" not in definitions["ck_studio_audio_timelines_origin"]

    assert list_index is not None
    assert "(audio_version_id, version_number DESC)" in list_index
    assert all(other_indexes)
    assert {name: column[:2] for name, column in timeline_columns.items()} == {
        "id": ("uuid", "NO"),
        "product_id": ("character varying", "NO"),
        "user_id": ("uuid", "NO"),
        "track_id": ("uuid", "NO"),
        "audio_version_id": ("uuid", "NO"),
        "version_number": ("integer", "NO"),
        "parent_id": ("uuid", "YES"),
        "origin": ("character varying", "NO"),
        "schema_version": ("smallint", "NO"),
        "content": ("jsonb", "NO"),
        "content_hash": ("character varying", "NO"),
        "is_complete": ("boolean", "NO"),
        "created_at": ("timestamp with time zone", "NO"),
    }

    assert audio_columns["next_timeline_number"] == ("integer", "NO", "1")
    assert audio_columns["current_timeline_id"][:2] == ("uuid", "YES")
    pointer = audio["fk_studio_track_audio_current_timeline"]
    assert pointer[:2] == ("f", "a")  # NO ACTION
    assert (
        "FOREIGN KEY (current_timeline_id, id) REFERENCES studio_audio_timelines(id, "
        "audio_version_id)" in pointer[2]
    )

    assert profile_columns["detail_peaks"][:2] == ("bytea", "YES")
    assert profile_columns["detail_peaks_version"][:2] == ("smallint", "YES")
    pair = profiles["ck_library_audio_profiles_detail_peaks_pair"][2]
    size = profiles["ck_library_audio_profiles_detail_peaks_size"][2]
    assert "(detail_peaks IS NULL) = (detail_peaks_version IS NULL)" in pair
    assert "% 2) = 0" in size
    assert "<= 90100" in size
    assert storage == "e", "detail_peaks must be STORAGE EXTERNAL so a window read stays partial"


async def _assert_downgraded(url: str) -> None:
    engine = create_async_engine(url, poolclass=NullPool)
    try:
        async with engine.connect() as connection:
            exists = (
                await connection.execute(
                    text("SELECT to_regclass('studio_audio_timelines') IS NOT NULL")
                )
            ).scalar_one()
            audio_columns = await _columns(connection, "studio_track_audio")
            profile_columns = await _columns(connection, "library_audio_profiles")
            audio = await _constraints(connection, "studio_track_audio")
            profiles = await _constraints(connection, "library_audio_profiles")
            revision = (
                await connection.execute(text("SELECT version_num FROM alembic_version"))
            ).scalar_one()
    finally:
        await engine.dispose()

    assert revision == "062"
    assert exists is False
    assert not {"next_timeline_number", "current_timeline_id"} & set(audio_columns)
    assert not {"detail_peaks", "detail_peaks_version"} & set(profile_columns)
    assert "fk_studio_track_audio_current_timeline" not in audio
    assert not [name for name in profiles if "detail_peaks" in name]
    # what 062 made is untouched
    assert "peaks" in profile_columns
    assert "hold_id" in audio_columns


async def test_upgrade_downgrade_upgrade_on_a_fresh_database(fresh_database_url: str) -> None:
    _alembic(fresh_database_url, "upgrade", "063")
    await _assert_upgraded(fresh_database_url)

    _alembic(fresh_database_url, "downgrade", "-1")
    await _assert_downgraded(fresh_database_url)

    _alembic(fresh_database_url, "upgrade", "head")
    await _assert_upgraded(fresh_database_url)
    # the models and the migrations agree after the round trip (``alembic check``)
    _alembic(fresh_database_url, "check")


def test_revision_chain() -> None:
    source = _PATH.read_text()
    assert 'revision: str = "063"' in source
    assert 'down_revision: str | Sequence[str] | None = "062"' in source


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
