"""Migration 056 (invariant 13): every v1 plan version becomes v2, rehashed, with the app's hash.

Plan rows are created through the services (so numbers, pointers and foreign keys are real), then
rewritten to exactly what P1 stored — no ``instrumental`` key, ``schema_version`` 1, the v1
``content_hash`` — and the real migration is run over them on the same connection.
"""

from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
from typing import TYPE_CHECKING, Any

import msgspec
import pytest
from alembic.operations import Operations
from alembic.runtime.migration import MigrationContext
from sqlalchemy import text

from src.core.enums import StudioDocumentKind as D
from src.core.enums import StudioSectionKind as K
from src.studio.documents import PlanDocument, decode_stored
from src.studio.errors import StudioDocumentSchemaError, VersionUnchanged
from src.studio.validation import content_hash, to_content
from tests.studio_support import golden_song, make_lyrics, make_plan, make_section

if TYPE_CHECKING:
    from types import ModuleType
    from uuid import UUID

    from sqlalchemy import Connection

    from tests.studio_world import StudioWorld

_MIGRATION_PATH = (
    Path(__file__).resolve().parents[2] / "alembic" / "versions" / "056_studio_plan_schema_v2.py"
)


def _load_migration() -> ModuleType:
    spec = importlib.util.spec_from_file_location("revision_056", _MIGRATION_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run(connection: Connection, action: str, batch: int | None = None) -> None:
    migration = _load_migration()
    if batch is not None:
        # The migration reads ``_BATCH`` from its module globals; ``ModuleType`` only types reads.
        vars(migration)["_BATCH"] = batch
    with Operations.context(MigrationContext.configure(connection)):
        getattr(migration, action)()


async def _migrate(w: StudioWorld, action: str, *, batch: int | None = None) -> None:
    await (await w.session.connection()).run_sync(_run, action, batch)


# The field order of the v1 documents, as P1 declared them (``git show 23ef4ee:src/studio/
# documents.py``). JSONB does not keep key order, so the P1 canonical encoding is rebuilt from
# this independent statement — not from the migration's own copy of the Structs.
_V1_PLAN = ("title", "idea", "tempo_bpm", "global_style", "references", "sections")
_V1_SECTION = (
    "id",
    "kind",
    "label",
    "duration_seconds",
    "style",
    "delivery",
    "instrumental",
    "notes",
)
_V1_STYLE = ("tags", "avoid")


def _ordered(content: dict[str, Any], order: tuple[str, ...]) -> dict[str, Any]:
    return {key: content[key] for key in order if key in content}


def _v1_canonical(content: dict[str, Any]) -> bytes:
    """The bytes P1 hashed: the compact JSON of the document in declared field order."""
    plan = _ordered(content, _V1_PLAN)
    plan["global_style"] = _ordered(plan["global_style"], _V1_STYLE)
    plan["sections"] = [
        {**_ordered(section, _V1_SECTION), "style": _ordered(section["style"], _V1_STYLE)}
        for section in plan["sections"]
    ]
    return msgspec.json.encode(plan)


def _v1_hash(content: dict[str, Any]) -> str:
    """What P1 stored as ``content_hash``: ``sha256("plan\\n\\n" + canonical json)``."""
    return hashlib.sha256(b"plan\n\n" + _v1_canonical(content)).hexdigest()


async def _make_v1(w: StudioWorld, version_id: UUID) -> dict[str, Any]:
    """Rewrite a (service-created, v2) plan row into the form P1 stored; return its v1 content."""
    row = await w.version_row(version_id)
    content = {k: v for k, v in row.content.items() if k != "instrumental"}
    assert "instrumental" not in content
    await w.session.execute(
        text(
            "UPDATE studio_document_versions SET content = CAST(:c AS jsonb), "
            "schema_version = 1, content_hash = :h WHERE id = :id"
        ),
        {"c": msgspec.json.encode(content).decode(), "h": _v1_hash(content), "id": version_id},
    )
    return content


async def _stored(w: StudioWorld, version_id: UUID) -> tuple[int, dict[str, Any], str]:
    row = (
        await w.session.execute(
            text(
                "SELECT schema_version, content, content_hash FROM studio_document_versions "
                "WHERE id = :id"
            ),
            {"id": version_id},
        )
    ).one()
    return row[0], row[1], row[2]


async def _plans(w: StudioWorld, track: UUID, *documents: PlanDocument) -> list[UUID]:
    """Create ``documents`` as a chain of plan versions; return their ids."""
    ids: list[UUID] = []
    parent: UUID | None = None
    for document in documents:
        out, _ = await w.plan(track, document, parent=parent)
        parent = out.version.id
        ids.append(parent)
    return ids


def test_revision_chain() -> None:
    migration = _load_migration()
    assert migration.revision == "056"
    assert migration.down_revision == "055"


def test_the_migration_imports_no_application_code() -> None:
    import re

    source = _MIGRATION_PATH.read_text()
    assert not re.search(r"^\s*(from|import)\s+src\b", source, flags=re.MULTILINE)


class TestUpgrade:
    async def test_every_v1_plan_becomes_v2_with_the_apps_hash(
        self, enabled_studio: StudioWorld
    ) -> None:
        w = enabled_studio
        song = golden_song()
        verse = make_section(
            K.VERSE,
            delivery=("breathy",),
            tags=("Café",),
            avoid=("n",),
            duration_seconds=7,
            notes="multi\nline",
        )
        track = await w.new_track()
        plans = [
            make_plan(verse, title="Привет, мир 🎵", tags=("a",), idea="idea", references=("r",)),
            song.plan,
            make_plan(make_section(K.DROP, instrumental=True), title="Wordless"),
            make_plan(title="Bare", tempo_bpm=90),
        ]
        ids = await _plans(w, track, *plans)
        v1 = {version_id: await _make_v1(w, version_id) for version_id in ids}

        await _migrate(w, "upgrade")

        for version_id in ids:
            schema_version, content, digest = await _stored(w, version_id)
            assert schema_version == 2
            assert content["instrumental"] is False
            decoded = decode_stored(D.PLAN, 2, content)
            assert isinstance(decoded, PlanDocument)
            assert decoded == msgspec.convert(
                {**v1[version_id], "instrumental": False}, PlanDocument
            )
            assert digest == content_hash(D.PLAN, None, decoded)
            # nothing but the new key was added: the rest of the document is as P1 wrote it
            assert {k: v for k, v in content.items() if k != "instrumental"} == v1[version_id]
            assert content == to_content(decoded)

    async def test_replay_and_version_unchanged_work_on_upgraded_rows(
        self, enabled_studio: StudioWorld
    ) -> None:
        w = enabled_studio
        track = await w.new_track()
        first = make_plan(make_section(K.VERSE), title="Song", tags=("x",))
        second = make_plan(make_section(K.VERSE), title="Song v2", tags=("x",))
        (first_id, second_id) = await _plans(w, track, first, second)
        for version_id in (first_id, second_id):
            await _make_v1(w, version_id)
        before = await w.version_count(track)

        await _migrate(w, "upgrade")

        # the same content as the migrated child, under the same parent: a replay, not a new row
        replay, created = await w.plan(track, second, parent=first_id)
        assert (replay.version.id, created) == (second_id, False)
        # the same content as its parent: nothing new to version
        with pytest.raises(VersionUnchanged):
            await w.plan(track, first, parent=first_id)
        assert await w.version_count(track) == before

    async def test_new_versions_after_the_upgrade_chain_from_migrated_ones(
        self, enabled_studio: StudioWorld
    ) -> None:
        w = enabled_studio
        track = await w.new_track()
        verse = make_section(K.VERSE)
        (first_id,) = await _plans(w, track, make_plan(verse))
        await _make_v1(w, first_id)
        await _migrate(w, "upgrade")

        out, created = await w.plan(track, make_plan(verse, instrumental=True), parent=first_id)
        assert created
        assert (await _stored(w, out.version.id))[0] == 2

    async def test_lyrics_rows_and_v2_rows_are_not_touched(
        self, enabled_studio: StudioWorld
    ) -> None:
        w = enabled_studio
        track = await w.new_track()
        verse = make_section(K.VERSE)
        plan_out, _ = await w.plan(track, make_plan(verse))
        lyrics_out, _ = await w.lyrics(track, make_lyrics((verse.id, ["la"])), plan_out.version.id)
        v2_out, _ = await w.plan(
            track, make_plan(verse, title="Second"), parent=plan_out.version.id
        )
        before = {
            version_id: await _stored(w, version_id)
            for version_id in (lyrics_out.version.id, v2_out.version.id)
        }
        await _make_v1(w, plan_out.version.id)

        await _migrate(w, "upgrade")

        assert {version_id: await _stored(w, version_id) for version_id in before} == before
        assert (await _stored(w, lyrics_out.version.id))[0] == 1

    async def test_rows_holding_names_that_are_now_reserved_still_upgrade_and_decode(
        self, enabled_studio: StudioWorld
    ) -> None:
        w = enabled_studio
        track = await w.new_track()
        (version_id,) = await _plans(w, track, make_plan(make_section(K.DROP), title="Old"))
        row = await w.version_row(version_id)
        legacy = {k: v for k, v in row.content.items() if k != "instrumental"}
        legacy["sections"][0]["style"]["tags"] = ["instrumental"]
        legacy["sections"][0]["delivery"] = ["Instrumental"]
        await w.session.execute(
            text(
                "UPDATE studio_document_versions SET content = CAST(:c AS jsonb), "
                "schema_version = 1, content_hash = :h WHERE id = :id"
            ),
            {"c": msgspec.json.encode(legacy).decode(), "h": _v1_hash(legacy), "id": version_id},
        )

        await _migrate(w, "upgrade")

        _, content, _ = await _stored(w, version_id)
        decoded = decode_stored(D.PLAN, 2, content)
        assert isinstance(decoded, PlanDocument)
        assert decoded.sections[0].style.tags == ("instrumental",)

    async def test_rows_are_processed_in_batches_without_skipping_or_repeating_any(
        self, enabled_studio: StudioWorld
    ) -> None:
        w = enabled_studio
        track = await w.new_track()
        ids = await _plans(w, track, *(make_plan(title=f"Plan {n}") for n in range(5)))
        for version_id in ids:
            await _make_v1(w, version_id)

        await _migrate(w, "upgrade", batch=2)  # three batches: 2 + 2 + 1

        for version_id in ids:
            schema_version, content, digest = await _stored(w, version_id)
            assert schema_version == 2
            assert digest == content_hash(D.PLAN, None, decode_stored(D.PLAN, 2, content))

        await _migrate(w, "downgrade", batch=2)
        assert {(await _stored(w, version_id))[0] for version_id in ids} == {1}

    async def test_with_nothing_to_migrate_it_is_a_no_op(self, enabled_studio: StudioWorld) -> None:
        w = enabled_studio
        await _migrate(w, "upgrade")
        await _migrate(w, "upgrade")

    async def test_running_it_twice_changes_nothing(self, enabled_studio: StudioWorld) -> None:
        w = enabled_studio
        track = await w.new_track()
        (version_id,) = await _plans(w, track, make_plan(title="Twice"))
        await _make_v1(w, version_id)
        await _migrate(w, "upgrade")
        once = await _stored(w, version_id)
        await _migrate(w, "upgrade")
        assert await _stored(w, version_id) == once

    async def test_a_v1_row_is_refused_by_the_app_until_it_is_migrated(
        self, enabled_studio: StudioWorld
    ) -> None:
        """D5: no read-time shim — the data migration is the only way across."""
        w = enabled_studio
        track = await w.new_track()
        (version_id,) = await _plans(w, track, make_plan(title="Shimless"))
        content = await _make_v1(w, version_id)
        with pytest.raises(StudioDocumentSchemaError):
            decode_stored(D.PLAN, 1, content)


class TestDowngrade:
    async def test_it_round_trips_when_every_plan_is_representable(
        self, enabled_studio: StudioWorld
    ) -> None:
        w = enabled_studio
        track = await w.new_track()
        song = golden_song()
        ids = await _plans(w, track, make_plan(title="One", tags=("a",)), song.plan)
        originals = {version_id: await _make_v1(w, version_id) for version_id in ids}
        hashes_v1 = {version_id: (await _stored(w, version_id))[2] for version_id in ids}

        await _migrate(w, "upgrade")
        await _migrate(w, "downgrade")

        for version_id in ids:
            schema_version, content, digest = await _stored(w, version_id)
            assert schema_version == 1
            assert content == originals[version_id]
            assert digest == hashes_v1[version_id]

    async def test_it_refuses_while_an_instrumental_plan_exists(
        self, enabled_studio: StudioWorld
    ) -> None:
        w = enabled_studio
        track = await w.new_track()
        keep, wordless = await _plans(
            w, track, make_plan(title="Keep"), make_plan(title="Wordless", instrumental=True)
        )
        before = {version_id: await _stored(w, version_id) for version_id in (keep, wordless)}

        with pytest.raises(RuntimeError, match="1 studio plan version"):
            await _migrate(w, "downgrade")

        # nothing was rewritten — not even the plan that could have been
        assert {version_id: await _stored(w, version_id) for version_id in before} == before

    async def test_it_refuses_while_a_plan_without_sections_exists(
        self, enabled_studio: StudioWorld
    ) -> None:
        w = enabled_studio
        track = await w.new_track()
        await _plans(w, track, make_plan(title="Empty", empty=True))
        with pytest.raises(RuntimeError, match="no sections"):
            await _migrate(w, "downgrade")
