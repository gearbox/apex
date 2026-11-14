"""The Suno-text importer against real rows: service behaviour and the HTTP route.

Invariants 2-8, 12, 14 and 15 of the S1-P1b prompt. The service tests run on the savepoint-wrapped
``db_session``; the HTTP tests commit rows and let the app open its own session per request (and
roll back when a handler raises), which is what proves a failed import leaves nothing behind.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any
from uuid import UUID, uuid4

import msgspec
import pytest
import pytest_asyncio
from litestar.testing import AsyncTestClient
from sqlalchemy import delete, event, func, select, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from structlog.testing import capture_logs

from src.api.schemas.studio import ArtistCreate, PlanVersionCreate
from src.core.enums import StudioDocumentKind, StudioDocumentOrigin
from src.core.enums import StudioSectionKind as K
from src.db.models.library import LibraryProject
from src.db.models.studio import StudioArtist, StudioDocumentVersion
from src.db.models.user import User
from src.studio.documents import LyricLine, PlanDocument, decode_stored
from src.studio.errors import ImportInvalid, StudioDocumentInvalid, StudioNotFound
from src.studio.services import version as version_module
from src.studio.validation import content_hash
from tests.integration.test_studio_routes import (
    S,
    _bearer,
    _build_app,
    _lyrics_body,
    _plan_body,
)
from tests.studio_support import (
    GOLDEN_EXCLUDE,
    GOLDEN_LYRICS,
    GOLDEN_STYLE,
    GoldenSong,
    golden_song,
    make_lyrics,
    make_plan,
    make_section,
)

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from litestar import Litestar
    from sqlalchemy.ext.asyncio import AsyncEngine

    from src.api.schemas.studio import SunoImportOut
    from tests.studio_world import StudioWorld


@pytest.fixture(autouse=True)
def studio_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "src.api.security.studio.get_settings", lambda: SimpleNamespace(studio_enabled=True)
    )


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Track:
    id: UUID
    song: GoldenSong
    plan_version_id: UUID
    lyrics_version_id: UUID

    @property
    def paste(self) -> dict[str, Any]:
        """The text the export renders for this track — what a user pastes back."""
        return {
            "title": self.song.plan.title,
            "style": GOLDEN_STYLE,
            "exclude": GOLDEN_EXCLUDE,
            "lyrics": GOLDEN_LYRICS,
        }


async def _golden(w: StudioWorld) -> Track:
    """A track holding the P1 golden song, with the artist whose tags the export mixes in."""
    song = golden_song()
    artist = await w.artists.create(
        w.user_id,
        w.product_id,
        ArtistCreate(
            name="Nova",
            style_tags=list(song.artist.style_tags),
            global_avoid=list(song.artist.global_avoid),
            default_language="en",
        ),
    )
    await w.enable(artist_id=artist.id)
    track = await w.new_track(song.plan.title)
    plan, _ = await w.plan(track, song.plan)
    lyrics, _ = await w.lyrics(track, song.lyrics, plan.version.id)
    exported = await w.exports.render_suno(track, w.user_id, w.product_id)
    assert (exported.style, exported.exclude, exported.lyrics) == (
        GOLDEN_STYLE,
        GOLDEN_EXCLUDE,
        GOLDEN_LYRICS,
    )
    return Track(track, song, plan.version.id, lyrics.version.id)


async def _state(w: StudioWorld, track_id: UUID) -> tuple[Any, ...]:
    """Pointers, version counters and row count: everything a write would move."""
    row = await w.track_row(track_id)
    return (
        row.current_plan_version_id,
        row.current_lyrics_version_id,
        row.next_plan_version_number,
        row.next_lyrics_version_number,
        await w.version_count(track_id),
        row.updated_at,
    )


async def _origins(w: StudioWorld, track_id: UUID) -> dict[UUID, str]:
    rows = await w.session.execute(
        select(StudioDocumentVersion.id, StudioDocumentVersion.origin).where(
            StudioDocumentVersion.track_id == track_id
        )
    )
    return dict(rows.all())


def _codes(out: SunoImportOut) -> list[tuple[str, int | None]]:
    return [(warning.code, warning.line) for warning in out.warnings]


async def _store_legacy_tempo_tag(session: AsyncSession, version_id: UUID, tag: str) -> None:
    """Rewrite a stored plan so that its global tags are ``[tag]``: a row from before ``80 BPM``
    was reserved. The service refuses to write one, so the row is changed directly (its hash is
    kept consistent with its content, as it was when P1b stored it)."""
    row = (
        await session.execute(
            select(StudioDocumentVersion.content, StudioDocumentVersion.schema_version).where(
                StudioDocumentVersion.id == version_id
            )
        )
    ).one()
    content = {**row[0], "global_style": {**row[0]["global_style"], "tags": [tag]}}
    document = decode_stored(StudioDocumentKind.PLAN, row[1], content)
    assert isinstance(document, PlanDocument)
    await session.execute(
        text(
            "UPDATE studio_document_versions SET content = CAST(:c AS jsonb), content_hash = :h "
            "WHERE id = :id"
        ),
        {
            "c": msgspec.json.encode(content).decode(),
            "h": content_hash(StudioDocumentKind.PLAN, None, document),
            "id": version_id,
        },
    )


# ---------------------------------------------------------------------------
# first import
# ---------------------------------------------------------------------------


class TestFirstImport:
    async def test_it_writes_plan_and_lyrics_as_current_with_origin_import(
        self, enabled_studio: StudioWorld
    ) -> None:
        w = enabled_studio
        track = await w.new_track("Track Title")
        out = await w.import_suno(
            track,
            style="lofi, 90 bpm",
            exclude="noise",
            lyrics="[Verse]\nhello\n\n[Chorus]\nsing",
            language="en",
        )

        assert out.created
        assert not (out.dry_run or out.plan_unchanged or out.lyrics_unchanged)
        assert out.plan_version is not None
        assert out.lyrics_version is not None
        assert (out.plan_version.version_number, out.lyrics_version.version_number) == (1, 1)
        assert (out.plan_version.origin, out.lyrics_version.origin) == (
            StudioDocumentOrigin.IMPORT,
            StudioDocumentOrigin.IMPORT,
        )
        assert out.lyrics_version.based_on_plan_version_id == out.plan_version.id
        assert (out.plan_version.is_current, out.lyrics_version.is_current) == (True, True)
        row = await w.track_row(track)
        assert (row.current_plan_version_id, row.current_lyrics_version_id) == (
            out.plan_version.id,
            out.lyrics_version.id,
        )
        assert (out.plan.title, out.plan.tempo_bpm) == ("Track Title", 90)
        assert out.plan.global_style.tags == ("lofi",)
        assert out.plan.global_style.avoid == ("noise",)
        assert [s.label for s in out.plan.sections] == [None, None]
        assert out.lyrics is not None
        assert [ln.text for sec in out.lyrics.sections for ln in sec.lines] == ["hello", "sing"]
        assert set((await _origins(w, track)).values()) == {"import"}

    async def test_the_title_comes_from_the_request_else_the_track(
        self, enabled_studio: StudioWorld
    ) -> None:
        w = enabled_studio
        track = await w.new_track("Track Title")
        assert (await w.import_suno(track, lyrics="[Verse]", dry_run=True)).plan.title == (
            "Track Title"
        )
        out = await w.import_suno(track, title="  Pasted   Title ", lyrics="[Verse]")
        assert out.plan.title == "Pasted Title"

    async def test_headers_only_writes_a_plan_and_no_lyrics_and_says_so(
        self, enabled_studio: StudioWorld
    ) -> None:
        """Invariant 7: instrumental is never inferred from an empty Lyrics box."""
        w = enabled_studio
        track = await w.new_track()
        out = await w.import_suno(track, lyrics="[Intro]\n\n[Verse]\n\n[Outro]")
        assert out.plan.instrumental is False
        assert len(out.plan.sections) == 3
        assert out.lyrics is None
        assert out.lyrics_version is None
        assert out.lyrics_unchanged  # no lines and no current lyrics: nothing to change
        assert _codes(out) == [("no_lyric_lines", None)]
        assert (await w.track_row(track)).current_lyrics_version_id is None
        assert await w.version_count(track) == 1

    async def test_an_empty_lyrics_box_is_a_plan_without_sections(
        self, enabled_studio: StudioWorld
    ) -> None:
        w = enabled_studio
        track = await w.new_track()
        out = await w.import_suno(track, style="lofi")
        assert (out.plan.sections, out.lyrics, out.warnings) == ((), None, [])
        export = await w.exports.render_suno(track, w.user_id, w.product_id)
        assert (export.lyrics, export.style, export.instrumental) == ("", "lofi", False)

    async def test_the_instrumental_switch_is_a_request_option(
        self, enabled_studio: StudioWorld
    ) -> None:
        w = enabled_studio
        track = await w.new_track()
        out = await w.import_suno(track, lyrics="[Drop]\n\n[Outro]", instrumental=True)
        assert out.plan.instrumental is True
        assert out.warnings == []  # nothing to warn about: lyrics were not expected
        export = await w.exports.render_suno(track, w.user_id, w.product_id)
        assert (export.instrumental, export.lyrics) == (True, "[Drop]\n\n[Outro]")

    async def test_the_artists_tags_are_not_copied_into_the_plan(
        self, enabled_studio: StudioWorld
    ) -> None:
        w = enabled_studio
        artist = await w.artists.create(
            w.user_id,
            w.product_id,
            ArtistCreate(name="Nova", style_tags=["husky", "Synthwave"], global_avoid=["metal"]),
        )
        await w.enable(artist_id=artist.id)
        track = await w.new_track()
        out = await w.import_suno(
            track, style="lofi, husky, Synthwave, 100 BPM", exclude="noise, metal", lyrics="[Verse]"
        )
        assert out.plan.global_style.tags == ("lofi",)
        assert out.plan.global_style.avoid == ("noise",)
        assert out.round_trip.style and out.round_trip.exclude
        export = await w.exports.render_suno(track, w.user_id, w.product_id)
        assert export.style == "lofi, husky, Synthwave, 100 BPM"
        assert export.exclude == "noise, metal"


# ---------------------------------------------------------------------------
# languages
# ---------------------------------------------------------------------------


class TestLanguage:
    async def test_request_then_current_lyrics_then_artist_default_then_422(
        self, enabled_studio: StudioWorld
    ) -> None:
        w = enabled_studio
        track = await w.new_track()

        with pytest.raises(ImportInvalid) as exc:  # nothing to fall back to
            await w.import_suno(track, lyrics="[Verse]\nhello")
        assert [(i.path, i.code) for i in exc.value.issues] == [("language", "language_required")]
        assert await w.version_count(track) == 0

        out = await w.import_suno(track, lyrics="[Verse]\nhello", language="fr")
        assert out.lyrics is not None
        assert out.lyrics.language == "fr"
        # the current lyrics' language is the next fallback
        out = await w.import_suno(track, lyrics="[Verse]\nhello again")
        assert out.lyrics is not None
        assert out.lyrics.language == "fr"
        # ... and the request beats it
        out = await w.import_suno(track, lyrics="[Verse]\nhello there", language="de")
        assert out.lyrics is not None
        assert out.lyrics.language == "de"

    async def test_the_artists_default_language_is_the_last_resort(
        self, enabled_studio: StudioWorld
    ) -> None:
        w = enabled_studio
        artist = await w.artists.create(
            w.user_id, w.product_id, ArtistCreate(name="Ana", default_language="es")
        )
        await w.enable(artist_id=artist.id)
        track = await w.new_track()
        out = await w.import_suno(track, lyrics="[Verse]\nhola")
        assert out.lyrics is not None
        assert out.lyrics.language == "es"

    async def test_no_lines_need_no_language(self, enabled_studio: StudioWorld) -> None:
        w = enabled_studio
        track = await w.new_track()
        await w.import_suno(track, lyrics="[Verse]\n\n[Chorus]")


# ---------------------------------------------------------------------------
# invariants 2, 3, 4: a song that did not change imports as a no-op
# ---------------------------------------------------------------------------


class TestNothingChanged:
    async def test_invariant_2_importing_the_export_writes_nothing(
        self, enabled_studio: StudioWorld
    ) -> None:
        w = enabled_studio
        t = await _golden(w)
        before = await _state(w, t.id)

        out = await w.import_suno(t.id, **t.paste)

        assert (out.plan_unchanged, out.lyrics_unchanged, out.created) == (True, True, False)
        assert out.plan_version is not None
        assert out.lyrics_version is not None
        assert (out.plan_version.id, out.lyrics_version.id) == (
            t.plan_version_id,
            t.lyrics_version_id,
        )
        assert await _state(w, t.id) == before
        # everything the export cannot say survived: kinds, delivery/tag splits, notes, idea...
        assert out.plan == t.song.plan
        assert out.lyrics == t.song.lyrics
        assert out.round_trip.style and out.round_trip.exclude and out.round_trip.lyrics
        assert [w_.code for w_ in out.warnings] == ["unknown_section_name"]  # "Break It Down"

    async def test_the_same_holds_for_a_dry_run(self, enabled_studio: StudioWorld) -> None:
        w = enabled_studio
        t = await _golden(w)
        out = await w.import_suno(t.id, dry_run=True, **t.paste)
        assert (out.plan_unchanged, out.lyrics_unchanged, out.created, out.dry_run) == (
            True,
            True,
            False,
            True,
        )

    async def test_invariant_3_the_same_text_twice_writes_rows_at_most_once(
        self, enabled_studio: StudioWorld
    ) -> None:
        w = enabled_studio
        track = await w.new_track("T")
        paste = {
            "style": "lofi, 80 BPM",
            "exclude": "noise",
            "lyrics": "[Intro: soft]\n\n[Verse 1]\n[whispered]\nhello\n\n[Chorus]\nsing\n\n[Verse 2]\nagain",
            "language": "en",
        }
        first = await w.import_suno(track, **paste)
        count = await w.version_count(track)
        second = await w.import_suno(track, **paste)

        assert first.created
        assert (second.plan_unchanged, second.lyrics_unchanged, second.created) == (
            True,
            True,
            False,
        )
        assert await w.version_count(track) == count == 2
        assert second.plan == first.plan
        assert second.lyrics == first.lyrics
        assert second.plan_version is not None
        assert first.plan_version is not None
        assert second.plan_version.id == first.plan_version.id

    async def test_importing_the_same_text_with_no_lines_twice_is_also_a_no_op(
        self, enabled_studio: StudioWorld
    ) -> None:
        w = enabled_studio
        track = await w.new_track()
        await w.import_suno(track, lyrics="[Verse]\n\n[Chorus]")
        again = await w.import_suno(track, lyrics="[Verse]\n\n[Chorus]")
        assert (again.plan_unchanged, again.lyrics_unchanged, again.created) == (True, True, False)

    async def test_invariant_4_an_edited_line_keeps_every_section_id(
        self, enabled_studio: StudioWorld
    ) -> None:
        w = enabled_studio
        t = await _golden(w)
        edited = {**t.paste, "lyrics": GOLDEN_LYRICS.replace("Chorus one", "Chorus uno")}

        out = await w.import_suno(t.id, **edited)

        assert out.plan_unchanged
        assert not out.lyrics_unchanged
        assert out.created
        assert out.plan_version is not None
        assert out.plan_version.id == t.plan_version_id  # the plan version did not move
        assert out.lyrics_version is not None
        assert out.lyrics_version.id != t.lyrics_version_id
        assert out.lyrics_version.parent_version_id == t.lyrics_version_id
        assert out.lyrics_version.based_on_plan_version_id == t.plan_version_id
        assert out.lyrics is not None
        assert [s.section_id for s in out.lyrics.sections] == [
            s.section_id for s in t.song.lyrics.sections
        ]
        assert [s.id for s in out.plan.sections] == [s.id for s in t.song.plan.sections]
        assert await w.version_count(t.id) == 3
        row = await w.track_row(t.id)
        assert row.current_plan_version_id == t.plan_version_id
        assert row.current_lyrics_version_id == out.lyrics_version.id


# ---------------------------------------------------------------------------
# a real change
# ---------------------------------------------------------------------------


class TestChanges:
    async def test_a_changed_plan_becomes_a_child_of_the_current_plan(
        self, enabled_studio: StudioWorld
    ) -> None:
        w = enabled_studio
        t = await _golden(w)
        out = await w.import_suno(t.id, **{**t.paste, "style": "synthwave, 130 BPM"})

        assert not out.plan_unchanged
        assert out.created
        assert out.plan_version is not None
        assert out.plan_version.parent_version_id == t.plan_version_id
        assert out.plan.tempo_bpm == 130
        assert out.plan.global_style.tags == ("synthwave",)
        # the lyrics are based on the plan version that is now current, hence a new version too
        assert out.lyrics_version is not None
        assert out.lyrics_version.based_on_plan_version_id == out.plan_version.id
        assert out.lyrics_version.parent_version_id == t.lyrics_version_id
        assert not out.lyrics_unchanged
        assert out.plan.idea == t.song.plan.idea  # carried
        assert out.plan.references == t.song.plan.references

    async def test_pasting_headers_only_clears_the_current_lyrics(
        self, enabled_studio: StudioWorld
    ) -> None:
        w = enabled_studio
        t = await _golden(w)
        headers_only = "\n\n".join(block.split("\n")[0] for block in GOLDEN_LYRICS.split("\n\n"))
        out = await w.import_suno(t.id, **{**t.paste, "lyrics": headers_only})

        assert out.lyrics is None
        assert out.lyrics_version is None
        assert not out.lyrics_unchanged
        assert [w_.code for w_ in out.warnings if w_.code == "no_lyric_lines"] == ["no_lyric_lines"]
        row = await w.track_row(t.id)
        assert row.current_lyrics_version_id is None  # the pair says: this song has no lyrics
        assert row.current_plan_version_id == t.plan_version_id  # headers unchanged: same plan
        assert out.plan_unchanged
        assert await w.version_count(t.id) == 2  # the lyrics version is kept, merely detached

    async def test_a_dropped_section_with_lyrics_detaches_the_old_ones_and_writes_new(
        self, enabled_studio: StudioWorld
    ) -> None:
        w = enabled_studio
        t = await _golden(w)
        without_pre = GOLDEN_LYRICS.replace("[Pre-Chorus]\nPre line\n\n", "")
        out = await w.import_suno(t.id, **{**t.paste, "lyrics": without_pre})

        assert out.created
        assert len(out.plan.sections) == len(t.song.plan.sections) - 1
        assert out.lyrics_version is not None
        row = await w.track_row(t.id)
        assert row.current_lyrics_version_id == out.lyrics_version.id
        assert row.current_plan_version_id == (out.plan_version.id if out.plan_version else None)

    async def test_a_second_import_after_a_change_is_a_no_op(
        self, enabled_studio: StudioWorld
    ) -> None:
        w = enabled_studio
        t = await _golden(w)
        changed = {
            **t.paste,
            "style": "synthwave, 130 BPM",
            "lyrics": GOLDEN_LYRICS + "\n\n[Outro]",
        }
        await w.import_suno(t.id, **changed)
        before = await _state(w, t.id)
        again = await w.import_suno(t.id, **changed)
        assert (again.plan_unchanged, again.lyrics_unchanged, again.created) == (True, True, False)
        assert await _state(w, t.id) == before

    async def test_non_exportable_fields_survive_a_header_edit(
        self, enabled_studio: StudioWorld
    ) -> None:
        w = enabled_studio
        t = await _golden(w)
        drop = t.song.drop
        edited = GOLDEN_LYRICS.replace("[Drop 1: heavy bass]", "[Drop 1: heavy bass, wide]")
        out = await w.import_suno(t.id, **{**t.paste, "lyrics": edited})
        section = next(s for s in out.plan.sections if s.id == drop.id)
        assert section.style.tags == ("heavy bass", "wide")
        assert (section.duration_seconds, section.style.avoid) == (20, ("distortion",))


# ---------------------------------------------------------------------------
# invariant 6: a dry run never writes
# ---------------------------------------------------------------------------


class TestDryRun:
    async def test_it_writes_nothing_and_equals_the_real_import(
        self, enabled_studio: StudioWorld
    ) -> None:
        w = enabled_studio
        t = await _golden(w)
        paste = {
            **t.paste,
            "style": "synthwave, 130 BPM",
            "lyrics": GOLDEN_LYRICS + "\n\n[Outro]\nfade",
        }
        before = await _state(w, t.id)

        dry = await w.import_suno(t.id, dry_run=True, **paste)

        assert await _state(w, t.id) == before  # rows, pointers, counters, updated_at
        assert dry.dry_run
        assert (dry.plan_version, dry.lyrics_version, dry.created) == (None, None, False)

        real = await w.import_suno(t.id, **paste)
        assert real.created
        assert not real.dry_run
        assert real.plan_version is not None
        assert real.lyrics_version is not None
        assert (dry.plan, dry.lyrics) == (real.plan, real.lyrics)
        assert dry.warnings == real.warnings
        assert dry.round_trip == real.round_trip
        assert (dry.plan_unchanged, dry.lyrics_unchanged) == (
            real.plan_unchanged,
            real.lyrics_unchanged,
        )

    async def test_it_takes_no_lock_and_issues_nothing_but_selects(
        self, enabled_studio: StudioWorld, db_engine: AsyncEngine
    ) -> None:
        w = enabled_studio
        t = await _golden(w)
        statements: list[str] = []

        def capture(_c: Any, _cur: Any, statement: str, *_a: Any) -> None:
            statements.append(statement)

        event.listen(db_engine.sync_engine, "before_cursor_execute", capture)
        try:
            await w.import_suno(t.id, dry_run=True, **{**t.paste, "style": "other"})
        finally:
            event.remove(db_engine.sync_engine, "before_cursor_execute", capture)

        assert statements
        assert all(s.lstrip().upper().startswith("SELECT") for s in statements), statements
        assert all("FOR UPDATE" not in s.upper() for s in statements)

    async def test_a_real_import_locks_the_track_before_it_reads_the_pointers(
        self, enabled_studio: StudioWorld, db_engine: AsyncEngine
    ) -> None:
        w = enabled_studio
        t = await _golden(w)
        statements: list[str] = []

        def capture(_c: Any, _cur: Any, statement: str, *_a: Any) -> None:
            statements.append(statement)

        event.listen(db_engine.sync_engine, "before_cursor_execute", capture)
        try:
            await w.import_suno(t.id, **t.paste)
        finally:
            event.remove(db_engine.sync_engine, "before_cursor_execute", capture)

        first_lock = next(i for i, s in enumerate(statements) if "FOR UPDATE" in s.upper())
        first_version_read = next(
            i for i, s in enumerate(statements) if "FROM studio_document_versions" in s
        )
        assert first_lock < first_version_read

    async def test_it_still_reports_errors(self, enabled_studio: StudioWorld) -> None:
        w = enabled_studio
        track = await w.new_track()
        with pytest.raises(ImportInvalid):
            await w.import_suno(track, lyrics="a [b]", dry_run=True)


# ---------------------------------------------------------------------------
# invariant 8 and the other refusals
# ---------------------------------------------------------------------------


class TestRefusals:
    async def test_invariant_8_lyric_lines_on_an_instrumental_track_are_a_422_that_writes_nothing(
        self, enabled_studio: StudioWorld
    ) -> None:
        w = enabled_studio
        t = await _golden(w)
        before = await _state(w, t.id)
        with pytest.raises(ImportInvalid) as exc:
            await w.import_suno(t.id, instrumental=True, **t.paste)
        assert [(i.path, i.code) for i in exc.value.issues] == [
            ("lyrics", "instrumental_with_lyric_lines")
        ]
        lines = exc.value.source_lines["lyrics"]
        assert isinstance(lines, tuple)
        assert lines[0] == 4  # the first lyric line of the pasted golden text
        assert await _state(w, t.id) == before

    async def test_an_inline_tag_is_refused_and_nothing_is_written(
        self, enabled_studio: StudioWorld
    ) -> None:
        w = enabled_studio
        track = await w.new_track()
        with pytest.raises(ImportInvalid) as exc:
            await w.import_suno(track, lyrics="[Verse]\nla [x] la")
        assert exc.value.source_lines == {"lyrics[2]": 2}
        assert await w.version_count(track) == 0

    async def test_validator_issues_are_addressed_to_pasted_lines(
        self, enabled_studio: StudioWorld
    ) -> None:
        w = enabled_studio
        track = await w.new_track()
        with pytest.raises(ImportInvalid) as exc:
            await w.import_suno(
                track, style="Lofi", exclude="lofi", lyrics="[Verse]\n\n[A [B]\nx", language="en"
            )
        assert {(i.path, i.code) for i in exc.value.issues} == {
            ("plan.global_style.avoid[0]", "tag_conflict"),
            ("plan.sections[1].label", "reserved_characters"),
        }
        assert exc.value.source_lines == {"plan.sections[1].label": 3}
        assert await w.version_count(track) == 0

    async def test_another_users_track_is_a_404_even_for_a_dry_run(
        self, enabled_studio: StudioWorld, make_user: Any
    ) -> None:
        w = enabled_studio
        track = await w.new_track()
        stranger = await make_user(email=f"stranger-{uuid4().hex[:8]}@example.com")
        for dry_run in (False, True):
            with pytest.raises(StudioNotFound):
                await w.imports.import_suno(
                    track,
                    request_for(lyrics="[Verse]", dry_run=dry_run),
                    stranger.id,
                    w.product_id,
                )
        with pytest.raises(StudioNotFound):
            await w.import_suno(uuid4(), lyrics="[Verse]")
        with pytest.raises(StudioNotFound):  # another product
            await w.imports.import_suno(track, request_for(lyrics="[Verse]"), w.user_id, "synthara")


def request_for(**fields: Any) -> Any:
    from src.api.schemas.studio import SunoImportRequest

    return SunoImportRequest(**fields)


# ---------------------------------------------------------------------------
# invariant 12: warnings carry a code and, for Lyrics-derived ones, a 1-based line
# ---------------------------------------------------------------------------


class TestWarnings:
    async def test_invariant_12_every_warning_has_a_code_and_lyrics_ones_a_line(
        self, enabled_studio: StudioWorld
    ) -> None:
        w = enabled_studio
        track = await w.new_track()
        lyrics = (
            "stray words\n"  # 1: text before the first header
            "\n"
            "[Guitar Duel: loud, LOUD]\n"  # 3: unknown name + duplicate descriptor
            "[whispered]\n"  # 4: dangling (the blank line ends the block)
            "\n"
            "[Chorus]\n"  # 6
            "[soft]\n"  # 7: dangling (a second delivery follows)
            "[loud]\n"  # 8
            "sing\n"  # 9
        )
        out = await w.import_suno(
            track, style="a, A, 100 bpm, 120 bpm", exclude="x, X", lyrics=lyrics, language="en"
        )
        assert all(warning.code for warning in out.warnings)
        assert set(_codes(out)) == {
            ("text_before_first_header", 1),
            ("unknown_section_name", 3),
            ("duplicate_descriptor", 3),
            ("dangling_delivery", 4),
            ("dangling_delivery", 7),
            ("duplicate_style_item", None),
            ("duplicate_tempo", None),
        }
        assert all(w_.line is None or w_.line >= 1 for w_ in out.warnings)


# ---------------------------------------------------------------------------
# invariant 14: the origin is stated by every caller
# ---------------------------------------------------------------------------


class TestOrigin:
    async def test_create_requires_an_origin(self, enabled_studio: StudioWorld) -> None:
        w = enabled_studio
        track = await w.new_track()
        with pytest.raises(TypeError, match="origin"):
            await w.versions.create(  # type: ignore[call-arg]
                track, PlanVersionCreate(content=make_plan()), w.user_id, w.product_id
            )

    async def test_typed_versions_are_user_and_imported_ones_are_import(
        self, enabled_studio: StudioWorld
    ) -> None:
        w = enabled_studio
        track = await w.new_track()
        verse = make_section(K.VERSE)
        typed, _ = await w.plan(track, make_plan(verse, title="Typed"))
        # a style the typed plan does not have: the import writes a plan version of its own (the
        # title is the typed plan's, carried over — it no longer changes by itself)
        imported = await w.import_suno(track, style="lofi", lyrics="[Verse]\nhello", language="en")
        assert imported.plan_version is not None
        assert imported.lyrics_version is not None
        assert imported.plan.title == "Typed"
        origins = await _origins(w, track)
        assert origins[typed.version.id] == "user"
        assert origins[imported.plan_version.id] == "import"
        assert origins[imported.lyrics_version.id] == "import"


# ---------------------------------------------------------------------------
# invariant 15: logs carry ids, counts and codes only
# ---------------------------------------------------------------------------


class TestLogging:
    MARK = "ZXQ-SECRET"

    async def test_invariant_15_no_pasted_text_reaches_the_logs(
        self, enabled_studio: StudioWorld
    ) -> None:
        w = enabled_studio
        m = self.MARK
        track = await w.new_track(f"{m}-track-title")
        paste = {
            "title": f"{m}-title",
            "style": f"{m}-style, {m}-tag, 100 BPM",
            "exclude": f"{m}-exclude",
            "lyrics": f"[{m}-Section: {m}-desc]\n[{m}-delivery]\n{m} lyric line\n\n[Chorus]\n{m} again\n[{m}-end]",
            "language": "en",
        }
        with capture_logs() as logs:
            await w.import_suno(track, dry_run=True, **paste)
            await w.import_suno(track, **paste)
            await w.import_suno(track, **paste)  # unchanged
            with pytest.raises(ImportInvalid):
                await w.import_suno(track, lyrics=f"{m} [{m}]")
        events = [entry["event"] for entry in logs]
        assert "studio.suno_import_dry_run" in events
        assert events.count("studio.suno_imported") == 2
        for entry in logs:
            assert m.lower() not in repr(entry).lower(), entry
        imported = next(e for e in logs if e["event"] == "studio.suno_imported")
        assert {"track_id", "sections", "lyric_lines", "warnings", "created"} <= set(imported)
        assert isinstance(imported["warnings"], list)
        assert all(isinstance(code, str) for code in imported["warnings"])


# ---------------------------------------------------------------------------
# invariants 9-11: the plan schema v2 rules, through version create / set-current / export
# ---------------------------------------------------------------------------


class TestPlanSchemaV2Rules:
    async def test_invariant_9_an_instrumental_plan_cannot_hold_lyrics_with_lines(
        self, enabled_studio: StudioWorld
    ) -> None:
        from src.studio.errors import LyricsIncompatibleWithPlan, StudioDocumentInvalid

        w = enabled_studio
        verse = make_section(K.VERSE)
        track = await w.new_track()
        plan, _ = await w.plan(track, make_plan(verse, title="Sung"))
        lyrics, _ = await w.lyrics(track, make_lyrics((verse.id, ["la"])), plan.version.id)
        wordless, _ = await w.plan(
            track, make_plan(verse, title="Wordless", instrumental=True), make_current=False
        )

        # on lyrics create: 422
        with pytest.raises(StudioDocumentInvalid) as invalid:
            await w.lyrics(track, make_lyrics((verse.id, ["la"])), wordless.version.id)
        assert ("sections", "instrumental_plan_has_lines") in {
            (i.path, i.code) for i in invalid.value.issues
        }
        # on make-current of the plan: 409 (reject) ...
        with pytest.raises(LyricsIncompatibleWithPlan) as rejected:
            await w.versions.set_current(
                track,
                w.user_id,
                w.product_id,
                plan_version_id=wordless.version.id,
                lyrics_version_id=lyrics.version.id,
            )
        assert [(i.path, i.code) for i in rejected.value.issues] == [
            ("sections", "instrumental_plan_has_lines")
        ]
        # ... and on a plan create with make_current over lyrics that have lines
        with pytest.raises(LyricsIncompatibleWithPlan):
            await w.plan(track, make_plan(verse, title="Wordless 2", instrumental=True))
        # an export of a stored incompatible pair is refused too
        with pytest.raises(LyricsIncompatibleWithPlan):
            await w.exports.render_suno(
                track,
                w.user_id,
                w.product_id,
                plan_version_id=wordless.version.id,
                lyrics_version_id=lyrics.version.id,
            )

    async def test_invariant_10_zero_section_plans_are_valid(
        self, enabled_studio: StudioWorld
    ) -> None:
        from src.studio.documents import LyricsDocument
        from src.studio.errors import StudioDocumentInvalid

        w = enabled_studio
        track = await w.new_track()
        plan, created = await w.plan(track, make_plan(empty=True, tags=("lofi",), title="Bare"))
        assert created
        assert plan.version.content.sections == ()  # type: ignore[union-attr]
        export = await w.exports.render_suno(track, w.user_id, w.product_id)
        assert (export.lyrics, export.style) == ("", "lofi")
        # lyrics are accepted only without sections
        await w.lyrics(track, LyricsDocument(language="en"), plan.version.id)
        with pytest.raises(StudioDocumentInvalid) as invalid:
            await w.lyrics(track, make_lyrics((uuid4(), ["la"])), plan.version.id)
        assert ("sections[0].section_id", "unknown_section") in {
            (i.path, i.code) for i in invalid.value.issues
        }

    async def test_invariant_11_reserved_names_are_refused_on_create(
        self, enabled_studio: StudioWorld
    ) -> None:
        from src.studio.errors import StudioDocumentInvalid

        w = enabled_studio
        track = await w.new_track()
        with pytest.raises(StudioDocumentInvalid) as plan_issues:
            await w.plan(track, make_plan(make_section(K.DROP, tags=("Instrumental",))))
        assert [(i.path, i.code) for i in plan_issues.value.issues] == [
            ("sections[0].style.tags[0]", "reserved_descriptor")
        ]
        verse = make_section(K.VERSE)
        plan, _ = await w.plan(track, make_plan(verse))
        with pytest.raises(StudioDocumentInvalid) as lyrics_issues:
            await w.lyrics(
                track,
                make_lyrics((verse.id, [LyricLine(text="la", delivery=("Chorus",))])),
                plan.version.id,
            )
        assert [(i.path, i.code) for i in lyrics_issues.value.issues] == [
            ("sections[0].lines[0].delivery", "reserved_delivery_tag")
        ]
        assert await w.version_count(track) == 1

    async def test_the_export_reports_the_instrumental_flag(
        self, enabled_studio: StudioWorld
    ) -> None:
        w = enabled_studio
        track = await w.new_track()
        await w.plan(track, make_plan(make_section(K.DROP), instrumental=True))
        export = await w.exports.render_suno(track, w.user_id, w.product_id)
        assert export.instrumental is True
        assert export.lyrics == "[Drop]"


# ---------------------------------------------------------------------------
# round 1 of the remediation: R1 (tempo-like tags), R2 (colon labels), R3 (title / flag)
# ---------------------------------------------------------------------------


class TestLegacyTempoTags:
    async def test_importing_over_a_legacy_plan_with_a_tempo_tag_fails_loudly(
        self, enabled_studio: StudioWorld
    ) -> None:
        """Row 1 of the reproduction table: tags ``[80 BPM]``, tempo 120 → ``80 BPM, 120 BPM``."""
        w = enabled_studio
        track = await w.new_track("T")
        plan, _ = await w.plan(track, make_plan(tags=("lofi",), tempo_bpm=120))
        await _store_legacy_tempo_tag(w.session, plan.version.id, "80 BPM")
        exported = await w.exports.render_suno(track, w.user_id, w.product_id)
        assert exported.style == "80 BPM, 120 BPM"  # stored rows still render and export
        before = await _state(w, track)

        for dry_run in (False, True):
            with pytest.raises(ImportInvalid) as exc:
                await w.import_suno(
                    track, style=exported.style, lyrics=exported.lyrics, dry_run=dry_run
                )
            assert [(i.path, i.code) for i in exc.value.issues] == [
                ("plan.global_style.tags[0]", "reserved_tempo_tag")
            ]
            assert exc.value.source_lines == {}
        assert await _state(w, track) == before  # nothing was written, nothing moved

    async def test_new_writes_of_such_a_plan_or_artist_are_refused(
        self, enabled_studio: StudioWorld
    ) -> None:
        w = enabled_studio
        track = await w.new_track("T")
        with pytest.raises(StudioDocumentInvalid) as exc:
            await w.plan(track, make_plan(tags=("80 BPM",), tempo_bpm=120))
        assert [(i.path, i.code) for i in exc.value.issues] == [
            ("global_style.tags[0]", "reserved_tempo_tag")
        ]
        assert await w.version_count(track) == 0
        with pytest.raises(StudioDocumentInvalid) as artist_exc:
            await w.artists.create(
                w.user_id, w.product_id, ArtistCreate(name="Nova", style_tags=["rock", "90 BPM"])
            )
        assert [(i.path, i.code) for i in artist_exc.value.issues] == [
            ("style_tags[1]", "reserved_tempo_tag")
        ]

    async def test_patching_the_avoid_list_of_a_legacy_artist_fails_at_its_tempo_tag(
        self, enabled_studio: StudioWorld
    ) -> None:
        from src.api.schemas.studio import ArtistPatch

        w = enabled_studio
        artist = await w.artists.create(
            w.user_id, w.product_id, ArtistCreate(name="Nova", style_tags=["rock"])
        )
        await w.session.execute(
            text("UPDATE studio_artists SET style_tags = ARRAY['80 BPM'] WHERE id = :id"),
            {"id": artist.id},
        )
        stored = await w.artists.get(artist.id, w.user_id, w.product_id)
        assert stored.style_tags == ["80 BPM"]  # a stored row still reads

        with pytest.raises(StudioDocumentInvalid) as exc:
            await w.artists.patch(
                artist.id, w.user_id, w.product_id, ArtistPatch(global_avoid=["metal"])
            )
        assert [(i.path, i.code) for i in exc.value.issues] == [
            ("style_tags[0]", "reserved_tempo_tag")
        ]
        row = (
            await w.session.execute(
                select(StudioArtist.global_avoid, StudioArtist.style_tags)
                .where(StudioArtist.id == artist.id)
                .execution_options(populate_existing=True)
            )
        ).one()
        assert (row[0], row[1]) == ([], ["80 BPM"])  # nothing was written
        # fixing the tag is the way out
        fixed = await w.artists.patch(
            artist.id, w.user_id, w.product_id, ArtistPatch(style_tags=["rock"])
        )
        assert fixed.style_tags == ["rock"]


class TestColonLabels:
    async def test_a_label_with_a_colon_keeps_its_section_id_and_writes_nothing(
        self, enabled_studio: StudioWorld
    ) -> None:
        """R2 / invariant 19: ``[Verse 1: The Return]`` is a label, not name + descriptor."""
        w = enabled_studio
        track = await w.new_track("Track Title")
        verse = make_section(K.VERSE, label="Verse 1: The Return")
        plan, _ = await w.plan(track, make_plan(verse, title="Plan Title"))
        lyrics, _ = await w.lyrics(track, make_lyrics((verse.id, ["la la"])), plan.version.id)
        exported = await w.exports.render_suno(track, w.user_id, w.product_id)
        assert exported.lyrics == "[Verse 1: The Return]\nla la"
        before = await _state(w, track)

        out = await w.import_suno(
            track, style=exported.style, exclude=exported.exclude, lyrics=exported.lyrics
        )

        assert (out.plan_unchanged, out.lyrics_unchanged, out.created) == (True, True, False)
        assert out.round_trip.lyrics
        assert out.plan.sections[0].id == verse.id
        assert out.plan.sections[0].label == "Verse 1: The Return"
        assert out.lyrics is not None
        assert out.lyrics.sections[0].section_id == verse.id
        assert (out.plan_version.id, out.lyrics_version.id) == (  # type: ignore[union-attr]
            plan.version.id,
            lyrics.version.id,
        )
        assert await _state(w, track) == before


class TestTitleAndFlagCarryOver:
    async def _instrumental_track(self, w: StudioWorld) -> tuple[UUID, Any]:
        track = await w.new_track("Track Title")
        drop = make_section(K.DROP)
        await w.plan(track, make_plan(drop, title="Plan Title", instrumental=True))
        exported = await w.exports.render_suno(track, w.user_id, w.product_id)
        assert (exported.instrumental, exported.lyrics) == (True, "[Drop]")
        return track, exported

    async def test_a_request_without_title_and_flag_is_a_no_op(
        self, enabled_studio: StudioWorld
    ) -> None:
        """R3 / invariant 18: the plan's title and flag are not reset by what the text omits."""
        w = enabled_studio
        track, exported = await self._instrumental_track(w)
        before = await _state(w, track)

        for dry_run in (True, False):
            out = await w.import_suno(
                track,
                style=exported.style,
                exclude=exported.exclude,
                lyrics=exported.lyrics,
                dry_run=dry_run,
            )
            assert (out.plan_unchanged, out.lyrics_unchanged, out.created) == (True, True, False)
            assert (out.plan.title, out.plan.instrumental) == ("Plan Title", True)
            assert out.warnings == []
            assert await _state(w, track) == before
        assert (await w.track_row(track)).title == "Track Title"  # the track is never renamed

    async def test_lyric_lines_over_an_instrumental_plan_need_an_explicit_false(
        self, enabled_studio: StudioWorld
    ) -> None:
        w = enabled_studio
        track, _ = await self._instrumental_track(w)
        before = await _state(w, track)

        with pytest.raises(ImportInvalid) as exc:
            await w.import_suno(track, lyrics="[Drop]\nsing", language="en")
        assert [(i.path, i.code) for i in exc.value.issues] == [
            ("lyrics", "instrumental_with_lyric_lines")
        ]
        assert await _state(w, track) == before

        out = await w.import_suno(track, lyrics="[Drop]\nsing", language="en", instrumental=False)
        assert out.created
        assert (out.plan.title, out.plan.instrumental) == ("Plan Title", False)
        assert out.lyrics is not None

    async def test_an_explicit_value_still_wins_and_a_new_track_uses_its_own_title(
        self, enabled_studio: StudioWorld
    ) -> None:
        w = enabled_studio
        track, _ = await self._instrumental_track(w)
        out = await w.import_suno(track, title="Renamed", lyrics="[Drop]", instrumental=True)
        assert (out.plan.title, out.plan.instrumental) == ("Renamed", True)
        assert out.created
        fresh = await w.new_track("Fresh Track")
        out = await w.import_suno(fresh, lyrics="[Verse]", dry_run=True)
        assert (out.plan.title, out.plan.instrumental) == ("Fresh Track", False)


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------


@pytest.fixture
def session_factory(db_engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(bind=db_engine, expire_on_commit=False)


@pytest_asyncio.fixture
async def owner(
    session_factory: async_sessionmaker[AsyncSession],
) -> AsyncGenerator[SimpleNamespace]:
    users: list[UUID] = []

    async def make() -> SimpleNamespace:
        user = User(
            id=uuid4(),
            email=f"import-{uuid4().hex}@example.com",
            password_hash="x",
            product_id="vex",
        )
        project = LibraryProject(id=uuid4(), product_id="vex", user_id=user.id, name="Album")
        async with session_factory() as session, session.begin():
            session.add(user)
            await session.flush()
            session.add(project)
        users.append(user.id)
        return SimpleNamespace(id=user.id, project_id=project.id, headers=_bearer(user.id))

    primary = await make()
    primary.make = make
    yield primary
    async with session_factory() as session, session.begin():
        await session.execute(delete(User).where(User.id.in_(users)))


@pytest_asyncio.fixture
async def client(
    session_factory: async_sessionmaker[AsyncSession],
) -> AsyncGenerator[AsyncTestClient[Litestar]]:
    async with AsyncTestClient(app=_build_app(session_factory)) as test_client:
        yield test_client


async def _http_track(client: AsyncTestClient[Litestar], owner: SimpleNamespace) -> str:
    resp = await client.put(
        f"{S}/projects/{owner.project_id}", headers=owner.headers, json={"release_kind": "album"}
    )
    assert resp.status_code == 201, resp.text  # the project is enabled for the studio
    resp = await client.post(
        f"{S}/projects/{owner.project_id}/tracks", headers=owner.headers, json={"title": "Song"}
    )
    assert resp.status_code == 201, resp.text
    return str(resp.json()["id"])


async def _count(factory: async_sessionmaker[AsyncSession], track_id: str) -> int:
    async with factory() as session:
        return (
            await session.execute(
                select(func.count())
                .select_from(StudioDocumentVersion)
                .where(StudioDocumentVersion.track_id == UUID(track_id))
            )
        ).scalar_one()


PASTE = {
    "style": "lofi, 90 BPM",
    "exclude": "noise",
    "lyrics": "[Verse]\nhello\n\n[Chorus]\nsing",
    "language": "en",
}


class TestOverHttp:
    async def test_201_when_written_then_200_when_nothing_changed(
        self, client: AsyncTestClient[Litestar], owner: SimpleNamespace
    ) -> None:
        track = await _http_track(client, owner)
        url = f"{S}/tracks/{track}/imports/suno"

        first = await client.post(url, headers=owner.headers, json=PASTE)
        assert first.status_code == 201, first.text
        body = first.json()
        assert body["created"] is True
        assert body["dry_run"] is False
        assert body["plan"]["tempo_bpm"] == 90
        assert body["plan"]["instrumental"] is False
        assert body["lyrics"]["language"] == "en"
        assert body["plan_version"]["origin"] == "import"
        assert body["lyrics_version"]["origin"] == "import"
        assert body["round_trip"] == {"style": True, "exclude": True, "lyrics": True}
        assert body["warnings"] == []
        assert (body["plan_unchanged"], body["lyrics_unchanged"]) == (False, False)

        second = await client.post(url, headers=owner.headers, json=PASTE)
        assert second.status_code == 200, second.text
        again = second.json()
        assert (again["created"], again["plan_unchanged"], again["lyrics_unchanged"]) == (
            False,
            True,
            True,
        )
        assert again["plan_version"]["id"] == body["plan_version"]["id"]

        # the versions are visible through the ordinary endpoints, with their origin
        listed = await client.get(f"{S}/tracks/{track}/versions", headers=owner.headers)
        assert {v["origin"] for v in listed.json()["items"]} == {"import"}
        detail = await client.get(f"{S}/tracks/{track}", headers=owner.headers)
        assert detail.json()["current_plan"]["content"]["global_style"]["tags"] == ["lofi"]

    async def test_a_dry_run_is_200_and_writes_nothing(
        self,
        client: AsyncTestClient[Litestar],
        owner: SimpleNamespace,
        session_factory: async_sessionmaker[AsyncSession],
    ) -> None:
        track = await _http_track(client, owner)
        resp = await client.post(
            f"{S}/tracks/{track}/imports/suno",
            headers=owner.headers,
            json={**PASTE, "dry_run": True},
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert (body["dry_run"], body["created"]) == (True, False)
        assert (body["plan_version"], body["lyrics_version"]) == (None, None)
        assert body["plan"]["tempo_bpm"] == 90
        assert await _count(session_factory, track) == 0
        detail = await client.get(f"{S}/tracks/{track}", headers=owner.headers)
        assert detail.json()["current_plan"] is None

    async def test_the_export_round_trips_through_the_routes(
        self, client: AsyncTestClient[Litestar], owner: SimpleNamespace
    ) -> None:
        track = await _http_track(client, owner)
        url = f"{S}/tracks/{track}/imports/suno"
        await client.post(url, headers=owner.headers, json=PASTE)
        exported = (
            await client.get(f"{S}/tracks/{track}/exports/suno", headers=owner.headers)
        ).json()
        assert exported["instrumental"] is False
        resp = await client.post(
            url,
            headers=owner.headers,
            json={
                "title": exported["title"],
                "style": exported["style"],
                "exclude": exported["exclude"],
                "lyrics": exported["lyrics"],
            },
        )
        assert resp.status_code == 200, resp.text
        assert resp.json()["created"] is False
        assert resp.json()["round_trip"] == {"style": True, "exclude": True, "lyrics": True}

    async def test_404_for_an_unknown_track_and_for_another_users_track(
        self, client: AsyncTestClient[Litestar], owner: SimpleNamespace
    ) -> None:
        track = await _http_track(client, owner)
        stranger = await owner.make()
        for dry_run in (False, True):
            body = {**PASTE, "dry_run": dry_run}
            unknown = await client.post(
                f"{S}/tracks/{uuid4()}/imports/suno", headers=owner.headers, json=body
            )
            assert unknown.status_code == 404
            other = await client.post(
                f"{S}/tracks/{track}/imports/suno", headers=stranger.headers, json=body
            )
            assert other.status_code == 404

    async def test_422_import_invalid_has_issues_and_source_lines(
        self,
        client: AsyncTestClient[Litestar],
        owner: SimpleNamespace,
        session_factory: async_sessionmaker[AsyncSession],
    ) -> None:
        track = await _http_track(client, owner)
        url = f"{S}/tracks/{track}/imports/suno"

        inline = await client.post(
            url, headers=owner.headers, json={**PASTE, "lyrics": "[Verse]\nla [x] la"}
        )
        assert inline.status_code == 422
        error = inline.json()
        assert error["error"] == "import_invalid"
        assert error["detail"] == {
            "issues": [{"path": "lyrics[2]", "code": "unsupported_inline_tag"}],
            "source_lines": {"lyrics[2]": 2},
        }
        assert "la [x] la" not in inline.text

        instrumental = await client.post(
            url, headers=owner.headers, json={**PASTE, "instrumental": True}
        )
        assert instrumental.status_code == 422
        assert instrumental.json()["detail"] == {
            "issues": [{"path": "lyrics", "code": "instrumental_with_lyric_lines"}],
            "source_lines": {"lyrics": [2, 5]},
        }

        no_language = await client.post(
            url, headers=owner.headers, json={k: v for k, v in PASTE.items() if k != "language"}
        )
        assert no_language.status_code == 422
        assert no_language.json()["detail"]["issues"] == [
            {"path": "language", "code": "language_required"}
        ]
        assert await _count(session_factory, track) == 0

    async def test_a_malformed_body_is_a_400(
        self, client: AsyncTestClient[Litestar], owner: SimpleNamespace
    ) -> None:
        track = await _http_track(client, owner)
        url = f"{S}/tracks/{track}/imports/suno"
        for body in (
            {"surprise": 1},
            {"style": "x" * 1001},
            {"exclude": "x" * 1001},
            {"lyrics": "x" * 20_001},
            {"title": ""},
            {"title": "x" * 201},
            {"language": "not a language"},
            {"instrumental": "yes"},
        ):
            resp = await client.post(url, headers=owner.headers, json=body)
            assert resp.status_code == 400, (list(body), resp.text)

    async def test_an_empty_body_is_a_plan_with_no_sections(
        self, client: AsyncTestClient[Litestar], owner: SimpleNamespace
    ) -> None:
        track = await _http_track(client, owner)
        resp = await client.post(f"{S}/tracks/{track}/imports/suno", headers=owner.headers, json={})
        assert resp.status_code == 201, resp.text
        assert resp.json()["plan"]["sections"] == []
        assert resp.json()["lyrics"] is None

    async def test_the_route_is_legal_enforced_and_hidden_when_studio_is_off(
        self,
        client: AsyncTestClient[Litestar],
        owner: SimpleNamespace,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        track = await _http_track(client, owner)
        url = f"{S}/tracks/{track}/imports/suno"
        stale = _bearer(owner.id, digest="stale-digest")
        blocked = await client.post(url, headers=stale, json=PASTE)
        assert (blocked.status_code, blocked.json()["error"]) == (428, "legal_acceptance_required")
        assert (await client.post(url, json=PASTE)).status_code == 401

        monkeypatch.setattr(
            "src.api.security.studio.get_settings", lambda: SimpleNamespace(studio_enabled=False)
        )
        hidden = await client.post(url, headers=owner.headers, json=PASTE)
        assert hidden.status_code == 404

    async def test_invariant_5_a_failed_lyrics_step_rolls_the_plan_back(
        self,
        client: AsyncTestClient[Litestar],
        owner: SimpleNamespace,
        session_factory: async_sessionmaker[AsyncSession],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Plan version written, lyrics step 409s (cap reached): the whole import is undone."""
        track = await _http_track(client, owner)
        url = f"{S}/tracks/{track}/imports/suno"
        assert (await client.post(url, headers=owner.headers, json=PASTE)).status_code == 201
        # a second lyrics version, so that the cap of two is reached for lyrics but not for plans
        again = await client.post(
            url,
            headers=owner.headers,
            json={**PASTE, "lyrics": "[Verse]\nhello 2\n\n[Chorus]\nsing"},
        )
        assert again.status_code == 201, again.text
        monkeypatch.setattr(version_module, "MAX_VERSIONS_PER_KIND", 2)

        async with session_factory() as session:
            before = (
                await session.execute(
                    text(
                        "SELECT current_plan_version_id, current_lyrics_version_id, "
                        "next_plan_version_number, next_lyrics_version_number "
                        "FROM studio_tracks WHERE id = :id"
                    ),
                    {"id": UUID(track)},
                )
            ).one()
        count = await _count(session_factory, track)

        with capture_logs() as logs:
            resp = await client.post(
                url,
                headers=owner.headers,
                json={
                    **PASTE,
                    "style": "synthwave",
                    "lyrics": "[Verse]\nhello 3\n\n[Chorus]\nsing",
                },
            )

        # the plan version WAS written inside the request (so the rollback below means something)
        assert any(
            entry["event"] == "studio.version_created" and entry["kind"] == "plan" for entry in logs
        )
        assert resp.status_code == 409, resp.text
        assert resp.json()["error"] == "version_limit_reached"
        assert await _count(session_factory, track) == count  # no plan version survived
        async with session_factory() as session:
            after = (
                await session.execute(
                    text(
                        "SELECT current_plan_version_id, current_lyrics_version_id, "
                        "next_plan_version_number, next_lyrics_version_number "
                        "FROM studio_tracks WHERE id = :id"
                    ),
                    {"id": UUID(track)},
                )
            ).one()
        assert tuple(after) == tuple(before)


class TestRemediationOverHttp:
    async def _post_version(
        self, client: AsyncTestClient[Litestar], owner: SimpleNamespace, track: str, body: Any
    ) -> dict[str, Any]:
        resp = await client.post(f"{S}/tracks/{track}/versions", headers=owner.headers, json=body)
        assert resp.status_code == 201, resp.text
        return dict(resp.json())

    async def test_a_legacy_tempo_tag_is_a_422_import_invalid_and_writes_nothing(
        self,
        client: AsyncTestClient[Litestar],
        owner: SimpleNamespace,
        session_factory: async_sessionmaker[AsyncSession],
    ) -> None:
        track = await _http_track(client, owner)
        url = f"{S}/tracks/{track}/imports/suno"
        first = await client.post(url, headers=owner.headers, json=PASTE)
        assert first.status_code == 201, first.text
        async with session_factory() as session, session.begin():
            await _store_legacy_tempo_tag(
                session, UUID(first.json()["plan_version"]["id"]), "80 BPM"
            )
        count = await _count(session_factory, track)

        resp = await client.post(url, headers=owner.headers, json=PASTE)

        assert resp.status_code == 422, resp.text
        assert resp.json()["error"] == "import_invalid"
        assert resp.json()["detail"]["issues"] == [
            {"path": "plan.global_style.tags[0]", "code": "reserved_tempo_tag"}
        ]
        assert await _count(session_factory, track) == count

    async def test_patching_the_avoid_list_of_a_legacy_artist_is_a_422_at_the_tempo_tag(
        self,
        client: AsyncTestClient[Litestar],
        owner: SimpleNamespace,
        session_factory: async_sessionmaker[AsyncSession],
    ) -> None:
        created = await client.post(
            f"{S}/artists", headers=owner.headers, json={"name": "Nova", "style_tags": ["rock"]}
        )
        assert created.status_code == 201, created.text
        artist_id = created.json()["id"]
        async with session_factory() as session, session.begin():
            await session.execute(
                text("UPDATE studio_artists SET style_tags = ARRAY['80 BPM'] WHERE id = :id"),
                {"id": UUID(artist_id)},
            )

        resp = await client.patch(
            f"{S}/artists/{artist_id}", headers=owner.headers, json={"global_avoid": ["metal"]}
        )

        assert resp.status_code == 422, resp.text
        assert resp.json()["error"] == "document_invalid"
        assert resp.json()["detail"]["issues"] == [
            {"path": "style_tags[0]", "code": "reserved_tempo_tag"}
        ]
        read = await client.get(f"{S}/artists/{artist_id}", headers=owner.headers)
        assert (read.json()["style_tags"], read.json()["global_avoid"]) == (["80 BPM"], [])

    async def test_a_colon_label_re_imports_as_a_no_op_without_title_and_flag(
        self,
        client: AsyncTestClient[Litestar],
        owner: SimpleNamespace,
        session_factory: async_sessionmaker[AsyncSession],
    ) -> None:
        track = await _http_track(client, owner)
        verse = make_section(K.VERSE, label="Verse 1: The Return")
        plan = make_plan(verse, title="Plan Title")
        plan_version = await self._post_version(client, owner, track, _plan_body(plan))
        await self._post_version(
            client,
            owner,
            track,
            _lyrics_body(make_lyrics((verse.id, ["la la"])), plan_version["version"]["id"]),
        )
        exported = (
            await client.get(f"{S}/tracks/{track}/exports/suno", headers=owner.headers)
        ).json()
        assert exported["lyrics"] == "[Verse 1: The Return]\nla la"
        count = await _count(session_factory, track)

        resp = await client.post(
            f"{S}/tracks/{track}/imports/suno",
            headers=owner.headers,
            json={k: exported[k] for k in ("style", "exclude", "lyrics")},
        )

        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert (body["created"], body["plan_unchanged"], body["lyrics_unchanged"]) == (
            False,
            True,
            True,
        )
        assert body["round_trip"] == {"style": True, "exclude": True, "lyrics": True}
        assert body["plan"]["title"] == "Plan Title"  # not the track's "Song"
        assert body["plan"]["sections"][0]["id"] == str(verse.id)
        assert await _count(session_factory, track) == count

    async def test_an_instrumental_plan_needs_an_explicit_false_for_lyric_lines(
        self,
        client: AsyncTestClient[Litestar],
        owner: SimpleNamespace,
        session_factory: async_sessionmaker[AsyncSession],
    ) -> None:
        track = await _http_track(client, owner)
        plan = make_plan(make_section(K.DROP), title="Plan Title", instrumental=True)
        await self._post_version(client, owner, track, _plan_body(plan))
        url = f"{S}/tracks/{track}/imports/suno"
        count = await _count(session_factory, track)

        same = await client.post(url, headers=owner.headers, json={"lyrics": "[Drop]"})
        assert same.status_code == 200, same.text
        assert (same.json()["created"], same.json()["plan"]["instrumental"]) == (False, True)

        sung = {"lyrics": "[Drop]\nsing", "language": "en"}
        refused = await client.post(url, headers=owner.headers, json=sung)
        assert refused.status_code == 422, refused.text
        assert refused.json()["detail"]["issues"] == [
            {"path": "lyrics", "code": "instrumental_with_lyric_lines"}
        ]
        assert await _count(session_factory, track) == count

        written = await client.post(
            url, headers=owner.headers, json={**sung, "instrumental": False}
        )
        assert written.status_code == 201, written.text
        assert written.json()["plan"]["instrumental"] is False
        assert written.json()["plan"]["title"] == "Plan Title"

        explicit_null = await client.post(
            url, headers=owner.headers, json={**sung, "instrumental": None}
        )
        assert explicit_null.status_code == 200, explicit_null.text  # null is omitted: carried over
        assert explicit_null.json()["created"] is False
