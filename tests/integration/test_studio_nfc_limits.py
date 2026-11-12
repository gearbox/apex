"""RF1 / RD6 / RD7 against real PostgreSQL: NFC expansion must never reach storage.

NFC can make a value longer than the wire allowed (U+0958 → 2 code points, U+FB2C → 3). Before
the post-normalization re-check, such values were stored: unreadable versions (500 on every
read), Postgres rejecting inserts (500), artist tag arrays silently truncated by the driver's
``varchar(60)[]`` cast, and ``Text`` columns accepting values past their cap.

Invariants 31 (no stored value exceeds its wire limit), 32 (every stored version decodes) and 33
(no length-capped column is ever truncated by the database). Rows are read back from the
database: "nothing was truncated" cannot be proven by expecting an exception.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import msgspec
import pytest
from sqlalchemy import func, select

from src.api.schemas.studio import (
    ArtistCreate,
    ArtistPatch,
    ProjectProfilePut,
    TrackPatch,
)
from src.core.enums import StudioDocumentKind, StudioReleaseKind
from src.core.enums import StudioSectionKind as K
from src.db.models.studio import StudioArtist, StudioProjectProfile, StudioTrack
from src.studio import validation as validation_module
from src.studio.documents import CURRENT_SCHEMA_VERSION, PlanDocument, decode_stored
from src.studio.errors import StudioDocumentInvalid
from src.studio.services.version import DocumentVersionService
from src.studio.validation import normalize_text
from tests.studio_support import make_lyrics, make_plan, make_section

if TYPE_CHECKING:
    from uuid import UUID

    from tests.studio_world import StudioWorld

QA = "क़"  # NFC: 2 code points
SHIN = "שּׁ"  # NFC: 3 code points
TOO_LONG = "too_long_after_normalization"

pytestmark = pytest.mark.usefixtures("db_session")


def _issues(exc: pytest.ExceptionInfo[StudioDocumentInvalid]) -> set[tuple[str, str]]:
    return {(i.path, i.code) for i in exc.value.issues}


async def _artist_count(w: StudioWorld) -> int:
    return (
        await w.session.execute(
            select(func.count()).select_from(StudioArtist).where(StudioArtist.user_id == w.user_id)
        )
    ).scalar_one()


async def _stored(w: StudioWorld, artist_id: UUID, column: str) -> object:
    """A column straight from the database (a column select never serves the identity map)."""
    return (
        await w.session.execute(
            select(getattr(StudioArtist, column)).where(StudioArtist.id == artist_id)
        )
    ).scalar_one()


async def _track_titles(w: StudioWorld) -> list[str]:
    return list(
        (await w.session.execute(select(StudioTrack.title).where(StudioTrack.user_id == w.user_id)))
        .scalars()
        .all()
    )


# ---------------------------------------------------------------------------
# Documents (plan / lyrics versions)
# ---------------------------------------------------------------------------


class TestDocuments:
    async def test_plan_title_probe_is_now_a_422_and_writes_nothing(
        self, enabled_studio: StudioWorld
    ) -> None:
        # The round-2 probe, flipped: this used to create (201) a version whose title was 202
        # code points, after which every read of it raised msgspec.ValidationError (500).
        w = enabled_studio
        track = await w.new_track()
        before = await w.track_row(track)

        raw = msgspec.to_builtins(make_plan(make_section(K.VERSE)))
        raw["title"] = QA * 101
        plan = msgspec.convert(raw, PlanDocument)  # passes the wire limit (101 <= 200)

        with pytest.raises(StudioDocumentInvalid) as exc:
            await w.plan(track, plan)

        assert _issues(exc) == {("title", TOO_LONG)}
        assert await w.version_count(track) == 0
        after = await w.track_row(track)
        assert after.current_plan_version_id is None
        assert after.next_plan_version_number == before.next_plan_version_number

    async def test_plan_title_at_the_limit_after_expansion_is_stored_and_readable(
        self, enabled_studio: StudioWorld
    ) -> None:
        w = enabled_studio
        track = await w.new_track()
        out, created = await w.plan(track, make_plan(title=QA * 100))
        assert created
        assert len(out.version.content.title) == 200  # type: ignore[union-attr]
        # Every read path decodes it.
        read = await w.versions.get(out.version.id, w.user_id, w.product_id)
        assert len(read.content.title) == 200  # type: ignore[union-attr]
        detail = await w.tracks.get_detail(track, w.user_id, w.product_id)
        assert detail.current_plan is not None
        export = await w.exports.render_suno(track, w.user_id, w.product_id)
        assert export.plan_version_id == out.version.id

    async def test_lyric_line_that_expands_past_300_is_refused(
        self, enabled_studio: StudioWorld
    ) -> None:
        w = enabled_studio
        track = await w.new_track()
        verse = make_section(K.VERSE)
        plan, _ = await w.plan(track, make_plan(verse))
        with pytest.raises(StudioDocumentInvalid) as exc:
            await w.lyrics(track, make_lyrics((verse.id, [SHIN * 101])), plan.version.id)
        assert _issues(exc) == {("sections[0].lines[0].text", TOO_LONG)}
        assert await w.version_count(track) == 1  # only the plan

    async def test_lyric_line_at_the_limit_after_expansion_is_stored_and_readable(
        self, enabled_studio: StudioWorld
    ) -> None:
        w = enabled_studio
        track = await w.new_track()
        verse = make_section(K.VERSE)
        plan, _ = await w.plan(track, make_plan(verse))
        out, _ = await w.lyrics(track, make_lyrics((verse.id, [SHIN * 100])), plan.version.id)
        read = await w.versions.get(out.version.id, w.user_id, w.product_id)
        assert len(read.content.sections[0].lines[0].text) == 300  # type: ignore[union-attr]

    async def test_section_style_tag_that_expands_past_60_is_refused(
        self, enabled_studio: StudioWorld
    ) -> None:
        w = enabled_studio
        track = await w.new_track()
        plan = make_plan(make_section(K.VERSE), make_section(K.CHORUS, tags=("ok", QA * 31)))
        with pytest.raises(StudioDocumentInvalid) as exc:
            await w.plan(track, plan)
        assert _issues(exc) == {("sections[1].style.tags[1]", TOO_LONG)}
        assert await w.version_count(track) == 0

    async def test_the_violation_is_collected_with_every_other_issue(
        self, enabled_studio: StudioWorld
    ) -> None:
        w = enabled_studio
        track = await w.new_track()
        plan = make_plan(make_section(K.VERSE, tags=("a", "A")), title=QA * 101, idea=SHIN * 1334)
        with pytest.raises(StudioDocumentInvalid) as exc:
            await w.plan(track, plan)
        assert _issues(exc) == {
            ("title", TOO_LONG),
            ("idea", TOO_LONG),
            ("sections[0].style.tags[1]", "duplicate_tag"),
        }


# ---------------------------------------------------------------------------
# RD7: every stored version decodes
# ---------------------------------------------------------------------------


class TestStoredVersionsDecode:
    async def test_a_value_the_length_check_missed_is_never_written(
        self, enabled_studio: StudioWorld, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Simulate the bug RD7 backstops: the RD6 re-check misses a field.
        monkeypatch.setattr(validation_module, "collect_length_issues", lambda *_a, **_k: None)
        w = enabled_studio
        track = await w.new_track()
        before = await w.track_row(track)

        with pytest.raises(RuntimeError, match="would not decode"):
            await w.plan(track, make_plan(title=QA * 101))

        assert await w.version_count(track) == 0
        after = await w.track_row(track)
        assert after.current_plan_version_id is None
        # Decoded before a number is allocated: nothing was burned either.
        assert after.next_plan_version_number == before.next_plan_version_number

    async def test_a_replay_is_not_decoded_again(
        self, enabled_studio: StudioWorld, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Nothing is written on a replay, so there is nothing to check (and no extra work).
        w = enabled_studio
        track = await w.new_track()
        doc = make_plan(title="same")
        first, created = await w.plan(track, doc)
        assert created

        calls: list[object] = []
        original = DocumentVersionService._assert_decodes

        def spy(kind: StudioDocumentKind, content: dict[str, object]) -> None:
            calls.append(kind)
            original(kind, content)

        monkeypatch.setattr(DocumentVersionService, "_assert_decodes", staticmethod(spy))
        again, created_again = await w.plan(track, doc)
        assert (again.version.id, created_again) == (first.version.id, False)
        assert calls == []

    async def test_what_is_stored_decodes_with_the_read_path(
        self, enabled_studio: StudioWorld
    ) -> None:
        w = enabled_studio
        track = await w.new_track()
        out, _ = await w.plan(track, make_plan(title=QA * 100, tags=(SHIN * 20,)))
        row = await w.version_row(out.version.id)
        decoded = decode_stored(StudioDocumentKind(row.kind), row.schema_version, row.content)
        assert isinstance(decoded, PlanDocument)
        assert row.schema_version == CURRENT_SCHEMA_VERSION[StudioDocumentKind.PLAN]


# ---------------------------------------------------------------------------
# Tracks
# ---------------------------------------------------------------------------


class TestTrackTitle:
    async def test_create_is_a_422_not_a_500(self, enabled_studio: StudioWorld) -> None:
        w = enabled_studio
        with pytest.raises(StudioDocumentInvalid) as exc:
            await w.tracks.create(w.project_id, w.user_id, w.product_id, QA * 101)
        assert _issues(exc) == {("title", TOO_LONG)}
        assert await _track_titles(w) == []

    async def test_create_at_the_limit_after_expansion_is_stored_whole(
        self, enabled_studio: StudioWorld
    ) -> None:
        w = enabled_studio
        out = await w.tracks.create(w.project_id, w.user_id, w.product_id, QA * 100)
        assert len(out.title) == 200
        assert await _track_titles(w) == [normalize_text(QA * 100)]

    async def test_patch_is_a_422_and_the_title_is_unchanged(
        self, enabled_studio: StudioWorld
    ) -> None:
        w = enabled_studio
        track = await w.new_track("Original")
        with pytest.raises(StudioDocumentInvalid) as exc:
            await w.tracks.patch(track, w.user_id, w.product_id, TrackPatch(title=QA * 101))
        assert _issues(exc) == {("title", TOO_LONG)}
        assert await _track_titles(w) == ["Original"]

    async def test_a_blank_title_keeps_its_own_code(self, enabled_studio: StudioWorld) -> None:
        w = enabled_studio
        with pytest.raises(StudioDocumentInvalid) as exc:
            await w.tracks.create(w.project_id, w.user_id, w.product_id, "  \n ")
        assert _issues(exc) == {("title", "empty_after_normalization")}


# ---------------------------------------------------------------------------
# Artists (varchar(60)[] arrays are truncated silently by the driver; Text columns have no cap)
# ---------------------------------------------------------------------------


class TestArtistFields:
    async def test_style_tags_create_is_a_422_and_no_row_is_written(
        self, studio: StudioWorld
    ) -> None:
        w = studio
        with pytest.raises(StudioDocumentInvalid) as exc:
            await w.artists.create(
                w.user_id,
                w.product_id,
                ArtistCreate(name="Nova", style_tags=["ok", QA * 31]),
            )
        assert _issues(exc) == {("style_tags[1]", TOO_LONG)}
        assert await _artist_count(w) == 0  # not even a truncated one

    async def test_style_tags_at_the_limit_are_stored_whole_never_truncated(
        self, studio: StudioWorld
    ) -> None:
        w = studio
        tag = QA * 30  # 60 code points after NFC: exactly varchar(60)
        out = await w.artists.create(
            w.user_id, w.product_id, ArtistCreate(name="Nova", style_tags=[tag])
        )
        stored = await _stored(w, out.id, "style_tags")
        assert stored == [normalize_text(tag)]
        assert len(stored[0]) == 60  # type: ignore[index]

    @pytest.mark.parametrize(
        ("field", "value", "path"),
        [
            ("persona", QA * 2001, "persona"),  # Text column: no database backstop
            ("voice_notes", QA * 1001, "voice_notes"),
            ("aesthetic_references", [QA * 101], "aesthetic_references[0]"),
            ("global_avoid", [SHIN * 21], "global_avoid[0]"),
        ],
    )
    async def test_every_other_artist_field_is_rechecked_on_create(
        self, studio: StudioWorld, field: str, value: object, path: str
    ) -> None:
        w = studio
        with pytest.raises(StudioDocumentInvalid) as exc:
            await w.artists.create(
                w.user_id,
                w.product_id,
                msgspec.structs.replace(ArtistCreate(name="Nova"), **{field: value}),
            )
        assert _issues(exc) == {(path, TOO_LONG)}
        assert await _artist_count(w) == 0

    async def test_name_that_expands_past_100_has_the_too_long_code(
        self, studio: StudioWorld
    ) -> None:
        w = studio
        with pytest.raises(StudioDocumentInvalid) as exc:
            await w.artists.create(w.user_id, w.product_id, ArtistCreate(name=QA * 51))
        assert _issues(exc) == {("name", TOO_LONG)}
        assert await _artist_count(w) == 0

    async def test_a_blank_name_keeps_its_own_code(self, studio: StudioWorld) -> None:
        w = studio
        with pytest.raises(StudioDocumentInvalid) as exc:
            await w.artists.create(w.user_id, w.product_id, ArtistCreate(name=" \t "))
        assert _issues(exc) == {("name", "empty_after_normalization")}

    async def test_create_collects_every_issue_at_once(self, studio: StudioWorld) -> None:
        w = studio
        with pytest.raises(StudioDocumentInvalid) as exc:
            await w.artists.create(
                w.user_id,
                w.product_id,
                ArtistCreate(
                    name=QA * 51,
                    persona=QA * 2001,
                    style_tags=["a", "A"],
                    global_avoid=["a"],
                ),
            )
        assert _issues(exc) == {
            ("name", TOO_LONG),
            ("persona", TOO_LONG),
            ("style_tags[1]", "duplicate_tag"),
            ("global_avoid[0]", "tag_conflict"),
        }

    async def test_patch_is_a_422_and_the_stored_row_is_untouched(
        self, studio: StudioWorld
    ) -> None:
        w = studio
        artist = await w.artists.create(
            w.user_id,
            w.product_id,
            ArtistCreate(name="Nova", persona="p", style_tags=["ok"], global_avoid=["bad"]),
        )
        with pytest.raises(StudioDocumentInvalid) as exc:
            await w.artists.patch(
                artist.id,
                w.user_id,
                w.product_id,
                ArtistPatch(style_tags=[QA * 31], persona=QA * 2001),
            )
        assert _issues(exc) == {("style_tags[0]", TOO_LONG), ("persona", TOO_LONG)}
        assert await _stored(w, artist.id, "style_tags") == ["ok"]
        assert await _stored(w, artist.id, "global_avoid") == ["bad"]
        assert await _stored(w, artist.id, "persona") == "p"

    async def test_patch_checks_only_what_the_client_sent(self, studio: StudioWorld) -> None:
        # A stored value is never re-judged by an unrelated PATCH (it was checked on write).
        w = studio
        artist = await w.artists.create(w.user_id, w.product_id, ArtistCreate(name="Nova"))
        out = await w.artists.patch(
            artist.id, w.user_id, w.product_id, ArtistPatch(persona=QA * 2000)
        )
        assert len(out.persona or "") == 4000

    async def test_patch_name_that_expands_past_100(self, studio: StudioWorld) -> None:
        w = studio
        artist = await w.artists.create(w.user_id, w.product_id, ArtistCreate(name="Nova"))
        with pytest.raises(StudioDocumentInvalid) as exc:
            await w.artists.patch(artist.id, w.user_id, w.product_id, ArtistPatch(name=QA * 51))
        assert _issues(exc) == {("name", TOO_LONG)}
        assert await _stored(w, artist.id, "name") == "Nova"


# ---------------------------------------------------------------------------
# Project profile concept
# ---------------------------------------------------------------------------


class TestProfileConcept:
    async def test_concept_that_expands_past_8000_is_a_422_and_not_stored(
        self, studio: StudioWorld
    ) -> None:
        w = studio
        with pytest.raises(StudioDocumentInvalid) as exc:
            await w.projects.upsert_profile(
                w.project_id,
                w.user_id,
                w.product_id,
                ProjectProfilePut(release_kind=StudioReleaseKind.EP, concept=QA * 4001),
            )
        assert _issues(exc) == {("concept", TOO_LONG)}
        rows = (
            await w.session.execute(
                select(func.count())
                .select_from(StudioProjectProfile)
                .where(StudioProjectProfile.library_project_id == w.project_id)
            )
        ).scalar_one()
        assert rows == 0

    async def test_concept_at_the_limit_after_expansion_is_stored(
        self, studio: StudioWorld
    ) -> None:
        w = studio
        out, created = await w.projects.upsert_profile(
            w.project_id,
            w.user_id,
            w.product_id,
            ProjectProfilePut(release_kind=StudioReleaseKind.EP, concept=QA * 4000),
        )
        assert created
        assert len(out.concept or "") == 8000
