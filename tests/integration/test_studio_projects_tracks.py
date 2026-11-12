"""Studio profiles, tracks and cross-tenant isolation (invariants 2, 3, 15) at the service level."""

from __future__ import annotations

from typing import TYPE_CHECKING
from uuid import uuid4

import msgspec
import pytest
from sqlalchemy.exc import IntegrityError

from src.api.schemas.studio import ArtistCreate, ProjectProfilePut, TrackPatch
from src.core.enums import StudioReleaseKind
from src.studio.errors import (
    ArtistNotFound,
    InvalidCursor,
    ProjectHasTracks,
    StudioDocumentInvalid,
    StudioNotFound,
    TrackLimitReached,
    TrackOrderMismatch,
)
from src.studio.services import track as track_module
from tests.studio_support import make_lyrics, make_plan, make_section
from tests.studio_world import StudioWorld

if TYPE_CHECKING:
    from uuid import UUID


async def _ids(w: StudioWorld) -> list[UUID]:
    return [t.id for t in await w.tracks.list_for_project(w.project_id, w.user_id, w.product_id)]


class TestProfiles:
    async def test_put_creates_then_replaces(self, studio: StudioWorld) -> None:
        w = studio
        body = ProjectProfilePut(release_kind=StudioReleaseKind.EP, concept="  a   concept ")
        created, was_created = await w.projects.upsert_profile(
            w.project_id, w.user_id, w.product_id, body
        )
        assert was_created is True
        assert (created.release_kind, created.concept, created.track_count) == (
            StudioReleaseKind.EP,
            "a   concept",
            0,
        )
        assert created.name == w.library_project.name

        replaced, was_created = await w.projects.upsert_profile(
            w.project_id,
            w.user_id,
            w.product_id,
            ProjectProfilePut(release_kind=StudioReleaseKind.ALBUM),
        )
        assert was_created is False
        assert (replaced.release_kind, replaced.concept) == (
            StudioReleaseKind.ALBUM,
            None,
        )  # PUT replaces
        assert replaced.created_at == created.created_at

    async def test_only_an_owned_library_project_can_be_enabled(
        self, studio: StudioWorld, make_user, make_library_project
    ) -> None:
        w = studio
        stranger = await make_user(email=f"s-{uuid4().hex[:6]}@example.com")
        theirs = await make_library_project(user=stranger)
        body = ProjectProfilePut(release_kind=StudioReleaseKind.SINGLE)
        with pytest.raises(StudioNotFound):
            await w.projects.upsert_profile(theirs.id, w.user_id, w.product_id, body)
        with pytest.raises(StudioNotFound):
            await w.projects.upsert_profile(uuid4(), w.user_id, w.product_id, body)
        with pytest.raises(StudioNotFound):  # right user, wrong product
            await w.projects.upsert_profile(w.project_id, w.user_id, "synthara", body)

    async def test_the_artist_must_be_the_callers(self, studio: StudioWorld, make_user) -> None:
        w = studio
        stranger = await make_user(email=f"s-{uuid4().hex[:6]}@example.com")
        theirs = await w.artists.create(stranger.id, w.product_id, ArtistCreate(name="Theirs"))
        with pytest.raises(ArtistNotFound):
            await w.projects.upsert_profile(
                w.project_id,
                w.user_id,
                w.product_id,
                ProjectProfilePut(release_kind=StudioReleaseKind.SINGLE, artist_id=theirs.id),
            )
        with pytest.raises(ArtistNotFound):
            await w.projects.upsert_profile(
                w.project_id,
                w.user_id,
                w.product_id,
                ProjectProfilePut(release_kind=StudioReleaseKind.SINGLE, artist_id=uuid4()),
            )
        mine = await w.artists.create(w.user_id, w.product_id, ArtistCreate(name="Mine"))
        out, _ = await w.projects.upsert_profile(
            w.project_id,
            w.user_id,
            w.product_id,
            ProjectProfilePut(release_kind=StudioReleaseKind.SINGLE, artist_id=mine.id),
        )
        assert out.artist_id == mine.id

    @staticmethod
    def _vanishes_after_first_read(service: object, attribute: str, monkeypatch) -> None:
        """Make ``service.<attribute>.get`` find its row once, then report it gone (a race)."""
        repo = getattr(service, attribute)
        real = repo.get
        calls = 0

        async def vanishing(*args: object, **kwargs: object) -> object:
            nonlocal calls
            calls += 1
            return await real(*args, **kwargs) if calls == 1 else None

        monkeypatch.setattr(repo, "get", vanishing)

    @staticmethod
    def _foreign_key_failure(w: StudioWorld, monkeypatch) -> None:
        async def refuse(*_args: object, **_kwargs: object) -> object:
            raise IntegrityError("INSERT studio_project_profiles", {}, Exception("fk"))

        monkeypatch.setattr(w.projects._profiles, "upsert", refuse)

    async def test_a_library_project_deleted_mid_request_is_not_found(
        self, studio: StudioWorld, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The FK failure used to be reported as artist_not_found whatever had vanished.
        w = studio
        self._vanishes_after_first_read(w.projects, "_library_projects", monkeypatch)
        self._foreign_key_failure(w, monkeypatch)
        with pytest.raises(StudioNotFound):
            await w.projects.upsert_profile(
                w.project_id,
                w.user_id,
                w.product_id,
                ProjectProfilePut(release_kind=StudioReleaseKind.SINGLE),
            )

    async def test_an_artist_deleted_mid_request_is_artist_not_found(
        self, studio: StudioWorld, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        w = studio
        artist = await w.artists.create(w.user_id, w.product_id, ArtistCreate(name="Fleeting"))
        self._vanishes_after_first_read(w.projects, "_artists", monkeypatch)
        self._foreign_key_failure(w, monkeypatch)
        with pytest.raises(ArtistNotFound):
            await w.projects.upsert_profile(
                w.project_id,
                w.user_id,
                w.product_id,
                ProjectProfilePut(release_kind=StudioReleaseKind.SINGLE, artist_id=artist.id),
            )

    async def test_an_unexplained_integrity_error_propagates(
        self, studio: StudioWorld, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Neither the project nor the artist is gone: do not guess a code, let it surface.
        w = studio
        artist = await w.artists.create(w.user_id, w.product_id, ArtistCreate(name="Steady"))
        self._foreign_key_failure(w, monkeypatch)
        with pytest.raises(IntegrityError):
            await w.projects.upsert_profile(
                w.project_id,
                w.user_id,
                w.product_id,
                ProjectProfilePut(release_kind=StudioReleaseKind.SINGLE, artist_id=artist.id),
            )
        with pytest.raises(IntegrityError):  # and without an artist at all
            await w.projects.upsert_profile(
                w.project_id,
                w.user_id,
                w.product_id,
                ProjectProfilePut(release_kind=StudioReleaseKind.SINGLE),
            )

    async def test_reads_and_pages(self, enabled_studio: StudioWorld, make_library_project) -> None:
        w = enabled_studio
        await w.new_track()
        await w.new_track()
        got = await w.projects.get_profile(w.project_id, w.user_id, w.product_id)
        assert got.track_count == 2

        second = StudioWorld(w.session, w.user, await make_library_project(user=w.user))
        await second.enable(kind=StudioReleaseKind.SINGLE)
        page = await w.projects.list_profiles(w.user_id, w.product_id, limit=1, cursor=None)
        assert (len(page.items), page.has_more) == (1, True)
        assert page.items[0].library_project_id == second.project_id  # newest first
        rest = await w.projects.list_profiles(
            w.user_id, w.product_id, limit=1, cursor=page.next_cursor
        )
        assert [p.library_project_id for p in rest.items] == [w.project_id]
        assert rest.items[0].track_count == 2
        with pytest.raises(InvalidCursor):
            await w.projects.list_profiles(w.user_id, w.product_id, limit=1, cursor="bad")

    async def test_unenabled_projects_have_no_profile(self, studio: StudioWorld) -> None:
        with pytest.raises(StudioNotFound):
            await studio.projects.get_profile(studio.project_id, studio.user_id, studio.product_id)
        page = await studio.projects.list_profiles(
            studio.user_id, studio.product_id, limit=10, cursor=None
        )
        assert page.items == []

    async def test_disable_is_refused_while_tracks_exist(self, enabled_studio: StudioWorld) -> None:
        w = enabled_studio
        track = await w.new_track()
        with pytest.raises(ProjectHasTracks):
            await w.projects.disable(w.project_id, w.user_id, w.product_id)
        await w.tracks.delete(track, w.user_id, w.product_id)
        await w.projects.disable(w.project_id, w.user_id, w.product_id)
        with pytest.raises(StudioNotFound):
            await w.projects.get_profile(w.project_id, w.user_id, w.product_id)
        with pytest.raises(StudioNotFound):  # nothing left to disable
            await w.projects.disable(w.project_id, w.user_id, w.product_id)
        # the library project itself is untouched, and studio can be enabled again
        await w.enable()


class TestTracks:
    async def test_tracks_are_appended_in_order(self, enabled_studio: StudioWorld) -> None:
        w = enabled_studio
        made = [
            await w.tracks.create(w.project_id, w.user_id, w.product_id, f" t{n} ")
            for n in range(3)
        ]
        assert [(t.title, t.position) for t in made] == [("t0", 0), ("t1", 1), ("t2", 2)]
        listed = await w.tracks.list_for_project(w.project_id, w.user_id, w.product_id)
        assert [t.id for t in listed] == [t.id for t in made]

    async def test_tracks_need_an_enabled_profile(self, studio: StudioWorld) -> None:
        with pytest.raises(StudioNotFound):
            await studio.tracks.create(studio.project_id, studio.user_id, studio.product_id, "x")
        with pytest.raises(StudioNotFound):
            await studio.tracks.list_for_project(
                studio.project_id, studio.user_id, studio.product_id
            )

    async def test_blank_titles_are_rejected(self, enabled_studio: StudioWorld) -> None:
        w = enabled_studio
        with pytest.raises(StudioDocumentInvalid) as exc:
            await w.tracks.create(w.project_id, w.user_id, w.product_id, "   ")
        assert exc.value.issues[0].path == "title"

    async def test_the_cap(
        self, enabled_studio: StudioWorld, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(track_module, "MAX_TRACKS_PER_PROJECT", 2)
        w = enabled_studio
        await w.new_track()
        await w.new_track()
        with pytest.raises(TrackLimitReached):
            await w.new_track()

    async def test_reorder_applies_exactly_the_given_order(
        self, enabled_studio: StudioWorld
    ) -> None:
        w = enabled_studio
        a, b, c = [await w.new_track(n) for n in "abc"]
        out = await w.tracks.reorder(w.project_id, w.user_id, w.product_id, [c, a, b])
        assert [(t.id, t.position) for t in out] == [(c, 0), (a, 1), (b, 2)]
        out = await w.tracks.reorder(
            w.project_id, w.user_id, w.product_id, [b, a, c]
        )  # swap round trip
        assert [t.id for t in out] == [b, a, c]
        assert [t.position for t in out] == [0, 1, 2]

    async def test_reorder_demands_the_exact_id_set(self, enabled_studio: StudioWorld) -> None:
        w = enabled_studio
        a, b, c = [await w.new_track(n) for n in "abc"]
        for bad in ([a, b], [a, b, c, uuid4()], [a, a, b], [a, b, uuid4()], []):
            with pytest.raises(TrackOrderMismatch):
                await w.tracks.reorder(w.project_id, w.user_id, w.product_id, bad)
        assert await _ids(w) == [a, b, c]  # nothing moved

    async def test_reorder_rejects_another_projects_track(
        self, enabled_studio: StudioWorld, make_library_project
    ) -> None:
        w = enabled_studio
        a = await w.new_track("a")
        other = StudioWorld(w.session, w.user, await make_library_project(user=w.user))
        await other.enable()
        theirs = await other.new_track("theirs")
        with pytest.raises(TrackOrderMismatch):
            await w.tracks.reorder(w.project_id, w.user_id, w.product_id, [a, theirs])

    async def test_delete_keeps_the_others_and_new_tracks_append_after_the_max(
        self, enabled_studio: StudioWorld
    ) -> None:
        w = enabled_studio
        a, b, c = [await w.new_track(n) for n in "abc"]
        await w.tracks.delete(b, w.user_id, w.product_id)
        d = await w.new_track("d")
        listed = await w.tracks.list_for_project(w.project_id, w.user_id, w.product_id)
        assert [(t.id, t.position) for t in listed] == [(a, 0), (c, 2), (d, 3)]
        compacted = await w.tracks.reorder(w.project_id, w.user_id, w.product_id, [d, c, a])
        assert [t.position for t in compacted] == [0, 1, 2]

    async def test_rename(self, enabled_studio: StudioWorld) -> None:
        w = enabled_studio
        t = await w.new_track("old")
        assert (
            await w.tracks.patch(t, w.user_id, w.product_id, TrackPatch(title="  new  name "))
        ).title == "new name"
        assert (await w.tracks.patch(t, w.user_id, w.product_id, TrackPatch())).title == "new name"
        with pytest.raises(StudioDocumentInvalid):
            await w.tracks.patch(t, w.user_id, w.product_id, TrackPatch(title=" "))

    async def test_track_detail_opens_a_track_with_its_current_documents(
        self, enabled_studio: StudioWorld
    ) -> None:
        w = enabled_studio
        t = await w.new_track()
        empty = await w.tracks.get_detail(t, w.user_id, w.product_id)
        assert (empty.current_plan, empty.current_lyrics) == (None, None)
        verse = make_section()
        plan, _ = await w.plan(t, make_plan(verse, title="Detail"))
        lyr, _ = await w.lyrics(t, make_lyrics((verse.id, ["hello"])), plan.version.id)
        detail = await w.tracks.get_detail(t, w.user_id, w.product_id)
        assert detail.current_plan is not None
        assert detail.current_plan.content.title == "Detail"
        assert detail.current_lyrics is not None
        assert detail.current_lyrics.id == lyr.version.id
        assert detail.current_plan_version_id == plan.version.id


class TestIsolation:
    """Invariant 2: another user's or product's ids are always 'not found'."""

    @pytest.fixture
    async def populated(self, enabled_studio: StudioWorld):
        w = enabled_studio
        verse = make_section()
        track = await w.new_track()
        plan, _ = await w.plan(track, make_plan(verse, title="Mine"))
        lyr, _ = await w.lyrics(track, make_lyrics((verse.id, ["x"])), plan.version.id)
        return w, track, plan.version.id, lyr.version.id

    @pytest.mark.parametrize("scope", ["other_user", "other_product"])
    async def test_every_operation_is_hidden(self, populated, scope: str) -> None:
        w, track, plan_id, lyrics_id = populated
        user_id = uuid4() if scope == "other_user" else w.user_id
        product = w.product_id if scope == "other_user" else "synthara"

        async def get_track() -> object:
            return await w.tracks.get_detail(track, user_id, product)

        calls = {
            "track.get": get_track,
            "track.patch": lambda: w.tracks.patch(track, user_id, product, TrackPatch(title="x")),
            "track.delete": lambda: w.tracks.delete(track, user_id, product),
            "tracks.list": lambda: w.tracks.list_for_project(w.project_id, user_id, product),
            "tracks.create": lambda: w.tracks.create(w.project_id, user_id, product, "x"),
            "tracks.reorder": lambda: w.tracks.reorder(w.project_id, user_id, product, [track]),
            "project.get": lambda: w.projects.get_profile(w.project_id, user_id, product),
            "project.disable": lambda: w.projects.disable(w.project_id, user_id, product),
            "versions.get": lambda: w.versions.get(plan_id, user_id, product),
            "versions.list": lambda: w.versions.list_page(
                track, None, user_id, product, limit=5, cursor=None
            ),
            "versions.delete": lambda: w.versions.delete(lyrics_id, user_id, product),
            "versions.set_current": lambda: w.versions.set_current(
                track, user_id, product, plan_version_id=plan_id, lyrics_version_id=msgspec.UNSET
            ),
            "export": lambda: w.exports.render_suno(track, user_id, product),
        }
        for name, call in calls.items():
            with pytest.raises(StudioNotFound):
                await call()  # type: ignore[operator]
            assert name  # (the parametrized failure message names the operation)

    async def test_creating_versions_on_a_foreign_track_is_hidden(self, populated) -> None:
        from src.api.schemas.studio import PlanVersionCreate

        w, track, *_ = populated
        with pytest.raises(StudioNotFound):
            await w.versions.create(
                track, PlanVersionCreate(content=make_plan()), uuid4(), w.product_id
            )
        with pytest.raises(StudioNotFound):
            await w.versions.create(
                track, PlanVersionCreate(content=make_plan()), w.user_id, "synthara"
            )

    async def test_nothing_leaked_or_changed(self, populated) -> None:
        w, track, *_ = populated
        assert await w.version_count(track) == 2
        assert len(await _ids(w)) == 1
