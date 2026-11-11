"""Artists against real PostgreSQL: uniqueness, the 50 cap, tri-state PATCH, artist_in_use."""

from __future__ import annotations

from typing import TYPE_CHECKING
from uuid import uuid4

import msgspec
import pytest

from src.api.schemas.studio import ArtistCreate, ArtistPatch, ProjectProfilePut
from src.core.enums import StudioReleaseKind
from src.studio.errors import (
    ArtistInUse,
    ArtistLimitReached,
    ArtistNameConflict,
    ArtistNotFound,
    InvalidCursor,
    StudioDocumentInvalid,
)
from src.studio.services import artist as artist_module

if TYPE_CHECKING:
    from tests.studio_world import StudioWorld

pytestmark = pytest.mark.usefixtures("db_session")


def _create(name: str = "Nova", **kwargs: object) -> ArtistCreate:
    return ArtistCreate(name=name, **kwargs)  # type: ignore[arg-type]


class TestCrud:
    async def test_create_normalizes_and_roundtrips(self, studio: StudioWorld) -> None:
        w = studio
        out = await w.artists.create(
            w.user_id,
            w.product_id,
            _create(
                "  Nova   Rae ",
                persona="  a  persona\r\nsecond line ",
                voice_notes="",
                default_language="en",
                aesthetic_references=[" 80s  Berlin "],
                style_tags=[" husky   vocals"],
                global_avoid=["autotune"],
            ),
        )
        assert out.name == "Nova Rae"
        assert out.persona == "a  persona\nsecond line"
        assert out.voice_notes is None
        assert out.aesthetic_references == ["80s Berlin"]
        assert out.style_tags == ["husky vocals"]
        assert out.global_avoid == ["autotune"]
        assert await w.artists.get(out.id, w.user_id, w.product_id) == out

    async def test_defaults_are_empty_lists(self, studio: StudioWorld) -> None:
        out = await studio.artists.create(studio.user_id, studio.product_id, _create())
        assert (out.aesthetic_references, out.style_tags, out.global_avoid) == ([], [], [])

    async def test_blank_name_after_normalization_is_a_document_issue(
        self, studio: StudioWorld
    ) -> None:
        with pytest.raises(StudioDocumentInvalid) as exc:
            await studio.artists.create(studio.user_id, studio.product_id, _create("   "))
        assert exc.value.issues[0].path == "name"

    async def test_names_are_unique_per_owner_case_insensitively(self, studio: StudioWorld) -> None:
        w = studio
        await w.artists.create(w.user_id, w.product_id, _create("Nova"))
        for variant in ("nova", "NOVA", "  Nova  "):
            with pytest.raises(ArtistNameConflict):
                await w.artists.create(w.user_id, w.product_id, _create(variant))
        # the session is still usable after the savepoint rollback
        assert (await w.artists.create(w.user_id, w.product_id, _create("Other"))).name == "Other"

    async def test_another_owner_may_reuse_a_name(self, studio: StudioWorld, make_user) -> None:
        w = studio
        await w.artists.create(w.user_id, w.product_id, _create("Nova"))
        other = await make_user(email=f"o-{uuid4().hex[:6]}@example.com")
        assert (await w.artists.create(other.id, w.product_id, _create("Nova"))).name == "Nova"

    async def test_the_cap_is_per_owner(
        self, studio: StudioWorld, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(artist_module, "MAX_ARTISTS_PER_OWNER", 2)
        w = studio
        for n in range(2):
            await w.artists.create(w.user_id, w.product_id, _create(f"a{n}"))
        with pytest.raises(ArtistLimitReached):
            await w.artists.create(w.user_id, w.product_id, _create("a3"))

    async def test_lookup_is_owner_and_product_scoped(self, studio: StudioWorld) -> None:
        w = studio
        out = await w.artists.create(w.user_id, w.product_id, _create())
        for user_id, product_id in ((uuid4(), w.product_id), (w.user_id, "synthara")):
            with pytest.raises(ArtistNotFound):
                await w.artists.get(out.id, user_id, product_id)
            with pytest.raises(ArtistNotFound):
                await w.artists.patch(out.id, user_id, product_id, ArtistPatch(name="x"))
            with pytest.raises(ArtistNotFound):
                await w.artists.delete(out.id, user_id, product_id)

    async def test_pages(self, studio: StudioWorld) -> None:
        w = studio
        for n in range(3):
            await w.artists.create(w.user_id, w.product_id, _create(f"a{n}"))
        page = await w.artists.list_page(w.user_id, w.product_id, limit=2, cursor=None)
        assert (len(page.items), page.has_more) == (2, True)
        rest = await w.artists.list_page(w.user_id, w.product_id, limit=2, cursor=page.next_cursor)
        assert (len(rest.items), rest.has_more) == (1, False)
        with pytest.raises(InvalidCursor):
            await w.artists.list_page(w.user_id, w.product_id, limit=2, cursor="nope")


class TestStyleValidation:
    async def test_style_tags_are_validated_like_a_style_spec(self, studio: StudioWorld) -> None:
        with pytest.raises(StudioDocumentInvalid) as exc:
            await studio.artists.create(
                studio.user_id,
                studio.product_id,
                _create(style_tags=["Soul", "soul", "[x]"], global_avoid=["SOUL"]),
            )
        assert {(i.path, i.code) for i in exc.value.issues} == {
            ("style_tags[1]", "duplicate_tag"),
            ("style_tags[2]", "reserved_characters"),
            ("global_avoid[0]", "tag_conflict"),
        }

    async def test_patch_validates_the_resulting_pair(self, studio: StudioWorld) -> None:
        w = studio
        out = await w.artists.create(w.user_id, w.product_id, _create(global_avoid=["Loud"]))
        with pytest.raises(StudioDocumentInvalid) as exc:  # clashes with the *stored* avoid list
            await w.artists.patch(out.id, w.user_id, w.product_id, ArtistPatch(style_tags=["loud"]))
        assert exc.value.issues[0].path == "global_avoid[0]"
        ok = await w.artists.patch(
            out.id, w.user_id, w.product_id, ArtistPatch(style_tags=["calm"])
        )
        assert (ok.style_tags, ok.global_avoid) == (["calm"], ["Loud"])


class TestTriStatePatch:
    async def test_omitted_fields_are_kept_and_null_clears_nullable_ones(
        self, studio: StudioWorld
    ) -> None:
        w = studio
        out = await w.artists.create(
            w.user_id,
            w.product_id,
            _create(persona="p", voice_notes="v", default_language="en", style_tags=["t"]),
        )
        same = await w.artists.patch(out.id, w.user_id, w.product_id, ArtistPatch())
        assert same == out
        cleared = await w.artists.patch(
            out.id, w.user_id, w.product_id, ArtistPatch(persona=None, default_language=None)
        )
        assert (cleared.persona, cleared.default_language) == (None, None)
        assert (cleared.voice_notes, cleared.style_tags) == ("v", ["t"])  # untouched
        renamed = await w.artists.patch(
            out.id, w.user_id, w.product_id, ArtistPatch(name=" New  Name ")
        )
        assert renamed.name == "New Name"
        assert renamed.updated_at >= out.updated_at

    async def test_rename_conflicts_are_409_and_leave_the_session_usable(
        self, studio: StudioWorld
    ) -> None:
        w = studio
        await w.artists.create(w.user_id, w.product_id, _create("Taken"))
        mine = await w.artists.create(w.user_id, w.product_id, _create("Mine"))
        with pytest.raises(ArtistNameConflict):
            await w.artists.patch(mine.id, w.user_id, w.product_id, ArtistPatch(name="taken"))
        assert (await w.artists.get(mine.id, w.user_id, w.product_id)).name == "Mine"

    def test_wire_decoding_distinguishes_null_from_omitted(self) -> None:
        assert msgspec.json.decode(b"{}", type=ArtistPatch).persona is msgspec.UNSET
        assert msgspec.json.decode(b'{"persona": null}', type=ArtistPatch).persona is None


class TestArtistInUse:
    async def test_delete_unused_then_gone(self, studio: StudioWorld) -> None:
        w = studio
        out = await w.artists.create(w.user_id, w.product_id, _create())
        await w.artists.delete(out.id, w.user_id, w.product_id)
        with pytest.raises(ArtistNotFound):
            await w.artists.get(out.id, w.user_id, w.product_id)

    async def test_an_artist_used_by_a_profile_cannot_be_deleted(self, studio: StudioWorld) -> None:
        w = studio
        out = await w.artists.create(w.user_id, w.product_id, _create())
        await w.enable(artist_id=out.id)
        with pytest.raises(ArtistInUse) as exc:
            await w.artists.delete(out.id, w.user_id, w.product_id)
        assert exc.value.detail == {"project_count": 1}
        assert (await w.artists.get(out.id, w.user_id, w.product_id)).id == out.id

    async def test_deleting_works_again_once_the_profile_lets_go(self, studio: StudioWorld) -> None:
        w = studio
        out = await w.artists.create(w.user_id, w.product_id, _create())
        await w.enable(artist_id=out.id)
        await w.projects.upsert_profile(
            w.project_id,
            w.user_id,
            w.product_id,
            ProjectProfilePut(release_kind=StudioReleaseKind.SINGLE),
        )
        await w.artists.delete(out.id, w.user_id, w.product_id)

    async def test_the_fk_race_is_mapped_to_artist_in_use(
        self, studio: StudioWorld, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        w = studio
        out = await w.artists.create(w.user_id, w.product_id, _create())
        await w.enable(artist_id=out.id)

        repo = w.artists._repo
        real = repo.count_profiles_using
        calls = 0

        async def stale_once(*args: object, **kwargs: object) -> int:
            nonlocal calls
            calls += 1
            return 0 if calls == 1 else await real(*args, **kwargs)  # type: ignore[arg-type]

        monkeypatch.setattr(repo, "count_profiles_using", stale_once)
        with pytest.raises(ArtistInUse) as exc:  # the count said 0; the NO ACTION FK said no
            await w.artists.delete(out.id, w.user_id, w.product_id)
        assert exc.value.detail == {"project_count": 1}
        assert (await w.artists.get(out.id, w.user_id, w.product_id)).id == out.id  # session intact
