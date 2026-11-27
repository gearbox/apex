"""Track audio over HTTP: real controllers, guards, DI, handlers and PostgreSQL (S2a, W8).

Rows are committed and the app opens its own session per request (see ``build_app``): a 409 leaves
nothing behind, a 428 never reaches the service, the studio kill switch answers 404 first.
"""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any
from uuid import uuid4

import pytest
import pytest_asyncio
from litestar.testing import AsyncTestClient
from sqlalchemy.ext.asyncio import async_sessionmaker

from src.api.routes.studio_track_audio import StudioTrackAudioController
from src.api.routes.studio_tracks import StudioTrackController
from src.core.product_registry import VEX_CONFIG
from src.studio.constraints import MAX_AUDIO_VERSIONS_PER_TRACK
from tests.library_holds_support import HoldWorld, bearer, build_app
from tests.track_audio_support import assert_audio_reconciled, committed_track

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator
    from uuid import UUID

    from litestar import Litestar
    from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession

    from src.db.models.user import User


@pytest.fixture(autouse=True)
def studio_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "src.api.security.studio.get_settings", lambda: SimpleNamespace(studio_enabled=True)
    )


@pytest.fixture
def factory(db_engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(bind=db_engine, expire_on_commit=False)


@pytest_asyncio.fixture
async def client(
    factory: async_sessionmaker[AsyncSession],
) -> AsyncGenerator[AsyncTestClient[Litestar]]:
    app = build_app(factory, [StudioTrackAudioController, StudioTrackController])
    async with AsyncTestClient(app=app) as test_client:
        yield test_client


@pytest_asyncio.fixture
async def checked(hold_world: HoldWorld) -> AsyncGenerator[HoldWorld]:
    """The hold world; every test ends with the reconciliation check (invariant 4)."""
    yield hold_world
    async with hold_world.factory() as session:
        await assert_audio_reconciled(session)


async def _attach(
    client: AsyncTestClient[Litestar], user: User, track_id: UUID, asset_ref: str, **body: Any
) -> dict[str, Any]:
    response = await client.post(
        f"/v1/studio/tracks/{track_id}/audio",
        headers=bearer(user.id),
        json={"asset": asset_ref, **body},
    )
    assert response.status_code == 201, response.text
    return response.json()  # type: ignore[no-any-return]


class TestLifecycleOverHttp:
    async def test_attach_list_read_edit_switch_and_delete(
        self, client: AsyncTestClient[Litestar], checked: HoldWorld
    ) -> None:
        user = await checked.user()
        headers = bearer(user.id)
        track_id = await committed_track(checked, user)
        first = await checked.audio(user, duration_ms=120_000)
        second = await checked.audio(user, duration_ms=95_500)

        created = await _attach(client, user, track_id, f"upload:{first.id}", label=" Demo  1 ")
        again = await _attach(client, user, track_id, f"upload:{second.id}")

        assert created["version_number"] == 1
        assert again["version_number"] == 2
        assert set(created) == {
            "id",
            "track_id",
            "version_number",
            "label",
            "origin",
            "asset",
            "duration_ms",
            "plan_version_id",
            "lyrics_version_id",
            "is_current",
            # S3a: the realized-timeline facts
            "current_timeline_id",
            "timeline_count",
            "timeline_complete",
            "created_at",
            "updated_at",
        }
        assert created["current_timeline_id"] is None
        assert created["timeline_count"] == 0
        assert created["timeline_complete"] is None
        assert created["label"] == "Demo 1"
        assert created["origin"] == "upload"
        assert created["duration_ms"] == 120_000
        assert created["is_current"] is False
        assert created["track_id"] == str(track_id)
        assert created["asset"]["media_type"] == "audio"
        assert created["asset"]["asset_ref"] == f"upload:{first.id}"
        assert created["asset"]["original"]["content_type"] == "audio/flac"
        assert created["asset"]["variants"] == []

        listed = await client.get(f"/v1/studio/tracks/{track_id}/audio", headers=headers)
        assert listed.status_code == 200
        page = listed.json()
        assert [item["version_number"] for item in page["items"]] == [2, 1]
        assert page["has_more"] is False
        assert page["next_cursor"] is None

        one = await client.get(f"/v1/studio/audio/{created['id']}", headers=headers)
        assert one.status_code == 200
        assert one.json()["id"] == created["id"]

        patched = await client.patch(
            f"/v1/studio/audio/{created['id']}", headers=headers, json={"label": "Final"}
        )
        assert patched.status_code == 200
        assert patched.json()["label"] == "Final"

        switched = await client.put(
            f"/v1/studio/tracks/{track_id}/current",
            headers=headers,
            json={"audio_version_id": created["id"]},
        )
        assert switched.status_code == 200
        assert switched.json()["current_audio_version_id"] == created["id"]
        detail = await client.get(f"/v1/studio/tracks/{track_id}", headers=headers)
        assert detail.status_code == 200
        assert detail.json()["current_audio"]["id"] == created["id"]
        assert detail.json()["current_audio"]["is_current"] is True

        refused = await client.delete(f"/v1/studio/audio/{created['id']}", headers=headers)
        assert refused.status_code == 409
        assert refused.json()["error"] == "audio_version_is_current"

        cleared = await client.put(
            f"/v1/studio/tracks/{track_id}/current",
            headers=headers,
            json={"audio_version_id": None},
        )
        assert cleared.status_code == 200
        assert cleared.json()["current_audio_version_id"] is None
        gone = await client.delete(f"/v1/studio/audio/{created['id']}", headers=headers)
        assert gone.status_code == 204
        assert (
            await client.get(f"/v1/studio/audio/{created['id']}", headers=headers)
        ).status_code == 404

    async def test_paging_with_limit_and_cursor(
        self, client: AsyncTestClient[Litestar], checked: HoldWorld
    ) -> None:
        user = await checked.user()
        track_id = await committed_track(checked, user)
        for _ in range(5):
            await _attach(client, user, track_id, f"upload:{(await checked.audio(user)).id}")
        seen: list[int] = []
        cursor: str | None = None

        for _ in range(5):
            url = f"/v1/studio/tracks/{track_id}/audio?limit=2" + (
                f"&cursor={cursor}" if cursor else ""
            )
            body = (await client.get(url, headers=bearer(user.id))).json()
            seen.extend(item["version_number"] for item in body["items"])
            cursor = body["next_cursor"]
            if cursor is None:
                break

        assert seen == [5, 4, 3, 2, 1]

    async def test_a_bad_cursor_is_400_and_a_bad_limit_is_rejected(
        self, client: AsyncTestClient[Litestar], checked: HoldWorld
    ) -> None:
        user = await checked.user()
        track_id = await committed_track(checked, user)
        base = f"/v1/studio/tracks/{track_id}/audio"

        bad_cursor = await client.get(f"{base}?cursor=nope", headers=bearer(user.id))
        too_many = await client.get(f"{base}?limit=51", headers=bearer(user.id))
        zero = await client.get(f"{base}?limit=0", headers=bearer(user.id))

        assert bad_cursor.status_code == 400
        assert bad_cursor.json()["error"] == "invalid_cursor"
        assert too_many.status_code == 400
        assert zero.status_code == 400

    async def test_links_and_version_deletion_over_http(
        self, client: AsyncTestClient[Litestar], checked: HoldWorld
    ) -> None:
        """A realized plan version cannot be deleted (409 version_has_audio with its count)."""
        user = await checked.user()
        headers = bearer(user.id)
        track_id = await committed_track(checked, user)
        audio = await checked.audio(user)
        section = {
            "id": str(uuid4()),
            "kind": "verse",
            "label": None,
            "duration_seconds": None,
            "style": {"tags": [], "avoid": []},
            "delivery": [],
            "notes": None,
        }
        plan = {
            "kind": "plan",
            "content": {
                "title": "Song",
                "tempo_bpm": None,
                "global_style": {"tags": [], "avoid": []},
                "idea": None,
                "references": [],
                "instrumental": False,
                "sections": [section],
            },
            "make_current": False,
        }
        # two plan versions: the first is the one the audio realizes, the second is current
        first = await client.post(
            f"/v1/studio/tracks/{track_id}/versions", headers=headers, json=plan
        )
        assert first.status_code == 201, first.text
        plan["content"]["title"] = "Song two"  # type: ignore[index]
        await client.post(f"/v1/studio/tracks/{track_id}/versions", headers=headers, json=plan)
        plan_id = first.json()["version"]["id"]

        audio_body = await _attach(
            client, user, track_id, f"upload:{audio.id}", plan_version_id=plan_id
        )
        assert audio_body["plan_version_id"] == plan_id
        blocked = await client.delete(f"/v1/studio/versions/{plan_id}", headers=headers)
        assert blocked.status_code == 409
        assert blocked.json()["error"] == "version_has_audio"
        assert blocked.json()["detail"] == {"count": 1}

        unlinked = await client.patch(
            f"/v1/studio/audio/{audio_body['id']}", headers=headers, json={"plan_version_id": None}
        )
        assert unlinked.status_code == 200
        assert unlinked.json()["plan_version_id"] is None
        deleted = await client.delete(f"/v1/studio/versions/{plan_id}", headers=headers)
        assert deleted.status_code == 204


class TestErrorsOverHttp:
    async def test_an_unknown_or_foreign_track_is_404_not_found(
        self, client: AsyncTestClient[Litestar], checked: HoldWorld
    ) -> None:
        user = await checked.user()
        other = await checked.user()
        theirs = await committed_track(checked, other)
        asset = await checked.audio(user)

        for track_id in (uuid4(), theirs):
            attach = await client.post(
                f"/v1/studio/tracks/{track_id}/audio",
                headers=bearer(user.id),
                json={"asset": f"upload:{asset.id}"},
            )
            listed = await client.get(
                f"/v1/studio/tracks/{track_id}/audio", headers=bearer(user.id)
            )
            assert (attach.status_code, listed.status_code) == (404, 404)
            assert attach.json()["error"] == "not_found"

    async def test_an_unknown_or_foreign_audio_version_is_404(
        self, client: AsyncTestClient[Litestar], checked: HoldWorld
    ) -> None:
        user = await checked.user()
        other = await checked.user()
        track = await committed_track(checked, other)
        theirs = await _attach(client, other, track, f"upload:{(await checked.audio(other)).id}")

        for audio_id in (uuid4(), theirs["id"]):
            for method, body in (("get", None), ("patch", {"label": "x"}), ("delete", None)):
                response = await getattr(client, method)(
                    f"/v1/studio/audio/{audio_id}",
                    headers=bearer(user.id),
                    **({"json": body} if body else {}),
                )
                assert response.status_code == 404, (method, response.text)
                assert response.json()["error"] == "audio_version_not_found"

    async def test_an_asset_that_is_not_audio_is_422_asset_not_eligible(
        self, client: AsyncTestClient[Litestar], checked: HoldWorld
    ) -> None:
        user = await checked.user()
        track_id = await committed_track(checked, user)
        image = await checked.upload(user)

        response = await client.post(
            f"/v1/studio/tracks/{track_id}/audio",
            headers=bearer(user.id),
            json={"asset": f"upload:{image.id}"},
        )

        assert response.status_code == 422
        assert response.json()["error"] == "asset_not_eligible"
        assert response.json()["detail"] == {"reason": "media_kind_not_accepted"}

    async def test_someone_else_s_or_a_missing_asset_is_404_asset_not_found(
        self, client: AsyncTestClient[Litestar], checked: HoldWorld
    ) -> None:
        user = await checked.user()
        other = await checked.user()
        track_id = await committed_track(checked, user)
        theirs = await checked.audio(other)

        for ref in (f"upload:{theirs.id}", f"upload:{uuid4()}"):
            response = await client.post(
                f"/v1/studio/tracks/{track_id}/audio", headers=bearer(user.id), json={"asset": ref}
            )
            assert response.status_code == 404
            assert response.json()["error"] == "asset_not_found"

    @pytest.mark.parametrize("ref", ["nope", "image:abc", "upload:not-a-uuid"])
    async def test_a_malformed_asset_ref_is_a_422_document_issue(
        self, client: AsyncTestClient[Litestar], checked: HoldWorld, ref: str
    ) -> None:
        user = await checked.user()
        track_id = await committed_track(checked, user)

        response = await client.post(
            f"/v1/studio/tracks/{track_id}/audio", headers=bearer(user.id), json={"asset": ref}
        )

        assert response.status_code == 422
        assert response.json()["error"] == "document_invalid"
        assert response.json()["detail"]["issues"] == [
            {"path": "asset", "code": "invalid_asset_ref"}
        ]

    async def test_the_cap_is_a_409_with_its_limit(
        self, client: AsyncTestClient[Litestar], checked: HoldWorld
    ) -> None:
        user = await checked.user()
        track_id = await committed_track(checked, user)
        for _ in range(MAX_AUDIO_VERSIONS_PER_TRACK):
            await _attach(client, user, track_id, f"upload:{(await checked.audio(user)).id}")
        extra = await checked.audio(user)

        response = await client.post(
            f"/v1/studio/tracks/{track_id}/audio",
            headers=bearer(user.id),
            json={"asset": f"upload:{extra.id}"},
        )

        assert response.status_code == 409
        assert response.json()["error"] == "audio_version_limit_reached"
        assert response.json()["detail"] == {"limit": 20}

    async def test_a_link_to_another_track_is_404_and_to_the_wrong_kind_is_422(
        self, client: AsyncTestClient[Litestar], checked: HoldWorld
    ) -> None:
        user = await checked.user()
        track_id = await committed_track(checked, user)
        asset = await checked.audio(user)

        unknown = await client.post(
            f"/v1/studio/tracks/{track_id}/audio",
            headers=bearer(user.id),
            json={"asset": f"upload:{asset.id}", "plan_version_id": str(uuid4())},
        )

        assert unknown.status_code == 404
        assert unknown.json()["error"] == "version_not_found"

    async def test_the_current_pointer_names_only_this_tracks_audio(
        self, client: AsyncTestClient[Litestar], checked: HoldWorld
    ) -> None:
        user = await checked.user()
        mine = await committed_track(checked, user)
        theirs = await committed_track(checked, user, "Other")
        other_audio = await _attach(
            client, user, theirs, f"upload:{(await checked.audio(user)).id}"
        )

        response = await client.put(
            f"/v1/studio/tracks/{mine}/current",
            headers=bearer(user.id),
            json={"audio_version_id": other_audio["id"]},
        )

        assert response.status_code == 404
        assert response.json()["error"] == "audio_version_not_found"

    @pytest.mark.parametrize(
        ("method", "url_part", "body"),
        [
            ("post", "tracks/{track}/audio", {"asset": "upload:{asset}", "origin": "generated"}),
            ("post", "tracks/{track}/audio", {"asset_ref": "upload:{asset}"}),
            ("patch", "audio/{audio}", {"asset": "upload:{asset}"}),
            ("patch", "audio/{audio}", {"origin": "upload"}),
            ("put", "tracks/{track}/current", {"audio": "x"}),
        ],
    )
    async def test_unknown_fields_are_refused_and_the_asset_cannot_be_swapped(
        self,
        client: AsyncTestClient[Litestar],
        checked: HoldWorld,
        method: str,
        url_part: str,
        body: dict[str, str],
    ) -> None:
        user = await checked.user()
        track_id = await committed_track(checked, user)
        asset = await checked.audio(user)
        audio = await _attach(client, user, track_id, f"upload:{asset.id}")
        fields = {"track": str(track_id), "asset": str(asset.id), "audio": audio["id"]}

        response = await getattr(client, method)(
            f"/v1/studio/{url_part.format(**fields)}",
            headers=bearer(user.id),
            json={k: v.format(**fields) for k, v in body.items()},
        )

        assert response.status_code == 400, response.text
        unchanged = await client.get(f"/v1/studio/audio/{audio['id']}", headers=bearer(user.id))
        assert unchanged.json()["asset"]["asset_ref"] == f"upload:{asset.id}"

    @pytest.mark.parametrize("label", ["", "x" * 61])
    async def test_a_label_over_its_wire_limit_is_refused(
        self, client: AsyncTestClient[Litestar], checked: HoldWorld, label: str
    ) -> None:
        user = await checked.user()
        track_id = await committed_track(checked, user)
        asset = await checked.audio(user)

        response = await client.post(
            f"/v1/studio/tracks/{track_id}/audio",
            headers=bearer(user.id),
            json={"asset": f"upload:{asset.id}", "label": label},
        )

        assert response.status_code == 400


class TestGuards:
    async def test_every_route_needs_authentication(
        self, client: AsyncTestClient[Litestar]
    ) -> None:
        track_id, audio_id = uuid4(), uuid4()
        calls = [
            await client.post(f"/v1/studio/tracks/{track_id}/audio", json={"asset": "upload:x"}),
            await client.get(f"/v1/studio/tracks/{track_id}/audio"),
            await client.get(f"/v1/studio/audio/{audio_id}"),
            await client.patch(f"/v1/studio/audio/{audio_id}", json={}),
            await client.delete(f"/v1/studio/audio/{audio_id}"),
        ]

        assert {call.status_code for call in calls} <= {401, 403}

    async def test_a_stale_legal_digest_blocks_mutations_with_428_but_not_reads(
        self, client: AsyncTestClient[Litestar], checked: HoldWorld
    ) -> None:
        user = await checked.user()
        track_id = await committed_track(checked, user)
        asset = await checked.audio(user)
        audio = await _attach(client, user, track_id, f"upload:{asset.id}")
        stale = bearer(user.id, digest="stale")

        post = await client.post(
            f"/v1/studio/tracks/{track_id}/audio",
            headers=stale,
            json={"asset": f"upload:{asset.id}"},
        )
        patch = await client.patch(
            f"/v1/studio/audio/{audio['id']}", headers=stale, json={"label": "x"}
        )
        delete = await client.delete(f"/v1/studio/audio/{audio['id']}", headers=stale)
        read = await client.get(f"/v1/studio/audio/{audio['id']}", headers=stale)

        assert (post.status_code, patch.status_code, delete.status_code) == (428, 428, 428)
        assert read.status_code == 200

    async def test_a_disabled_studio_answers_404_even_unauthenticated(
        self,
        factory: async_sessionmaker[AsyncSession],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(
            "src.api.security.studio.get_settings", lambda: SimpleNamespace(studio_enabled=False)
        )
        app = build_app(factory, [StudioTrackAudioController])
        async with AsyncTestClient(app=app) as client:
            for method, url in (
                ("get", f"/v1/studio/tracks/{uuid4()}/audio"),
                ("post", f"/v1/studio/tracks/{uuid4()}/audio"),
                ("get", f"/v1/studio/audio/{uuid4()}"),
                ("patch", f"/v1/studio/audio/{uuid4()}"),
                ("delete", f"/v1/studio/audio/{uuid4()}"),
            ):
                response = await getattr(client, method)(url)
                assert response.status_code == 404, (method, url)
                assert response.json()["error"] == "not_found"

    async def test_a_product_without_the_studio_feature_answers_404(
        self, factory: async_sessionmaker[AsyncSession], checked: HoldWorld
    ) -> None:
        app = build_app(
            factory,
            [StudioTrackAudioController],
            config=replace(VEX_CONFIG, features=frozenset()),
        )
        user = await checked.user()
        async with AsyncTestClient(app=app) as client:
            response = await client.get(f"/v1/studio/audio/{uuid4()}", headers=bearer(user.id))

        assert response.status_code == 404
        assert response.json()["error"] == "not_found"

    async def test_a_cross_product_user_cannot_reach_anything(
        self, client: AsyncTestClient[Litestar], checked: HoldWorld
    ) -> None:
        user = await checked.user()
        track_id = await committed_track(checked, user)
        audio = await _attach(client, user, track_id, f"upload:{(await checked.audio(user)).id}")
        stranger = await checked.user(product_id="synthara")

        response = await client.get(
            f"/v1/studio/audio/{audio['id']}", headers=bearer(stranger.id, product="synthara")
        )

        assert response.status_code == 401  # a token minted for another product
