"""The cast over HTTP: real controllers, guards, DI, handlers and PostgreSQL (S1-P3, W7).

Rows are committed and the app opens its own session per request (see ``build_app``): a 409 leaves
nothing behind, a 428 never reaches the service, the studio kill switch answers 404 first.
"""

from __future__ import annotations

import re
from dataclasses import replace
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any
from uuid import uuid4

import pytest
import pytest_asyncio
from litestar.testing import AsyncTestClient
from sqlalchemy.ext.asyncio import async_sessionmaker

from src.api.routes.studio_artists import StudioArtistController
from src.api.routes.studio_cast import StudioCastController
from src.core.product_registry import VEX_CONFIG
from src.studio.constraints import MAX_CAST_REFERENCES
from tests.cast_support import assert_reconciled
from tests.library_holds_support import HoldWorld, bearer, build_app

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from litestar import Litestar
    from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession

    from src.db.models.user import User

C = "/v1/studio/cast"


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
    app = build_app(factory, [StudioCastController, StudioArtistController])
    async with AsyncTestClient(app=app) as test_client:
        yield test_client


@pytest_asyncio.fixture
async def checked(hold_world: HoldWorld) -> AsyncGenerator[HoldWorld]:
    """The hold world; every test ends with the reconciliation check (invariant 3)."""
    yield hold_world
    async with hold_world.factory() as session:
        await assert_reconciled(session)


async def _create(
    client: AsyncTestClient[Litestar], user: User, name: str = "Mira", **body: Any
) -> dict[str, Any]:
    response = await client.post(C, headers=bearer(user.id), json={"name": name, **body})
    assert response.status_code == 201, response.text
    return response.json()  # type: ignore[no-any-return]


class TestCrudOverHttp:
    async def test_the_whole_lifecycle(
        self, client: AsyncTestClient[Litestar], checked: HoldWorld
    ) -> None:
        user = await checked.user()
        headers = bearer(user.id)
        front = await checked.upload(user)
        side = await checked.upload(user)
        extra = await checked.upload(user)

        created = await _create(
            client,
            user,
            description="Lead",
            appearance="silver hair",
            references=[
                {"label": "front", "asset": f"upload:{front.id}"},
                {"label": "side", "asset": f"upload:{side.id}"},
            ],
        )
        member_id = created["id"]
        assert set(created) == {
            "id", "name", "description", "appearance", "references", "looks",
            "created_at", "updated_at",
        }  # fmt: skip
        assert [(r["label"], r["position"], r["asset"]) for r in created["references"]] == [
            ("front", 0, f"upload:{front.id}"),
            ("side", 1, f"upload:{side.id}"),
        ]
        reference = created["references"][0]
        assert set(reference) == {"id", "label", "position", "asset", "media"}
        assert reference["media"]["original"]["url"] == f"/v1/content/uploads/{front.id}"

        got = await client.get(f"{C}/{member_id}", headers=headers)
        assert got.status_code == 200
        assert got.json() == created

        patched = await client.patch(
            f"{C}/{member_id}", headers=headers, json={"name": "Nova", "description": None}
        )
        assert patched.status_code == 200
        assert (patched.json()["name"], patched.json()["description"]) == ("Nova", None)
        assert patched.json()["appearance"] == "silver hair"  # omitted: kept

        added = await client.post(
            f"{C}/{member_id}/references",
            headers=headers,
            json={"label": "back", "asset": f"upload:{extra.id}"},
        )
        assert added.status_code == 201
        assert added.json()["position"] == 2

        relabelled = await client.patch(
            f"{C}/{member_id}/references/{added.json()['id']}",
            headers=headers,
            json={"label": "rear"},
        )
        assert (relabelled.status_code, relabelled.json()["label"]) == (200, "rear")

        ids = [r["id"] for r in created["references"]] + [added.json()["id"]]
        reordered = await client.put(
            f"{C}/{member_id}/references/order",
            headers=headers,
            json={"reference_ids": [ids[2], ids[0], ids[1]]},
        )
        assert reordered.status_code == 200
        assert [(r["id"], r["position"]) for r in reordered.json()["references"]] == [
            (ids[2], 0),
            (ids[0], 1),
            (ids[1], 2),
        ]

        look = await client.post(
            f"{C}/{member_id}/looks",
            headers=headers,
            json={"name": "Stage", "reference_ids": [ids[1], ids[0]]},
        )
        assert look.status_code == 201
        assert set(look.json()) == {"id", "name", "reference_ids"}
        assert look.json()["reference_ids"] == [ids[0], ids[1]]  # reference position order
        look_patch = await client.patch(
            f"{C}/{member_id}/looks/{look.json()['id']}",
            headers=headers,
            json={"name": "Street", "reference_ids": [ids[2]]},
        )
        assert look_patch.json() == {
            "id": look.json()["id"],
            "name": "Street",
            "reference_ids": [ids[2]],
        }

        removed = await client.delete(f"{C}/{member_id}/references/{ids[2]}", headers=headers)
        assert removed.status_code == 204
        after = (await client.get(f"{C}/{member_id}", headers=headers)).json()
        assert [r["position"] for r in after["references"]] == [0, 1]
        assert after["looks"][0]["reference_ids"] == []  # emptied by the deletion, kept

        assert (
            await client.delete(f"{C}/{member_id}/looks/{look.json()['id']}", headers=headers)
        ).status_code == 204
        assert (await client.delete(f"{C}/{member_id}", headers=headers)).status_code == 204
        assert (await client.get(f"{C}/{member_id}", headers=headers)).status_code == 404
        assert await checked.hold_count() == 0

    async def test_the_list_pages_newest_first_with_summaries(
        self, client: AsyncTestClient[Litestar], checked: HoldWorld
    ) -> None:
        user = await checked.user()
        image = await checked.upload(user)
        await _create(
            client, user, "First", references=[{"label": "a", "asset": f"upload:{image.id}"}]
        )
        await _create(client, user, "Second", appearance="tall")

        first_page = await client.get(f"{C}?limit=1", headers=bearer(user.id))
        second_page = await client.get(
            f"{C}?limit=1&cursor={first_page.json()['next_cursor']}", headers=bearer(user.id)
        )

        assert first_page.status_code == 200
        assert [m["name"] for m in first_page.json()["items"]] == ["Second"]
        assert first_page.json()["has_more"] is True
        item = second_page.json()["items"][0]
        assert set(item) == {
            "id", "name", "appearance", "reference_count", "look_count", "cover",
            "created_at", "updated_at",
        }  # fmt: skip
        assert (item["name"], item["reference_count"], item["look_count"]) == ("First", 1, 0)
        assert item["cover"]["asset_ref"] == f"upload:{image.id}"
        assert second_page.json()["has_more"] is False

    async def test_the_holding_filter_over_http_is_exact(
        self, client: AsyncTestClient[Litestar], checked: HoldWorld
    ) -> None:
        user = await checked.user()
        job = await checked.job(user)
        x = await checked.upload(user)
        y = await checked.upload(user)
        out = await checked.output(user, job)
        await _create(client, user, "A", references=[{"label": "x", "asset": f"upload:{x.id}"}])
        await _create(
            client,
            user,
            "B",
            references=[
                {"label": "x", "asset": f"upload:{x.id}"},
                {"label": "o", "asset": f"output:{out.id}"},
            ],
        )

        async def names(asset: str) -> list[str]:
            response = await client.get(f"{C}?holding={asset}", headers=bearer(user.id))
            assert response.status_code == 200, response.text
            return sorted(m["name"] for m in response.json()["items"])

        assert await names(f"upload:{x.id}") == ["A", "B"]
        assert await names(f"output:{out.id}") == ["B"]
        assert await names(f"upload:{y.id}") == []
        assert (await client.get(f"{C}?holding=nonsense", headers=bearer(user.id))).json()[
            "error"
        ] == "invalid_asset_ref"
        assert (
            await client.get(f"{C}?holding=nonsense", headers=bearer(user.id))
        ).status_code == 400


class TestErrorsOverHttp:
    async def test_the_documented_errors_and_their_shapes(
        self, client: AsyncTestClient[Litestar], checked: HoldWorld
    ) -> None:
        user = await checked.user()
        headers = bearer(user.id)
        image = await checked.upload(user)
        video = await checked.upload(user, content_type="video/mp4")
        thumb_parent = await checked.upload(user)
        thumb = await checked.upload(user, is_thumbnail=True, parent=thumb_parent)
        member = await _create(
            client, user, references=[{"label": "a", "asset": f"upload:{image.id}"}]
        )
        member_id = member["id"]
        reference_id = member["references"][0]["id"]

        async def error(method: str, path: str, **kwargs: Any) -> tuple[int, dict[str, Any]]:
            response = await client.request(method, path, headers=headers, **kwargs)
            return response.status_code, response.json()

        # 404s
        for path in (f"{C}/{uuid4()}", f"{C}/{member_id}/references/{uuid4()}"):
            status, body = await error("GET" if "references" not in path else "DELETE", path)
            assert status == 404
        assert (await error("GET", f"{C}/{uuid4()}"))[1]["error"] == "cast_member_not_found"
        assert (await error("DELETE", f"{C}/{member_id}/references/{uuid4()}"))[1][
            "error"
        ] == "cast_reference_not_found"
        assert (await error("DELETE", f"{C}/{member_id}/looks/{uuid4()}"))[1][
            "error"
        ] == "cast_look_not_found"
        status, body = await error(
            "POST", f"{C}/{member_id}/references", json={"label": "z", "asset": f"upload:{uuid4()}"}
        )
        assert (status, body["error"]) == (404, "asset_not_found")

        # 409s
        status, body = await error("POST", C, json={"name": "MIRA"})
        assert (status, body["error"]) == (409, "cast_name_taken")
        status, body = await error(
            "POST",
            f"{C}/{member_id}/references",
            json={"label": "A", "asset": f"upload:{image.id}"},
        )
        assert (status, body["error"]) == (409, "cast_label_taken")
        look = {"name": "Look", "reference_ids": [reference_id]}
        assert (await error("POST", f"{C}/{member_id}/looks", json=look))[0] == 201
        status, body = await error("POST", f"{C}/{member_id}/looks", json=look)
        assert (status, body["error"]) == (409, "cast_look_name_taken")

        # 422s
        status, body = await error(
            "POST",
            f"{C}/{member_id}/references",
            json={"label": "v", "asset": f"upload:{video.id}"},
        )
        assert (status, body["error"], body["detail"]) == (
            422, "asset_not_eligible", {"reason": "media_kind_not_accepted"},
        )  # fmt: skip
        status, body = await error(
            "POST",
            f"{C}/{member_id}/references",
            json={"label": "t", "asset": f"upload:{thumb.id}"},
        )
        assert (status, body["detail"]) == (422, {"reason": "derivative"})
        status, body = await error(
            "PUT", f"{C}/{member_id}/references/order", json={"reference_ids": [str(uuid4())]}
        )
        assert (status, body["error"]) == (422, "invalid_order")
        status, body = await error(
            "POST", f"{C}/{member_id}/looks", json={"name": "Empty", "reference_ids": []}
        )
        assert (status, body["error"]) == (422, "document_invalid")
        assert body["detail"]["issues"] == [{"path": "reference_ids", "code": "empty"}]
        status, body = await error("POST", C, json={"name": "   "})
        assert (status, body["detail"]["issues"]) == (
            422,
            [{"path": "name", "code": "empty_after_normalization"}],
        )
        status, body = await error(
            "POST", C, json={"name": "Bad", "references": [{"label": "x", "asset": "nonsense"}]}
        )
        assert body["detail"]["issues"] == [
            {"path": "references[0].asset", "code": "invalid_asset_ref"}
        ]

    async def test_the_reference_cap_is_a_409_with_its_kind_and_limit(
        self, client: AsyncTestClient[Litestar], checked: HoldWorld
    ) -> None:
        user = await checked.user()
        assets = [await checked.upload(user) for _ in range(MAX_CAST_REFERENCES + 1)]
        member = await _create(
            client,
            user,
            references=[
                {"label": f"r{i}", "asset": f"upload:{a.id}"}
                for i, a in enumerate(assets[:MAX_CAST_REFERENCES])
            ],
        )

        response = await client.post(
            f"{C}/{member['id']}/references",
            headers=bearer(user.id),
            json={"label": "one more", "asset": f"upload:{assets[-1].id}"},
        )

        assert response.status_code == 409
        assert response.json()["error"] == "cast_limit_reached"
        assert response.json()["detail"] == {"kind": "references", "limit": MAX_CAST_REFERENCES}

    async def test_a_performer_cannot_be_deleted_and_the_409_names_the_blockers(
        self, client: AsyncTestClient[Litestar], checked: HoldWorld
    ) -> None:
        user = await checked.user()
        headers = bearer(user.id)
        member = await _create(client, user)
        artist = await client.post(
            "/v1/studio/artists",
            headers=headers,
            json={"name": "Band", "performer_cast_member_id": member["id"]},
        )
        assert artist.status_code == 201
        assert artist.json()["performer_cast_member_id"] == member["id"]

        refused = await client.delete(f"{C}/{member['id']}", headers=headers)

        assert refused.status_code == 409
        assert refused.json()["error"] == "cast_member_in_use"
        assert refused.json()["detail"] == {"blockers": [{"code": "artists", "count": 1}]}
        unlinked = await client.patch(
            f"/v1/studio/artists/{artist.json()['id']}",
            headers=headers,
            json={"performer_cast_member_id": None},
        )
        assert unlinked.json()["performer_cast_member_id"] is None
        assert (await client.delete(f"{C}/{member['id']}", headers=headers)).status_code == 204
        # linking a member that is not the caller's is the cast 404
        missing = await client.patch(
            f"/v1/studio/artists/{artist.json()['id']}",
            headers=headers,
            json={"performer_cast_member_id": str(uuid4())},
        )
        assert (missing.status_code, missing.json()["error"]) == (404, "cast_member_not_found")

    async def test_unknown_fields_and_bad_limits_are_rejected_by_the_wire(
        self, client: AsyncTestClient[Litestar], checked: HoldWorld
    ) -> None:
        user = await checked.user()
        headers = bearer(user.id)

        for body in (
            {"name": "Mira", "surprise": 1},
            {"name": ""},
            {"name": "x" * 101},
            {"name": "ok", "appearance": "x" * 321},
            {"name": "ok", "description": "x" * 2001},
            {"name": "ok", "references": [{"label": "x" * 41, "asset": "upload:" + str(uuid4())}]},
        ):
            response = await client.post(C, headers=headers, json=body)
            assert response.status_code == 400, body


class TestOwnershipOverHttp:
    async def test_someone_elses_member_is_a_404_on_every_route(
        self, client: AsyncTestClient[Litestar], checked: HoldWorld
    ) -> None:
        owner = await checked.user()
        stranger = await checked.user()
        image = await checked.upload(owner)
        member = await _create(
            client, owner, references=[{"label": "a", "asset": f"upload:{image.id}"}]
        )
        member_id, reference_id = member["id"], member["references"][0]["id"]
        look = (
            await client.post(
                f"{C}/{member_id}/looks",
                headers=bearer(owner.id),
                json={"name": "L", "reference_ids": [reference_id]},
            )
        ).json()
        theirs = bearer(stranger.id)

        calls = [
            ("GET", f"{C}/{member_id}", None),
            ("PATCH", f"{C}/{member_id}", {"name": "x"}),
            ("DELETE", f"{C}/{member_id}", None),
            ("POST", f"{C}/{member_id}/references", {"label": "x", "asset": f"upload:{uuid4()}"}),
            ("PATCH", f"{C}/{member_id}/references/{reference_id}", {"label": "x"}),
            ("DELETE", f"{C}/{member_id}/references/{reference_id}", None),
            ("PUT", f"{C}/{member_id}/references/order", {"reference_ids": [reference_id]}),
            ("POST", f"{C}/{member_id}/looks", {"name": "y", "reference_ids": [reference_id]}),
            ("PATCH", f"{C}/{member_id}/looks/{look['id']}", {"name": "y"}),
            ("DELETE", f"{C}/{member_id}/looks/{look['id']}", None),
        ]
        for method, path, body in calls:
            response = await client.request(method, path, headers=theirs, json=body)
            assert response.status_code == 404, (method, path, response.text)
            assert response.json()["error"] == "cast_member_not_found"
        listed = await client.get(C, headers=theirs)
        assert listed.json()["items"] == []
        # the owner's data is untouched
        mine = (await client.get(f"{C}/{member_id}", headers=bearer(owner.id))).json()
        assert (mine["name"], len(mine["references"]), len(mine["looks"])) == ("Mira", 1, 1)

    async def test_another_users_asset_cannot_be_referenced(
        self, client: AsyncTestClient[Litestar], checked: HoldWorld
    ) -> None:
        owner = await checked.user()
        stranger = await checked.user()
        theirs = await checked.upload(stranger)

        response = await client.post(
            C,
            headers=bearer(owner.id),
            json={"name": "Mira", "references": [{"label": "x", "asset": f"upload:{theirs.id}"}]},
        )

        assert (response.status_code, response.json()["error"]) == (404, "asset_not_found")
        assert (await client.get(C, headers=bearer(owner.id))).json()["items"] == []
        assert await checked.hold_count() == 0


class TestGuardsOverHttp:
    async def test_a_stale_legal_token_is_428_on_every_mutation_but_reads_still_work(
        self, client: AsyncTestClient[Litestar], checked: HoldWorld
    ) -> None:
        user = await checked.user()
        image = await checked.upload(user)
        member = await _create(
            client, user, references=[{"label": "a", "asset": f"upload:{image.id}"}]
        )
        stale = bearer(user.id, digest="stale-digest")
        member_id, reference_id = member["id"], member["references"][0]["id"]

        mutations = [
            ("POST", C, {"name": "New"}),
            ("PATCH", f"{C}/{member_id}", {"name": "x"}),
            ("DELETE", f"{C}/{member_id}", None),
            ("POST", f"{C}/{member_id}/references", {"label": "x", "asset": f"upload:{image.id}"}),
            ("PATCH", f"{C}/{member_id}/references/{reference_id}", {"label": "x"}),
            ("DELETE", f"{C}/{member_id}/references/{reference_id}", None),
            ("PUT", f"{C}/{member_id}/references/order", {"reference_ids": [reference_id]}),
            ("POST", f"{C}/{member_id}/looks", {"name": "y", "reference_ids": [reference_id]}),
            ("PATCH", f"{C}/{member_id}/looks/{uuid4()}", {"name": "y"}),
            ("DELETE", f"{C}/{member_id}/looks/{uuid4()}", None),
        ]
        for method, path, body in mutations:
            response = await client.request(method, path, headers=stale, json=body)
            assert response.status_code == 428, (method, path, response.text)
            assert response.json()["error"] == "legal_acceptance_required"

        assert (await client.get(C, headers=stale)).status_code == 200
        assert (await client.get(f"{C}/{member_id}", headers=stale)).status_code == 200
        # nothing was written by the refused calls
        unchanged = (await client.get(f"{C}/{member_id}", headers=bearer(user.id))).json()
        assert (unchanged["name"], len(unchanged["references"])) == ("Mira", 1)

    async def test_unauthenticated_requests_are_401(
        self, client: AsyncTestClient[Litestar]
    ) -> None:
        assert (await client.get(C)).status_code == 401
        assert (await client.post(C, json={"name": "x"})).status_code == 401

    async def test_a_disabled_studio_answers_404_before_auth_and_the_database(
        self,
        factory: async_sessionmaker[AsyncSession],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(
            "src.api.security.studio.get_settings", lambda: SimpleNamespace(studio_enabled=False)
        )
        app = build_app(factory, [StudioCastController])
        placeholder = str(uuid4())
        requests = [
            (method, re.sub(r"\{[^}]+\}", placeholder, route.path))
            for route in app.routes
            if route.path.startswith(C)
            for method in sorted(route.methods)
            if method not in {"HEAD", "OPTIONS"}
        ]
        assert len(requests) == 12  # the endpoint table of the contract
        async with AsyncTestClient(app=app) as off:
            for method, path in requests:
                assert (await off.request(method, path)).status_code == 404, (method, path)

    async def test_a_product_without_the_studio_feature_answers_404(
        self,
        factory: async_sessionmaker[AsyncSession],
        checked: HoldWorld,
    ) -> None:
        user = await checked.user()
        no_studio = replace(VEX_CONFIG, features=frozenset({"organizations"}))
        app = build_app(factory, [StudioCastController], config=no_studio)
        async with AsyncTestClient(app=app) as other:
            response = await other.get(C, headers=bearer(user.id))
        assert response.status_code == 404
