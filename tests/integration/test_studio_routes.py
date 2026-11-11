"""End-to-end studio HTTP behaviour: real controllers, guards, DI, handlers and PostgreSQL.

Rows are committed and the app opens its own session per request (``AsyncTestClient`` runs
handlers on another event loop, so the savepoint ``db_session`` cannot be shared). The session
provider mirrors ``get_db_session``: commit on success, rollback when the handler raises — which
is what makes a 409 leave nothing behind.
"""

from __future__ import annotations

import re
from dataclasses import replace
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any
from uuid import UUID, uuid4

import pytest
import pytest_asyncio
from litestar import Litestar
from litestar.di import Provide
from litestar.exceptions import HTTPException
from litestar.testing import AsyncTestClient
from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from src.api.app import (
    global_exception_handler,
    http_exception_handler,
    legal_acceptance_required_handler,
    library_project_in_use_handler,
    studio_error_handler,
)
from src.api.dependencies.common import (
    get_library_project_service,
    get_product_id,
    get_studio_artist_service,
    get_studio_export_service,
    get_studio_project_service,
    get_studio_track_service,
    get_studio_version_service,
)
from src.api.routes.library_project import LibraryProjectController
from src.api.routes.studio_artists import StudioArtistController
from src.api.routes.studio_projects import StudioProjectController
from src.api.routes.studio_tracks import StudioTrackController
from src.api.security.jwt import JWTConfig, JWTService
from src.api.services.legal.errors import LegalAcceptanceRequiredError
from src.api.services.library_project import LibraryProjectInUseError
from src.api.services.token_revocation import TokenRevocationService
from src.core.enums import StudioSectionKind as K
from src.core.product_registry import SYNTHARA_CONFIG, VEX_CONFIG
from src.db.models.library import LibraryProject
from src.db.models.studio import StudioDocumentVersion
from src.db.models.user import User
from src.studio.errors import StudioError
from tests.legal_support import TEST_LEGAL_DIGEST, TEST_LEGAL_REGISTRY
from tests.studio_support import make_lyrics, make_plan, make_section, to_json

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Callable

    from litestar.types import ASGIApp, Receive, Scope, Send
    from sqlalchemy.ext.asyncio import AsyncEngine

    from src.core.product import ProductConfig

SECRET = "test_secret_key_for_testing_only_256bits_long"
_NO_STUDIO = replace(VEX_CONFIG, features=frozenset({"organizations"}))
S = "/v1/studio"


def _scope_for(config: ProductConfig) -> Callable[[ASGIApp], ASGIApp]:
    def factory(app: ASGIApp) -> ASGIApp:
        async def middleware(scope: Scope, receive: Receive, send: Send) -> None:
            if scope["type"] in ("http", "websocket"):
                state = scope.setdefault("state", {})
                state["product_config"] = config
                state["product_id"] = config.slug
            await app(scope, receive, send)

        return middleware

    return factory


def _build_app(
    factory: async_sessionmaker[AsyncSession], config: ProductConfig = VEX_CONFIG
) -> Litestar:
    async def provide_session() -> AsyncGenerator[AsyncSession]:
        async with factory() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise

    def dep(fn: Callable[..., Any]) -> Provide:
        return Provide(fn, sync_to_thread=False)

    app = Litestar(
        route_handlers=[
            StudioArtistController,
            StudioProjectController,
            StudioTrackController,
            LibraryProjectController,
        ],
        dependencies={
            "session": Provide(provide_session),
            "product_id": dep(get_product_id),
            "studio_artist_service": dep(get_studio_artist_service),
            "studio_project_service": dep(get_studio_project_service),
            "studio_track_service": dep(get_studio_track_service),
            "studio_version_service": dep(get_studio_version_service),
            "studio_export_service": dep(get_studio_export_service),
            "library_project_service": dep(get_library_project_service),
        },
        exception_handlers={
            HTTPException: http_exception_handler,
            StudioError: studio_error_handler,
            LibraryProjectInUseError: library_project_in_use_handler,
            LegalAcceptanceRequiredError: legal_acceptance_required_handler,
            Exception: global_exception_handler,
        },
        middleware=[_scope_for(config)],
    )
    app.state["jwt_service"] = JWTService(JWTConfig(secret_key=SECRET))
    app.state["token_revocation"] = TokenRevocationService(None, max_token_ttl_seconds=0)
    app.state["legal_registry"] = TEST_LEGAL_REGISTRY
    return app


def _bearer(
    user_id: UUID, *, product: str = "vex", digest: str | None = TEST_LEGAL_DIGEST
) -> dict[str, str]:
    token, _ = JWTService(JWTConfig(secret_key=SECRET)).create_access_token(
        user_id, product_id=product, legal_digest=digest
    )
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture(autouse=True)
def studio_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "src.api.security.studio.get_settings", lambda: SimpleNamespace(studio_enabled=True)
    )


@pytest.fixture
def session_factory(db_engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(bind=db_engine, expire_on_commit=False)


@pytest_asyncio.fixture
async def owner(
    session_factory: async_sessionmaker[AsyncSession],
) -> AsyncGenerator[SimpleNamespace]:
    """A committed vex user with one library project; deleted (cascading) afterwards."""
    users: list[UUID] = []

    async def make(name: str = "Album") -> SimpleNamespace:
        user = User(
            id=uuid4(),
            email=f"routes-{uuid4().hex}@example.com",
            password_hash="x",
            product_id="vex",
        )
        project = LibraryProject(id=uuid4(), product_id="vex", user_id=user.id, name=name)
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


def _plan_body(plan: Any, **extra: Any) -> dict[str, Any]:
    return {"kind": "plan", "content": to_json(plan), **extra}


def _lyrics_body(lyrics: Any, based_on: str, **extra: Any) -> dict[str, Any]:
    return {
        "kind": "lyrics",
        "content": to_json(lyrics),
        "based_on_plan_version_id": based_on,
        **extra,
    }


async def _enable(client: AsyncTestClient[Litestar], owner: SimpleNamespace, **body: Any) -> Any:
    return await client.put(
        f"{S}/projects/{owner.project_id}",
        headers=owner.headers,
        json={"release_kind": "album", **body},
    )


async def _track(
    client: AsyncTestClient[Litestar], owner: SimpleNamespace, title: str = "Track"
) -> str:
    resp = await client.post(
        f"{S}/projects/{owner.project_id}/tracks", headers=owner.headers, json={"title": title}
    )
    assert resp.status_code == 201, resp.text
    return resp.json()["id"]


# ---------------------------------------------------------------------------
# Feature gating (invariant 1)
# ---------------------------------------------------------------------------


def _every_studio_request(app: Litestar) -> list[tuple[str, str]]:
    placeholder = str(uuid4())
    requests = []
    for route in app.routes:
        if not route.path.startswith(S):
            continue
        path = re.sub(r"\{[^}]+\}", placeholder, route.path)
        requests.extend(
            (method, path) for method in sorted(route.methods) if method not in {"HEAD", "OPTIONS"}
        )
    return requests


class TestFeatureGating:
    def test_the_table_of_routes_is_complete(
        self, session_factory: async_sessionmaker[AsyncSession]
    ) -> None:
        requests = _every_studio_request(_build_app(session_factory))
        assert len(requests) == 21  # 2+3+1+3+2+1+3+2+2+1+1 — the endpoint table in the contract

    async def test_kill_switch_hides_every_method_before_auth_and_db(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        owner: SimpleNamespace,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(
            "src.api.security.studio.get_settings", lambda: SimpleNamespace(studio_enabled=False)
        )
        app = _build_app(session_factory)
        async with AsyncTestClient(app=app) as client:
            for method, path in _every_studio_request(app):
                anonymous = await client.request(method, path, json={})
                authed = await client.request(method, path, headers=owner.headers, json={})
                assert anonymous.status_code == 404, (method, path)  # not 401
                assert authed.status_code == 404, (method, path)  # not 403, not 200
                assert anonymous.json()["error"] == "not_found"

    async def test_a_product_without_the_feature_is_hidden_even_when_enabled(
        self, session_factory: async_sessionmaker[AsyncSession], owner: SimpleNamespace
    ) -> None:
        app = _build_app(session_factory, _NO_STUDIO)
        async with AsyncTestClient(app=app) as client:
            resp = await client.get(f"{S}/artists", headers=owner.headers)
        assert resp.status_code == 404

    async def test_enabled_requests_without_credentials_are_401(
        self, client: AsyncTestClient[Litestar]
    ) -> None:
        assert (await client.get(f"{S}/artists")).status_code == 401

    async def test_library_routes_are_unaffected_by_the_studio_switch(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        owner: SimpleNamespace,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(
            "src.api.security.studio.get_settings", lambda: SimpleNamespace(studio_enabled=False)
        )
        async with AsyncTestClient(app=_build_app(session_factory)) as client:
            resp = await client.get(
                f"/v1/library/projects/{owner.project_id}", headers=owner.headers
            )
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# The whole journey over HTTP
# ---------------------------------------------------------------------------


class TestJourney:
    async def test_artist_to_suno_export(
        self, client: AsyncTestClient[Litestar], owner: SimpleNamespace
    ) -> None:
        h = owner.headers

        artist = await client.post(
            f"{S}/artists",
            headers=h,
            json={
                "name": "Nova",
                "style_tags": ["husky vocals"],
                "global_avoid": ["autotune"],
                "persona": "MARKER",
            },
        )
        assert artist.status_code == 201
        artist_id = artist.json()["id"]

        enabled = await _enable(client, owner, artist_id=artist_id, concept="A concept")
        assert enabled.status_code == 201
        assert enabled.json()["track_count"] == 0
        assert (await _enable(client, owner, artist_id=artist_id)).status_code == 200  # replace

        track_id = await _track(client, owner, "Opening")
        verse, chorus = make_section(K.VERSE), make_section(K.CHORUS)
        plan = make_plan(
            verse, chorus, title="Opening", tempo_bpm=100, tags=("synth pop",), avoid=("metal",)
        )

        created = await client.post(
            f"{S}/tracks/{track_id}/versions", headers=h, json=_plan_body(plan)
        )
        assert created.status_code == 201, created.text
        body = created.json()
        plan_id = body["version"]["id"]
        assert body["version"]["kind"] == "plan"
        assert body["version"]["version_number"] == 1
        assert body["version"]["is_current"] is True
        assert body["lyrics_detached"] is False

        replay = await client.post(
            f"{S}/tracks/{track_id}/versions", headers=h, json=_plan_body(plan)
        )
        assert replay.status_code == 200  # naturally idempotent
        assert replay.json()["version"]["id"] == plan_id

        lyrics = make_lyrics((verse.id, ["first line"]), (chorus.id, ["the hook"]))
        lyr = await client.post(
            f"{S}/tracks/{track_id}/versions", headers=h, json=_lyrics_body(lyrics, plan_id)
        )
        assert lyr.status_code == 201, lyr.text
        assert lyr.json()["version"]["kind"] == "lyrics"
        assert lyr.json()["version"]["based_on_plan_version_id"] == plan_id

        detail = await client.get(f"{S}/tracks/{track_id}", headers=h)
        assert detail.status_code == 200
        assert detail.json()["current_plan"]["content"]["title"] == "Opening"
        assert (
            detail.json()["current_lyrics"]["content"]["sections"][0]["lines"][0]["text"]
            == "first line"
        )

        export = await client.get(f"{S}/tracks/{track_id}/exports/suno", headers=h)
        assert export.status_code == 200
        out = export.json()
        assert out["target"] == "suno"
        assert out["renderer_version"] == "suno-v6.2026-11"
        assert out["title"] == "Opening"
        assert out["style"] == "synth pop, husky vocals, 100 BPM"
        assert out["exclude"] == "metal, autotune"
        assert out["lyrics"] == "[Verse]\nfirst line\n\n[Chorus]\nthe hook"
        assert out["fields"]["style"] == {
            "length": len(out["style"]),
            "limit": 1000,
            "within_limit": True,
        }
        assert out["warnings"] == []
        assert "MARKER" not in export.text

        listing = await client.get(f"{S}/tracks/{track_id}/versions?kind=plan", headers=h)
        assert [v["kind"] for v in listing.json()["items"]] == ["plan"]
        assert listing.json()["items"][0]["is_current"] is True

        project = await client.get(f"{S}/projects/{owner.project_id}", headers=h)
        assert project.json()["track_count"] == 1
        assert project.json()["release_kind"] == "album"

    async def test_conflicts_and_validation_use_the_unified_envelope(
        self, client: AsyncTestClient[Litestar], owner: SimpleNamespace
    ) -> None:
        h = owner.headers
        await _enable(client, owner)
        track_id = await _track(client, owner)
        plan = make_plan(title="Same")
        first = await client.post(
            f"{S}/tracks/{track_id}/versions", headers=h, json=_plan_body(plan)
        )
        parent = first.json()["version"]["id"]

        unchanged = await client.post(
            f"{S}/tracks/{track_id}/versions",
            headers=h,
            json=_plan_body(plan, parent_version_id=parent),
        )
        assert unchanged.status_code == 409
        assert unchanged.json() == {
            "error": "version_unchanged",
            "message": "The content is identical to the parent version.",
            "status_code": 409,
            "detail": None,
        }

        shared = make_section()
        invalid = await client.post(
            f"{S}/tracks/{track_id}/versions",
            headers=h,
            json=_plan_body(
                make_plan(shared, make_section(section_id=shared.id), title="x", tags=("a", "A"))
            ),
        )
        assert invalid.status_code == 422
        assert invalid.json()["error"] == "document_invalid"
        assert {(i["path"], i["code"]) for i in invalid.json()["detail"]["issues"]} == {
            ("global_style.tags[1]", "duplicate_tag"),
            ("sections[1].id", "duplicate_section_id"),
        }

    async def test_malformed_bodies_are_400_not_500(
        self, client: AsyncTestClient[Litestar], owner: SimpleNamespace
    ) -> None:
        h = owner.headers
        await _enable(client, owner)
        track_id = await _track(client, owner)
        for payload in (
            {"kind": "nonsense", "content": {}},
            {"kind": "plan", "content": {"title": "x", "sections": []}},
            {
                "kind": "plan",
                "content": {"title": "x", "sections": [{"id": str(uuid4()), "kind": "verse"}]},
                "extra": 1,
            },
            {"kind": "lyrics", "content": {"language": "en"}},
        ):
            resp = await client.post(f"{S}/tracks/{track_id}/versions", headers=h, json=payload)
            assert resp.status_code == 400, (payload, resp.text)

    async def test_incompatible_plan_is_a_409_and_detach_is_explicit(
        self,
        client: AsyncTestClient[Litestar],
        owner: SimpleNamespace,
        session_factory: async_sessionmaker[AsyncSession],
    ) -> None:
        h = owner.headers
        await _enable(client, owner)
        track_id = await _track(client, owner)
        verse, chorus = make_section(K.VERSE), make_section(K.CHORUS)
        plan_id = (
            await client.post(
                f"{S}/tracks/{track_id}/versions",
                headers=h,
                json=_plan_body(make_plan(verse, chorus)),
            )
        ).json()["version"]["id"]
        await client.post(
            f"{S}/tracks/{track_id}/versions",
            headers=h,
            json=_lyrics_body(make_lyrics((chorus.id, ["x"])), plan_id),
        )

        async def row_count() -> int:
            async with session_factory() as session:
                return (
                    await session.execute(select(func.count()).select_from(StudioDocumentVersion))
                ).scalar_one()

        before = await row_count()
        narrow = make_plan(verse, title="narrow")
        rejected = await client.post(
            f"{S}/tracks/{track_id}/versions", headers=h, json=_plan_body(narrow)
        )
        assert rejected.status_code == 409
        assert rejected.json()["error"] == "lyrics_incompatible_with_plan"
        assert await row_count() == before  # a 409 writes nothing

        detached = await client.post(
            f"{S}/tracks/{track_id}/versions",
            headers=h,
            json=_plan_body(narrow, incompatible_lyrics="detach"),
        )
        assert detached.status_code == 201
        assert detached.json()["lyrics_detached"] is True
        track = (await client.get(f"{S}/tracks/{track_id}", headers=h)).json()
        assert track["current_lyrics_version_id"] is None
        assert track["current_plan_version_id"] == detached.json()["version"]["id"]

    async def test_switching_and_deleting_versions(
        self, client: AsyncTestClient[Litestar], owner: SimpleNamespace
    ) -> None:
        h = owner.headers
        await _enable(client, owner)
        track_id = await _track(client, owner)
        a = (
            await client.post(
                f"{S}/tracks/{track_id}/versions", headers=h, json=_plan_body(make_plan(title="a"))
            )
        ).json()["version"]["id"]
        b = (
            await client.post(
                f"{S}/tracks/{track_id}/versions", headers=h, json=_plan_body(make_plan(title="b"))
            )
        ).json()["version"]["id"]

        assert (await client.delete(f"{S}/versions/{b}", headers=h)).json()[
            "error"
        ] == "version_is_current"
        switched = await client.put(
            f"{S}/tracks/{track_id}/current", headers=h, json={"plan_version_id": a}
        )
        assert switched.status_code == 200
        assert switched.json()["current_plan_version_id"] == a
        assert (await client.delete(f"{S}/versions/{b}", headers=h)).status_code == 204
        assert (await client.get(f"{S}/versions/{b}", headers=h)).status_code == 404
        got = await client.get(f"{S}/versions/{a}", headers=h)
        assert got.json()["kind"] == "plan"
        assert got.json()["is_current"] is True

    async def test_reorder_and_rename_and_delete_tracks(
        self, client: AsyncTestClient[Litestar], owner: SimpleNamespace
    ) -> None:
        h = owner.headers
        await _enable(client, owner)
        a, b = await _track(client, owner, "a"), await _track(client, owner, "b")
        reordered = await client.put(
            f"{S}/projects/{owner.project_id}/tracks/order", headers=h, json={"track_ids": [b, a]}
        )
        assert reordered.status_code == 200
        assert [t["id"] for t in reordered.json()] == [b, a]
        mismatch = await client.put(
            f"{S}/projects/{owner.project_id}/tracks/order", headers=h, json={"track_ids": [a]}
        )
        assert (mismatch.status_code, mismatch.json()["error"]) == (422, "track_order_mismatch")
        renamed = await client.patch(f"{S}/tracks/{a}", headers=h, json={"title": "Renamed"})
        assert renamed.json()["title"] == "Renamed"
        listing = await client.get(f"{S}/projects/{owner.project_id}/tracks", headers=h)
        assert [t["title"] for t in listing.json()] == ["b", "Renamed"]
        assert (await client.delete(f"{S}/tracks/{a}", headers=h)).status_code == 204
        assert (await client.get(f"{S}/tracks/{a}", headers=h)).status_code == 404
        blocked = await client.delete(f"{S}/projects/{owner.project_id}", headers=h)
        assert blocked.json()["error"] == "project_has_tracks"

    async def test_artist_crud_pagination_and_in_use(
        self, client: AsyncTestClient[Litestar], owner: SimpleNamespace
    ) -> None:
        h = owner.headers
        ids = []
        for name in ("A", "B", "C"):
            ids.append(
                (await client.post(f"{S}/artists", headers=h, json={"name": name})).json()["id"]
            )
        page = await client.get(f"{S}/artists?limit=2", headers=h)
        assert (len(page.json()["items"]), page.json()["has_more"]) == (2, True)
        rest = await client.get(
            f"{S}/artists?limit=2&cursor={page.json()['next_cursor']}", headers=h
        )
        assert [a["name"] for a in rest.json()["items"]] == ["A"]
        assert (await client.get(f"{S}/artists?cursor=bad", headers=h)).json()[
            "error"
        ] == "invalid_cursor"
        assert (await client.post(f"{S}/artists", headers=h, json={"name": "a"})).json()[
            "error"
        ] == "artist_name_conflict"

        patched = await client.patch(
            f"{S}/artists/{ids[0]}", headers=h, json={"persona": "p", "default_language": None}
        )
        assert patched.json()["persona"] == "p"
        await _enable(client, owner, artist_id=ids[0])
        in_use = await client.delete(f"{S}/artists/{ids[0]}", headers=h)
        assert (in_use.status_code, in_use.json()["error"], in_use.json()["detail"]) == (
            409,
            "artist_in_use",
            {"project_count": 1},
        )
        assert (await client.delete(f"{S}/artists/{ids[1]}", headers=h)).status_code == 204

    async def test_project_listing(
        self, client: AsyncTestClient[Litestar], owner: SimpleNamespace
    ) -> None:
        await _enable(client, owner)
        listing = await client.get(f"{S}/projects", headers=owner.headers)
        assert [p["library_project_id"] for p in listing.json()["items"]] == [str(owner.project_id)]
        unknown = await client.put(
            f"{S}/projects/{uuid4()}", headers=owner.headers, json={"release_kind": "ep"}
        )
        assert unknown.status_code == 404
        bad_kind = await client.put(
            f"{S}/projects/{owner.project_id}",
            headers=owner.headers,
            json={"release_kind": "box_set"},
        )
        assert bad_kind.status_code == 400


# ---------------------------------------------------------------------------
# Library project deletion over HTTP (invariant 4)
# ---------------------------------------------------------------------------


class TestProjectDeletionOverHttp:
    async def test_project_in_use_then_free(
        self, client: AsyncTestClient[Litestar], owner: SimpleNamespace
    ) -> None:
        h = owner.headers
        await _enable(client, owner)
        track_id = await _track(client, owner)
        blocked = await client.delete(f"/v1/library/projects/{owner.project_id}", headers=h)
        assert blocked.status_code == 409
        assert blocked.json()["error"] == "project_in_use"
        assert blocked.json()["detail"] == {"blockers": [{"code": "studio_tracks", "count": 1}]}
        assert (
            await client.get(f"/v1/library/projects/{owner.project_id}", headers=h)
        ).status_code == 200

        await client.delete(f"{S}/tracks/{track_id}", headers=h)
        assert (
            await client.delete(f"/v1/library/projects/{owner.project_id}", headers=h)
        ).status_code == 204
        assert (
            await client.get(f"{S}/projects/{owner.project_id}", headers=h)
        ).status_code == 404  # profile cascaded

    async def test_a_plain_library_project_deletes_as_before(
        self, client: AsyncTestClient[Litestar], owner: SimpleNamespace
    ) -> None:
        resp = await client.delete(
            f"/v1/library/projects/{owner.project_id}", headers=owner.headers
        )
        assert resp.status_code == 204


# ---------------------------------------------------------------------------
# Scoping (invariant 2) and legal enforcement
# ---------------------------------------------------------------------------


class TestScoping:
    async def test_other_users_and_other_products_see_404(
        self,
        client: AsyncTestClient[Litestar],
        owner: SimpleNamespace,
        session_factory: async_sessionmaker[AsyncSession],
    ) -> None:
        h = owner.headers
        await _enable(client, owner)
        track_id = await _track(client, owner)
        plan = (
            await client.post(
                f"{S}/tracks/{track_id}/versions", headers=h, json=_plan_body(make_plan())
            )
        ).json()["version"]["id"]
        artist = (await client.post(f"{S}/artists", headers=h, json={"name": "Mine"})).json()["id"]

        stranger = await owner.make("Theirs")
        requests = [
            ("GET", f"{S}/artists/{artist}"),
            ("PATCH", f"{S}/artists/{artist}"),
            ("DELETE", f"{S}/artists/{artist}"),
            ("GET", f"{S}/projects/{owner.project_id}"),
            ("GET", f"{S}/projects/{owner.project_id}/tracks"),
            ("GET", f"{S}/tracks/{track_id}"),
            ("GET", f"{S}/tracks/{track_id}/versions"),
            ("GET", f"{S}/versions/{plan}"),
            ("DELETE", f"{S}/versions/{plan}"),
            ("GET", f"{S}/tracks/{track_id}/exports/suno"),
        ]
        for method, path in requests:
            resp = await client.request(
                method, path, headers=stranger.headers, json={} if method == "PATCH" else None
            )
            assert resp.status_code == 404, (method, path, resp.text)
        assert (await client.get(f"{S}/artists", headers=stranger.headers)).json()["items"] == []

        # right user id, wrong product: a synthara token for the same user id
        async with AsyncTestClient(app=_build_app(session_factory, SYNTHARA_CONFIG)) as synthara:
            cross = _bearer(owner.id, product="synthara")
            for method, path in requests:
                resp = await synthara.request(
                    method, path, headers=cross, json={} if method == "PATCH" else None
                )
                assert resp.status_code == 404, (method, path, resp.text)

        assert (
            await client.get(f"{S}/tracks/{track_id}", headers=h)
        ).status_code == 200  # untouched

    async def test_the_artist_of_another_user_cannot_be_attached(
        self, client: AsyncTestClient[Litestar], owner: SimpleNamespace
    ) -> None:
        stranger = await owner.make()
        theirs = (
            await client.post(f"{S}/artists", headers=stranger.headers, json={"name": "Theirs"})
        ).json()["id"]
        resp = await _enable(client, owner, artist_id=theirs)
        assert (resp.status_code, resp.json()["error"]) == (404, "artist_not_found")


class TestLegalEnforcement:
    async def test_mutations_need_current_legal_acceptance_reads_do_not(
        self, session_factory: async_sessionmaker[AsyncSession], owner: SimpleNamespace
    ) -> None:
        stale = _bearer(owner.id, digest="stale-digest")
        async with AsyncTestClient(app=_build_app(session_factory)) as client:
            assert (await client.get(f"{S}/artists", headers=stale)).status_code == 200
            posted = await client.post(f"{S}/artists", headers=stale, json={"name": "Blocked"})
            assert (posted.status_code, posted.json()["error"]) == (
                428,
                "legal_acceptance_required",
            )
            put = await client.put(
                f"{S}/projects/{owner.project_id}", headers=stale, json={"release_kind": "ep"}
            )
            assert put.status_code == 428
            assert (await client.delete(f"{S}/tracks/{uuid4()}", headers=stale)).status_code == 428

    def test_no_studio_route_is_legal_exempt(
        self, session_factory: async_sessionmaker[AsyncSession]
    ) -> None:
        app = _build_app(session_factory)
        exempt = [
            (route.path, handler.name)
            for route in app.routes
            if route.path.startswith(S)
            for handler in route.route_handlers  # type: ignore[attr-defined]
            if handler.opt.get("legal_exempt")
        ]
        assert exempt == []


class TestOpenApi:
    def test_the_studio_surface_is_documented(
        self, session_factory: async_sessionmaker[AsyncSession]
    ) -> None:
        schema = _build_app(session_factory).openapi_schema.to_schema()
        paths = schema["paths"]
        for path in (
            "/v1/studio/artists",
            "/v1/studio/artists/{artist_id}",
            "/v1/studio/projects/{project_id}/tracks/order",
            "/v1/studio/tracks/{track_id}/versions",
            "/v1/studio/tracks/{track_id}/current",
            "/v1/studio/tracks/{track_id}/exports/suno",
            "/v1/studio/versions/{version_id}",
        ):
            assert path in paths, path
        body = paths["/v1/studio/tracks/{track_id}/versions"]["post"]["requestBody"]
        assert "oneOf" in str(body)  # the tagged union is rendered


class TestErrorLogging:
    async def test_error_responses_and_logs_never_echo_document_text(
        self, client: AsyncTestClient[Litestar], owner: SimpleNamespace
    ) -> None:
        from structlog.testing import capture_logs

        h = owner.headers
        await _enable(client, owner)
        track_id = await _track(client, owner)
        marker = "ZXQ-SECRET"
        bad = make_plan(title=marker, tags=(f"{marker}-tag", f"{marker.lower()}-tag"))
        with capture_logs() as logs:
            resp = await client.post(
                f"{S}/tracks/{track_id}/versions", headers=h, json=_plan_body(bad)
            )
        assert resp.status_code == 422
        assert marker.lower() not in resp.text.lower()
        studio_logs = [entry for entry in logs if entry["event"] == "studio.error"]
        assert len(studio_logs) == 1
        assert studio_logs[0]["code"] == "document_invalid"
        assert all(marker.lower() not in repr(entry).lower() for entry in logs)
