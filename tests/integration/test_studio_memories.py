"""Artist memories over HTTP against real PostgreSQL (S1-P2b): CRUD, scoping, validation, read-back.

Same shape as ``test_studio_routes``: committed rows, a session per request (``AsyncTestClient``
runs handlers on another event loop), a commit on success and a rollback when the handler raises.
The overlapping-request tests (the 200 cap, RD8) are in ``test_studio_memory_concurrency``.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, Any
from uuid import uuid4

import pytest
import pytest_asyncio
from litestar import Litestar
from litestar.di import Provide
from litestar.exceptions import HTTPException
from litestar.testing import AsyncTestClient
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from src.api.app import (
    global_exception_handler,
    http_exception_handler,
    legal_acceptance_required_handler,
    studio_error_handler,
)
from src.api.dependencies.common import (
    get_product_id,
    get_studio_artist_service,
    get_studio_memory_service,
)
from src.api.routes.studio_artists import StudioArtistController
from src.api.routes.studio_memories import StudioMemoryController
from src.api.security.jwt import JWTConfig, JWTService
from src.api.services.legal.errors import LegalAcceptanceRequiredError
from src.api.services.token_revocation import TokenRevocationService
from src.core.product_registry import VEX_CONFIG
from src.db.models.studio import StudioArtistMemory
from src.db.models.user import User
from src.studio.constraints import (
    MEMORY_BODY_MAX,
    MEMORY_IMAGE_MAX,
    MEMORY_IMAGERY_MAX,
    MEMORY_THEME_MAX,
    MEMORY_THEMES_MAX,
    MEMORY_TITLE_MAX,
)
from src.studio.errors import MemoryNotFound, StudioError
from src.studio.services.memory import MemoryService
from tests.legal_support import TEST_LEGAL_DIGEST, TEST_LEGAL_REGISTRY

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Callable
    from uuid import UUID

    from litestar.types import ASGIApp, Receive, Scope, Send
    from sqlalchemy.ext.asyncio import AsyncEngine

SECRET = "test_secret_key_for_testing_only_256bits_long"
S = "/v1/studio"
QA = "क़"  # NFC decomposes it to two code points and never recomposes it


def _scope(app: ASGIApp) -> ASGIApp:
    async def middleware(scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] in ("http", "websocket"):
            state = scope.setdefault("state", {})
            state["product_config"] = VEX_CONFIG
            state["product_id"] = VEX_CONFIG.slug
        await app(scope, receive, send)

    return middleware


def _build_app(factory: async_sessionmaker[AsyncSession]) -> Litestar:
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
        route_handlers=[StudioArtistController, StudioMemoryController],
        dependencies={
            "session": Provide(provide_session),
            "product_id": dep(get_product_id),
            "studio_artist_service": dep(get_studio_artist_service),
            "studio_memory_service": dep(get_studio_memory_service),
        },
        exception_handlers={
            HTTPException: http_exception_handler,
            StudioError: studio_error_handler,
            LegalAcceptanceRequiredError: legal_acceptance_required_handler,
            Exception: global_exception_handler,
        },
        middleware=[_scope],
    )
    app.state["jwt_service"] = JWTService(JWTConfig(secret_key=SECRET))
    app.state["token_revocation"] = TokenRevocationService(None, max_token_ttl_seconds=0)
    app.state["legal_registry"] = TEST_LEGAL_REGISTRY
    return app


def _bearer(user_id: UUID) -> dict[str, str]:
    token, _ = JWTService(JWTConfig(secret_key=SECRET)).create_access_token(
        user_id, product_id="vex", legal_digest=TEST_LEGAL_DIGEST
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
async def make_owner(
    session_factory: async_sessionmaker[AsyncSession],
) -> AsyncGenerator[Callable[[], Any]]:
    """Factory of committed vex users; deleted (cascading to artists and memories) afterwards."""
    users: list[UUID] = []

    async def make() -> SimpleNamespace:
        user = User(
            id=uuid4(),
            email=f"memories-{uuid4().hex}@example.com",
            password_hash="x",
            product_id="vex",
        )
        async with session_factory() as session, session.begin():
            session.add(user)
        users.append(user.id)
        return SimpleNamespace(id=user.id, headers=_bearer(user.id))

    yield make
    async with session_factory() as session, session.begin():
        await session.execute(delete(User).where(User.id.in_(users)))


@pytest_asyncio.fixture
async def owner(make_owner: Callable[[], Any]) -> SimpleNamespace:
    return await make_owner()


@pytest_asyncio.fixture
async def client(
    session_factory: async_sessionmaker[AsyncSession],
) -> AsyncGenerator[AsyncTestClient[Litestar]]:
    async with AsyncTestClient(app=_build_app(session_factory)) as test_client:
        yield test_client


async def _artist(
    client: AsyncTestClient[Litestar], owner: SimpleNamespace, name: str = "Nova"
) -> str:
    response = await client.post(f"{S}/artists", headers=owner.headers, json={"name": name})
    assert response.status_code == 201, response.text
    return response.json()["id"]


def _body(**overrides: Any) -> dict[str, Any]:
    return {"title": "Rooftop night", "body": "We watched the trains.", **overrides}


async def _create(
    client: AsyncTestClient[Litestar], owner: SimpleNamespace, artist_id: str, **overrides: Any
) -> dict[str, Any]:
    response = await client.post(
        f"{S}/artists/{artist_id}/memories", headers=owner.headers, json=_body(**overrides)
    )
    assert response.status_code == 201, response.text
    return response.json()


# ---------------------------------------------------------------------------
# CRUD
# ---------------------------------------------------------------------------


class TestCrud:
    async def test_create_normalizes_and_answers_the_stored_row(
        self, client: AsyncTestClient[Litestar], owner: SimpleNamespace
    ) -> None:
        artist_id = await _artist(client, owner)
        response = await client.post(
            f"{S}/artists/{artist_id}/memories",
            headers=owner.headers,
            json={
                "title": "  Rooftop\n night ",
                "body": " First line.\r\n\r\nSecond. ",
                "occurred_on": "2019-06-04",
                "themes": [" loss ", "home"],
                "imagery": ["wet  asphalt"],
                "salience": 5,
            },
        )
        assert response.status_code == 201
        created = response.json()
        assert created["title"] == "Rooftop night"
        assert created["body"] == "First line.\n\nSecond."
        assert (created["themes"], created["imagery"]) == (["loss", "home"], ["wet asphalt"])
        assert (created["occurred_on"], created["salience"]) == ("2019-06-04", 5)
        assert (created["status"], created["origin"]) == ("active", "user")
        assert created["artist_id"] == artist_id
        fetched = await client.get(f"{S}/memories/{created['id']}", headers=owner.headers)
        assert (fetched.status_code, fetched.json()) == (200, created)

    async def test_the_defaults(
        self, client: AsyncTestClient[Litestar], owner: SimpleNamespace
    ) -> None:
        artist_id = await _artist(client, owner)
        created = await _create(client, owner, artist_id)
        assert (created["salience"], created["themes"], created["imagery"]) == (3, [], [])
        assert created["occurred_on"] is None

    async def test_patch_is_tri_state(
        self, client: AsyncTestClient[Litestar], owner: SimpleNamespace
    ) -> None:
        artist_id = await _artist(client, owner)
        created = await _create(
            client, owner, artist_id, occurred_on="2019-06-04", themes=["a"], imagery=["b"]
        )
        url = f"{S}/memories/{created['id']}"
        response = await client.patch(url, headers=owner.headers, json={"salience": 1})
        assert response.status_code == 200
        kept = response.json()
        assert kept["salience"] == 1
        for field in ("title", "body", "occurred_on", "themes", "imagery"):
            assert kept[field] == created[field], field  # omitted: kept

        response = await client.patch(
            url, headers=owner.headers, json={"occurred_on": None, "title": " New  title "}
        )
        cleared = response.json()
        assert (cleared["occurred_on"], cleared["title"]) == (None, "New title")
        assert cleared["themes"] == ["a"]

        response = await client.patch(url, headers=owner.headers, json={})
        assert (response.status_code, response.json()) == (
            200,
            cleared,
        )  # an empty PATCH changes nothing

    async def test_archive_and_restore(
        self, client: AsyncTestClient[Litestar], owner: SimpleNamespace
    ) -> None:
        artist_id = await _artist(client, owner)
        created = await _create(client, owner, artist_id)
        url = f"{S}/memories/{created['id']}"
        archived = await client.patch(url, headers=owner.headers, json={"status": "archived"})
        assert archived.json()["status"] == "archived"
        restored = await client.patch(url, headers=owner.headers, json={"status": "active"})
        assert restored.json()["status"] == "active"

    async def test_delete_is_always_allowed_and_then_the_memory_is_gone(
        self, client: AsyncTestClient[Litestar], owner: SimpleNamespace
    ) -> None:
        artist_id = await _artist(client, owner)
        created = await _create(client, owner, artist_id)
        url = f"{S}/memories/{created['id']}"
        assert (await client.delete(url, headers=owner.headers)).status_code == 204
        gone = await client.get(url, headers=owner.headers)
        assert (gone.status_code, gone.json()["error"]) == (404, "memory_not_found")
        again = await client.delete(url, headers=owner.headers)
        assert (again.status_code, again.json()["error"]) == (404, "memory_not_found")


class TestList:
    async def test_pages_newest_first_and_defaults_to_active(
        self, client: AsyncTestClient[Litestar], owner: SimpleNamespace
    ) -> None:
        artist_id = await _artist(client, owner)
        created = [await _create(client, owner, artist_id, title=f"m{n}") for n in range(4)]
        archived = created[0]
        await client.patch(
            f"{S}/memories/{archived['id']}", headers=owner.headers, json={"status": "archived"}
        )
        url = f"{S}/artists/{artist_id}/memories"

        first = (await client.get(url, headers=owner.headers, params={"limit": 2})).json()
        assert [m["title"] for m in first["items"]] == ["m3", "m2"]  # newest first
        assert first["has_more"] is True
        rest = (
            await client.get(
                url, headers=owner.headers, params={"limit": 2, "cursor": first["next_cursor"]}
            )
        ).json()
        assert [m["title"] for m in rest["items"]] == ["m1"]  # the archived one is not listed
        assert rest["has_more"] is False

        only_archived = (
            await client.get(url, headers=owner.headers, params={"status": "archived"})
        ).json()
        assert [m["id"] for m in only_archived["items"]] == [archived["id"]]

    @pytest.mark.parametrize(
        ("params", "status"),
        [
            ({"limit": 0}, 400),
            ({"limit": 51}, 400),
            ({"status": "deleted"}, 400),
            ({"cursor": "nope"}, 400),
        ],
    )
    async def test_bad_query_parameters_are_400(
        self,
        client: AsyncTestClient[Litestar],
        owner: SimpleNamespace,
        params: dict[str, Any],
        status: int,
    ) -> None:
        artist_id = await _artist(client, owner)
        response = await client.get(
            f"{S}/artists/{artist_id}/memories", headers=owner.headers, params=params
        )
        assert response.status_code == status

    async def test_an_invalid_cursor_has_its_own_code(
        self, client: AsyncTestClient[Litestar], owner: SimpleNamespace
    ) -> None:
        artist_id = await _artist(client, owner)
        response = await client.get(
            f"{S}/artists/{artist_id}/memories", headers=owner.headers, params={"cursor": "nope"}
        )
        assert response.json()["error"] == "invalid_cursor"


# ---------------------------------------------------------------------------
# Scoping: another user's, another artist's and unknown ids are the same 404
# ---------------------------------------------------------------------------


class TestScoping:
    async def test_another_users_memory_and_artist_are_not_found(
        self,
        client: AsyncTestClient[Litestar],
        owner: SimpleNamespace,
        make_owner: Callable[[], Any],
    ) -> None:
        other = await make_owner()
        artist_id = await _artist(client, owner)
        memory = await _create(client, owner, artist_id)
        url = f"{S}/memories/{memory['id']}"

        for method, kwargs in (("get", {}), ("patch", {"json": {"salience": 1}}), ("delete", {})):
            response = await getattr(client, method)(url, headers=other.headers, **kwargs)
            assert (response.status_code, response.json()["error"]) == (404, "memory_not_found")
        unknown = await client.get(f"{S}/memories/{uuid4()}", headers=owner.headers)
        assert (unknown.status_code, unknown.json()["error"]) == (404, "memory_not_found")

        listing = await client.get(f"{S}/artists/{artist_id}/memories", headers=other.headers)
        assert (listing.status_code, listing.json()["error"]) == (404, "artist_not_found")
        creating = await client.post(
            f"{S}/artists/{artist_id}/memories", headers=other.headers, json=_body()
        )
        assert (creating.status_code, creating.json()["error"]) == (404, "artist_not_found")
        # nothing the other user did changed or removed the memory
        assert (await client.get(url, headers=owner.headers)).json() == memory

    async def test_another_product_cannot_see_it(
        self,
        client: AsyncTestClient[Litestar],
        owner: SimpleNamespace,
        session_factory: async_sessionmaker[AsyncSession],
    ) -> None:
        artist_id = await _artist(client, owner)
        memory = await _create(client, owner, artist_id)
        async with session_factory() as session:
            service = MemoryService(session)
            with pytest.raises(MemoryNotFound):
                await service.get(memory["id"], owner.id, "synthara")

    async def test_a_memory_is_listed_only_under_its_own_artist(
        self, client: AsyncTestClient[Litestar], owner: SimpleNamespace
    ) -> None:
        first = await _artist(client, owner, "First")
        second = await _artist(client, owner, "Second")
        memory = await _create(client, owner, first)
        listing = (await client.get(f"{S}/artists/{second}/memories", headers=owner.headers)).json()
        assert listing["items"] == []
        own = (await client.get(f"{S}/artists/{first}/memories", headers=owner.headers)).json()
        assert [m["id"] for m in own["items"]] == [memory["id"]]

    async def test_the_studio_switch_and_authentication_apply(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        owner: SimpleNamespace,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        app = _build_app(session_factory)
        async with AsyncTestClient(app=app) as client:
            anonymous = await client.get(f"{S}/artists/{uuid4()}/memories")
            assert anonymous.status_code == 401
            monkeypatch.setattr(
                "src.api.security.studio.get_settings",
                lambda: SimpleNamespace(studio_enabled=False),
            )
            for method, path in (
                ("get", f"{S}/artists/{uuid4()}/memories"),
                ("post", f"{S}/artists/{uuid4()}/memories"),
                ("get", f"{S}/memories/{uuid4()}"),
                ("patch", f"{S}/memories/{uuid4()}"),
                ("delete", f"{S}/memories/{uuid4()}"),
            ):
                response = await client.request(method, path, headers=owner.headers, json={})
                assert (response.status_code, response.json()["error"]) == (404, "not_found")


# ---------------------------------------------------------------------------
# Validation: every issue at once, at its wire path
# ---------------------------------------------------------------------------


def _issues(response: Any) -> list[tuple[str, str]]:
    assert response.status_code == 422, response.text
    assert response.json()["error"] == "document_invalid"
    return [(i["path"], i["code"]) for i in response.json()["detail"]["issues"]]


class TestValidation:
    async def test_every_issue_is_reported_together(
        self, client: AsyncTestClient[Litestar], owner: SimpleNamespace
    ) -> None:
        artist_id = await _artist(client, owner)
        response = await client.post(
            f"{S}/artists/{artist_id}/memories",
            headers=owner.headers,
            json={
                "title": "   ",
                "body": "\n \n",
                "themes": ["ok", " ", "OK", "a [b]"],
                "imagery": ["x", "X"],
            },
        )
        assert _issues(response) == [
            ("title", "empty_after_normalization"),
            ("body", "empty_after_normalization"),
            ("themes[1]", "empty_after_normalization"),
            ("themes[2]", "duplicate_tag"),
            ("themes[3]", "reserved_characters"),
            ("imagery[1]", "duplicate_tag"),
        ]
        listing = await client.get(f"{S}/artists/{artist_id}/memories", headers=owner.headers)
        assert listing.json()["items"] == []  # nothing was written

    async def test_nfc_expansion_is_reported_and_never_truncated_on_create(
        self, client: AsyncTestClient[Litestar], owner: SimpleNamespace
    ) -> None:
        """Each value is within its wire limit as sent and past it once normalized (RD6)."""
        artist_id = await _artist(client, owner)
        response = await client.post(
            f"{S}/artists/{artist_id}/memories",
            headers=owner.headers,
            json={
                "title": QA * (MEMORY_TITLE_MAX // 2 + 1),
                "body": QA * (MEMORY_BODY_MAX // 2 + 1),
                "themes": ["fine", QA * (MEMORY_THEME_MAX // 2 + 1)],
                "imagery": [QA * (MEMORY_IMAGE_MAX // 2 + 1)],
            },
        )
        assert _issues(response) == [
            ("title", "too_long_after_normalization"),
            ("body", "too_long_after_normalization"),
            ("themes[1]", "too_long_after_normalization"),
            ("imagery[0]", "too_long_after_normalization"),
        ]

    async def test_nfc_expansion_is_reported_on_patch_too(
        self, client: AsyncTestClient[Litestar], owner: SimpleNamespace
    ) -> None:
        artist_id = await _artist(client, owner)
        memory = await _create(client, owner, artist_id, themes=["keep"])
        response = await client.patch(
            f"{S}/memories/{memory['id']}",
            headers=owner.headers,
            json={"themes": [QA * (MEMORY_THEME_MAX // 2 + 1)]},
        )
        assert _issues(response) == [("themes[0]", "too_long_after_normalization")]
        stored = await client.get(f"{S}/memories/{memory['id']}", headers=owner.headers)
        assert stored.json()["themes"] == ["keep"]

    @pytest.mark.parametrize(
        "body",
        [
            {"title": "t", "body": ""},
            {"title": "t", "body": "b", "salience": 6},
            {"title": "t", "body": "b", "origin": "chat"},
            {"title": "t", "body": "b", "status": "archived"},
        ],
    )
    async def test_the_wire_refuses_before_the_service(
        self, client: AsyncTestClient[Litestar], owner: SimpleNamespace, body: dict[str, Any]
    ) -> None:
        artist_id = await _artist(client, owner)
        response = await client.post(
            f"{S}/artists/{artist_id}/memories", headers=owner.headers, json=body
        )
        assert response.status_code == 400

    async def test_patch_refuses_origin_and_empty_values(
        self, client: AsyncTestClient[Litestar], owner: SimpleNamespace
    ) -> None:
        artist_id = await _artist(client, owner)
        memory = await _create(client, owner, artist_id)
        url = f"{S}/memories/{memory['id']}"
        assert (
            await client.patch(url, headers=owner.headers, json={"origin": "chat"})
        ).status_code == 400
        blank = await client.patch(url, headers=owner.headers, json={"title": "   "})
        assert _issues(blank) == [("title", "empty_after_normalization")]


# ---------------------------------------------------------------------------
# Read-back (invariant 14): what is stored is what was validated
# ---------------------------------------------------------------------------


class TestReadBack:
    async def test_values_at_their_limits_are_stored_exactly(
        self,
        client: AsyncTestClient[Litestar],
        owner: SimpleNamespace,
        session_factory: async_sessionmaker[AsyncSession],
    ) -> None:
        """Varchar arrays truncate silently on the bind cast: read the rows back and compare."""
        artist_id = await _artist(client, owner)
        themes = [f"{n:02d}" + "t" * (MEMORY_THEME_MAX - 2) for n in range(MEMORY_THEMES_MAX)]
        imagery = [f"{n:02d}" + "i" * (MEMORY_IMAGE_MAX - 2) for n in range(MEMORY_IMAGERY_MAX)]
        created = await _create(
            client,
            owner,
            artist_id,
            title="T" * MEMORY_TITLE_MAX,
            body="b" * MEMORY_BODY_MAX,
            themes=themes,
            imagery=imagery,
        )
        async with session_factory() as session:
            row = (
                await session.execute(
                    select(StudioArtistMemory).where(StudioArtistMemory.id == created["id"])
                )
            ).scalar_one()
        assert row.title == "T" * MEMORY_TITLE_MAX
        assert row.body == "b" * MEMORY_BODY_MAX
        assert list(row.themes) == themes
        assert list(row.imagery) == imagery
        assert (created["themes"], created["imagery"]) == (themes, imagery)

    async def test_multi_codepoint_values_are_stored_as_the_nfc_form_that_was_validated(
        self,
        client: AsyncTestClient[Litestar],
        owner: SimpleNamespace,
        session_factory: async_sessionmaker[AsyncSession],
    ) -> None:
        artist_id = await _artist(client, owner)
        decomposed = "é" * 30  # NFC composes each pair to one code point: 30 characters
        created = await _create(client, owner, artist_id, themes=[decomposed], title=decomposed)
        async with session_factory() as session:
            row = (
                await session.execute(
                    select(StudioArtistMemory).where(StudioArtistMemory.id == created["id"])
                )
            ).scalar_one()
        assert list(row.themes) == ["é" * 30]
        assert row.title == "é" * 30


class TestCascade:
    async def test_deleting_an_artist_deletes_its_memories(
        self,
        client: AsyncTestClient[Litestar],
        owner: SimpleNamespace,
        session_factory: async_sessionmaker[AsyncSession],
    ) -> None:
        artist_id = await _artist(client, owner)
        first = await _create(client, owner, artist_id)
        second = await _create(client, owner, artist_id, title="second")
        assert (
            await client.delete(f"{S}/artists/{artist_id}", headers=owner.headers)
        ).status_code == 204
        for memory in (first, second):
            response = await client.get(f"{S}/memories/{memory['id']}", headers=owner.headers)
            assert response.status_code == 404
        async with session_factory() as session:
            remaining = (
                await session.execute(
                    select(StudioArtistMemory.id).where(StudioArtistMemory.artist_id == artist_id)
                )
            ).all()
        assert remaining == []
