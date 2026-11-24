"""The chat routes answer 404 while ``STUDIO_CHAT_ENABLED`` is off (invariant 16)."""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, Any
from uuid import UUID

import pytest
from litestar import Litestar
from litestar.exceptions import NotFoundException
from litestar.testing import AsyncTestClient

from src.api.routes.studio_chat import StudioChatController
from src.api.security.studio import studio_chat_guard, studio_feature_guard
from src.core.product_registry import VEX_CONFIG

if TYPE_CHECKING:
    from litestar.types import ASGIApp, Receive, Scope, Send

pytestmark = pytest.mark.unit

_ID = UUID(int=1)
CHAT_ROUTES = [
    ("GET", "/v1/studio/artists/{id}/chats"),
    ("POST", "/v1/studio/artists/{id}/chats"),
    ("GET", "/v1/studio/chats/{id}"),
    ("PATCH", "/v1/studio/chats/{id}"),
    ("DELETE", "/v1/studio/chats/{id}"),
    ("GET", "/v1/studio/chats/{id}/messages"),
    ("POST", "/v1/studio/chats/{id}/messages"),
    ("GET", "/v1/studio/chat-messages/{id}"),
    ("POST", "/v1/studio/chat-messages/{id}/memory-draft"),
    ("GET", "/v1/studio/memory-drafts/{id}"),
]
CHAT_REQUESTS = [(method, path.format(id=_ID)) for method, path in CHAT_ROUTES]
CHAT_IDS = [f"{method}-{path}" for method, path in CHAT_ROUTES]


def _settings(*, studio: bool, chat: bool) -> Any:
    return lambda: SimpleNamespace(studio_enabled=studio, studio_chat_enabled=chat)


async def test_the_chat_guard_is_a_404_when_the_switch_is_off(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("src.api.security.studio.get_settings", _settings(studio=True, chat=False))
    with pytest.raises(NotFoundException):
        await studio_chat_guard(None, None)  # type: ignore[arg-type]


async def test_the_chat_guard_passes_when_the_switch_is_on(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("src.api.security.studio.get_settings", _settings(studio=True, chat=True))
    await studio_chat_guard(None, None)  # type: ignore[arg-type]


def test_the_controller_runs_the_feature_guard_then_the_chat_guard_before_auth() -> None:
    guards = StudioChatController.guards or []
    assert [g.__name__ for g in guards] == [
        "studio_feature_guard",
        "studio_chat_guard",
        "auth_guard",
    ]
    assert guards[0] is studio_feature_guard


def _app() -> Litestar:
    """The chat controller behind a middleware that does what ``ProductMiddleware`` does."""

    def middleware_factory(app: ASGIApp) -> ASGIApp:
        async def middleware(scope: Scope, receive: Receive, send: Send) -> None:
            if scope["type"] == "http":
                state = scope.setdefault("state", {})
                state["product_config"] = VEX_CONFIG
                state["product_id"] = VEX_CONFIG.slug
            await app(scope, receive, send)

        return middleware

    return Litestar(route_handlers=[StudioChatController], middleware=[middleware_factory])


@pytest.mark.parametrize(("method", "path"), CHAT_REQUESTS, ids=CHAT_IDS)
async def test_every_chat_route_is_404_with_chat_off_even_unauthenticated(
    method: str, path: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("src.api.security.studio.get_settings", _settings(studio=True, chat=False))
    async with AsyncTestClient(app=_app()) as client:
        response = await client.request(method, path)
    assert response.status_code == 404


@pytest.mark.parametrize(("method", "path"), CHAT_REQUESTS, ids=CHAT_IDS)
async def test_with_chat_on_authentication_is_what_answers_next(
    method: str, path: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("src.api.security.studio.get_settings", _settings(studio=True, chat=True))
    async with AsyncTestClient(app=_app()) as client:
        response = await client.request(method, path)
    assert response.status_code != 404


@pytest.mark.parametrize(("method", "path"), CHAT_REQUESTS, ids=CHAT_IDS)
async def test_the_studio_switch_off_hides_the_chat_routes_too(
    method: str, path: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("src.api.security.studio.get_settings", _settings(studio=False, chat=True))
    async with AsyncTestClient(app=_app()) as client:
        response = await client.request(method, path)
    assert response.status_code == 404


def test_the_route_table_is_exactly_the_documented_endpoints() -> None:
    app = Litestar(route_handlers=[StudioChatController])
    found = sorted(
        (method, route.path)
        for route in app.routes
        if route.path.startswith("/v1/studio")
        for method in route.methods
        if method not in {"HEAD", "OPTIONS"}
    )
    assert found == sorted(
        [
            ("GET", "/v1/studio/artists/{artist_id:uuid}/chats"),
            ("POST", "/v1/studio/artists/{artist_id:uuid}/chats"),
            ("GET", "/v1/studio/chats/{chat_id:uuid}"),
            ("PATCH", "/v1/studio/chats/{chat_id:uuid}"),
            ("DELETE", "/v1/studio/chats/{chat_id:uuid}"),
            ("GET", "/v1/studio/chats/{chat_id:uuid}/messages"),
            ("POST", "/v1/studio/chats/{chat_id:uuid}/messages"),
            ("GET", "/v1/studio/chat-messages/{message_id:uuid}"),
            ("POST", "/v1/studio/chat-messages/{message_id:uuid}/memory-draft"),
            ("GET", "/v1/studio/memory-drafts/{draft_id:uuid}"),
        ]
    )
