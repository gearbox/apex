"""The run routes answer 404 while ``STUDIO_RUNS_ENABLED`` is off (invariant 20)."""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, Any
from uuid import uuid4

import pytest
from litestar import Litestar
from litestar.exceptions import NotFoundException
from litestar.testing import AsyncTestClient

from src.api.routes.studio_runs import StudioRunController
from src.api.security.studio import studio_feature_guard, studio_runs_guard
from src.core.product_registry import VEX_CONFIG

if TYPE_CHECKING:
    from litestar.types import ASGIApp, Receive, Scope, Send

pytestmark = pytest.mark.unit

RUN_REQUESTS = [
    ("POST", f"/v1/studio/tracks/{uuid4()}/runs"),
    ("GET", f"/v1/studio/runs/{uuid4()}"),
    ("GET", f"/v1/studio/tracks/{uuid4()}/runs"),
    ("POST", f"/v1/studio/runs/{uuid4()}/cancel"),
]


def _settings(*, studio: bool, runs: bool) -> Any:
    return lambda: SimpleNamespace(studio_enabled=studio, studio_runs_enabled=runs)


async def test_the_runs_guard_is_a_404_when_the_switch_is_off(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("src.api.security.studio.get_settings", _settings(studio=True, runs=False))
    with pytest.raises(NotFoundException):
        await studio_runs_guard(None, None)  # type: ignore[arg-type]


async def test_the_runs_guard_passes_when_the_switch_is_on(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("src.api.security.studio.get_settings", _settings(studio=True, runs=True))
    await studio_runs_guard(None, None)  # type: ignore[arg-type]


def test_the_controller_runs_the_feature_guard_then_the_runs_guard_before_auth() -> None:
    guards = StudioRunController.guards or []
    assert [g.__name__ for g in guards] == [
        "studio_feature_guard",
        "studio_runs_guard",
        "auth_guard",
    ]
    assert guards[0] is studio_feature_guard


def _app() -> Litestar:
    """The run controller behind a middleware that does what ``ProductMiddleware`` does."""

    def middleware_factory(app: ASGIApp) -> ASGIApp:
        async def middleware(scope: Scope, receive: Receive, send: Send) -> None:
            if scope["type"] == "http":
                state = scope.setdefault("state", {})
                state["product_config"] = VEX_CONFIG
                state["product_id"] = VEX_CONFIG.slug
            await app(scope, receive, send)

        return middleware

    return Litestar(route_handlers=[StudioRunController], middleware=[middleware_factory])


@pytest.mark.parametrize(("method", "path"), RUN_REQUESTS)
async def test_every_run_route_is_404_with_runs_off_even_unauthenticated(
    method: str, path: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("src.api.security.studio.get_settings", _settings(studio=True, runs=False))
    async with AsyncTestClient(app=_app()) as client:
        response = await client.request(method, path)
    assert response.status_code == 404


@pytest.mark.parametrize(("method", "path"), RUN_REQUESTS)
async def test_with_runs_on_authentication_is_what_answers_next(
    method: str, path: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("src.api.security.studio.get_settings", _settings(studio=True, runs=True))
    async with AsyncTestClient(app=_app()) as client:
        response = await client.request(method, path)
    assert response.status_code != 404


@pytest.mark.parametrize(("method", "path"), RUN_REQUESTS)
async def test_the_studio_switch_off_hides_the_run_routes_too(
    method: str, path: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("src.api.security.studio.get_settings", _settings(studio=False, runs=True))
    async with AsyncTestClient(app=_app()) as client:
        response = await client.request(method, path)
    assert response.status_code == 404


def test_the_route_table_has_exactly_the_four_endpoints() -> None:
    app = Litestar(route_handlers=[StudioRunController])
    found = sorted(
        (method, route.path)
        for route in app.routes
        if route.path.startswith("/v1/studio")
        for method in route.methods
        if method not in {"HEAD", "OPTIONS"}
    )
    assert found == sorted(
        [
            ("POST", "/v1/studio/tracks/{track_id:uuid}/runs"),
            ("GET", "/v1/studio/runs/{run_id:uuid}"),
            ("GET", "/v1/studio/tracks/{track_id:uuid}/runs"),
            ("POST", "/v1/studio/runs/{run_id:uuid}/cancel"),
        ]
    )
