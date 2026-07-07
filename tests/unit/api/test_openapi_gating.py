"""OpenAPI docs gating — /docs must be 404 unless ENABLE_DOCS=true."""

from __future__ import annotations

from collections.abc import Generator

import pytest
from litestar.status_codes import HTTP_200_OK, HTTP_404_NOT_FOUND
from litestar.testing import AsyncTestClient

from src.core.config import get_settings

pytestmark = pytest.mark.unit

# ProductMiddleware runs on every route (incl. /docs) and 400s if it can't
# resolve a product from Origin/Host/X-Product-Id — supply it explicitly.
_PRODUCT_HEADERS = {"X-Product-Id": "vex"}


@pytest.fixture(autouse=True)
def _clear_settings_cache() -> Generator[None]:
    """get_settings() is lru_cache'd — flipping env-driven settings mid-suite
    requires clearing it before *and* after each test to avoid ordering-dependent
    flakes against other test modules.
    """
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def _set_minimal_env(monkeypatch: pytest.MonkeyPatch, *, enable_docs: bool) -> None:
    monkeypatch.setenv("JWT_SECRET_KEY", "ci-unit-test-secret-key-32-bytes-long")
    monkeypatch.setenv("ENABLE_DOCS", "true" if enable_docs else "false")
    # Avoid the aisha poller's startup domain-guard — irrelevant to this test.
    monkeypatch.setenv("AISHA_POLLER_ENABLED", "false")
    get_settings.cache_clear()


class TestOpenAPIGatingDisabled:
    async def test_docs_json_returns_404_by_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _set_minimal_env(monkeypatch, enable_docs=False)
        from src.api.app import create_app

        async with AsyncTestClient(app=create_app()) as client:
            resp = await client.get("/docs/openapi.json", headers=_PRODUCT_HEADERS)
        assert resp.status_code == HTTP_404_NOT_FOUND

    async def test_docs_swagger_returns_404_by_default(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _set_minimal_env(monkeypatch, enable_docs=False)
        from src.api.app import create_app

        async with AsyncTestClient(app=create_app()) as client:
            resp = await client.get("/docs", headers=_PRODUCT_HEADERS)
        assert resp.status_code == HTTP_404_NOT_FOUND


class TestOpenAPIGatingEnabled:
    async def test_docs_json_returns_200_when_enabled(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _set_minimal_env(monkeypatch, enable_docs=True)
        from src.api.app import create_app

        async with AsyncTestClient(app=create_app()) as client:
            resp = await client.get("/docs/openapi.json", headers=_PRODUCT_HEADERS)
        assert resp.status_code == HTTP_200_OK
