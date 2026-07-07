"""GET /health/ready echoes the running build's SHA (H3).

The staging deploy verification step polls this endpoint and compares
build_sha against github.sha to confirm the redeploy actually took effect
rather than the old container still serving traffic.
"""

from __future__ import annotations

import pytest
from litestar.testing import AsyncTestClient

from src.core.config import get_settings

pytestmark = pytest.mark.unit

_PRODUCT_HEADERS = {"X-Product-Id": "vex"}


@pytest.fixture(autouse=True)
def _clear_settings_cache() -> None:
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def _set_minimal_env(monkeypatch: pytest.MonkeyPatch, *, build_sha: str | None) -> None:
    monkeypatch.setenv("JWT_SECRET_KEY", "ci-unit-test-secret-key-32-bytes-long")
    monkeypatch.setenv("AISHA_POLLER_ENABLED", "false")
    if build_sha is not None:
        monkeypatch.setenv("BUILD_SHA", build_sha)
    get_settings.cache_clear()


class TestBuildShaEcho:
    async def test_build_sha_reflects_env_var(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _set_minimal_env(monkeypatch, build_sha="abc123")
        from src.api.app import create_app

        async with AsyncTestClient(app=create_app()) as client:
            resp = await client.get("/health/ready", headers=_PRODUCT_HEADERS)

        body = resp.json()
        assert body["build_sha"] == "abc123"

    async def test_build_sha_defaults_to_unknown(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _set_minimal_env(monkeypatch, build_sha=None)
        from src.api.app import create_app

        async with AsyncTestClient(app=create_app()) as client:
            resp = await client.get("/health/ready", headers=_PRODUCT_HEADERS)

        body = resp.json()
        assert body["build_sha"] == "unknown"

    async def test_status_and_checks_unchanged(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Regression: adding build_sha must not alter existing readiness fields."""
        _set_minimal_env(monkeypatch, build_sha="abc123")
        from src.api.app import create_app

        async with AsyncTestClient(app=create_app()) as client:
            resp = await client.get("/health/ready", headers=_PRODUCT_HEADERS)

        body = resp.json()
        assert "status" in body
        assert "checks" in body
        assert isinstance(body["checks"], dict)
