"""Integration tests verifying CORS configuration.

Checks that credentialed preflight requests are allowed from registered
product origins and rejected from unknown origins, and that no wildcard
origin is ever emitted (which would invalidate credentials).
"""

from __future__ import annotations

import re

import pytest
from litestar import Litestar, get
from litestar.config.cors import CORSConfig
from litestar.testing import TestClient

from src.core.product_registry import PRODUCT_REGISTRY

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_cors_config(debug: bool = False) -> CORSConfig:
    """Mirror the logic in src/api/app.py create_app()."""
    domains = sorted({d for pc in PRODUCT_REGISTRY.values() for d in pc.domains})
    escaped = "|".join(re.escape(d) for d in domains)
    if debug:
        origin_regex = rf"^(https://([a-z0-9-]+\.)?({escaped})|http://localhost(:\d+)?)$"
    else:
        origin_regex = rf"^https://([a-z0-9-]+\.)?({escaped})$"
    return CORSConfig(
        allow_origins=[],  # must be empty — origin matching is done by allow_origin_regex
        allow_origin_regex=origin_regex,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PATCH", "DELETE", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type", "X-Product-Id"],
    )


def _make_app(debug: bool = False) -> Litestar:
    @get("/test")
    async def dummy() -> dict[str, str]:
        return {"ok": "1"}

    return Litestar(route_handlers=[dummy], cors_config=_build_cors_config(debug))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestCORSAllowedOrigins:
    """Preflight from registered product origins is accepted with credentials."""

    @pytest.mark.parametrize(
        "origin",
        [
            "https://vex.pics",
            "https://app.vex.pics",
            "https://staging.vex.pics",
            "https://synthara.app",
            "https://app.synthara.app",
            "https://preview.synthara.app",
        ],
    )
    def test_preflight_allowed_for_product_origin(self, origin: str) -> None:
        app = _make_app()
        with TestClient(app=app) as client:
            resp = client.options(
                "/test",
                headers={
                    "Origin": origin,
                    "Access-Control-Request-Method": "GET",
                    "Access-Control-Request-Headers": "Authorization",
                },
            )
        assert resp.status_code in (200, 204)
        assert resp.headers.get("Access-Control-Allow-Credentials") == "true"
        echoed_origin = resp.headers.get("Access-Control-Allow-Origin")
        assert echoed_origin == origin, f"Expected echoed origin, got {echoed_origin!r}"

    def test_no_wildcard_origin_emitted(self) -> None:
        app = _make_app()
        with TestClient(app=app) as client:
            resp = client.options(
                "/test",
                headers={
                    "Origin": "https://vex.pics",
                    "Access-Control-Request-Method": "GET",
                },
            )
        assert resp.headers.get("Access-Control-Allow-Origin") != "*"


class TestCORSRejectedOrigins:
    """Unknown origins receive no ACAO header (or a non-matching one)."""

    @pytest.mark.parametrize(
        "origin",
        [
            "https://evil.com",
            "https://notvex.pics",
            "https://vex.pics.evil.com",
            "http://vex.pics",  # http not https
        ],
    )
    def test_unknown_origin_rejected(self, origin: str) -> None:
        app = _make_app()
        with TestClient(app=app) as client:
            resp = client.options(
                "/test",
                headers={
                    "Origin": origin,
                    "Access-Control-Request-Method": "GET",
                },
            )
        acao = resp.headers.get("Access-Control-Allow-Origin", "")
        assert acao not in (origin, "*")


class TestCORSDevMode:
    """In debug mode localhost is also an allowed origin."""

    def test_localhost_allowed_in_debug(self) -> None:
        app = _make_app(debug=True)
        with TestClient(app=app) as client:
            resp = client.options(
                "/test",
                headers={
                    "Origin": "http://localhost:3000",
                    "Access-Control-Request-Method": "GET",
                },
            )
        assert resp.headers.get("Access-Control-Allow-Origin") == "http://localhost:3000"
        assert resp.headers.get("Access-Control-Allow-Credentials") == "true"

    def test_localhost_rejected_in_prod(self) -> None:
        app = _make_app(debug=False)
        with TestClient(app=app) as client:
            resp = client.options(
                "/test",
                headers={
                    "Origin": "http://localhost:3000",
                    "Access-Control-Request-Method": "GET",
                },
            )
        acao = resp.headers.get("Access-Control-Allow-Origin", "")
        assert acao != "http://localhost:3000"
