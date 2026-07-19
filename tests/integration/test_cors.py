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

from src.core.product_registry import PRODUCT_REGISTRY, SYNTHARA_CONFIG, VEX_CONFIG

# Derived from the real registry rather than hardcoded, so this file never
# needs to spell out the product's actual domain.
_VEX_DOMAIN = next(iter(VEX_CONFIG.domains))
_SYNTHARA_DOMAIN = next(iter(SYNTHARA_CONFIG.domains))

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
        allow_headers=[
            "Authorization",
            "Content-Type",
            "X-Product-Id",
            "Idempotency-Key",
            "X-Request-Id",
        ],
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
            f"https://{_VEX_DOMAIN}",
            f"https://app.{_VEX_DOMAIN}",
            f"https://staging.{_VEX_DOMAIN}",
            f"https://{_SYNTHARA_DOMAIN}",
            f"https://app.{_SYNTHARA_DOMAIN}",
            f"https://preview.{_SYNTHARA_DOMAIN}",
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
                    "Origin": f"https://{_VEX_DOMAIN}",
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
            f"https://not{_VEX_DOMAIN}",
            f"https://{_VEX_DOMAIN}.evil.com",
            f"http://{_VEX_DOMAIN}",  # http not https
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


class TestCORSIdempotencyKey:
    """Regression: generation + billing preflights must include Idempotency-Key."""

    @pytest.mark.parametrize(
        "origin",
        [f"https://{_VEX_DOMAIN}", f"https://{_SYNTHARA_DOMAIN}"],
    )
    def test_preflight_allows_idempotency_key(self, origin: str) -> None:
        app = _make_app()
        with TestClient(app=app) as client:
            resp = client.options(
                "/test",
                headers={
                    "Origin": origin,
                    "Access-Control-Request-Method": "POST",
                    "Access-Control-Request-Headers": "Authorization, Content-Type, Idempotency-Key",
                },
            )
        assert resp.status_code in (200, 204)
        allowed = resp.headers.get("Access-Control-Allow-Headers", "")
        allowed_lower = allowed.lower()
        assert "idempotency-key" in allowed_lower, (
            f"Expected Idempotency-Key in Access-Control-Allow-Headers, got: {allowed!r}"
        )

    def test_preflight_allows_x_request_id(self) -> None:
        app = _make_app()
        with TestClient(app=app) as client:
            resp = client.options(
                "/test",
                headers={
                    "Origin": f"https://{_VEX_DOMAIN}",
                    "Access-Control-Request-Method": "POST",
                    "Access-Control-Request-Headers": "X-Request-Id",
                },
            )
        assert resp.status_code in (200, 204)
        allowed = resp.headers.get("Access-Control-Allow-Headers", "")
        assert "x-request-id" in allowed.lower()


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
