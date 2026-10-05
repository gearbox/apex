"""Credentialed-CORS and ``Vary: Origin`` behaviour of the content proxy (/v1/content/*).

The frontend frame extractor loads content through a hidden
``<video crossorigin="use-credentials">`` — a CORS-mode request — while the rest of
the app plays the same URLs without ``crossorigin`` (a no-cors request carrying no
``Origin`` header). Content responses are ``private, max-age=N, immutable``, so a
response cached for the no-cors request *must* carry ``Vary: Origin``; otherwise the
browser reuses it for the CORS-mode request, finds no ``Access-Control-Allow-Origin``,
fails the CORS check, and (being ``immutable``) never revalidates.

Runs the real controller + guard + CORS middleware with in-memory service doubles
(see ``test_content_range_streaming``), authenticated by a real content cookie.
"""

from __future__ import annotations

from datetime import timedelta
from typing import TYPE_CHECKING
from uuid import uuid4

import pytest
from litestar.status_codes import (
    HTTP_200_OK,
    HTTP_206_PARTIAL_CONTENT,
    HTTP_304_NOT_MODIFIED,
    HTTP_404_NOT_FOUND,
    HTTP_416_REQUESTED_RANGE_NOT_SATISFIABLE,
)
from litestar.testing import TestClient

from src.api.security.jwt import JWTConfig, JWTService
from src.core.product_registry import VEX_CONFIG
from tests.integration.test_content_range_streaming import (
    COOKIE_NAME,
    ETAG,
    PAYLOAD,
    PRODUCT_ID,
    TEST_SECRET,
    _make_app,
    _NotFoundContentProxy,
    _StubContentProxy,
    _StubR2,
)
from tests.integration.test_cors import _build_cors_config

if TYPE_CHECKING:
    from litestar import Litestar

ALLOWED_ORIGIN = f"https://{next(iter(VEX_CONFIG.domains))}"
DISALLOWED_ORIGIN = "https://evil.example.org"

_KINDS = ["outputs", "uploads"]


@pytest.fixture
def jwt_service() -> JWTService:
    return JWTService(JWTConfig(secret_key=TEST_SECRET))


@pytest.fixture
def content_cookie(jwt_service: JWTService) -> dict[str, str]:
    token, _ = jwt_service.create_content_token(
        uuid4(), product_id=PRODUCT_ID, ttl=timedelta(hours=1)
    )
    return {COOKIE_NAME: token}


def _app(jwt_service: JWTService, *, found: bool = True) -> Litestar:
    return _make_app(
        _StubContentProxy() if found else _NotFoundContentProxy(),
        _StubR2(),
        jwt_service,
        cors_config=_build_cors_config(),
    )


def _vary_tokens(headers: dict[str, str] | object) -> list[str]:
    raw = headers.get("vary", "")  # type: ignore[attr-defined]
    return [t.strip() for t in raw.split(",") if t.strip()]


@pytest.mark.parametrize("kind", _KINDS)
class TestCredentialedCors:
    def test_allowed_origin_gets_exact_acao_credentials_and_vary(
        self, kind: str, jwt_service: JWTService, content_cookie: dict[str, str]
    ) -> None:
        with TestClient(app=_app(jwt_service), cookies=content_cookie) as client:
            resp = client.get(f"/v1/content/{kind}/{uuid4()}", headers={"Origin": ALLOWED_ORIGIN})

        assert resp.status_code == HTTP_200_OK
        assert resp.headers["access-control-allow-origin"] == ALLOWED_ORIGIN
        assert resp.headers["access-control-allow-credentials"] == "true"
        assert _vary_tokens(resp.headers) == ["Origin"]
        assert resp.content == PAYLOAD

    def test_range_request_206_keeps_cors_headers(
        self, kind: str, jwt_service: JWTService, content_cookie: dict[str, str]
    ) -> None:
        with TestClient(app=_app(jwt_service), cookies=content_cookie) as client:
            resp = client.get(
                f"/v1/content/{kind}/{uuid4()}",
                headers={"Origin": ALLOWED_ORIGIN, "Range": "bytes=0-99"},
            )

        assert resp.status_code == HTTP_206_PARTIAL_CONTENT
        assert resp.headers["content-range"] == f"bytes 0-99/{len(PAYLOAD)}"
        assert resp.headers["access-control-allow-origin"] == ALLOWED_ORIGIN
        assert resp.headers["access-control-allow-credentials"] == "true"
        assert _vary_tokens(resp.headers) == ["Origin"]

    def test_no_origin_header_still_varies_on_origin(
        self, kind: str, jwt_service: JWTService, content_cookie: dict[str, str]
    ) -> None:
        """The cache-poisoning case: a no-cors response must already say ``Vary: Origin``."""
        with TestClient(app=_app(jwt_service), cookies=content_cookie) as client:
            resp = client.get(f"/v1/content/{kind}/{uuid4()}")

        assert resp.status_code == HTTP_200_OK
        assert _vary_tokens(resp.headers) == ["Origin"]
        assert "access-control-allow-origin" not in resp.headers

    def test_disallowed_origin_gets_no_acao(
        self, kind: str, jwt_service: JWTService, content_cookie: dict[str, str]
    ) -> None:
        with TestClient(app=_app(jwt_service), cookies=content_cookie) as client:
            resp = client.get(
                f"/v1/content/{kind}/{uuid4()}", headers={"Origin": DISALLOWED_ORIGIN}
            )

        assert "access-control-allow-origin" not in resp.headers
        assert "origin" in [t.lower() for t in _vary_tokens(resp.headers)]

    @pytest.mark.parametrize("origin", [None, ALLOWED_ORIGIN])
    def test_304_carries_vary(
        self,
        kind: str,
        origin: str | None,
        jwt_service: JWTService,
        content_cookie: dict[str, str],
    ) -> None:
        headers = {"If-None-Match": f'"{ETAG}"'}
        if origin is not None:
            headers["Origin"] = origin
        with TestClient(app=_app(jwt_service), cookies=content_cookie) as client:
            resp = client.get(f"/v1/content/{kind}/{uuid4()}", headers=headers)

        assert resp.status_code == HTTP_304_NOT_MODIFIED
        assert _vary_tokens(resp.headers) == ["Origin"]

    @pytest.mark.parametrize("origin", [None, ALLOWED_ORIGIN])
    def test_404_carries_vary(
        self,
        kind: str,
        origin: str | None,
        jwt_service: JWTService,
        content_cookie: dict[str, str],
    ) -> None:
        headers = {"Origin": origin} if origin is not None else {}
        with TestClient(app=_app(jwt_service, found=False), cookies=content_cookie) as client:
            resp = client.get(f"/v1/content/{kind}/{uuid4()}", headers=headers)

        assert resp.status_code == HTTP_404_NOT_FOUND
        assert _vary_tokens(resp.headers) == ["Origin"]

    @pytest.mark.parametrize("origin", [None, ALLOWED_ORIGIN])
    def test_416_carries_vary(
        self,
        kind: str,
        origin: str | None,
        jwt_service: JWTService,
        content_cookie: dict[str, str],
    ) -> None:
        headers = {"Range": f"bytes={len(PAYLOAD) + 10}-{len(PAYLOAD) + 20}"}
        if origin is not None:
            headers["Origin"] = origin
        with TestClient(app=_app(jwt_service), cookies=content_cookie) as client:
            resp = client.get(f"/v1/content/{kind}/{uuid4()}", headers=headers)

        assert resp.status_code == HTTP_416_REQUESTED_RANGE_NOT_SATISFIABLE
        assert _vary_tokens(resp.headers) == ["Origin"]
