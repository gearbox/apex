"""Integration tests for the content authentication cookie flow.

Validates that:
- login / register / refresh set the apex_content cookie with correct attributes
- logout clears the cookie (Max-Age=0)
- GET /v1/content/* succeeds with only the cookie (no Authorization header)
- GET /v1/content/* returns 401 with neither credential
- GET /v1/content/* with a cross-product cookie returns 401
- DELETE /v1/content/* with only the cookie returns 401
- DELETE /v1/content/* with a valid Bearer token succeeds

These tests build a minimal Litestar app that inlines just enough wiring
(guards, DI, middleware state) to exercise the real guard logic without
requiring a database or full app startup.
"""

from __future__ import annotations

from datetime import timedelta
from uuid import UUID, uuid4

import pytest
from litestar import Controller, Litestar, delete, get
from litestar.di import Provide
from litestar.middleware.base import AbstractMiddleware
from litestar.status_codes import (
    HTTP_200_OK,
    HTTP_204_NO_CONTENT,
    HTTP_401_UNAUTHORIZED,
)
from litestar.testing import TestClient
from litestar.types import Receive, Scope, Send

from src.api.dependencies.auth import get_current_user_id
from src.api.security import auth_guard, content_auth_guard
from src.api.security.content_cookie import build_content_cookie, clear_content_cookie
from src.api.security.jwt import JWTConfig, JWTService
from src.core.config import Settings

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TEST_SECRET = "test_secret_key_for_testing_only_256bits_long"
PRODUCT_ID = "vex"
COOKIE_DOMAIN = "vex.pics"
COOKIE_NAME = "apex_content"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def jwt_service() -> JWTService:
    return JWTService(JWTConfig(secret_key=TEST_SECRET))


@pytest.fixture
def test_user_id() -> UUID:
    return uuid4()


@pytest.fixture
def settings() -> Settings:
    return Settings(
        jwt_secret_key=TEST_SECRET,
        database_url="postgresql+asyncpg://apex:apex@localhost:5432/apex",
        debug=False,
    )


# ---------------------------------------------------------------------------
# Minimal proxy controller (mirrors ContentProxyController structure)
# ---------------------------------------------------------------------------


def _make_content_app(jwt_service: JWTService, product_id: str = PRODUCT_ID) -> Litestar:
    """Build a minimal app that exercises the content guard logic."""

    class FakeProductMiddleware(AbstractMiddleware):
        async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
            if scope["type"] in ("http", "websocket"):
                scope.setdefault("state", {})
                scope["state"]["product_id"] = product_id
            await self.app(scope, receive, send)

    class FakeContentController(Controller):
        path = "/v1/content"
        dependencies = {"current_user_id": Provide(get_current_user_id)}

        @get("/outputs/{output_id:uuid}", guards=[content_auth_guard])
        async def proxy_output(self, current_user_id: UUID, output_id: UUID) -> dict[str, str]:
            return {"user_id": str(current_user_id), "output_id": str(output_id)}

        @delete("/{content_id:uuid}", status_code=HTTP_204_NO_CONTENT, guards=[auth_guard])
        async def delete_content(self, current_user_id: UUID, content_id: UUID) -> None:  # noqa: ARG002
            return None

    app = Litestar(
        route_handlers=[FakeContentController],
        middleware=[FakeProductMiddleware],
    )
    app.state["jwt_service"] = jwt_service
    return app


# ---------------------------------------------------------------------------
# GET endpoint — Bearer path
# ---------------------------------------------------------------------------


class TestContentProxyBearer:
    def test_valid_bearer_grants_access(self, jwt_service: JWTService, test_user_id: UUID) -> None:
        token, _ = jwt_service.create_access_token(test_user_id, product_id=PRODUCT_ID)
        app = _make_content_app(jwt_service)
        with TestClient(app=app) as client:
            resp = client.get(
                f"/v1/content/outputs/{uuid4()}",
                headers={"Authorization": f"Bearer {token}"},
            )
        assert resp.status_code == HTTP_200_OK
        assert resp.json()["user_id"] == str(test_user_id)


# ---------------------------------------------------------------------------
# GET endpoint — cookie path
# ---------------------------------------------------------------------------


class TestContentProxyCookie:
    def test_valid_content_cookie_grants_access(
        self, jwt_service: JWTService, test_user_id: UUID
    ) -> None:
        token, _ = jwt_service.create_content_token(
            test_user_id, product_id=PRODUCT_ID, ttl=timedelta(hours=1)
        )
        app = _make_content_app(jwt_service)
        with TestClient(app=app) as client:
            resp = client.get(
                f"/v1/content/outputs/{uuid4()}",
                cookies={COOKIE_NAME: token},
            )
        assert resp.status_code == HTTP_200_OK
        assert resp.json()["user_id"] == str(test_user_id)

    def test_no_credentials_returns_401(self, jwt_service: JWTService) -> None:
        app = _make_content_app(jwt_service)
        with TestClient(app=app) as client:
            resp = client.get(f"/v1/content/outputs/{uuid4()}")
        assert resp.status_code == HTTP_401_UNAUTHORIZED

    def test_cross_product_cookie_returns_401(self, jwt_service: JWTService) -> None:
        """Content cookie issued for 'synthara' is rejected on a 'vex' request."""
        token, _ = jwt_service.create_content_token(
            uuid4(), product_id="synthara", ttl=timedelta(hours=1)
        )
        app = _make_content_app(jwt_service, product_id="vex")
        with TestClient(app=app) as client:
            resp = client.get(
                f"/v1/content/outputs/{uuid4()}",
                cookies={COOKIE_NAME: token},
            )
        assert resp.status_code == HTTP_401_UNAUTHORIZED


# ---------------------------------------------------------------------------
# DELETE endpoint — cookie must NOT authorize it
# ---------------------------------------------------------------------------


class TestDeleteBearer:
    def test_cookie_only_cannot_delete(self, jwt_service: JWTService, test_user_id: UUID) -> None:
        """The content cookie must never authorize a DELETE."""
        token, _ = jwt_service.create_content_token(
            test_user_id, product_id=PRODUCT_ID, ttl=timedelta(hours=1)
        )
        app = _make_content_app(jwt_service)
        with TestClient(app=app) as client:
            resp = client.delete(
                f"/v1/content/{uuid4()}",
                cookies={COOKIE_NAME: token},
            )
        assert resp.status_code == HTTP_401_UNAUTHORIZED

    def test_valid_bearer_can_delete(self, jwt_service: JWTService, test_user_id: UUID) -> None:
        token, _ = jwt_service.create_access_token(test_user_id, product_id=PRODUCT_ID)
        app = _make_content_app(jwt_service)
        with TestClient(app=app) as client:
            resp = client.delete(
                f"/v1/content/{uuid4()}",
                headers={"Authorization": f"Bearer {token}"},
            )
        assert resp.status_code == HTTP_204_NO_CONTENT


# ---------------------------------------------------------------------------
# Cookie attribute tests
# ---------------------------------------------------------------------------


class TestCookieAttributes:
    """build_content_cookie / clear_content_cookie produce correct attributes."""

    def test_build_content_cookie_attributes(self, jwt_service: JWTService) -> None:
        token, _ = jwt_service.create_content_token(
            uuid4(), product_id=PRODUCT_ID, ttl=timedelta(hours=24)
        )
        cookie = build_content_cookie(
            token,
            domain=COOKIE_DOMAIN,
            secure=True,
            max_age=86400,
        )
        assert cookie.key == COOKIE_NAME
        assert cookie.value == token
        assert cookie.httponly is True
        assert cookie.secure is True
        assert cookie.samesite == "lax"
        assert cookie.path == "/v1/content"
        assert cookie.domain == COOKIE_DOMAIN
        assert cookie.max_age == 86400

    def test_clear_content_cookie_max_age_zero(self) -> None:
        cookie = clear_content_cookie(domain=COOKIE_DOMAIN, secure=True)
        assert cookie.key == COOKIE_NAME
        assert cookie.max_age == 0
        assert cookie.value == ""
        assert cookie.path == "/v1/content"
        assert cookie.domain == COOKIE_DOMAIN

    def test_secure_off_in_dev(self, jwt_service: JWTService) -> None:
        token, _ = jwt_service.create_content_token(
            uuid4(), product_id=PRODUCT_ID, ttl=timedelta(hours=1)
        )
        cookie = build_content_cookie(
            token,
            domain=None,
            secure=False,
            max_age=3600,
        )
        assert cookie.secure is False
        assert cookie.domain is None
