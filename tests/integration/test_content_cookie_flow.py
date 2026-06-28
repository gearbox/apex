"""Integration tests for the content authentication cookie flow.

Validates that:
- login / register / refresh set the apex_content cookie with correct attributes
- logout clears the cookie (Max-Age=0)
- GET /v1/content/* succeeds with only the cookie (no Authorization header)
- GET /v1/content/* returns 401 with neither credential
- GET /v1/content/* with a cross-product cookie returns 401
- DELETE /v1/content/* with only the cookie returns 401
- DELETE /v1/content/* with a valid Bearer token succeeds

Auth-route tests use the real AuthController handlers via .fn() with mocked
AuthService (no DB required). Guard tests build a minimal Litestar app.
"""

from __future__ import annotations

from datetime import timedelta
from typing import Any
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

    def test_default_cookie_ttl_is_1_hour(self) -> None:
        s = Settings(
            jwt_secret_key=TEST_SECRET,
            database_url="postgresql+asyncpg://apex:apex@localhost:5432/apex",
        )
        assert s.content_cookie_ttl_hours == 1

    def test_cookie_ttl_above_max_rejected(self) -> None:
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            Settings(
                jwt_secret_key=TEST_SECRET,
                database_url="postgresql+asyncpg://apex:apex@localhost:5432/apex",
                content_cookie_ttl_hours=25,
            )

    def test_cookie_ttl_below_min_rejected(self) -> None:
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            Settings(
                jwt_secret_key=TEST_SECRET,
                database_url="postgresql+asyncpg://apex:apex@localhost:5432/apex",
                content_cookie_ttl_hours=0,
            )


# ---------------------------------------------------------------------------
# Real AuthController handler tests (no DB — AuthService is mocked)
# ---------------------------------------------------------------------------


def _make_product_config(domain: str = COOKIE_DOMAIN) -> Any:
    from unittest.mock import MagicMock

    cfg = MagicMock()
    cfg.cookie_domain = domain
    return cfg


def _make_token_pair() -> Any:
    from unittest.mock import MagicMock

    pair = MagicMock()
    pair.access_token = "access.token.here"
    pair.refresh_token = "refresh.token.here"
    pair.expires_in = 3600
    pair.expires_at = None
    return pair


class TestAuthControllerCookies:
    """AuthController register / login / refresh set; logout clears the content cookie."""

    async def test_register_sets_content_cookie(
        self, jwt_service: JWTService, settings: Settings
    ) -> None:
        from unittest.mock import AsyncMock, MagicMock

        from src.api.routes.auth import AuthController
        from src.api.schemas.auth import RegisterRequest

        user = MagicMock()
        user.id = uuid4()
        pair = _make_token_pair()

        mock_auth = AsyncMock()
        mock_auth.register = AsyncMock(return_value=(user, pair))

        response = await AuthController.register.fn(  # type: ignore[attr-defined]
            MagicMock(),
            data=RegisterRequest(email="a@b.com", password="pass1234"),
            auth_service=mock_auth,
            jwt_service=jwt_service,
            product_id=PRODUCT_ID,
            product_config=_make_product_config(),
            settings=settings,
        )

        assert len(response.cookies) == 1
        cookie = response.cookies[0]
        assert cookie.key == COOKIE_NAME
        assert cookie.httponly is True
        assert cookie.samesite == "lax"
        assert cookie.path == "/v1/content"
        assert cookie.domain == COOKIE_DOMAIN
        assert cookie.max_age == settings.content_cookie_ttl_hours * 3600
        assert cookie.secure == settings.content_cookie_secure

    async def test_login_sets_content_cookie(
        self, jwt_service: JWTService, settings: Settings
    ) -> None:
        from unittest.mock import AsyncMock, MagicMock

        from litestar import Request

        from src.api.routes.auth import AuthController
        from src.api.schemas.auth import LoginRequest

        user = MagicMock()
        user.id = uuid4()
        pair = _make_token_pair()

        mock_auth = AsyncMock()
        mock_auth.login = AsyncMock(return_value=(user, pair))

        mock_request = MagicMock(spec=Request)
        mock_request.headers.get.return_value = None
        mock_request.client = None

        response = await AuthController.login.fn(  # type: ignore[attr-defined]
            MagicMock(),
            request=mock_request,
            data=LoginRequest(email="a@b.com", password="pass1234"),
            auth_service=mock_auth,
            jwt_service=jwt_service,
            product_id=PRODUCT_ID,
            product_config=_make_product_config(),
            settings=settings,
        )

        assert len(response.cookies) == 1
        cookie = response.cookies[0]
        assert cookie.key == COOKIE_NAME
        assert cookie.httponly is True
        assert cookie.samesite == "lax"
        assert cookie.path == "/v1/content"
        assert cookie.domain == COOKIE_DOMAIN

    async def test_refresh_sets_content_cookie_without_decoding(
        self, jwt_service: JWTService, settings: Settings
    ) -> None:
        """refresh sets the cookie via user_id from AuthService — no access-token decode."""
        from unittest.mock import AsyncMock, MagicMock

        from litestar import Request

        from src.api.routes.auth import AuthController
        from src.api.schemas.auth import RefreshTokenRequest
        from src.api.security.jwt import JWTConfig
        from src.api.security.jwt import JWTService as _JS

        user_id = uuid4()
        pair = _make_token_pair()
        # AuthService.refresh_tokens now returns (TokenPair, UUID)
        mock_auth = AsyncMock()
        mock_auth.refresh_tokens = AsyncMock(return_value=(pair, user_id))

        mock_request = MagicMock(spec=Request)
        mock_request.headers.get.return_value = None
        mock_request.client = None

        response = await AuthController.refresh_tokens.fn(  # type: ignore[attr-defined]
            MagicMock(),
            request=mock_request,
            data=RefreshTokenRequest(refresh_token="some.refresh.token"),
            auth_service=mock_auth,
            jwt_service=jwt_service,
            product_id=PRODUCT_ID,
            product_config=_make_product_config(),
            settings=settings,
        )

        assert len(response.cookies) == 1
        cookie = response.cookies[0]
        assert cookie.key == COOKIE_NAME
        assert cookie.httponly is True
        assert cookie.path == "/v1/content"
        assert cookie.domain == COOKIE_DOMAIN

        # Verify the cookie is for the correct user by decoding it
        svc = _JS(JWTConfig(secret_key=TEST_SECRET))
        payload = svc.decode_content_token(cookie.value)
        assert payload is not None
        from uuid import UUID as _UUID

        assert _UUID(payload.sub) == user_id

    async def test_logout_clears_content_cookie(
        self,
        jwt_service: JWTService,  # noqa: ARG002
        settings: Settings,  # noqa: ARG002
    ) -> None:
        from unittest.mock import AsyncMock, MagicMock

        from src.api.routes.auth import AuthController
        from src.api.schemas.auth import RefreshTokenRequest

        mock_auth = AsyncMock()
        mock_auth.logout = AsyncMock(return_value=True)

        response = await AuthController.logout.fn(  # type: ignore[attr-defined]
            MagicMock(),
            data=RefreshTokenRequest(refresh_token="tok"),
            auth_service=mock_auth,
            product_config=_make_product_config(),
            settings=settings,
        )

        assert len(response.cookies) == 1
        cookie = response.cookies[0]
        assert cookie.key == COOKIE_NAME
        assert cookie.max_age == 0
        assert cookie.value == ""
        assert cookie.path == "/v1/content"
        assert cookie.domain == COOKIE_DOMAIN

    async def test_register_secure_false_in_debug(self, jwt_service: JWTService) -> None:
        """When debug=True, content_cookie_secure=False so Secure attribute is absent."""
        from unittest.mock import AsyncMock, MagicMock

        from src.api.routes.auth import AuthController
        from src.api.schemas.auth import RegisterRequest

        debug_settings = Settings(
            jwt_secret_key=TEST_SECRET,
            database_url="postgresql+asyncpg://apex:apex@localhost:5432/apex",
            debug=True,
        )

        user = MagicMock()
        user.id = uuid4()
        pair = _make_token_pair()
        mock_auth = AsyncMock()
        mock_auth.register = AsyncMock(return_value=(user, pair))

        response = await AuthController.register.fn(  # type: ignore[attr-defined]
            MagicMock(),
            data=RegisterRequest(email="a@b.com", password="pass1234"),
            auth_service=mock_auth,
            jwt_service=jwt_service,
            product_id=PRODUCT_ID,
            product_config=_make_product_config(),
            settings=debug_settings,
        )

        cookie = response.cookies[0]
        assert cookie.secure is False
