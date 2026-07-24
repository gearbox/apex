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

from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any
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

from src.api.dependencies.auth import get_current_user_id
from src.api.security import auth_guard, content_auth_guard
from src.api.security.content_cookie import build_content_cookie, clear_content_cookie
from src.api.security.jwt import JWTConfig, JWTService
from src.core.config import Settings

if TYPE_CHECKING:
    from litestar.types import Receive, Scope, Send

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TEST_SECRET = "test_secret_key_for_testing_only_256bits_long"
PRODUCT_ID = "vex"
COOKIE_DOMAIN = "vex-domain.com"  # arbitrary fixture domain, not the real product config value
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
        dependencies = {"current_user_id": Provide(get_current_user_id)}  # noqa: RUF012

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

    def test_default_cookie_ttl_is_24_hours(self) -> None:
        s = Settings(
            jwt_secret_key=TEST_SECRET,
            database_url="postgresql+asyncpg://apex:apex@localhost:5432/apex",
        )
        assert s.content_cookie_ttl_hours == 24

    def test_cookie_ttl_up_to_168_hours_accepted(self) -> None:
        s = Settings(
            jwt_secret_key=TEST_SECRET,
            database_url="postgresql+asyncpg://apex:apex@localhost:5432/apex",
            content_cookie_ttl_hours=168,
        )
        assert s.content_cookie_ttl_hours == 168

    def test_cookie_ttl_above_max_rejected(self) -> None:
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            Settings(
                jwt_secret_key=TEST_SECRET,
                database_url="postgresql+asyncpg://apex:apex@localhost:5432/apex",
                content_cookie_ttl_hours=169,
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
# content_cookie_lifetime — single source of truth for max_age / expires_at (D3)
# ---------------------------------------------------------------------------


class TestContentCookieLifetimeHelper:
    """A cookie built via build_content_cookie and the advertised expires_at must agree."""

    def test_max_age_and_expires_at_agree_to_the_second(self) -> None:
        from src.api.security.content_cookie import content_cookie_lifetime

        s = Settings(
            jwt_secret_key=TEST_SECRET,
            database_url="postgresql+asyncpg://apex:apex@localhost:5432/apex",
            content_cookie_ttl_hours=48,
        )
        max_age, expires_at = content_cookie_lifetime(s)
        assert max_age == 48 * 3600

        cookie = build_content_cookie("token", domain=None, secure=True, max_age=max_age)
        assert cookie.max_age == max_age
        derived_expiry = datetime.now(UTC) + timedelta(seconds=max_age)
        assert abs((expires_at - derived_expiry).total_seconds()) < 1

    def test_mint_content_cookie_returns_matching_max_age_and_expiry(
        self, jwt_service: JWTService
    ) -> None:
        from src.api.security.content_cookie import mint_content_cookie
        from src.core.product_registry import resolve_product_by_slug

        s = Settings(
            jwt_secret_key=TEST_SECRET,
            database_url="postgresql+asyncpg://apex:apex@localhost:5432/apex",
            content_cookie_ttl_hours=24,
        )
        product_config = resolve_product_by_slug(PRODUCT_ID)
        assert product_config is not None

        cookie, expires_at = mint_content_cookie(
            user_id=uuid4(),
            product_id=PRODUCT_ID,
            jwt_service=jwt_service,
            settings=s,
            product_config=product_config,
        )
        assert cookie.max_age == 24 * 3600
        derived_expiry = datetime.now(UTC) + timedelta(seconds=24 * 3600)
        assert abs((expires_at - derived_expiry).total_seconds()) < 1

        # The JWT's own exp claim (the actual revocation authority) agrees too.
        assert isinstance(cookie.value, str)
        payload = jwt_service.decode_content_token(cookie.value)
        assert payload is not None
        token_expiry = datetime.fromtimestamp(payload.exp, tz=UTC)
        assert abs((expires_at - token_expiry).total_seconds()) < 1


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

        assert response.content is not None
        expected_expiry = datetime.now(UTC) + timedelta(
            seconds=settings.content_cookie_ttl_hours * 3600
        )
        assert (
            abs((response.content.content_cookie_expires_at - expected_expiry).total_seconds()) < 2
        )

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

        assert response.content is not None
        expected_expiry = datetime.now(UTC) + timedelta(
            seconds=settings.content_cookie_ttl_hours * 3600
        )
        assert (
            abs((response.content.content_cookie_expires_at - expected_expiry).total_seconds()) < 2
        )

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

        assert response.content is not None
        expected_expiry = datetime.now(UTC) + timedelta(
            seconds=settings.content_cookie_ttl_hours * 3600
        )
        assert (
            abs((response.content.content_cookie_expires_at - expected_expiry).total_seconds()) < 2
        )

    async def test_logout_clears_content_cookie(
        self,
        jwt_service: JWTService,  # noqa: ARG002
        settings: Settings,
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


class TestRemintContentCookieHandler:
    """POST /v1/auth/content-cookie (D4) — direct handler tests, no DB."""

    async def test_remint_sets_content_cookie_200(
        self, jwt_service: JWTService, settings: Settings
    ) -> None:
        from unittest.mock import MagicMock

        from src.api.routes.auth import AuthController

        user_id = uuid4()

        response = await AuthController.remint_content_cookie.fn(
            MagicMock(),
            current_user_id=user_id,
            jwt_service=jwt_service,
            product_id=PRODUCT_ID,
            product_config=_make_product_config(),
            settings=settings,
        )

        assert response.status_code == HTTP_200_OK
        assert response.content is not None
        assert len(response.cookies) == 1
        cookie = response.cookies[0]
        assert cookie.key == COOKIE_NAME
        assert cookie.httponly is True
        assert cookie.samesite == "lax"
        assert cookie.path == "/v1/content"
        assert cookie.domain == COOKIE_DOMAIN
        assert cookie.max_age == settings.content_cookie_ttl_hours * 3600
        assert cookie.secure == settings.content_cookie_secure

        expected_expiry = datetime.now(UTC) + timedelta(
            seconds=settings.content_cookie_ttl_hours * 3600
        )
        assert abs((response.content.expires_at - expected_expiry).total_seconds()) < 2

    async def test_remint_cookie_attributes_match_login(
        self, jwt_service: JWTService, settings: Settings
    ) -> None:
        """The cookie D4 mints has identical attributes to login's — same helper, same call shape."""
        from unittest.mock import AsyncMock, MagicMock

        from src.api.routes.auth import AuthController

        user_id = uuid4()

        remint_response = await AuthController.remint_content_cookie.fn(
            MagicMock(),
            current_user_id=user_id,
            jwt_service=jwt_service,
            product_id=PRODUCT_ID,
            product_config=_make_product_config(),
            settings=settings,
        )

        mock_auth = AsyncMock()
        user = MagicMock()
        user.id = user_id
        mock_auth.login = AsyncMock(return_value=(user, _make_token_pair()))
        mock_request = MagicMock()
        mock_request.headers.get.return_value = None
        mock_request.client = None

        from src.api.schemas.auth import LoginRequest

        login_response = await AuthController.login.fn(
            MagicMock(),
            request=mock_request,
            data=LoginRequest(email="a@b.com", password="pass1234"),
            auth_service=mock_auth,
            jwt_service=jwt_service,
            product_id=PRODUCT_ID,
            product_config=_make_product_config(),
            settings=settings,
        )

        remint_cookie = remint_response.cookies[0]
        login_cookie = login_response.cookies[0]
        assert remint_cookie.httponly == login_cookie.httponly
        assert remint_cookie.secure == login_cookie.secure
        assert remint_cookie.samesite == login_cookie.samesite
        assert remint_cookie.path == login_cookie.path
        assert remint_cookie.domain == login_cookie.domain
        assert remint_cookie.max_age == login_cookie.max_age

        # And the minted token really does authenticate this user/product.
        svc = JWTService(JWTConfig(secret_key=TEST_SECRET))
        payload = svc.decode_content_token(remint_cookie.value)
        assert payload is not None
        assert UUID(payload.sub) == user_id
        assert payload.product_id == PRODUCT_ID

    async def test_remint_scoped_to_requesting_users_product(
        self, jwt_service: JWTService, settings: Settings
    ) -> None:
        """The minted cookie is scoped to the product_id resolved for the request."""
        from unittest.mock import MagicMock

        from src.api.routes.auth import AuthController

        user_id = uuid4()

        response = await AuthController.remint_content_cookie.fn(
            MagicMock(),
            current_user_id=user_id,
            jwt_service=jwt_service,
            product_id="synthara",
            product_config=_make_product_config(),
            settings=settings,
        )

        cookie = response.cookies[0]
        svc = JWTService(JWTConfig(secret_key=TEST_SECRET))
        payload = svc.decode_content_token(cookie.value)
        assert payload is not None
        assert payload.product_id == "synthara"


class TestRemintContentCookieHTTP:
    """POST /v1/auth/content-cookie over real HTTP — guard behavior (bearer-only)."""

    @staticmethod
    def _make_app(jwt_service: JWTService, settings: Settings) -> Litestar:
        from src.api.routes.auth import AuthController
        from src.core.product_registry import resolve_product_by_slug

        # A real ProductConfig (not a MagicMock) — Litestar's signature model
        # validates injected DI values by isinstance, which a bare mock fails.
        product_config = resolve_product_by_slug(PRODUCT_ID)
        assert product_config is not None

        app = Litestar(
            route_handlers=[AuthController],
            dependencies={
                "product_id": Provide(lambda: PRODUCT_ID, sync_to_thread=False),
                "product_config": Provide(lambda: product_config, sync_to_thread=False),
                "settings": Provide(lambda: settings, sync_to_thread=False),
                "jwt_service": Provide(lambda: jwt_service, sync_to_thread=False),
            },
        )
        app.state["jwt_service"] = jwt_service
        return app

    def test_no_bearer_returns_401(self, jwt_service: JWTService, settings: Settings) -> None:
        app = self._make_app(jwt_service, settings)
        with TestClient(app=app) as client:
            resp = client.post("/v1/auth/content-cookie")
        assert resp.status_code == HTTP_401_UNAUTHORIZED

    def test_valid_bearer_returns_200_with_cookie_and_body(
        self, jwt_service: JWTService, settings: Settings, test_user_id: UUID
    ) -> None:
        token, _ = jwt_service.create_access_token(test_user_id, product_id=PRODUCT_ID)
        app = self._make_app(jwt_service, settings)
        with TestClient(app=app) as client:
            resp = client.post(
                "/v1/auth/content-cookie", headers={"Authorization": f"Bearer {token}"}
            )
        assert resp.status_code == HTTP_200_OK
        body = resp.json()
        assert "expires_at" in body
        # httpx's parsed cookie jar drops this (Domain=vex.pics, Secure, over a
        # plain-http testserver request) — assert on the raw header instead.
        set_cookie = resp.headers.get("set-cookie", "")
        assert set_cookie.startswith(f"{COOKIE_NAME}=")
        assert "HttpOnly" in set_cookie
        assert "Secure" in set_cookie
        assert "Path=/v1/content" in set_cookie

    def test_content_cookie_alone_cannot_authorize_remint(
        self, jwt_service: JWTService, settings: Settings, test_user_id: UUID
    ) -> None:
        """The content cookie itself must not authorize minting a fresh one — bearer-only."""
        token, _ = jwt_service.create_content_token(
            test_user_id, product_id=PRODUCT_ID, ttl=timedelta(hours=1)
        )
        app = self._make_app(jwt_service, settings)
        with TestClient(app=app) as client:
            resp = client.post("/v1/auth/content-cookie", cookies={COOKIE_NAME: token})
        assert resp.status_code == HTTP_401_UNAUTHORIZED


# ---------------------------------------------------------------------------
# effective_cookie_domain unit tests
# ---------------------------------------------------------------------------


class TestEffectiveCookieDomain:
    def test_returns_none_in_dev(self) -> None:
        from src.api.security.content_cookie import effective_cookie_domain
        from src.core.product_registry import resolve_product_by_slug

        debug_settings = Settings(
            jwt_secret_key=TEST_SECRET,
            database_url="postgresql+asyncpg://apex:apex@localhost:5432/apex",
            debug=True,
        )
        vex_config = resolve_product_by_slug("vex")
        assert vex_config is not None
        assert effective_cookie_domain(debug_settings, vex_config) is None

    def test_returns_product_domain_in_prod(self) -> None:
        from src.api.security.content_cookie import effective_cookie_domain
        from src.core.product_registry import resolve_product_by_slug

        prod_settings = Settings(
            jwt_secret_key=TEST_SECRET,
            database_url="postgresql+asyncpg://apex:apex@localhost:5432/apex",
            debug=False,
        )
        vex_config = resolve_product_by_slug("vex")
        assert vex_config is not None
        assert effective_cookie_domain(prod_settings, vex_config) == vex_config.cookie_domain


# ---------------------------------------------------------------------------
# attach_content_cookie domain behaviour
# ---------------------------------------------------------------------------


class TestAttachContentCookieDomain:
    async def test_host_only_in_dev(self, jwt_service: JWTService) -> None:
        from litestar import Response

        from src.api.security.content_cookie import attach_content_cookie

        debug_settings = Settings(
            jwt_secret_key=TEST_SECRET,
            database_url="postgresql+asyncpg://apex:apex@localhost:5432/apex",
            debug=True,
        )
        product_config = _make_product_config(domain=COOKIE_DOMAIN)
        response: Response[dict[str, str]] = Response(content={"ok": "true"})
        attach_content_cookie(
            response,
            user_id=uuid4(),
            product_id=PRODUCT_ID,
            jwt_service=jwt_service,
            settings=debug_settings,
            product_config=product_config,
        )
        assert len(response.cookies) == 1
        cookie = response.cookies[0]
        assert cookie.domain is None
        assert cookie.secure is False

    async def test_domain_set_in_prod(self, jwt_service: JWTService, settings: Settings) -> None:
        from litestar import Response

        from src.api.security.content_cookie import attach_content_cookie

        product_config = _make_product_config(domain=COOKIE_DOMAIN)
        response: Response[dict[str, str]] = Response(content={"ok": "true"})
        attach_content_cookie(
            response,
            user_id=uuid4(),
            product_id=PRODUCT_ID,
            jwt_service=jwt_service,
            settings=settings,
            product_config=product_config,
        )
        assert len(response.cookies) == 1
        cookie = response.cookies[0]
        assert cookie.domain == COOKIE_DOMAIN
        assert cookie.secure is True
