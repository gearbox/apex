"""Integration tests for the token-revocation feature (issue #142).

Exercises the real call chain — AuthService.logout_all / UserService.change_password
/ UserService.deactivate_account / AuthController.logout — writing into a real
TokenRevocationService, then verifies rejection (or continued acceptance) via
real HTTP requests through auth_guard / content_auth_guard on a Litestar
TestClient.

Uses an in-memory fake Redis client (only the `set`/`mget` subset
TokenRevocationService needs, with real TTL-expiry semantics) rather than a
live Redis server — consistent with this repo's other infra-free
"integration" tests (test_content_cookie_flow.py, test_content_range_streaming.py)
that exercise real guards/DI/HTTP without a live external dependency.
"""

from __future__ import annotations

import asyncio
import time
from datetime import timedelta
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID, uuid4

import pytest
from litestar import Litestar, get
from litestar.di import Provide
from litestar.status_codes import HTTP_200_OK, HTTP_401_UNAUTHORIZED
from litestar.testing import TestClient

from src.api.dependencies.auth import get_current_user_id
from src.api.routes.auth import AuthController
from src.api.security import auth_guard, content_auth_guard
from src.api.security.jwt import JWTConfig, JWTService
from src.api.services.auth import AuthService
from src.api.services.token_revocation import TokenRevocationService
from src.api.services.user import UserService

TEST_SECRET = "test_secret_key_for_testing_only_256bits_long"
PRODUCT_ID = "vex"

# A token minted in the same wall-clock second as a revocation epoch must
# survive by design (D2 — strictly `<`, not `<=`). Tests asserting rejection
# need a real gap past that second to be deterministic; tests asserting
# survival (the D2 regression itself) need no gap since real time only moves
# forward between mint and revoke.
_PAST_SECOND_BOUNDARY = 1.1


class _FakeRedis:
    """Minimal in-memory stand-in for redis.asyncio.Redis.

    Implements only `set`/`mget` — the subset TokenRevocationService uses —
    with real TTL-expiry semantics, so epoch/jti keys actually age out.
    """

    def __init__(self) -> None:
        self._store: dict[str, tuple[str, float | None]] = {}

    async def set(self, key: str, value: object, ex: int | None = None) -> None:
        deadline = time.time() + ex if ex is not None else None
        self._store[key] = (str(value), deadline)

    async def mget(self, keys: list[str]) -> list[str | None]:
        now = time.time()
        result: list[str | None] = []
        for key in keys:
            entry = self._store.get(key)
            if entry is None:
                result.append(None)
                continue
            value, deadline = entry
            if deadline is not None and deadline < now:
                del self._store[key]
                result.append(None)
            else:
                result.append(value)
        return result


def _make_repo() -> AsyncMock:
    repo = AsyncMock()
    repo.revoke_all_user_tokens.return_value = 1
    repo.get_refresh_token_by_hash.return_value = None
    return repo


def _make_password_service() -> MagicMock:
    pwd = MagicMock()
    pwd.averify = AsyncMock(return_value=True)
    pwd.ahash = AsyncMock(return_value="new_hash")
    return pwd


def _make_app(jwt_service: JWTService, token_revocation: TokenRevocationService) -> Litestar:
    """A minimal app exposing one auth_guard route and one content_auth_guard
    route — the two guards that consult TokenRevocationService."""

    @get(
        "/ping",
        guards=[auth_guard],
        dependencies={"current_user_id": Provide(get_current_user_id)},
    )
    async def ping(current_user_id: UUID) -> dict[str, str]:
        return {"user_id": str(current_user_id)}

    @get(
        "/content-ping",
        guards=[content_auth_guard],
        dependencies={"current_user_id": Provide(get_current_user_id)},
    )
    async def content_ping(current_user_id: UUID) -> dict[str, str]:
        return {"user_id": str(current_user_id)}

    app = Litestar(route_handlers=[ping, content_ping])
    app.state["jwt_service"] = jwt_service
    app.state["token_revocation"] = token_revocation
    return app


@pytest.fixture
def jwt_service() -> JWTService:
    return JWTService(JWTConfig(secret_key=TEST_SECRET))


@pytest.fixture
def token_revocation() -> TokenRevocationService:
    return TokenRevocationService(_FakeRedis(), max_token_ttl_seconds=3600)  # type: ignore[arg-type]


@pytest.fixture
def app(jwt_service: JWTService, token_revocation: TokenRevocationService) -> Litestar:
    return _make_app(jwt_service, token_revocation)


class TestLogoutAllRevokesAccessTokens:
    """AuthService.logout_all — bulk revocation via the user epoch (D1a)."""

    async def test_token_401s_after_logout_all(
        self,
        jwt_service: JWTService,
        token_revocation: TokenRevocationService,
        app: Litestar,
    ) -> None:
        user_id = uuid4()
        token, _ = jwt_service.create_access_token(user_id, product_id=PRODUCT_ID)

        with TestClient(app=app) as client:
            resp = client.get("/ping", headers={"Authorization": f"Bearer {token}"})
        assert resp.status_code == HTTP_200_OK

        await asyncio.sleep(_PAST_SECOND_BOUNDARY)
        auth_service = AuthService(
            repository=_make_repo(),
            jwt_service=jwt_service,
            password_service=_make_password_service(),
            token_revocation_service=token_revocation,
        )
        await auth_service.logout_all(user_id)

        with TestClient(app=app) as client:
            resp = client.get("/ping", headers={"Authorization": f"Bearer {token}"})
        assert resp.status_code == HTTP_401_UNAUTHORIZED

    async def test_relogin_immediately_after_logout_all_yields_working_token(
        self,
        jwt_service: JWTService,
        token_revocation: TokenRevocationService,
        app: Litestar,
    ) -> None:
        """D2 regression — this is the assertion that catches a `<`→`<=` change.

        No sleep here: real time only moves forward between the
        logout_all() call and the fresh mint that follows it, so the fresh
        token's `iat` is always >= the epoch just written, and must survive.
        """
        user_id = uuid4()
        auth_service = AuthService(
            repository=_make_repo(),
            jwt_service=jwt_service,
            password_service=_make_password_service(),
            token_revocation_service=token_revocation,
        )
        await auth_service.logout_all(user_id)

        fresh_token, _ = jwt_service.create_access_token(user_id, product_id=PRODUCT_ID)

        with TestClient(app=app) as client:
            resp = client.get("/ping", headers={"Authorization": f"Bearer {fresh_token}"})
        assert resp.status_code == HTTP_200_OK


class TestChangePasswordRevokesAccessTokens:
    async def test_token_401s_after_change_password(
        self,
        jwt_service: JWTService,
        token_revocation: TokenRevocationService,
        app: Litestar,
    ) -> None:
        user_id = uuid4()
        token, _ = jwt_service.create_access_token(user_id, product_id=PRODUCT_ID)

        with TestClient(app=app) as client:
            resp = client.get("/ping", headers={"Authorization": f"Bearer {token}"})
        assert resp.status_code == HTTP_200_OK

        await asyncio.sleep(_PAST_SECOND_BOUNDARY)
        repo = _make_repo()
        user = MagicMock()
        user.password_hash = "hashed"
        repo.get_user.return_value = user
        repo.update_user.return_value = user
        user_service = UserService(
            repository=repo,
            password_service=_make_password_service(),
            age_verification_service=MagicMock(),
            token_revocation_service=token_revocation,
        )
        await user_service.change_password(user_id, current_password="old", new_password="new")

        with TestClient(app=app) as client:
            resp = client.get("/ping", headers={"Authorization": f"Bearer {token}"})
        assert resp.status_code == HTTP_401_UNAUTHORIZED


class TestDeactivateAccountRevokesAccessTokens:
    async def test_token_401s_after_deactivate_account(
        self,
        jwt_service: JWTService,
        token_revocation: TokenRevocationService,
        app: Litestar,
    ) -> None:
        user_id = uuid4()
        token, _ = jwt_service.create_access_token(user_id, product_id=PRODUCT_ID)

        with TestClient(app=app) as client:
            resp = client.get("/ping", headers={"Authorization": f"Bearer {token}"})
        assert resp.status_code == HTTP_200_OK

        await asyncio.sleep(_PAST_SECOND_BOUNDARY)
        repo = _make_repo()
        repo.soft_delete_user.return_value = MagicMock()
        user_service = UserService(
            repository=repo,
            password_service=_make_password_service(),
            age_verification_service=MagicMock(),
            token_revocation_service=token_revocation,
        )
        await user_service.deactivate_account(user_id)

        with TestClient(app=app) as client:
            resp = client.get("/ping", headers={"Authorization": f"Bearer {token}"})
        assert resp.status_code == HTTP_401_UNAUTHORIZED


class TestSingleDeviceLogoutDenylistsOnlyThatToken:
    """POST /v1/auth/logout — per-jti denylist (D1b), scoped to one device."""

    async def test_logged_out_device_401s_other_device_still_works(
        self,
        jwt_service: JWTService,
        token_revocation: TokenRevocationService,
        app: Litestar,
    ) -> None:
        user_id = uuid4()
        device_a_token, _device_a_expires_at = jwt_service.create_access_token(
            user_id, product_id=PRODUCT_ID
        )
        device_b_token, _ = jwt_service.create_access_token(user_id, product_id=PRODUCT_ID)

        with TestClient(app=app) as client:
            resp_a = client.get("/ping", headers={"Authorization": f"Bearer {device_a_token}"})
            resp_b = client.get("/ping", headers={"Authorization": f"Bearer {device_b_token}"})
        assert resp_a.status_code == HTTP_200_OK
        assert resp_b.status_code == HTTP_200_OK

        # Simulate device A calling POST /v1/auth/logout with its own
        # presenting access token — mirrors AuthController.logout's body.
        payload_a = jwt_service.decode_access_token(device_a_token)
        assert payload_a is not None
        auth_service = AuthService(
            repository=_make_repo(),
            jwt_service=jwt_service,
            password_service=_make_password_service(),
            token_revocation_service=token_revocation,
        )
        request = MagicMock()
        request.headers.get.return_value = f"Bearer {device_a_token}"
        product_config = MagicMock()
        product_config.cookie_domain = "example.com"
        settings = MagicMock()
        settings.content_cookie_secure = True

        response = await AuthController.logout.fn(
            MagicMock(),
            request=request,
            data=MagicMock(refresh_token="device-a-refresh"),
            auth_service=auth_service,
            jwt_service=jwt_service,
            token_revocation_service=token_revocation,
            product_config=product_config,
            settings=settings,
        )
        assert response.status_code == HTTP_200_OK

        with TestClient(app=app) as client:
            resp_a_after = client.get(
                "/ping", headers={"Authorization": f"Bearer {device_a_token}"}
            )
            resp_b_after = client.get(
                "/ping", headers={"Authorization": f"Bearer {device_b_token}"}
            )
        assert resp_a_after.status_code == HTTP_401_UNAUTHORIZED
        assert resp_b_after.status_code == HTTP_200_OK


class TestContentCookieRevokedByLogoutAll:
    """content_auth_guard must also honor the bulk-revocation epoch (the
    issue's third gap: the 24h content cookie widened the exposure)."""

    async def test_content_cookie_401s_after_logout_all(
        self,
        jwt_service: JWTService,
        token_revocation: TokenRevocationService,
        app: Litestar,
    ) -> None:
        user_id = uuid4()
        content_token, _ = jwt_service.create_content_token(
            user_id, product_id=PRODUCT_ID, ttl=timedelta(hours=1)
        )

        with TestClient(app=app) as client:
            resp = client.get("/content-ping", cookies={"apex_content": content_token})
        assert resp.status_code == HTTP_200_OK

        await asyncio.sleep(_PAST_SECOND_BOUNDARY)
        auth_service = AuthService(
            repository=_make_repo(),
            jwt_service=jwt_service,
            password_service=_make_password_service(),
            token_revocation_service=token_revocation,
        )
        await auth_service.logout_all(user_id)

        with TestClient(app=app) as client:
            resp = client.get("/content-ping", cookies={"apex_content": content_token})
        assert resp.status_code == HTTP_401_UNAUTHORIZED
