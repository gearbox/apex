"""Integration tests for the token-revocation feature (issue #142, rounds 1-2).

Exercises the real call chain — AuthService.logout_all / UserService.change_password
/ UserService.deactivate_account / AuthController.logout — writing into a real
TokenRevocationService, then verifies rejection (or continued acceptance) via
real HTTP requests through auth_guard / content_auth_guard on a Litestar
TestClient.

Uses an in-memory fake Redis client (the `set`/`mget`/`get`/`eval`/`evalsha`
subset TokenRevocationService needs, with real TTL-expiry semantics and a
pluggable clock simulating Redis `TIME`) rather than a live Redis server —
consistent with this repo's other infra-free "integration" tests
(test_content_cookie_flow.py, test_content_range_streaming.py) that exercise
real guards/DI/HTTP without a live external dependency.
"""

from __future__ import annotations

import asyncio
import time
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID, uuid4

import pytest
from litestar import Litestar, get
from litestar.di import Provide
from litestar.status_codes import HTTP_200_OK, HTTP_401_UNAUTHORIZED
from litestar.testing import TestClient
from redis.exceptions import NoScriptError
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession

from src.api.dependencies.auth import get_current_user_id, get_optional_user_id
from src.api.routes.auth import AuthController
from src.api.routes.user import UserController
from src.api.security import auth_guard, content_auth_guard, hash_token, optional_auth_guard
from src.api.security.jwt import JWTConfig, JWTService
from src.api.services.auth import AuthService, InvalidRefreshTokenError, TokenReuseDetectedError
from src.api.services.email_verification import EmailVerificationService
from src.api.services.token_revocation import TokenRevocationService
from src.api.services.user import UserService
from src.core.uid import new_id
from src.db.models.user import RefreshToken, User
from src.db.repositories.user import UserRepository

if TYPE_CHECKING:
    from collections.abc import Callable

TEST_SECRET = "test_secret_key_for_testing_only_256bits_long"
PRODUCT_ID = "vex"

# Used only by the one deliberate "re-login on the FOLLOWING second succeeds"
# test (F1 round 2) — everywhere else revocation is asserted immediately,
# with no sleep, since `<=` (not `<`) means same-second rejection is now the
# correct, intended behavior rather than something to dodge.
_NEXT_SECOND_GAP = 1.1


class _FakeRedis:
    """Minimal in-memory stand-in for redis.asyncio.Redis.

    Implements `set`/`mget`/`get`/`eval`/`evalsha` — the subset
    TokenRevocationService uses — with real TTL-expiry semantics, so
    epoch/jti keys actually age out. `eval` simulates the production Lua
    epoch-write script (`redis.call('TIME')` + `SET ... EX`) using a
    pluggable clock so tests can pin the "Redis clock" deterministically
    instead of depending on real-clock timing.
    """

    def __init__(self, *, clock: Callable[[], float] = time.time) -> None:
        self._store: dict[str, tuple[str, float | None]] = {}
        self._clock = clock

    async def set(self, key: str, value: object, ex: int | None = None) -> None:
        deadline = self._clock() + ex if ex is not None else None
        self._store[key] = (str(value), deadline)

    async def mget(self, keys: list[str]) -> list[str | None]:
        now = self._clock()
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

    async def get(self, key: str) -> str | None:
        (result,) = await self.mget([key])
        return result

    async def evalsha(self, _sha: str, _numkeys: int, *_keys_and_args: object) -> int:
        raise NoScriptError("fake redis never has a cached script")

    async def eval(self, _script: str, numkeys: int, *keys_and_args: object) -> int:
        """Simulates the production epoch-write script: SET key=TIME, EX=ttl."""
        key = str(keys_and_args[0])
        ttl = int(str(keys_and_args[numkeys]))
        now = int(self._clock())
        await self.set(key, now, ex=ttl)
        return now


def _make_repo() -> AsyncMock:
    repo = AsyncMock()
    repo.revoke_all_user_tokens.return_value = 1
    repo.get_refresh_token_by_hash.return_value = None
    repo.get_refresh_token_owner.return_value = None
    repo.get_refresh_token_by_hash_for_update.return_value = None
    return repo


def _make_password_service() -> MagicMock:
    pwd = MagicMock()
    pwd.averify = AsyncMock(return_value=True)
    pwd.ahash = AsyncMock(return_value="new_hash")
    return pwd


def _make_mock_session() -> AsyncMock:
    """AsyncMock session whose begin_nested() behaves like a real SAVEPOINT
    context manager (push-cleanup-on-revocation's _delete_push_subscriptions
    uses ``async with session.begin_nested():``, which a bare AsyncMock()
    cannot satisfy — see test_grok_image_thumbnails.py for the same idiom)."""
    session = AsyncMock()

    def _begin_nested() -> AsyncMock:
        cm = AsyncMock()
        cm.__aenter__ = AsyncMock(return_value=None)
        cm.__aexit__ = AsyncMock(return_value=False)
        return cm

    session.begin_nested = MagicMock(side_effect=_begin_nested)
    return session


def _make_app(jwt_service: JWTService, token_revocation: TokenRevocationService) -> Litestar:
    """A minimal app exposing an auth_guard route, a content_auth_guard
    route, and an optional_auth_guard route — every guard that consults
    TokenRevocationService. The optional-auth route stands in for
    GET /v1/providers (issue #142 R3): it never 401s, so the only way to
    observe revocation is whether it returns user_id or degrades to
    anonymous."""

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

    @get(
        "/optional-ping",
        guards=[optional_auth_guard],
        dependencies={"current_user_id": Provide(get_optional_user_id)},
    )
    async def optional_ping(current_user_id: UUID | None) -> dict[str, str | None]:
        return {"user_id": str(current_user_id) if current_user_id is not None else None}

    app = Litestar(route_handlers=[ping, content_ping, optional_ping])
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


class TestSameSecondRevocationIsRejected:
    """F1 (round 2) — `<=`, not `<`: a token minted in the exact same
    wall-clock second as its revocation must be rejected. This is the case
    round 1's `_PAST_SECOND_BOUNDARY` sleeps existed to dodge; round 2
    reverses that decision, so this asserts the opposite outcome
    deterministically (pinned clock, no sleep) rather than depending on
    real-clock timing luck.
    """

    async def test_token_401s_when_revoked_in_the_same_second_as_mint(
        self, jwt_service: JWTService
    ) -> None:
        user_id = uuid4()
        token, _ = jwt_service.create_access_token(user_id, product_id=PRODUCT_ID)
        payload = jwt_service.decode_access_token(token)
        assert payload is not None

        # Pin the fake Redis clock to exactly the token's `iat` — guarantees
        # the epoch write lands in the identical integer second as the
        # mint, deterministically, rather than racing real wall-clock time.
        redis = _FakeRedis(clock=lambda: float(payload.iat))
        token_revocation = TokenRevocationService(redis, max_token_ttl_seconds=3600)  # type: ignore[arg-type]
        app = _make_app(jwt_service, token_revocation)

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

        auth_service = AuthService(
            repository=_make_repo(),
            jwt_service=jwt_service,
            password_service=_make_password_service(),
            token_revocation_service=token_revocation,
        )
        logout_all_response = await UserController.logout_all.fn(  # type: ignore[attr-defined]
            MagicMock(),
            current_user_id=user_id,
            auth_service=auth_service,
        )
        assert logout_all_response.headers["Clear-Site-Data"] == '"cache", "storage"'

        with TestClient(app=app) as client:
            resp = client.get("/ping", headers={"Authorization": f"Bearer {token}"})
        assert resp.status_code == HTTP_401_UNAUTHORIZED

    async def test_relogin_on_the_following_second_yields_working_token(
        self,
        jwt_service: JWTService,
        token_revocation: TokenRevocationService,
        app: Litestar,
    ) -> None:
        """The one canonical "re-login still works" test (F1 round 2 —
        "keep exactly one test proving re-login succeeds on the following
        second"). Requires an explicit gap now: with `<=`, a token minted in
        the *same* second as the revocation is correctly rejected (see
        TestSameSecondRevocationIsRejected) — only a token minted after that
        second has elapsed is guaranteed to survive.
        """
        user_id = uuid4()
        auth_service = AuthService(
            repository=_make_repo(),
            jwt_service=jwt_service,
            password_service=_make_password_service(),
            token_revocation_service=token_revocation,
        )
        await auth_service.logout_all(user_id)

        await asyncio.sleep(_NEXT_SECOND_GAP)
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
        data = MagicMock()
        data.current_password = "old"
        data.new_password = "new"
        change_password_response = await UserController.change_password.fn(  # type: ignore[attr-defined]
            MagicMock(),
            current_user_id=user_id,
            data=data,
            user_service=user_service,
        )
        assert change_password_response.headers["Clear-Site-Data"] == '"cache", "storage"'

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

        repo = _make_repo()
        repo.soft_delete_user.return_value = MagicMock()
        user_service = UserService(
            repository=repo,
            password_service=_make_password_service(),
            age_verification_service=MagicMock(),
            token_revocation_service=token_revocation,
        )
        delete_account_response = await UserController.delete_account.fn(  # type: ignore[attr-defined]
            MagicMock(),
            current_user_id=user_id,
            user_service=user_service,
        )
        assert delete_account_response.headers["Clear-Site-Data"] == '"cache", "storage"'

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
            product_config=product_config,
            settings=settings,
        )
        assert response.status_code == HTTP_200_OK
        assert response.headers["Clear-Site-Data"] == '"cache", "storage"'

        with TestClient(app=app) as client:
            resp_a_after = client.get(
                "/ping", headers={"Authorization": f"Bearer {device_a_token}"}
            )
            resp_b_after = client.get(
                "/ping", headers={"Authorization": f"Bearer {device_b_token}"}
            )
        assert resp_a_after.status_code == HTTP_401_UNAUTHORIZED
        assert resp_b_after.status_code == HTTP_200_OK


class TestUnknownRefreshTokenLogoutStillPurgesCache:
    """D4 (Clear-Site-Data coverage prompt) — POST /v1/auth/logout must
    respond 200 with Clear-Site-Data regardless of whether refresh_token was
    ever valid. This is a deliberate, permanent contract: a client that
    discovers its session was revoked remotely has no other way to purge
    this origin's HTTP cache, so it fires a best-effort logout purely to
    receive the header. Uses the real AuthService.logout call chain — the
    stub repo's get_refresh_token_by_hash returns None (see _make_repo),
    so this exercises a syntactically valid but unknown/already-revoked
    token exactly as AuthService.logout returning False does.
    """

    async def test_unknown_refresh_token_returns_200_with_clear_site_data(
        self,
        jwt_service: JWTService,
        token_revocation: TokenRevocationService,
    ) -> None:
        from src.api.routes.auth import AuthController

        auth_service = AuthService(
            repository=_make_repo(),
            jwt_service=jwt_service,
            password_service=_make_password_service(),
            token_revocation_service=token_revocation,
        )

        request = MagicMock()
        request.headers.get.return_value = None
        product_config = MagicMock()
        product_config.cookie_domain = "example.com"
        settings = MagicMock()
        settings.content_cookie_secure = True

        response = await AuthController.logout.fn(  # type: ignore[attr-defined]
            MagicMock(),
            request=request,
            data=MagicMock(refresh_token="never-issued-or-already-revoked"),
            auth_service=auth_service,
            jwt_service=jwt_service,
            product_config=product_config,
            settings=settings,
        )

        assert response.status_code == HTTP_200_OK
        assert response.headers["Clear-Site-Data"] == '"cache", "storage"'


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


def _make_email_verification_service(
    token_revocation: TokenRevocationService,
) -> EmailVerificationService:
    email_service = AsyncMock()
    return EmailVerificationService(
        email_service=email_service,
        app_url="https://app.example.com",
        token_revocation_service=token_revocation,
    )


class TestResetPasswordRevokesAccessTokens:
    """R1 (issue #142) — the account-recovery path a user reaches *because*
    they believe their account is compromised must bulk-revoke live access
    tokens/content cookies, not just refresh tokens."""

    async def test_token_401s_after_reset_password(
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

        svc = _make_email_verification_service(token_revocation)
        user = MagicMock()
        user.id = user_id
        session = _make_mock_session()

        with (
            patch("src.api.services.email_verification.UserRepository") as user_repo_cls,
            patch("src.api.services.email_verification.AuthTokenRepository") as token_repo_cls,
            patch("src.api.security.PasswordService") as pwd_cls,
        ):
            token_repo = AsyncMock()
            token_repo.consume_reset_token = AsyncMock(return_value=user_id)
            token_repo_cls.return_value = token_repo

            user_repo = AsyncMock()
            user_repo.update_user = AsyncMock(return_value=user)
            user_repo.revoke_all_refresh_tokens = AsyncMock(return_value=2)
            user_repo_cls.return_value = user_repo

            pwd_instance = MagicMock()
            pwd_instance.ahash = AsyncMock(return_value="hashed_pw")
            pwd_cls.return_value = pwd_instance

            data = MagicMock()
            data.token = "raw-reset-token"
            data.new_password = "new_password"
            reset_password_response = await AuthController.reset_password.fn(  # type: ignore[attr-defined]
                MagicMock(),
                data=data,
                session=session,
                email_verification_service=svc,
            )
            assert reset_password_response.headers["Clear-Site-Data"] == '"cache", "storage"'

        with TestClient(app=app) as client:
            resp = client.get("/ping", headers={"Authorization": f"Bearer {token}"})
        assert resp.status_code == HTTP_401_UNAUTHORIZED


class TestTokenReuseDetectionRevokesAccessTokens:
    """R2 (issue #142) — reuse detection must bulk-revoke live access
    tokens/content cookies before raising, so its "All sessions have been
    invalidated" message is accurate rather than aspirational."""

    async def test_token_401s_after_reuse_detected(
        self,
        jwt_service: JWTService,
        token_revocation: TokenRevocationService,
        app: Litestar,
    ) -> None:
        user_id = uuid4()
        access_token, _ = jwt_service.create_access_token(user_id, product_id=PRODUCT_ID)

        with TestClient(app=app) as client:
            resp = client.get("/ping", headers={"Authorization": f"Bearer {access_token}"})
        assert resp.status_code == HTTP_200_OK

        repo = _make_repo()
        stored_token = MagicMock()
        stored_token.is_revoked = True
        stored_token.user_id = user_id
        stored_token.family_id = uuid4()
        # revoked_reason is an unconfigured MagicMock attribute here, not
        # RefreshTokenRevocationReason.BULK_REVOCATION.value — correctly
        # takes the theft-detection path, not the B2 benign-race path.
        repo.get_refresh_token_owner.return_value = user_id
        repo.get_refresh_token_by_hash_for_update.return_value = stored_token
        repo.revoke_token_family = AsyncMock(return_value=3)

        auth_service = AuthService(
            repository=repo,
            jwt_service=jwt_service,
            password_service=_make_password_service(),
            token_revocation_service=token_revocation,
        )

        with pytest.raises(TokenReuseDetectedError):
            await auth_service.refresh_tokens("stolen-refresh-token")

        with TestClient(app=app) as client:
            resp = client.get("/ping", headers={"Authorization": f"Bearer {access_token}"})
        assert resp.status_code == HTTP_401_UNAUTHORIZED


class TestOptionalAuthGuardRevocation:
    """R3 (issue #142) — GET /v1/providers uses optional_auth_guard, which
    previously never consulted TokenRevocationService: a revoked token kept
    authenticating (and thus kept receiving user_context/session_state)
    instead of degrading to anonymous like auth_guard/content_auth_guard
    already did. /optional-ping stands in for that route."""

    async def test_revoked_token_yields_anonymous_response_not_401(
        self,
        jwt_service: JWTService,
        token_revocation: TokenRevocationService,
        app: Litestar,
    ) -> None:
        user_id = uuid4()
        token, _ = jwt_service.create_access_token(user_id, product_id=PRODUCT_ID)

        with TestClient(app=app) as client:
            resp = client.get("/optional-ping", headers={"Authorization": f"Bearer {token}"})
        assert resp.status_code == HTTP_200_OK
        assert resp.json()["user_id"] == str(user_id)

        auth_service = AuthService(
            repository=_make_repo(),
            jwt_service=jwt_service,
            password_service=_make_password_service(),
            token_revocation_service=token_revocation,
        )
        await auth_service.logout_all(user_id)

        with TestClient(app=app) as client:
            resp = client.get("/optional-ping", headers={"Authorization": f"Bearer {token}"})
        # Never 401s — degrades to anonymous instead, per the guard's contract.
        assert resp.status_code == HTTP_200_OK
        assert resp.json()["user_id"] is None


async def _seed_user_with_refresh_token(
    engine: AsyncEngine, *, user_id: UUID, family_id: UUID
) -> str:
    """Commit a real User + RefreshToken row on a dedicated connection.

    Returns the raw (pre-hash) refresh token string.
    """
    from src.api.security import generate_token, hash_token

    raw_token = generate_token(32)
    user = User(
        id=user_id,
        email=f"race-{uuid4().hex[:8]}@example.com",
        password_hash="x" * 64,
        product_id="vex",
        is_active=True,
    )
    refresh_token = RefreshToken(
        id=new_id(),
        user_id=user_id,
        token_hash=hash_token(raw_token),
        family_id=family_id,
        expires_at=datetime.now(UTC) + timedelta(days=7),
        product_id="vex",
    )
    async with AsyncSession(bind=engine, expire_on_commit=False) as session:
        session.add_all([user, refresh_token])
        await session.commit()
    return raw_token


async def _cleanup_user(engine: AsyncEngine, user_id: UUID) -> None:
    async with AsyncSession(bind=engine, expire_on_commit=False) as session:
        await session.execute(delete(RefreshToken).where(RefreshToken.user_id == user_id))
        await session.execute(delete(User).where(User.id == user_id))
        await session.commit()


async def _attempt_refresh(
    *,
    engine: AsyncEngine,
    jwt_service: JWTService,
    token_revocation: TokenRevocationService,
    raw_refresh_token: str,
    outcome: dict[str, str],
) -> None:
    async with (
        AsyncSession(bind=engine, expire_on_commit=False) as session,
        session.begin(),
    ):
        auth_service = AuthService(
            repository=UserRepository(session),
            jwt_service=jwt_service,
            password_service=_make_password_service(),
            token_revocation_service=token_revocation,
        )
        try:
            tokens, _uid = await auth_service.refresh_tokens(raw_refresh_token)
        except (TokenReuseDetectedError, InvalidRefreshTokenError):
            # TokenReuseDetectedError: refresh observed the row already
            # revoked with a non-bulk_revocation reason (shouldn't happen
            # in this race, but tolerated). InvalidRefreshTokenError: the
            # expected outcome when logout-all won the user-row lock first
            # — G1's lock means refresh then observes
            # revoked_reason=bulk_revocation and takes the benign B2 path.
            outcome["refresh"] = "rejected"
        else:
            outcome["refresh"] = tokens.access_token
            outcome["new_refresh_token"] = tokens.refresh_token


async def _attempt_logout_all(
    *,
    engine: AsyncEngine,
    jwt_service: JWTService,
    token_revocation: TokenRevocationService,
    user_id: UUID,
) -> None:
    async with (
        AsyncSession(bind=engine, expire_on_commit=False) as session,
        session.begin(),
    ):
        auth_service = AuthService(
            repository=UserRepository(session),
            jwt_service=jwt_service,
            password_service=_make_password_service(),
            token_revocation_service=token_revocation,
        )
        await auth_service.logout_all(user_id)


class TestRefreshVsLogoutAllRace:
    """F2 — a concurrent refresh-token rotation racing a logout-all must
    never leave a client holding a usable access token.

    Self-contained against a real Postgres (via `db_engine`, two independent
    connections) rather than the SAVEPOINT-per-test `db_session` fixture,
    which only allocates one connection — the row lock this test exercises
    requires two genuinely concurrent transactions.

    Two orderings are possible depending on which side wins the row lock on
    the refresh-token row (`UserRepository.get_refresh_token_by_hash_for_update`):

      - logout-all wins: refresh observes `is_revoked=True` on the row once
        it acquires the lock and raises `TokenReuseDetectedError` without
        minting anything.
      - refresh wins: it mints a fresh pair before logout-all's bulk UPDATE
        (blocked on the same row) can unblock, but the new access token's
        `iat` necessarily predates the epoch logout-all subsequently
        writes, so `is_revoked()` flags it correctly once that epoch has
        propagated (asserted below, after both sides have completed).

    Either way, no access token a client could actually present survives.
    Run repeatedly — a single pass proves nothing about a race.
    """

    async def test_no_usable_access_token_survives(self, db_engine: AsyncEngine) -> None:
        jwt_service = JWTService(JWTConfig(secret_key=TEST_SECRET))
        iterations = 50

        for _ in range(iterations):
            user_id = new_id()
            family_id = new_id()
            raw_refresh_token = await _seed_user_with_refresh_token(
                db_engine, user_id=user_id, family_id=family_id
            )

            # One shared fake Redis/TokenRevocationService — real Redis is
            # shared infrastructure both concurrent requests would hit, and
            # the whole point is that the epoch write is visible to both.
            redis = _FakeRedis()
            token_revocation = TokenRevocationService(
                redis,  # type: ignore[arg-type]
                max_token_ttl_seconds=3600,
            )

            outcome: dict[str, str] = {}

            try:
                await asyncio.gather(
                    _attempt_refresh(
                        engine=db_engine,
                        jwt_service=jwt_service,
                        token_revocation=token_revocation,
                        raw_refresh_token=raw_refresh_token,
                        outcome=outcome,
                    ),
                    _attempt_logout_all(
                        engine=db_engine,
                        jwt_service=jwt_service,
                        token_revocation=token_revocation,
                        user_id=user_id,
                    ),
                )

                access_token = outcome.get("refresh")
                if access_token is not None and access_token != "rejected":
                    payload = jwt_service.decode_access_token(access_token)
                    assert payload is not None
                    assert await token_revocation.is_revoked(payload) is True, (
                        "a refresh that raced logout-all minted an access token "
                        "that survives revocation"
                    )

                    # G1 — the new refresh-token row itself must also be
                    # dead in the DB; the epoch check above only proves the
                    # paired access token is dead, not that the credential
                    # chain terminates (a live, unrevoked refresh token
                    # would let the client mint indefinitely many more
                    # post-epoch access tokens).
                    new_refresh_token = outcome["new_refresh_token"]
                    async with AsyncSession(bind=db_engine, expire_on_commit=False) as session:
                        result = await session.execute(
                            select(RefreshToken).where(
                                RefreshToken.token_hash == hash_token(new_refresh_token)
                            )
                        )
                        new_token_row = result.scalar_one()
                    assert new_token_row.is_revoked is True, (
                        "a refresh that raced logout-all minted a refresh token "
                        "row that survives un-revoked in the DB, allowing "
                        "indefinite further rotation"
                    )
            finally:
                await _cleanup_user(db_engine, user_id)
