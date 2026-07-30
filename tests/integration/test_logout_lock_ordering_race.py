"""Integration tests for the C1 lock-ordering fix in AuthService.logout
(10r remediation).

AuthService.logout must lock the user row *before* acting on the
refresh-token row — G1's documented rule is user row -> refresh-token row,
in every path that acquires both (see
UserRepository.lock_user_for_session_change). Before the C1 fix, logout
acquired the refresh-token row's lock first (via revoke_refresh_token's
UPDATE) and only locked the user row afterward — the inverse of
AuthService.refresh_tokens and AuthService.logout_all (via
revoke_all_user_tokens), both of which lock the user row first. Two
requests for the same user racing on the same refresh token, one taking
each order, form a lock cycle Postgres aborts with SQLSTATE 40P01.

Exercises the real call chain (AuthController.logout.fn /
AuthController.refresh_tokens.fn / AuthService.logout_all) against a real
PostgreSQL database and a shared in-memory fake Redis. Follows the same
shape as test_admin_grant_role_revocation_race.py's TestMutualGrantDeadlock:
many concurrent iterations with no artificial synchronization, since a
consistent global lock order makes a cycle physically impossible rather
than merely unlikely — no signal-based interleaving control is needed to
prove that.
"""

from __future__ import annotations

import asyncio
import time
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING
from unittest.mock import MagicMock
from uuid import uuid4

from redis.exceptions import NoScriptError
from sqlalchemy import delete
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.routes.auth import AuthController
from src.api.schemas.auth import RefreshTokenRequest
from src.api.security import generate_token, hash_token
from src.api.security.jwt import JWTConfig, JWTService
from src.api.services.auth import AuthService
from src.api.services.ops_event_bus import OpsEventBus
from src.api.services.token_revocation import TokenRevocationService
from src.core.config import Settings
from src.core.enums import UserRole
from src.core.product_registry import VEX_CONFIG
from src.core.uid import new_id
from src.db.models.user import RefreshToken, User
from src.db.repositories.user import UserRepository

if TYPE_CHECKING:
    from collections.abc import Callable
    from uuid import UUID

    from sqlalchemy.ext.asyncio import AsyncEngine

TEST_SECRET = "test_secret_key_for_testing_only_256bits_long"
PRODUCT_ID = "vex"
ITERATIONS = 50


class _FakeRedis:
    """Same in-memory Redis double as test_admin_grant_role_revocation_race.py."""

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
        key = str(keys_and_args[0])
        ttl = int(str(keys_and_args[numkeys]))
        now = int(self._clock())
        await self.set(key, now, ex=ttl)
        return now


def _make_password_service() -> MagicMock:
    return MagicMock()


def _settings() -> Settings:
    return Settings(
        jwt_secret_key=TEST_SECRET,
        database_url="postgresql+asyncpg://apex:apex@localhost:5432/apex",
        debug=False,
    )


async def _seed_user(engine: AsyncEngine, *, user_id: UUID) -> None:
    user = User(
        id=user_id,
        email=f"lock-order-{uuid4().hex[:8]}@example.com",
        password_hash="x" * 64,
        product_id=PRODUCT_ID,
        is_active=True,
        role=UserRole.USER.value,
    )
    async with AsyncSession(bind=engine, expire_on_commit=False) as session:
        session.add(user)
        await session.commit()


async def _cleanup_user(engine: AsyncEngine, *, user_id: UUID) -> None:
    async with AsyncSession(bind=engine, expire_on_commit=False) as session:
        await session.execute(delete(RefreshToken).where(RefreshToken.user_id == user_id))
        await session.execute(delete(User).where(User.id == user_id))
        await session.commit()


async def _create_refresh_token(engine: AsyncEngine, *, user_id: UUID) -> str:
    """Persist a real refresh-token row so both logout and refresh can find it by hash."""
    raw_token = generate_token(32)
    async with AsyncSession(bind=engine, expire_on_commit=False) as session:
        session.add(
            RefreshToken(
                id=new_id(),
                user_id=user_id,
                token_hash=hash_token(raw_token),
                family_id=new_id(),
                expires_at=datetime.now(UTC) + timedelta(days=7),
                product_id=PRODUCT_ID,
            )
        )
        await session.commit()
    return raw_token


def _auth_service(
    session: AsyncSession,
    *,
    jwt_service: JWTService,
    token_revocation: TokenRevocationService,
) -> AuthService:
    return AuthService(
        repository=UserRepository(session),
        jwt_service=jwt_service,
        password_service=_make_password_service(),
        token_revocation_service=token_revocation,
        session=session,
        ops_event_bus=OpsEventBus(enabled=False),
    )


async def _run_logout(
    *,
    engine: AsyncEngine,
    access_token: str,
    refresh_token: str,
    jwt_service: JWTService,
    token_revocation: TokenRevocationService,
    settings: Settings,
) -> str:
    """Invokes the real AuthController.logout handler presenting both the
    access token (whose sub is locked) and the refresh token (whose row is
    revoked) — the exact pair that raced against refresh/logout_all in the
    C1 bug report."""
    request = MagicMock()
    request.headers.get.return_value = f"Bearer {access_token}"

    async with (
        AsyncSession(bind=engine, expire_on_commit=False) as session,
        session.begin(),
    ):
        auth_service = _auth_service(
            session, jwt_service=jwt_service, token_revocation=token_revocation
        )
        await AuthController.logout.fn(
            MagicMock(),
            request=request,
            data=RefreshTokenRequest(refresh_token=refresh_token),
            auth_service=auth_service,
            jwt_service=jwt_service,
            product_config=VEX_CONFIG,
            settings=settings,
        )
    return "logged_out"


async def _run_refresh(
    *,
    engine: AsyncEngine,
    refresh_token: str,
    jwt_service: JWTService,
    token_revocation: TokenRevocationService,
    settings: Settings,
) -> str:
    """Invokes the real AuthController.refresh_tokens handler on the same
    refresh token logout is racing. request.client is None so the ip
    lookup short-circuits without touching request.headers.get's return
    value for anything but user-agent."""
    request = MagicMock()
    request.headers.get.return_value = None
    request.client = None

    async with (
        AsyncSession(bind=engine, expire_on_commit=False) as session,
        session.begin(),
    ):
        auth_service = _auth_service(
            session, jwt_service=jwt_service, token_revocation=token_revocation
        )
        response = await AuthController.refresh_tokens.fn(
            MagicMock(),
            request=request,
            data=RefreshTokenRequest(refresh_token=refresh_token),
            auth_service=auth_service,
            jwt_service=jwt_service,
            product_id=PRODUCT_ID,
            product_config=VEX_CONFIG,
            settings=settings,
        )
    # AuthController.refresh_tokens catches InvalidRefreshTokenError/
    # TokenReuseDetectedError itself and returns an error Response rather
    # than raising — either outcome (200 or 401) is a legitimate result of
    # two operations racing on the same token, not a bug this test is
    # about. Only an uncaught exception (e.g. a deadlock) is.
    return f"refresh:{response.status_code}"


async def _run_logout_all(
    *,
    engine: AsyncEngine,
    user_id: UUID,
    jwt_service: JWTService,
    token_revocation: TokenRevocationService,
) -> str:
    async with (
        AsyncSession(bind=engine, expire_on_commit=False) as session,
        session.begin(),
    ):
        auth_service = _auth_service(
            session, jwt_service=jwt_service, token_revocation=token_revocation
        )
        count = await auth_service.logout_all(user_id)
    return f"logout_all:{count}"


class TestLogoutVsRefreshDeadlock:
    """C1 — a concurrent single-device logout and refresh-token rotation
    for the same user, racing on the *same* refresh token (e.g. a
    near-expiry auto-refresh firing at the same moment as a
    user-initiated logout on another tab), must never deadlock, in either
    scheduling order. This test fails on the pre-remediation SHA
    (6fb9e44) — logout's inverted lock order — and passes once both paths
    lock the user row first, since a cycle can no longer form regardless
    of interleaving.
    """

    async def _run_race(self, db_engine: AsyncEngine, *, logout_first: bool) -> None:
        jwt_service = JWTService(JWTConfig(secret_key=TEST_SECRET))
        settings = _settings()

        for _ in range(ITERATIONS):
            user_id = new_id()
            await _seed_user(db_engine, user_id=user_id)
            try:
                refresh_token = await _create_refresh_token(db_engine, user_id=user_id)
                access_token, _ = jwt_service.create_access_token(user_id, product_id=PRODUCT_ID)
                token_revocation = TokenRevocationService(
                    _FakeRedis(),  # type: ignore[arg-type]
                    max_token_ttl_seconds=3600,
                )

                logout_coro = _run_logout(
                    engine=db_engine,
                    access_token=access_token,
                    refresh_token=refresh_token,
                    jwt_service=jwt_service,
                    token_revocation=token_revocation,
                    settings=settings,
                )
                refresh_coro = _run_refresh(
                    engine=db_engine,
                    refresh_token=refresh_token,
                    jwt_service=jwt_service,
                    token_revocation=token_revocation,
                    settings=settings,
                )
                coros = [logout_coro, refresh_coro] if logout_first else [refresh_coro, logout_coro]

                results = await asyncio.gather(*coros, return_exceptions=True)
                for result in results:
                    assert not isinstance(result, BaseException), (
                        f"logout vs refresh must never raise (deadlock or otherwise): {result!r}"
                    )
            finally:
                await _cleanup_user(db_engine, user_id=user_id)

    async def test_logout_scheduled_first_never_deadlocks(self, db_engine: AsyncEngine) -> None:
        await self._run_race(db_engine, logout_first=True)

    async def test_refresh_scheduled_first_never_deadlocks(self, db_engine: AsyncEngine) -> None:
        await self._run_race(db_engine, logout_first=False)


class TestLogoutVsLogoutAllDeadlock:
    """C1 variant — exercises the bulk revocation path
    (AuthService.logout_all -> UserRepository.revoke_all_user_tokens)
    rather than refresh_tokens' rotation path. revoke_all_user_tokens also
    locks the user row first, then bulk-UPDATEs every active refresh-token
    row for that user — including the one row single-device logout is
    revoking — so the same cycle risk applies.
    """

    async def _run_race(self, db_engine: AsyncEngine, *, logout_first: bool) -> None:
        jwt_service = JWTService(JWTConfig(secret_key=TEST_SECRET))
        settings = _settings()

        for _ in range(ITERATIONS):
            user_id = new_id()
            await _seed_user(db_engine, user_id=user_id)
            try:
                refresh_token = await _create_refresh_token(db_engine, user_id=user_id)
                access_token, _ = jwt_service.create_access_token(user_id, product_id=PRODUCT_ID)
                token_revocation = TokenRevocationService(
                    _FakeRedis(),  # type: ignore[arg-type]
                    max_token_ttl_seconds=3600,
                )

                logout_coro = _run_logout(
                    engine=db_engine,
                    access_token=access_token,
                    refresh_token=refresh_token,
                    jwt_service=jwt_service,
                    token_revocation=token_revocation,
                    settings=settings,
                )
                logout_all_coro = _run_logout_all(
                    engine=db_engine,
                    user_id=user_id,
                    jwt_service=jwt_service,
                    token_revocation=token_revocation,
                )
                coros = (
                    [logout_coro, logout_all_coro]
                    if logout_first
                    else [logout_all_coro, logout_coro]
                )

                results = await asyncio.gather(*coros, return_exceptions=True)
                for result in results:
                    assert not isinstance(result, BaseException), (
                        f"logout vs logout_all must never raise (deadlock or otherwise): {result!r}"
                    )
            finally:
                await _cleanup_user(db_engine, user_id=user_id)

    async def test_logout_scheduled_first_never_deadlocks(self, db_engine: AsyncEngine) -> None:
        await self._run_race(db_engine, logout_first=True)

    async def test_logout_all_scheduled_first_never_deadlocks(self, db_engine: AsyncEngine) -> None:
        await self._run_race(db_engine, logout_first=False)
