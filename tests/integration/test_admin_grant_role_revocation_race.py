"""Integration test for the generalized pre-commit revocation re-check (Claude board item).

Exercises the real call chain — ``AdminManagementController.grant_role``
(the actual decorated handler function, invoked via ``.fn(...)`` the same
way tests/integration/test_push_subscription_revocation_race.py does) racing
``AuthService.logout_all`` for the *granting superadmin's own* session —
against a real PostgreSQL database and a shared in-memory fake Redis,
proving a persistent role grant cannot survive a concurrent bulk revocation
of the credentials that authorized it, in either lock-acquisition ordering.

This is the second endpoint wired to
``src/api/security/revocation_recheck.py`` (after
``POST /v1/push/subscriptions``) and follows the same test shape as
test_push_subscription_revocation_race.py — see that module's docstring for
why the interleavings are pinned deterministically via an ``asyncio.Event``
rather than left to natural scheduling.

Also covers the review r1 remediation (09r): ``TestGrantRoleVsSingleDeviceLogout``
(E1 — the per-jti path now participates in the same lock) and
``TestMutualGrantDeadlock`` (E2/C1 — ``also_lock`` prevents the actor/target
lock cycle two mirrored grants would otherwise form).
"""

from __future__ import annotations

import asyncio
import time
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock, patch
from uuid import UUID, uuid4

from litestar.exceptions import NotAuthorizedException
from redis.exceptions import NoScriptError
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.routes.admin_management import AdminManagementController
from src.api.routes.auth import AuthController
from src.api.schemas.admin import GrantRoleRequest
from src.api.schemas.auth import RefreshTokenRequest
from src.api.security import generate_token, hash_token
from src.api.security.jwt import JWTConfig, JWTService, TokenPayload
from src.api.services.admin_management import AdminManagementService
from src.api.services.auth import AuthService
from src.api.services.ops_event_bus import OpsEventBus
from src.api.services.token_revocation import TokenRevocationService
from src.core.config import Settings
from src.core.enums import UserRole
from src.core.product_registry import VEX_CONFIG
from src.core.uid import new_id
from src.db.models.admin import AdminAuditLog, AdminPermissionGrant
from src.db.models.user import RefreshToken, User
from src.db.repositories.user import UserRepository

if TYPE_CHECKING:
    from collections.abc import Callable

    from sqlalchemy.ext.asyncio import AsyncEngine

TEST_SECRET = "test_secret_key_for_testing_only_256bits_long"
PRODUCT_ID = "vex"


class _FakeRedis:
    """Same in-memory Redis double as test_push_subscription_revocation_race.py."""

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


async def _seed_users(engine: AsyncEngine, *, superadmin_id: UUID, target_id: UUID) -> None:
    superadmin = User(
        id=superadmin_id,
        email=f"race-superadmin-{uuid4().hex[:8]}@example.com",
        password_hash="x" * 64,
        product_id=PRODUCT_ID,
        is_active=True,
        role=UserRole.SUPERADMIN.value,
    )
    target = User(
        id=target_id,
        email=f"race-target-{uuid4().hex[:8]}@example.com",
        password_hash="x" * 64,
        product_id=PRODUCT_ID,
        is_active=True,
        role=UserRole.USER.value,
    )
    async with AsyncSession(bind=engine, expire_on_commit=False) as session:
        session.add_all([superadmin, target])
        await session.commit()


async def _cleanup_users(engine: AsyncEngine, *, superadmin_id: UUID, target_id: UUID) -> None:
    async with AsyncSession(bind=engine, expire_on_commit=False) as session:
        for user_id in (superadmin_id, target_id):
            await session.execute(
                delete(AdminPermissionGrant).where(AdminPermissionGrant.user_id == user_id)
            )
            await session.execute(
                delete(AdminAuditLog).where(AdminAuditLog.target_user_id == user_id)
            )
            await session.execute(delete(AdminAuditLog).where(AdminAuditLog.actor_id == user_id))
            await session.execute(delete(RefreshToken).where(RefreshToken.user_id == user_id))
            await session.execute(delete(User).where(User.id == user_id))
        await session.commit()


async def _seed_two_superadmins(engine: AsyncEngine, *, user_a_id: UUID, user_b_id: UUID) -> None:
    """Both users are superadmins — used by TestMutualGrantDeadlock, where
    each grants the other a role (mirrored actor/target pairs)."""
    user_a = User(
        id=user_a_id,
        email=f"race-mutual-a-{uuid4().hex[:8]}@example.com",
        password_hash="x" * 64,
        product_id=PRODUCT_ID,
        is_active=True,
        role=UserRole.SUPERADMIN.value,
    )
    user_b = User(
        id=user_b_id,
        email=f"race-mutual-b-{uuid4().hex[:8]}@example.com",
        password_hash="x" * 64,
        product_id=PRODUCT_ID,
        is_active=True,
        role=UserRole.SUPERADMIN.value,
    )
    async with AsyncSession(bind=engine, expire_on_commit=False) as session:
        session.add_all([user_a, user_b])
        await session.commit()


async def _create_refresh_token(engine: AsyncEngine, *, user_id: UUID) -> str:
    """Persist a real refresh-token row so AuthService.logout can find it by hash."""
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


def _patched_lock_that_signals(
    signal: asyncio.Event,
    *,
    watch_user_id: UUID,
) -> Callable[[UserRepository, UUID], Any]:
    """See test_push_subscription_revocation_race.py — same deterministic-ordering trick.

    Only signals when ``watch_user_id``'s row lock specifically is acquired.
    After the E2/C1 fix, ``grant_role``'s recheck locks the actor *and* the
    target together (``also_lock``) in one sorted pass — a signal that
    fired on any locked id could fire on the wrong one depending on how the
    two (randomly generated) UUIDs happen to sort relative to each other,
    reintroducing exactly the raciness this trick exists to eliminate.
    """
    original = UserRepository.lock_user_for_session_change

    async def _wrapped(self: UserRepository, user_id: UUID) -> None:
        await original(self, user_id)
        if user_id == watch_user_id:
            signal.set()

    return _wrapped


async def _run_grant_role(
    *,
    engine: AsyncEngine,
    superadmin_id: UUID,
    target_id: UUID,
    jwt_service: JWTService,
    token_revocation: TokenRevocationService,
    start_after: asyncio.Event | None,
    token_payload: TokenPayload | None = None,
) -> dict[str, object]:
    if start_after is not None:
        await start_after.wait()

    if token_payload is None:
        token, _ = jwt_service.create_access_token(superadmin_id, product_id=PRODUCT_ID)
        token_payload = jwt_service.decode_access_token(token)
        assert token_payload is not None

    superadmin = MagicMock()
    superadmin.id = superadmin_id

    async with AsyncSession(bind=engine, expire_on_commit=False) as session:
        admin_mgmt = AdminManagementService()
        try:
            result = await AdminManagementController.grant_role.fn(
                MagicMock(),
                superadmin=superadmin,
                user_id=target_id,
                data=GrantRoleRequest(role=UserRole.ADMIN),
                session=session,
                product_id=PRODUCT_ID,
                admin_mgmt=admin_mgmt,
                token_payload=token_payload,
                token_revocation_service=token_revocation,
            )
        except NotAuthorizedException:
            return {"status": "rejected"}
        return {"status": "granted", "message": result["message"]}


async def _run_single_device_logout(
    *,
    engine: AsyncEngine,
    access_token: str,
    refresh_token: str,
    jwt_service: JWTService,
    token_revocation: TokenRevocationService,
    settings: Settings,
    start_after: asyncio.Event | None,
) -> None:
    """Race partner for TestGrantRoleVsSingleDeviceLogout (E1) — invokes the
    real ``AuthController.logout`` handler presenting the exact access token
    used to authorize the concurrent grant, so denylisting its jti is the
    revocation the grant's re-check must actually observe.
    """
    if start_after is not None:
        await start_after.wait()

    request = MagicMock()
    request.headers.get.return_value = f"Bearer {access_token}"

    async with (
        AsyncSession(bind=engine, expire_on_commit=False) as session,
        session.begin(),
    ):
        auth_service = AuthService(
            repository=UserRepository(session),
            jwt_service=jwt_service,
            password_service=_make_password_service(),
            token_revocation_service=token_revocation,
            session=session,
            ops_event_bus=OpsEventBus(enabled=False),
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


async def _run_bulk_logout_all(
    *,
    engine: AsyncEngine,
    user_id: UUID,
    jwt_service: JWTService,
    token_revocation: TokenRevocationService,
    start_after: asyncio.Event | None,
) -> None:
    if start_after is not None:
        await start_after.wait()

    async with (
        AsyncSession(bind=engine, expire_on_commit=False) as session,
        session.begin(),
    ):
        auth_service = AuthService(
            repository=UserRepository(session),
            jwt_service=jwt_service,
            password_service=_make_password_service(),
            token_revocation_service=token_revocation,
            session=session,
            ops_event_bus=OpsEventBus(enabled=False),
        )
        await auth_service.logout_all(user_id)


async def _current_role(engine: AsyncEngine, user_id: UUID) -> str:
    async with AsyncSession(bind=engine, expire_on_commit=False) as session:
        result = await session.execute(select(User.role).where(User.id == user_id))
        role = result.scalar_one()
        return role if isinstance(role, str) else role.value


class TestGrantRoleVsLogoutAllRace:
    """A concurrent ``POST /v1/admin/manage/roles/{user_id}/grant`` and the
    granting superadmin's own ``logout-all`` must never leave a persisted
    role grant made with a token that was revoked mid-request, regardless of
    which side acquires the superadmin's user-row lock first.
    """

    async def test_grant_wins_the_lock(self, db_engine: AsyncEngine) -> None:
        """Grant wins the lock -> commits -> logout-all's subsequent
        revocation only affects sessions issued going forward; the grant
        made with the still-valid token stands."""
        jwt_service = JWTService(JWTConfig(secret_key=TEST_SECRET))
        iterations = 50

        for _ in range(iterations):
            superadmin_id, target_id = new_id(), new_id()
            await _seed_users(db_engine, superadmin_id=superadmin_id, target_id=target_id)
            redis = _FakeRedis()
            token_revocation = TokenRevocationService(redis, max_token_ttl_seconds=3600)  # type: ignore[arg-type]
            signal = asyncio.Event()

            try:
                with patch.object(
                    UserRepository,
                    "lock_user_for_session_change",
                    _patched_lock_that_signals(signal, watch_user_id=superadmin_id),
                ):
                    grant_result, _ = await asyncio.gather(
                        _run_grant_role(
                            engine=db_engine,
                            superadmin_id=superadmin_id,
                            target_id=target_id,
                            jwt_service=jwt_service,
                            token_revocation=token_revocation,
                            start_after=None,  # grant starts immediately
                        ),
                        _run_bulk_logout_all(
                            engine=db_engine,
                            user_id=superadmin_id,
                            jwt_service=jwt_service,
                            token_revocation=token_revocation,
                            start_after=signal,  # waits for grant to hold the lock first
                        ),
                    )

                assert grant_result["status"] == "granted", (
                    "the grant should win the lock and be accepted in this ordering"
                )
                assert await _current_role(db_engine, target_id) == UserRole.ADMIN.value
            finally:
                await _cleanup_users(db_engine, superadmin_id=superadmin_id, target_id=target_id)

    async def test_logout_all_wins_the_lock(self, db_engine: AsyncEngine) -> None:
        """Bulk logout-all wins the lock -> the grant request waits -> on
        acquiring, its re-check observes the epoch logout-all just wrote and
        401s instead of granting — the role must never change."""
        jwt_service = JWTService(JWTConfig(secret_key=TEST_SECRET))
        iterations = 50

        for _ in range(iterations):
            superadmin_id, target_id = new_id(), new_id()
            await _seed_users(db_engine, superadmin_id=superadmin_id, target_id=target_id)
            redis = _FakeRedis()
            token_revocation = TokenRevocationService(redis, max_token_ttl_seconds=3600)  # type: ignore[arg-type]
            signal = asyncio.Event()

            try:
                with patch.object(
                    UserRepository,
                    "lock_user_for_session_change",
                    _patched_lock_that_signals(signal, watch_user_id=superadmin_id),
                ):
                    grant_result, _ = await asyncio.gather(
                        _run_grant_role(
                            engine=db_engine,
                            superadmin_id=superadmin_id,
                            target_id=target_id,
                            jwt_service=jwt_service,
                            token_revocation=token_revocation,
                            start_after=signal,  # waits for logout-all to hold the lock first
                        ),
                        _run_bulk_logout_all(
                            engine=db_engine,
                            user_id=superadmin_id,
                            jwt_service=jwt_service,
                            token_revocation=token_revocation,
                            start_after=None,  # logout-all starts immediately
                        ),
                    )

                assert grant_result["status"] == "rejected", (
                    "the grant must observe the epoch logout-all wrote while holding "
                    "the lock, and reject with 401 instead of granting"
                )
                assert await _current_role(db_engine, target_id) == UserRole.USER.value, (
                    "no role change should be persisted at all in this ordering"
                )
            finally:
                await _cleanup_users(db_engine, superadmin_id=superadmin_id, target_id=target_id)


class TestGrantRoleVsSingleDeviceLogout:
    """E1 — the per-jti path must participate in the same lock protocol as
    the bulk paths. A concurrent grant and a single-device logout of the
    granting superadmin's own session (presenting the exact access token
    that authorized the grant) must never leave a persisted role grant made
    with a token whose jti was denylisted mid-request, regardless of which
    side acquires the superadmin's user-row lock first. Before the D-E1 fix
    (AuthService.logout taking lock_user_for_session_change before its
    Redis write), logout never took this lock at all, so this raced wide
    open — this test fails on the pre-remediation SHA.
    """

    async def test_grant_wins_the_lock(self, db_engine: AsyncEngine) -> None:
        """Grant wins the lock -> commits -> the single-device logout that
        follows only denylists this one jti; the grant made while it was
        still valid stands."""
        jwt_service = JWTService(JWTConfig(secret_key=TEST_SECRET))
        settings = Settings(
            jwt_secret_key=TEST_SECRET,
            database_url="postgresql+asyncpg://apex:apex@localhost:5432/apex",
            debug=False,
        )
        iterations = 50

        for _ in range(iterations):
            superadmin_id, target_id = new_id(), new_id()
            await _seed_users(db_engine, superadmin_id=superadmin_id, target_id=target_id)
            refresh_token = await _create_refresh_token(db_engine, user_id=superadmin_id)
            access_token, _ = jwt_service.create_access_token(superadmin_id, product_id=PRODUCT_ID)
            token_payload = jwt_service.decode_access_token(access_token)
            assert token_payload is not None
            redis = _FakeRedis()
            token_revocation = TokenRevocationService(redis, max_token_ttl_seconds=3600)  # type: ignore[arg-type]
            signal = asyncio.Event()

            try:
                with patch.object(
                    UserRepository,
                    "lock_user_for_session_change",
                    _patched_lock_that_signals(signal, watch_user_id=superadmin_id),
                ):
                    grant_result, _ = await asyncio.gather(
                        _run_grant_role(
                            engine=db_engine,
                            superadmin_id=superadmin_id,
                            target_id=target_id,
                            jwt_service=jwt_service,
                            token_revocation=token_revocation,
                            start_after=None,  # grant starts immediately
                            token_payload=token_payload,
                        ),
                        _run_single_device_logout(
                            engine=db_engine,
                            access_token=access_token,
                            refresh_token=refresh_token,
                            jwt_service=jwt_service,
                            token_revocation=token_revocation,
                            settings=settings,
                            start_after=signal,  # waits for grant to hold the lock first
                        ),
                    )

                assert grant_result["status"] == "granted", (
                    "the grant should win the lock and be accepted in this ordering"
                )
                assert await _current_role(db_engine, target_id) == UserRole.ADMIN.value
                assert await token_revocation.is_revoked(token_payload) is True, (
                    "the presenting jti must be denylisted once single-device logout runs, "
                    "even though the grant that shared its lock already committed"
                )
            finally:
                await _cleanup_users(db_engine, superadmin_id=superadmin_id, target_id=target_id)

    async def test_logout_wins_the_lock(self, db_engine: AsyncEngine) -> None:
        """Single-device logout wins the lock -> the grant request waits ->
        on acquiring, its re-check observes the jti denylisted while
        holding the lock and 401s instead of granting — the role must
        never change."""
        jwt_service = JWTService(JWTConfig(secret_key=TEST_SECRET))
        settings = Settings(
            jwt_secret_key=TEST_SECRET,
            database_url="postgresql+asyncpg://apex:apex@localhost:5432/apex",
            debug=False,
        )
        iterations = 50

        for _ in range(iterations):
            superadmin_id, target_id = new_id(), new_id()
            await _seed_users(db_engine, superadmin_id=superadmin_id, target_id=target_id)
            refresh_token = await _create_refresh_token(db_engine, user_id=superadmin_id)
            access_token, _ = jwt_service.create_access_token(superadmin_id, product_id=PRODUCT_ID)
            token_payload = jwt_service.decode_access_token(access_token)
            assert token_payload is not None
            redis = _FakeRedis()
            token_revocation = TokenRevocationService(redis, max_token_ttl_seconds=3600)  # type: ignore[arg-type]
            signal = asyncio.Event()

            try:
                with patch.object(
                    UserRepository,
                    "lock_user_for_session_change",
                    _patched_lock_that_signals(signal, watch_user_id=superadmin_id),
                ):
                    grant_result, _ = await asyncio.gather(
                        _run_grant_role(
                            engine=db_engine,
                            superadmin_id=superadmin_id,
                            target_id=target_id,
                            jwt_service=jwt_service,
                            token_revocation=token_revocation,
                            start_after=signal,  # waits for logout to hold the lock first
                            token_payload=token_payload,
                        ),
                        _run_single_device_logout(
                            engine=db_engine,
                            access_token=access_token,
                            refresh_token=refresh_token,
                            jwt_service=jwt_service,
                            token_revocation=token_revocation,
                            settings=settings,
                            start_after=None,  # logout starts immediately
                        ),
                    )

                assert grant_result["status"] == "rejected", (
                    "the grant must observe the jti logout denylisted while holding "
                    "the lock, and reject with 401 instead of granting"
                )
                assert await _current_role(db_engine, target_id) == UserRole.USER.value, (
                    "no role change should be persisted at all in this ordering"
                )
            finally:
                await _cleanup_users(db_engine, superadmin_id=superadmin_id, target_id=target_id)


class TestMutualGrantDeadlock:
    """E2/C1 — two superadmins granting each other a role concurrently must
    never deadlock. Mirrored actor/target pairs (A grants B, B grants A)
    would otherwise take the acting user's row lock first and the target's
    row lock second (explicit ``UPDATE users`` in ``update_user_admin``),
    forming a lock cycle Postgres aborts with SQLSTATE 40P01. ``also_lock``
    (D-E2/D-E3) closes this by locking the full ``{actor, target}`` set in
    one sorted pass per request, so no cycle can form regardless of
    interleaving. This test fails on the pre-remediation SHA.
    """

    async def test_mutual_grant_never_deadlocks(self, db_engine: AsyncEngine) -> None:
        jwt_service = JWTService(JWTConfig(secret_key=TEST_SECRET))
        iterations = 50

        for _ in range(iterations):
            user_a_id, user_b_id = new_id(), new_id()
            await _seed_two_superadmins(db_engine, user_a_id=user_a_id, user_b_id=user_b_id)
            redis = _FakeRedis()
            token_revocation = TokenRevocationService(redis, max_token_ttl_seconds=3600)  # type: ignore[arg-type]

            try:
                results = await asyncio.gather(
                    _run_grant_role(
                        engine=db_engine,
                        superadmin_id=user_a_id,
                        target_id=user_b_id,
                        jwt_service=jwt_service,
                        token_revocation=token_revocation,
                        start_after=None,
                    ),
                    _run_grant_role(
                        engine=db_engine,
                        superadmin_id=user_b_id,
                        target_id=user_a_id,
                        jwt_service=jwt_service,
                        token_revocation=token_revocation,
                        start_after=None,
                    ),
                    return_exceptions=True,
                )

                for result in results:
                    assert not isinstance(result, BaseException), (
                        f"mutual grant must never raise (deadlock or otherwise): {result!r}"
                    )

                statuses = {result["status"] for result in results}  # type: ignore[index]
                assert statuses == {"granted"}, (
                    f"both mutual grants should commit without contention rejection: {results}"
                )
            finally:
                await _cleanup_users(db_engine, superadmin_id=user_a_id, target_id=user_b_id)
