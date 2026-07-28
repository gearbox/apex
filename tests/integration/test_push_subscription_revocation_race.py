"""Integration test for be-push-subscription-race-fix R1 (the regression that matters).

Exercises the real call chain — ``PushController.create_subscription`` (the
actual decorated handler function, invoked via ``.fn(...)`` the same way
tests/unit/test_frames_routes.py and tests/unit/test_admin.py already do in
this repo) racing ``AuthService.logout_all`` — against a real PostgreSQL
database and a shared in-memory fake Redis, proving that a subscription
cannot survive a concurrent bulk revocation in either lock-acquisition
ordering.

Self-contained against ``db_engine`` (two independent connections) rather
than the SAVEPOINT-per-test ``db_session`` fixture, which only allocates one
connection — the row lock this test exercises requires two genuinely
concurrent transactions. Mirrors the structure of
``TestRefreshVsLogoutAllRace`` in test_token_revocation_flow.py.

Both interleavings are forced deterministically via an ``asyncio.Event``
rather than left to natural scheduling: the "first" side runs immediately,
the "second" side awaits the event (set right after the first side's own
``lock_user_for_session_change`` call returns) before even opening its
session. This guarantees the intended lock-acquisition order every
iteration without relying on timing luck, so there is no randomised ordering
here to seed — the two orderings are each pinned to their own test method
and run 50 times apiece, matching this repo's existing concurrency-test
convention (test_token_revocation_flow.py's ``TestRefreshVsLogoutAllRace``).
"""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock, patch
from uuid import UUID, uuid4

from litestar.exceptions import NotAuthorizedException
from redis.exceptions import NoScriptError
from sqlalchemy import delete
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.routes.push import PushController
from src.api.schemas.push import PushSubscriptionKeys, PushSubscriptionRequest
from src.api.security.jwt import JWTConfig, JWTService
from src.api.services.auth import AuthService
from src.api.services.ops_event_bus import OpsEventBus
from src.api.services.push import PushService
from src.api.services.token_revocation import TokenRevocationService
from src.core.uid import new_id
from src.db.models.push_subscription import PushSubscription
from src.db.models.user import RefreshToken, User
from src.db.repositories.push_subscription import PushSubscriptionRepository
from src.db.repositories.user import UserRepository

if TYPE_CHECKING:
    from collections.abc import Callable

    from sqlalchemy.ext.asyncio import AsyncEngine

TEST_SECRET = "test_secret_key_for_testing_only_256bits_long"
PRODUCT_ID = "vex"


class _FakeRedis:
    """Minimal in-memory stand-in for redis.asyncio.Redis — same subset and
    semantics as test_token_revocation_flow.py's ``_FakeRedis`` (real TTL
    expiry, pluggable clock). Shared by both sides of the race: real Redis
    is shared infrastructure both concurrent requests would hit, and the
    whole point of the "bulk wins the lock" ordering is that the epoch
    write is visible to the push-creation side's own re-check.
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
        key = str(keys_and_args[0])
        ttl = int(str(keys_and_args[numkeys]))
        now = int(self._clock())
        await self.set(key, now, ex=ttl)
        return now


class _DummySender:
    """A WebPushSender that is never actually called by these tests —
    create_subscription only upserts, it never sends."""

    async def send(
        self,
        *,
        endpoint: str,  # noqa: ARG002
        p256dh: str,  # noqa: ARG002
        auth: str,  # noqa: ARG002
        payload: dict[str, Any],  # noqa: ARG002
    ) -> None:
        raise AssertionError("send() should not be called by create_subscription")


def _make_password_service() -> MagicMock:
    return MagicMock()


async def _seed_user(engine: AsyncEngine, *, user_id: UUID) -> None:
    user = User(
        id=user_id,
        email=f"race-push-{uuid4().hex[:8]}@example.com",
        password_hash="x" * 64,
        product_id=PRODUCT_ID,
        is_active=True,
    )
    async with AsyncSession(bind=engine, expire_on_commit=False) as session:
        session.add(user)
        await session.commit()


async def _cleanup_user(engine: AsyncEngine, user_id: UUID) -> None:
    async with AsyncSession(bind=engine, expire_on_commit=False) as session:
        await session.execute(delete(PushSubscription).where(PushSubscription.user_id == user_id))
        await session.execute(delete(RefreshToken).where(RefreshToken.user_id == user_id))
        await session.execute(delete(User).where(User.id == user_id))
        await session.commit()


def _patched_lock_that_signals(
    signal: asyncio.Event,
) -> Callable[[UserRepository, UUID], Any]:
    """Wraps the real ``lock_user_for_session_change`` so the first caller
    to acquire the row lock sets ``signal`` right after acquiring it (before
    doing anything else). The second side waits on this same event before
    even opening its own session — see module docstring for why this makes
    the ordering deterministic rather than relying on scheduling luck.
    """
    original = UserRepository.lock_user_for_session_change

    async def _wrapped(self: UserRepository, user_id: UUID) -> None:
        await original(self, user_id)
        signal.set()

    return _wrapped


async def _run_create_subscription(
    *,
    engine: AsyncEngine,
    user_id: UUID,
    jwt_service: JWTService,
    token_revocation: TokenRevocationService,
    endpoint: str,
    start_after: asyncio.Event | None,
) -> dict[str, object]:
    if start_after is not None:
        await start_after.wait()

    token, _ = jwt_service.create_access_token(user_id, product_id=PRODUCT_ID)
    token_payload = jwt_service.decode_access_token(token)
    assert token_payload is not None

    async with AsyncSession(bind=engine, expire_on_commit=False) as session:
        push_service = PushService(sender=_DummySender())
        try:
            result = await PushController.create_subscription.fn(
                MagicMock(),
                data=PushSubscriptionRequest(
                    endpoint=endpoint,
                    keys=PushSubscriptionKeys(p256dh="p256dh-key", auth="auth-key"),
                    user_agent="test-agent",
                ),
                current_user_id=user_id,
                product_id=PRODUCT_ID,
                session=session,
                push_service=push_service,
                token_payload=token_payload,
                token_revocation_service=token_revocation,
            )
        except NotAuthorizedException:
            return {"status": "rejected"}
        return {"status": "created", "id": result.id}


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


class TestCreateSubscriptionVsLogoutAllRace:
    """R1 (be-push-subscription-race-fix) — a concurrent
    ``POST /v1/push/subscriptions`` and ``logout-all`` must never leave a
    subscribed row for the device the revocation was meant to unsubscribe,
    regardless of which side acquires the user-row lock first.
    """

    async def test_push_creation_wins_the_lock(self, db_engine: AsyncEngine) -> None:
        """Insert wins the lock -> commits -> logout-all's subsequent
        snapshot (taken after it acquires the lock) includes the new row,
        so delete_all_for_user removes it."""
        jwt_service = JWTService(JWTConfig(secret_key=TEST_SECRET))
        iterations = 50

        for _ in range(iterations):
            user_id = new_id()
            await _seed_user(db_engine, user_id=user_id)
            redis = _FakeRedis()
            token_revocation = TokenRevocationService(redis, max_token_ttl_seconds=3600)  # type: ignore[arg-type]
            signal = asyncio.Event()

            try:
                with patch.object(
                    UserRepository,
                    "lock_user_for_session_change",
                    _patched_lock_that_signals(signal),
                ):
                    push_result, _ = await asyncio.gather(
                        _run_create_subscription(
                            engine=db_engine,
                            user_id=user_id,
                            jwt_service=jwt_service,
                            token_revocation=token_revocation,
                            endpoint=f"https://push.example/{user_id}",
                            start_after=None,  # push creation starts immediately
                        ),
                        _run_bulk_logout_all(
                            engine=db_engine,
                            user_id=user_id,
                            jwt_service=jwt_service,
                            token_revocation=token_revocation,
                            start_after=signal,  # waits for push to hold the lock first
                        ),
                    )

                assert push_result["status"] == "created", (
                    "push creation should win the lock and be accepted in this ordering"
                )

                async with AsyncSession(bind=db_engine, expire_on_commit=False) as session:
                    remaining = await PushSubscriptionRepository(session).list_by_user(user_id)
                assert remaining == [], (
                    "a subscription created while winning the lock race must still be "
                    "deleted by the bulk revocation that follows it"
                )
            finally:
                await _cleanup_user(db_engine, user_id)

    async def test_logout_all_wins_the_lock(self, db_engine: AsyncEngine) -> None:
        """Bulk wins the lock -> the insert waits -> on acquiring, its
        re-check observes the epoch logout-all just wrote and 401s instead
        of inserting."""
        jwt_service = JWTService(JWTConfig(secret_key=TEST_SECRET))
        iterations = 50

        for _ in range(iterations):
            user_id = new_id()
            await _seed_user(db_engine, user_id=user_id)
            redis = _FakeRedis()
            token_revocation = TokenRevocationService(redis, max_token_ttl_seconds=3600)  # type: ignore[arg-type]
            signal = asyncio.Event()

            try:
                with patch.object(
                    UserRepository,
                    "lock_user_for_session_change",
                    _patched_lock_that_signals(signal),
                ):
                    push_result, _ = await asyncio.gather(
                        _run_create_subscription(
                            engine=db_engine,
                            user_id=user_id,
                            jwt_service=jwt_service,
                            token_revocation=token_revocation,
                            endpoint=f"https://push.example/{user_id}",
                            start_after=signal,  # waits for logout-all to hold the lock first
                        ),
                        _run_bulk_logout_all(
                            engine=db_engine,
                            user_id=user_id,
                            jwt_service=jwt_service,
                            token_revocation=token_revocation,
                            start_after=None,  # logout-all starts immediately
                        ),
                    )

                assert push_result["status"] == "rejected", (
                    "push creation must observe the epoch logout-all wrote while holding "
                    "the lock, and reject with 401 instead of inserting"
                )

                async with AsyncSession(bind=db_engine, expire_on_commit=False) as session:
                    remaining = await PushSubscriptionRepository(session).list_by_user(user_id)
                assert remaining == [], (
                    "no row should exist at all in this ordering — the insert must never "
                    "have committed"
                )
            finally:
                await _cleanup_user(db_engine, user_id)
