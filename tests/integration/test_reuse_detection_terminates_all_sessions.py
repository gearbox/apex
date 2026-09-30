"""W1-c — refresh-token reuse detection signs the user out on every device.

Real PostgreSQL rows, a real ``SessionTerminationService`` and an in-memory Redis
for the revocation epoch. The replay goes through ``AuthController.refresh_tokens``
inside a committing transaction, so it also proves the 401 still commits the
revocations (the route returns a ``Response`` instead of raising).
"""

from __future__ import annotations

import asyncio
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock, patch
from uuid import UUID, uuid4

import pytest
from litestar import Litestar, get
from litestar.di import Provide
from litestar.testing import TestClient
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.dependencies.auth import get_current_user_id
from src.api.routes.auth import AuthController
from src.api.schemas.auth import RefreshTokenRequest
from src.api.security import auth_guard, hash_token
from src.api.security.jwt import JWTConfig, JWTService
from src.api.services.auth import (
    AuthService,
    InvalidRefreshTokenError,
    TokenPair,
    TokenReuseDetectedError,
)
from src.api.services.session_termination import SessionTerminationService
from src.api.services.token_revocation import TokenRevocationService
from src.core.config import Settings
from src.core.enums import RefreshTokenRevocationReason
from src.core.product_registry import VEX_CONFIG
from src.db.models.user import RefreshToken, User
from src.db.repositories.push_subscription import PushSubscriptionRepository
from src.db.repositories.user import UserRepository
from tests.legal_support import TEST_REQUEST_CONTEXT, make_legal_acceptance_service
from tests.revocation_support import FakeRedis, make_session_termination

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Iterator

    from sqlalchemy.ext.asyncio import AsyncEngine

TEST_SECRET = "test_secret_key_for_testing_only_256bits_long"
PRODUCT_ID = "vex"


@pytest.fixture
def jwt_service() -> JWTService:
    return JWTService(JWTConfig(secret_key=TEST_SECRET))


@pytest.fixture
def token_revocation() -> TokenRevocationService:
    return TokenRevocationService(FakeRedis(), max_token_ttl_seconds=3600)  # type: ignore[arg-type]


@pytest.fixture
async def user_id(db_engine: AsyncEngine) -> AsyncGenerator[UUID]:
    uid = uuid4()
    async with AsyncSession(bind=db_engine, expire_on_commit=False) as session:
        session.add(
            User(
                id=uid,
                email=f"w1c-{uuid4().hex[:8]}@example.com",
                password_hash="x" * 64,
                product_id=PRODUCT_ID,
                is_active=True,
            )
        )
        await session.commit()
    yield uid
    async with AsyncSession(bind=db_engine, expire_on_commit=False) as session:
        await session.execute(delete(RefreshToken).where(RefreshToken.user_id == uid))
        await session.execute(delete(User).where(User.id == uid))
        await session.commit()


def _service(
    session: AsyncSession, jwt_service: JWTService, token_revocation: TokenRevocationService
) -> AuthService:
    repo = UserRepository(session)
    return AuthService(
        legal_acceptance_service=make_legal_acceptance_service(),
        repository=repo,
        jwt_service=jwt_service,
        password_service=MagicMock(),
        token_revocation_service=token_revocation,
        session=session,
        session_termination=make_session_termination(
            user_repo=repo, token_revocation=token_revocation, session=session
        ),
    )


def _app(jwt_service: JWTService, token_revocation: TokenRevocationService) -> Litestar:
    @get(
        "/ping",
        guards=[auth_guard],
        dependencies={"current_user_id": Provide(get_current_user_id)},
    )
    async def ping(current_user_id: UUID) -> dict[str, str]:
        return {"user_id": str(current_user_id)}

    app = Litestar(route_handlers=[ping])
    app.state["jwt_service"] = jwt_service
    app.state["token_revocation"] = token_revocation
    return app


class _Harness:
    def __init__(
        self,
        engine: AsyncEngine,
        jwt_service: JWTService,
        token_revocation: TokenRevocationService,
        user_id: UUID,
    ) -> None:
        self.engine = engine
        self.jwt = jwt_service
        self.revocation = token_revocation
        self.user_id = user_id

    async def issue(self) -> TokenPair:
        async with AsyncSession(bind=self.engine, expire_on_commit=False) as s, s.begin():
            return await _service(s, self.jwt, self.revocation).issue_session(
                self.user_id, product_id=PRODUCT_ID, context=TEST_REQUEST_CONTEXT
            )

    async def refresh(self, raw: str) -> TokenPair:
        async with AsyncSession(bind=self.engine, expire_on_commit=False) as s, s.begin():
            tokens, _ = await _service(s, self.jwt, self.revocation).refresh_tokens(raw)
        return tokens

    async def replay_via_route(self, raw: str) -> Any:
        """The route catches the reuse error, so the surrounding transaction commits."""
        async with AsyncSession(bind=self.engine, expire_on_commit=False) as s, s.begin():
            return await AuthController.refresh_tokens.fn(
                MagicMock(),
                data=RefreshTokenRequest(refresh_token=raw),
                auth_service=_service(s, self.jwt, self.revocation),
                jwt_service=self.jwt,
                product_id=PRODUCT_ID,
                product_config=VEX_CONFIG,
                settings=Settings(jwt_secret_key=TEST_SECRET, debug=True),
                request_context=TEST_REQUEST_CONTEXT,
            )

    async def logout(self, raw: str) -> None:
        async with AsyncSession(bind=self.engine, expire_on_commit=False) as s, s.begin():
            await _service(s, self.jwt, self.revocation).logout(raw)

    async def subscribe_push(self, endpoint: str) -> None:
        async with AsyncSession(bind=self.engine, expire_on_commit=False) as s, s.begin():
            await PushSubscriptionRepository(s).upsert(
                user_id=self.user_id,
                product_id=PRODUCT_ID,
                endpoint=endpoint,
                p256dh="p",
                auth="a",
                user_agent=None,
            )

    async def push_endpoints(self) -> list[str]:
        async with AsyncSession(bind=self.engine) as s:
            rows = await PushSubscriptionRepository(s).list_by_user(self.user_id)
        return [row.endpoint for row in rows]

    def ping(self, access_token: str) -> int:
        with TestClient(app=_app(self.jwt, self.revocation)) as client:
            return client.get(
                "/ping", headers={"Authorization": f"Bearer {access_token}"}
            ).status_code

    async def reason(self, raw: str) -> str | None:
        async with AsyncSession(bind=self.engine) as s:
            row = (
                await s.execute(
                    select(RefreshToken.is_revoked, RefreshToken.revoked_reason).where(
                        RefreshToken.token_hash == hash_token(raw)
                    )
                )
            ).one()
        assert row.is_revoked is True
        reason: str | None = row.revoked_reason
        return reason


async def test_w1_c_reuse_detection_signs_out_every_device(
    db_engine: AsyncEngine,
    jwt_service: JWTService,
    token_revocation: TokenRevocationService,
    user_id: UUID,
) -> None:
    h = _Harness(db_engine, jwt_service, token_revocation, user_id)
    app = _app(jwt_service, token_revocation)

    # Device A and device B log in; A rotates once, so its first token is now "stolen".
    device_a = await h.issue()
    device_b = await h.issue()
    device_a_rotated = await h.refresh(device_a.refresh_token)
    with TestClient(app=app) as client:
        ok = client.get("/ping", headers={"Authorization": f"Bearer {device_b.access_token}"})
    assert ok.status_code == 200

    # Replaying the rotated-away token is theft: 401 token_reuse_detected, all devices out.
    response = await h.replay_via_route(device_a.refresh_token)
    assert response.status_code == 401
    assert response.content.error == "token_reuse_detected"
    assert "signed out on all devices" in response.content.message

    # It committed: the stolen family is stamped reuse_detected (the replayed row
    # included, X1), device B bulk_revocation.
    assert await h.reason(device_a_rotated.refresh_token) == (
        RefreshTokenRevocationReason.REUSE_DETECTED.value
    )
    assert await h.reason(device_a.refresh_token) == (
        RefreshTokenRevocationReason.REUSE_DETECTED.value
    )
    assert await h.reason(device_b.refresh_token) == (
        RefreshTokenRevocationReason.BULK_REVOCATION.value
    )

    # Device B: refresh token refused as an ended session, access token refused by the guard.
    with pytest.raises(InvalidRefreshTokenError, match="session has ended"):
        await h.refresh(device_b.refresh_token)
    with TestClient(app=app) as client:
        denied = client.get("/ping", headers={"Authorization": f"Bearer {device_b.access_token}"})
    assert denied.status_code == 401

    # A second replay from the stolen family is still classified as theft, not as a
    # benign bulk-revocation race — for the old token and for the live-but-stamped one.
    for stolen in (device_a.refresh_token, device_a_rotated.refresh_token):
        with pytest.raises(TokenReuseDetectedError):
            await h.refresh(stolen)
    again = await h.replay_via_route(device_a_rotated.refresh_token)
    assert again.content.error == "token_reuse_detected"


_REPLAYED_MESSAGE = "Please sign in again."


@contextmanager
def _count_terminations() -> Iterator[list[str]]:
    """Record the ``op`` of every ``terminate_all`` call (the real method still runs)."""
    ops: list[str] = []
    original = SessionTerminationService.terminate_all

    async def counting(self: SessionTerminationService, user_id: UUID, **kwargs: str) -> Any:
        ops.append(kwargs["op"])
        return await original(self, user_id, **kwargs)

    with patch.object(SessionTerminationService, "terminate_all", counting):
        yield ops


async def test_x1_a_replaying_a_handled_stolen_token_never_signs_the_victim_out_again(
    db_engine: AsyncEngine,
    jwt_service: JWTService,
    token_revocation: TokenRevocationService,
    user_id: UUID,
) -> None:
    h = _Harness(db_engine, jwt_service, token_revocation, user_id)

    device_a = await h.issue()
    await h.refresh(device_a.refresh_token)  # device_a.refresh_token is now "stolen"

    with patch.object(
        token_revocation, "revoke_user_sessions", wraps=token_revocation.revoke_user_sessions
    ) as revoke_user_sessions:
        # First detection: signs everyone out, exactly once.
        first = await h.replay_via_route(device_a.refresh_token)
        assert first.status_code == 401
        assert "signed out on all devices" in first.content.message
        assert revoke_user_sessions.await_count == 1
        epoch_after_detection = await token_revocation.get_current_epoch(user_id)
        assert epoch_after_detection is not None

        # The victim signs in again (new family) and registers a push subscription.
        victim = await h.issue()
        await h.subscribe_push("https://push.example/x1-a")

        # The attacker replays the stolen token twice: refused, with no side effects.
        for _ in range(2):
            replay = await h.replay_via_route(device_a.refresh_token)
            assert replay.status_code == 401
            assert replay.content.error == "token_reuse_detected"
            assert _REPLAYED_MESSAGE in replay.content.message
            assert "signed out on all devices" not in replay.content.message

        # The victim's session is untouched.
        rotated = await h.refresh(victim.refresh_token)
        assert h.ping(victim.access_token) == 200
        assert h.ping(rotated.access_token) == 200
        assert await h.push_endpoints() == ["https://push.example/x1-a"]
        assert await token_revocation.get_current_epoch(user_id) == epoch_after_detection
        assert revoke_user_sessions.await_count == 1


async def test_x1_b_first_detection_stamps_the_replayed_row_and_the_family(
    db_engine: AsyncEngine,
    jwt_service: JWTService,
    token_revocation: TokenRevocationService,
    user_id: UUID,
) -> None:
    h = _Harness(db_engine, jwt_service, token_revocation, user_id)
    device_a = await h.issue()
    device_b = await h.issue()
    device_a_rotated = await h.refresh(device_a.refresh_token)

    response = await h.replay_via_route(device_a.refresh_token)

    assert response.status_code == 401
    # The replayed row (was ROTATED) and the live family row both read reuse_detected...
    reuse = RefreshTokenRevocationReason.REUSE_DETECTED.value
    assert await h.reason(device_a.refresh_token) == reuse
    assert await h.reason(device_a_rotated.refresh_token) == reuse
    # ...while the other device was signed out by the bulk revocation.
    assert await h.reason(device_b.refresh_token) == (
        RefreshTokenRevocationReason.BULK_REVOCATION.value
    )


async def test_x1_c_family_with_no_live_token_is_handled_once_then_only_refused(
    db_engine: AsyncEngine,
    jwt_service: JWTService,
    token_revocation: TokenRevocationService,
    user_id: UUID,
) -> None:
    h = _Harness(db_engine, jwt_service, token_revocation, user_id)
    device_a = await h.issue()
    device_b = await h.issue()
    device_a_rotated = await h.refresh(device_a.refresh_token)
    await h.logout(device_a_rotated.refresh_token)  # family A now has no active row
    assert await h.reason(device_a_rotated.refresh_token) == (
        RefreshTokenRevocationReason.SINGLE_LOGOUT.value
    )

    with _count_terminations() as terminations:
        first = await h.replay_via_route(device_a.refresh_token)
        assert first.status_code == 401
        assert terminations == ["token_reuse_detected"]

        # revoke_token_family found nothing active; the replayed row is the marker.
        assert await h.reason(device_a.refresh_token) == (
            RefreshTokenRevocationReason.REUSE_DETECTED.value
        )
        assert await h.reason(device_b.refresh_token) == (
            RefreshTokenRevocationReason.BULK_REVOCATION.value
        )

        # The victim signs in again; a second replay is refused and leaves them alone.
        victim = await h.issue()
        second = await h.replay_via_route(device_a.refresh_token)
        assert second.status_code == 401
        assert _REPLAYED_MESSAGE in second.content.message
        assert terminations == ["token_reuse_detected"]

    await h.refresh(victim.refresh_token)
    assert h.ping(victim.access_token) == 200


async def test_x1_d_concurrent_first_detections_terminate_exactly_once(
    db_engine: AsyncEngine,
    jwt_service: JWTService,
    token_revocation: TokenRevocationService,
    user_id: UUID,
) -> None:
    h = _Harness(db_engine, jwt_service, token_revocation, user_id)
    device_a = await h.issue()
    await h.issue()
    await h.refresh(device_a.refresh_token)

    with _count_terminations() as terminations:
        responses = await asyncio.gather(
            h.replay_via_route(device_a.refresh_token),
            h.replay_via_route(device_a.refresh_token),
        )

    assert [r.status_code for r in responses] == [401, 401]
    assert [r.content.error for r in responses] == ["token_reuse_detected"] * 2
    # The user-row lock serializes the two replays: one detects, the other only refuses.
    assert terminations == ["token_reuse_detected"]
    messages = sorted("signed out on all devices" in r.content.message for r in responses)
    assert messages == [False, True]
