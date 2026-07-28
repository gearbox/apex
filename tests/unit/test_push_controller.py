"""Controller-level tests for PushController.

Uses Litestar's TestClient with overridden dependencies, following the same
pattern as ``tests/unit/test_auth_guards.py`` — no real DB, no real Redis.
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID, uuid4

import pytest
from litestar import Litestar
from litestar.di import Provide
from litestar.exceptions import NotAuthorizedException, ServiceUnavailableException
from litestar.status_codes import (
    HTTP_200_OK,
    HTTP_201_CREATED,
    HTTP_204_NO_CONTENT,
    HTTP_401_UNAUTHORIZED,
    HTTP_503_SERVICE_UNAVAILABLE,
)
from litestar.testing import TestClient
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.routes.push import PushController
from src.api.security import JWTConfig, JWTService
from src.api.services.push import PushService
from src.api.services.token_revocation import TokenRevocationService

TEST_SECRET = "test_secret_key_for_testing_only_256bits_long"


@pytest.fixture
def jwt_service() -> JWTService:
    return JWTService(JWTConfig(secret_key=TEST_SECRET))


@pytest.fixture
def test_user_id() -> UUID:
    return uuid4()


@pytest.fixture
def auth_header(jwt_service: JWTService, test_user_id: UUID) -> dict[str, str]:
    token, _ = jwt_service.create_access_token(test_user_id)
    return {"Authorization": f"Bearer {token}"}


def _no_op_token_revocation() -> TokenRevocationService:
    """A TokenRevocationService with no Redis client — never reports revoked.

    Mirrors tests/unit/test_auth_guards.py's helper of the same name.
    """
    return TokenRevocationService(None, max_token_ttl_seconds=0)


def _create_app(
    jwt_service: JWTService,
    *,
    push_service: PushService | None = None,
    push_service_raises_503: bool = False,
    token_revocation_service: TokenRevocationService | None = None,
) -> Litestar:
    mock_session = MagicMock(spec=AsyncSession)
    mock_session.commit = AsyncMock()

    def _provide_push_service() -> PushService:
        if push_service_raises_503:
            raise ServiceUnavailableException(detail="Push notifications not available")
        assert push_service is not None
        return push_service

    # Same instance backs both the guard's app.state lookup and the
    # create_subscription handler's DI param (R2) — in production both
    # resolve to the same process-wide singleton, and a test wiring two
    # independent instances here would mask a real desync.
    resolved_token_revocation = token_revocation_service or _no_op_token_revocation()

    app = Litestar(
        route_handlers=[PushController],
        dependencies={
            "session": Provide(lambda: mock_session, sync_to_thread=False),
            "product_id": Provide(lambda: "vex", sync_to_thread=False),
            "push_service": Provide(_provide_push_service, sync_to_thread=False),
            "token_revocation_service": Provide(
                lambda: resolved_token_revocation, sync_to_thread=False
            ),
        },
    )
    app.state["jwt_service"] = jwt_service
    app.state["token_revocation"] = resolved_token_revocation
    app.state["mock_session"] = mock_session
    return app


# ---------------------------------------------------------------------------
# GET /v1/push/vapid-public-key
# ---------------------------------------------------------------------------


class TestVapidPublicKey:
    def test_requires_auth(self, jwt_service: JWTService) -> None:
        app = _create_app(jwt_service)
        with TestClient(app=app) as client:
            resp = client.get("/v1/push/vapid-public-key")
            assert resp.status_code == HTTP_401_UNAUTHORIZED

    def test_returns_503_when_push_disabled(
        self, jwt_service: JWTService, auth_header: dict[str, str]
    ) -> None:
        app = _create_app(jwt_service)
        fake_settings = MagicMock(push_enabled=False, vapid_public_key=None)
        with (
            patch("src.api.routes.push.get_settings", return_value=fake_settings),
            TestClient(app=app) as client,
        ):
            resp = client.get("/v1/push/vapid-public-key", headers=auth_header)
            assert resp.status_code == HTTP_503_SERVICE_UNAVAILABLE
            assert resp.json()["error"] == "service_unavailable"

    def test_returns_public_key_when_enabled(
        self, jwt_service: JWTService, auth_header: dict[str, str]
    ) -> None:
        app = _create_app(jwt_service)
        fake_settings = MagicMock(push_enabled=True, vapid_public_key="abc123")
        with (
            patch("src.api.routes.push.get_settings", return_value=fake_settings),
            TestClient(app=app) as client,
        ):
            resp = client.get("/v1/push/vapid-public-key", headers=auth_header)
            assert resp.status_code == HTTP_200_OK
            assert resp.json() == {"public_key": "abc123"}


# ---------------------------------------------------------------------------
# POST /v1/push/subscriptions
# ---------------------------------------------------------------------------


class TestCreateSubscription:
    def _body(self) -> dict:
        return {
            "endpoint": "https://push.example/abc",
            "keys": {"p256dh": "p256dh-key", "auth": "auth-key"},
            "user_agent": "test-agent",
        }

    def test_requires_auth(self, jwt_service: JWTService) -> None:
        app = _create_app(jwt_service, push_service_raises_503=True)
        with TestClient(app=app) as client:
            resp = client.post("/v1/push/subscriptions", json=self._body())
            assert resp.status_code == HTTP_401_UNAUTHORIZED

    def test_returns_503_when_push_disabled(
        self, jwt_service: JWTService, auth_header: dict[str, str]
    ) -> None:
        app = _create_app(jwt_service, push_service_raises_503=True)
        with TestClient(app=app) as client:
            resp = client.post("/v1/push/subscriptions", json=self._body(), headers=auth_header)
            assert resp.status_code == HTTP_503_SERVICE_UNAVAILABLE

    def test_creates_subscription_when_enabled(
        self, jwt_service: JWTService, test_user_id: UUID, auth_header: dict[str, str]
    ) -> None:
        mock_push_service = AsyncMock(spec=PushService)
        subscription_id = uuid4()
        mock_subscription = MagicMock()
        mock_subscription.id = subscription_id
        mock_subscription.endpoint = "https://push.example/abc"
        mock_subscription.created_at = datetime.now(UTC)
        mock_push_service.upsert_subscription.return_value = mock_subscription

        app = _create_app(jwt_service, push_service=mock_push_service)
        with TestClient(app=app) as client:
            resp = client.post("/v1/push/subscriptions", json=self._body(), headers=auth_header)

            assert resp.status_code == HTTP_201_CREATED
            body = resp.json()
            assert body["id"] == str(subscription_id)
            assert body["endpoint"] == "https://push.example/abc"

        mock_push_service.upsert_subscription.assert_awaited_once()
        call_kwargs = mock_push_service.upsert_subscription.call_args.kwargs
        assert call_kwargs["user_id"] == test_user_id
        assert call_kwargs["product_id"] == "vex"
        assert call_kwargs["endpoint"] == "https://push.example/abc"
        assert call_kwargs["p256dh"] == "p256dh-key"
        assert call_kwargs["auth"] == "auth-key"
        app.state["mock_session"].commit.assert_awaited_once()

    def test_rejects_unknown_fields(
        self, jwt_service: JWTService, auth_header: dict[str, str]
    ) -> None:
        mock_push_service = AsyncMock(spec=PushService)
        app = _create_app(jwt_service, push_service=mock_push_service)
        body = self._body()
        body["unexpected_field"] = "nope"
        with TestClient(app=app) as client:
            resp = client.post("/v1/push/subscriptions", json=body, headers=auth_header)
            assert resp.status_code >= 400
        mock_push_service.upsert_subscription.assert_not_awaited()


# ---------------------------------------------------------------------------
# POST /v1/push/subscriptions — R1 revocation re-check
# (be-push-subscription-race-fix): the handler locks the user row, then
# re-checks revocation, then upserts. The 50-iteration two-way concurrency
# test against a real database lives in
# tests/integration/test_push_subscription_revocation_race.py — these are
# fast, DB-free unit checks of the handler's own logic and ordering.
# ---------------------------------------------------------------------------


class TestCreateSubscriptionRevocationRecheck:
    def _body(self) -> dict[str, object]:
        return {
            "endpoint": "https://push.example/abc",
            "keys": {"p256dh": "p256dh-key", "auth": "auth-key"},
            "user_agent": "test-agent",
        }

    def test_revoked_token_rejected_end_to_end_writes_no_row(
        self, jwt_service: JWTService, auth_header: dict[str, str]
    ) -> None:
        """A token already revoked before the request is sent is rejected
        with 401 and never reaches push_service (guard rejects it first;
        the handler's own re-check exists for the race where revocation
        lands *during* the request — see the direct-call tests below)."""
        mock_push_service = AsyncMock(spec=PushService)
        revoked = AsyncMock(spec=TokenRevocationService)
        revoked.is_revoked.return_value = True

        app = _create_app(
            jwt_service, push_service=mock_push_service, token_revocation_service=revoked
        )
        with TestClient(app=app) as client:
            resp = client.post("/v1/push/subscriptions", json=self._body(), headers=auth_header)
            assert resp.status_code == HTTP_401_UNAUTHORIZED

        mock_push_service.upsert_subscription.assert_not_awaited()

    async def _call_handler(
        self,
        *,
        push_service: AsyncMock,
        token_revocation_service: AsyncMock,
        session: MagicMock | None = None,
        product_id: str = "vex",
    ) -> tuple[MagicMock, object]:
        """Invoke PushController.create_subscription.fn(...) directly —
        bypasses auth_guard entirely, isolating the handler's own
        lock-then-recheck-then-upsert logic from guard-level revocation
        checking (which test_revoked_token_rejected_end_to_end_writes_no_row
        above already covers)."""
        from src.api.schemas.push import PushSubscriptionKeys, PushSubscriptionRequest

        mock_session = session or MagicMock(spec=AsyncSession)
        mock_session.commit = AsyncMock()
        token_payload = MagicMock()
        token_payload.sub = str(uuid4())

        result = await PushController.create_subscription.fn(
            MagicMock(),
            data=PushSubscriptionRequest(
                endpoint="https://push.example/abc",
                keys=PushSubscriptionKeys(p256dh="p256dh-key", auth="auth-key"),
                user_agent="test-agent",
            ),
            current_user_id=uuid4(),
            product_id=product_id,
            session=mock_session,
            push_service=push_service,
            token_payload=token_payload,
            token_revocation_service=token_revocation_service,
        )
        return mock_session, result

    async def test_direct_call_rejects_when_recheck_finds_revoked(self) -> None:
        """Isolates R1's own re-check: even with the guard bypassed, a
        token_revocation_service that reports revoked at re-check time must
        raise 401 and never call upsert_subscription or commit."""
        mock_push_service = AsyncMock(spec=PushService)
        revoked = AsyncMock(spec=TokenRevocationService)
        revoked.is_revoked.return_value = True

        with (
            patch("src.api.routes.push.UserRepository") as mock_repo_cls,
            pytest.raises(NotAuthorizedException, match="Session has been revoked"),
        ):
            mock_repo_cls.return_value.lock_user_for_session_change = AsyncMock()
            await self._call_handler(
                push_service=mock_push_service, token_revocation_service=revoked
            )

        mock_push_service.upsert_subscription.assert_not_awaited()

    async def test_direct_call_creates_when_recheck_finds_not_revoked(self) -> None:
        """Ordinary create with a valid, non-revoked token is unchanged:
        same response shape as before this fix."""
        mock_push_service = AsyncMock(spec=PushService)
        subscription_id = uuid4()
        mock_subscription = MagicMock()
        mock_subscription.id = subscription_id
        mock_subscription.endpoint = "https://push.example/abc"
        mock_subscription.created_at = datetime.now(UTC)
        mock_push_service.upsert_subscription.return_value = mock_subscription

        not_revoked = AsyncMock(spec=TokenRevocationService)
        not_revoked.is_revoked.return_value = False

        with patch("src.api.routes.push.UserRepository") as mock_repo_cls:
            mock_repo_cls.return_value.lock_user_for_session_change = AsyncMock()
            session, result = await self._call_handler(
                push_service=mock_push_service, token_revocation_service=not_revoked
            )

        assert result.id == subscription_id  # type: ignore[attr-defined]
        mock_push_service.upsert_subscription.assert_awaited_once()
        session.commit.assert_awaited_once()

    async def test_direct_call_fails_open_when_revocation_check_raises(self) -> None:
        """D3 — is_revoked itself never raises (it fails open internally),
        but this asserts the call site doesn't add its own fail-closed
        wrapper: if is_revoked somehow does raise, the exception must
        propagate as a 500 rather than being silently swallowed into a
        401 — a caught-and-rejected exception here would be an accidental
        fail-closed regression of the documented posture."""
        mock_push_service = AsyncMock(spec=PushService)
        broken = AsyncMock(spec=TokenRevocationService)
        broken.is_revoked.side_effect = RuntimeError("redis unreachable")

        with (
            patch("src.api.routes.push.UserRepository") as mock_repo_cls,
            pytest.raises(RuntimeError, match="redis unreachable"),
        ):
            mock_repo_cls.return_value.lock_user_for_session_change = AsyncMock()
            await self._call_handler(
                push_service=mock_push_service, token_revocation_service=broken
            )

        mock_push_service.upsert_subscription.assert_not_awaited()

    async def test_fail_open_end_to_end_when_redis_unavailable(
        self, jwt_service: JWTService, auth_header: dict[str, str]
    ) -> None:
        """D3, full stack: a real TokenRevocationService whose Redis calls
        fail must still let the create through — both auth_guard's own
        check and the handler's re-check independently fail open."""
        import redis.exceptions

        class _BrokenRedis:
            async def mget(self, _keys: list[str]) -> list[str | None]:
                raise redis.exceptions.ConnectionError("connection refused")

        mock_push_service = AsyncMock(spec=PushService)
        subscription_id = uuid4()
        mock_subscription = MagicMock()
        mock_subscription.id = subscription_id
        mock_subscription.endpoint = "https://push.example/abc"
        mock_subscription.created_at = datetime.now(UTC)
        mock_push_service.upsert_subscription.return_value = mock_subscription

        broken_token_revocation = TokenRevocationService(
            _BrokenRedis(),  # type: ignore[arg-type]
            max_token_ttl_seconds=3600,
        )

        app = _create_app(
            jwt_service,
            push_service=mock_push_service,
            token_revocation_service=broken_token_revocation,
        )
        with TestClient(app=app) as client:
            resp = client.post("/v1/push/subscriptions", json=self._body(), headers=auth_header)
            assert resp.status_code == HTTP_201_CREATED

        mock_push_service.upsert_subscription.assert_awaited_once()

    async def test_lock_acquired_before_revocation_recheck(self) -> None:
        """DoD — the lock must be acquired strictly before the re-check.
        Reversing the order would silently reintroduce the race this fix
        closes, while every other test here still passes — so ordering
        must be asserted explicitly, not just inferred from outcomes."""
        call_order: list[str] = []

        mock_push_service = AsyncMock(spec=PushService)
        subscription_id = uuid4()
        mock_subscription = MagicMock()
        mock_subscription.id = subscription_id
        mock_subscription.endpoint = "https://push.example/abc"
        mock_subscription.created_at = datetime.now(UTC)
        mock_push_service.upsert_subscription.return_value = mock_subscription

        token_revocation_service = AsyncMock(spec=TokenRevocationService)

        async def _record_is_revoked(*_args: object, **_kwargs: object) -> bool:
            call_order.append("is_revoked")
            return False

        token_revocation_service.is_revoked.side_effect = _record_is_revoked

        with patch("src.api.routes.push.UserRepository") as mock_repo_cls:

            async def _record_lock(*_args: object, **_kwargs: object) -> None:
                call_order.append("lock")

            mock_repo_cls.return_value.lock_user_for_session_change = AsyncMock(
                side_effect=_record_lock
            )
            await self._call_handler(
                push_service=mock_push_service, token_revocation_service=token_revocation_service
            )

        assert call_order == ["lock", "is_revoked"]


# ---------------------------------------------------------------------------
# DELETE /v1/push/subscriptions
# ---------------------------------------------------------------------------


class TestDeleteSubscription:
    def test_requires_auth(self, jwt_service: JWTService) -> None:
        app = _create_app(jwt_service, push_service_raises_503=True)
        with TestClient(app=app) as client:
            resp = client.request(
                "DELETE",
                "/v1/push/subscriptions",
                json={"endpoint": "https://push.example/abc"},
            )
            assert resp.status_code == HTTP_401_UNAUTHORIZED

    def test_returns_503_when_push_disabled(
        self, jwt_service: JWTService, auth_header: dict[str, str]
    ) -> None:
        app = _create_app(jwt_service, push_service_raises_503=True)
        with TestClient(app=app) as client:
            resp = client.request(
                "DELETE",
                "/v1/push/subscriptions",
                json={"endpoint": "https://push.example/abc"},
                headers=auth_header,
            )
            assert resp.status_code == HTTP_503_SERVICE_UNAVAILABLE

    def test_delete_is_idempotent_returns_204_even_if_not_found(
        self, jwt_service: JWTService, test_user_id: UUID, auth_header: dict[str, str]
    ) -> None:
        mock_push_service = AsyncMock(spec=PushService)
        mock_push_service.delete_subscription.return_value = None

        app = _create_app(jwt_service, push_service=mock_push_service)
        with TestClient(app=app) as client:
            resp = client.request(
                "DELETE",
                "/v1/push/subscriptions",
                json={"endpoint": "https://push.example/does-not-exist"},
                headers=auth_header,
            )
            assert resp.status_code == HTTP_204_NO_CONTENT

        mock_push_service.delete_subscription.assert_awaited_once_with(
            user_id=test_user_id,
            endpoint="https://push.example/does-not-exist",
            session=app.state["mock_session"],
        )
        app.state["mock_session"].commit.assert_awaited_once()
