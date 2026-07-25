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
from litestar.exceptions import ServiceUnavailableException
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


def _create_app(
    jwt_service: JWTService,
    *,
    push_service: PushService | None = None,
    push_service_raises_503: bool = False,
) -> Litestar:
    mock_session = MagicMock(spec=AsyncSession)
    mock_session.commit = AsyncMock()

    def _provide_push_service() -> PushService:
        if push_service_raises_503:
            raise ServiceUnavailableException(detail="Push notifications not available")
        assert push_service is not None
        return push_service

    app = Litestar(
        route_handlers=[PushController],
        dependencies={
            "session": Provide(lambda: mock_session, sync_to_thread=False),
            "product_id": Provide(lambda: "vex", sync_to_thread=False),
            "push_service": Provide(_provide_push_service, sync_to_thread=False),
        },
    )
    app.state["jwt_service"] = jwt_service
    app.state["token_revocation"] = TokenRevocationService(None, max_token_ttl_seconds=0)
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
