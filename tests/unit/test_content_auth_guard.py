"""Unit tests for content_auth_guard."""

from __future__ import annotations

from datetime import timedelta
from typing import Any
from unittest.mock import MagicMock
from uuid import UUID, uuid4

import pytest
from litestar import Litestar, get
from litestar.di import Provide
from litestar.exceptions import NotAuthorizedException
from litestar.status_codes import HTTP_200_OK, HTTP_401_UNAUTHORIZED
from litestar.testing import TestClient

from src.api.dependencies.auth import get_current_user_id
from src.api.security import content_auth_guard
from src.api.security.guards import AuthenticatedUser
from src.api.security.jwt import JWTConfig, JWTService

TEST_SECRET = "test_secret_key_for_testing_only_256bits_long"
PRODUCT_ID = "vex"


@pytest.fixture
def jwt_config() -> JWTConfig:
    return JWTConfig(secret_key=TEST_SECRET)


@pytest.fixture
def jwt_service(jwt_config: JWTConfig) -> JWTService:
    return JWTService(jwt_config)


@pytest.fixture
def test_user_id() -> UUID:
    return uuid4()


def _make_app(jwt_service: JWTService, product_id: str = PRODUCT_ID) -> Litestar:
    """Minimal app wiring content_auth_guard on a test route."""

    @get("/content/test", guards=[content_auth_guard])
    async def content_route(current_user_id: UUID) -> dict[str, str]:
        return {"user_id": str(current_user_id)}

    app = Litestar(
        route_handlers=[content_route],
        dependencies={"current_user_id": Provide(get_current_user_id)},
    )
    app.state["jwt_service"] = jwt_service
    # Simulate ProductMiddleware setting product_id in state via middleware
    # For guard tests we inject it via a custom middleware-like fixture
    app.state["_test_product_id"] = product_id
    return app


# ---------------------------------------------------------------------------
# Integration tests via TestClient
# ---------------------------------------------------------------------------


class TestContentAuthGuardBearer:
    """content_auth_guard accepts a valid Bearer access token."""

    def test_valid_bearer_token_grants_access(
        self, jwt_service: JWTService, test_user_id: UUID
    ) -> None:
        app = _make_app(jwt_service)
        token, _ = jwt_service.create_access_token(test_user_id, product_id=PRODUCT_ID)
        with TestClient(app=app) as client:
            resp = client.get("/content/test", headers={"Authorization": f"Bearer {token}"})
        assert resp.status_code == HTTP_200_OK
        assert resp.json()["user_id"] == str(test_user_id)

    def test_expired_bearer_returns_401(self, jwt_service: JWTService) -> None:
        expired = JWTService(JWTConfig(secret_key=TEST_SECRET, access_token_expire_minutes=-1))
        token, _ = expired.create_access_token(uuid4(), product_id=PRODUCT_ID)
        app = _make_app(jwt_service)
        with TestClient(app=app) as client:
            resp = client.get("/content/test", headers={"Authorization": f"Bearer {token}"})
        assert resp.status_code == HTTP_401_UNAUTHORIZED

    def test_tampered_bearer_returns_401(self, jwt_service: JWTService, test_user_id: UUID) -> None:
        token, _ = jwt_service.create_access_token(test_user_id, product_id=PRODUCT_ID)
        app = _make_app(jwt_service)
        with TestClient(app=app) as client:
            resp = client.get(
                "/content/test",
                headers={"Authorization": f"Bearer {token[:-5]}XXXXX"},
            )
        assert resp.status_code == HTTP_401_UNAUTHORIZED


class TestContentAuthGuardCookie:
    """content_auth_guard accepts a valid content cookie with no Bearer header."""

    def test_valid_content_cookie_grants_access(
        self, jwt_service: JWTService, test_user_id: UUID
    ) -> None:
        token, _ = jwt_service.create_content_token(
            test_user_id, product_id=PRODUCT_ID, ttl=timedelta(hours=1)
        )
        app = _make_app(jwt_service)
        with TestClient(app=app) as client:
            resp = client.get("/content/test", cookies={"apex_content": token})
        assert resp.status_code == HTTP_200_OK
        assert resp.json()["user_id"] == str(test_user_id)

    def test_expired_content_cookie_returns_401(self, jwt_service: JWTService) -> None:
        expired = JWTService(JWTConfig(secret_key=TEST_SECRET))
        token, _ = expired.create_content_token(
            uuid4(), product_id=PRODUCT_ID, ttl=timedelta(seconds=-1)
        )
        app = _make_app(jwt_service)
        with TestClient(app=app) as client:
            resp = client.get("/content/test", cookies={"apex_content": token})
        assert resp.status_code == HTTP_401_UNAUTHORIZED

    def test_access_token_in_cookie_slot_rejected(
        self, jwt_service: JWTService, test_user_id: UUID
    ) -> None:
        """An access token must not authenticate via the content cookie slot."""
        access_token, _ = jwt_service.create_access_token(test_user_id, product_id=PRODUCT_ID)
        app = _make_app(jwt_service)
        with TestClient(app=app) as client:
            resp = client.get("/content/test", cookies={"apex_content": access_token})
        assert resp.status_code == HTTP_401_UNAUTHORIZED


class TestContentAuthGuardNoCredentials:
    def test_no_header_no_cookie_returns_401(self, jwt_service: JWTService) -> None:
        app = _make_app(jwt_service)
        with TestClient(app=app) as client:
            resp = client.get("/content/test")
        assert resp.status_code == HTTP_401_UNAUTHORIZED


class TestContentAuthGuardSetsState:
    """Guard sets connection.state exactly like auth_guard."""

    def test_user_id_set_in_state(self, jwt_service: JWTService, test_user_id: UUID) -> None:
        token, _ = jwt_service.create_content_token(
            test_user_id, product_id=PRODUCT_ID, ttl=timedelta(hours=1)
        )
        app = _make_app(jwt_service)
        with TestClient(app=app) as client:
            resp = client.get("/content/test", cookies={"apex_content": token})
        assert resp.status_code == HTTP_200_OK
        # current_user_id dependency reads from state["user_id"] — if it
        # returns the correct UUID, state was set correctly.
        assert UUID(resp.json()["user_id"]) == test_user_id


# ---------------------------------------------------------------------------
# Unit test: product mismatch is rejected
# ---------------------------------------------------------------------------


class TestContentAuthGuardProductCheck:
    """Token product_id must match the request product_id when both are present."""

    def test_cross_product_cookie_rejected(self, jwt_service: JWTService) -> None:
        """A content cookie for 'synthara' must not authenticate a 'vex' request."""
        token, _ = jwt_service.create_content_token(
            uuid4(), product_id="synthara", ttl=timedelta(hours=1)
        )

        state: dict[str, Any] = {"product_id": "vex"}
        mock_connection = MagicMock()
        mock_connection.headers.get.return_value = None  # no Authorization header
        mock_connection.cookies.get.return_value = token
        mock_connection.state = state
        mock_connection.app.state.get.return_value = jwt_service

        import asyncio

        with pytest.raises(NotAuthorizedException):
            asyncio.run(content_auth_guard(mock_connection, MagicMock()))

    def test_unscoped_token_rejected_on_product_request(self, jwt_service: JWTService) -> None:
        """Content cookie whose product_id is None is rejected when request has a product scope."""
        from unittest.mock import MagicMock, patch

        class DictState(dict):  # type: ignore[type-arg]
            pass

        uid = uuid4()
        token, _ = jwt_service.create_content_token(uid, product_id="vex", ttl=timedelta(hours=1))

        state = DictState({"product_id": "vex"})
        mock_connection = MagicMock()
        mock_connection.headers.get.return_value = None
        mock_connection.cookies.get.return_value = token
        mock_connection.state = state
        mock_connection.app.state.get.return_value = jwt_service

        # Patch decode_content_token to return a payload with product_id=None
        fake_payload = MagicMock()
        fake_payload.sub = str(uid)
        fake_payload.product_id = None

        import asyncio

        with (
            patch.object(jwt_service, "decode_content_token", return_value=fake_payload),
            pytest.raises(NotAuthorizedException, match="not scoped to the requested product"),
        ):
            asyncio.run(content_auth_guard(mock_connection, MagicMock()))

    def test_matching_product_cookie_accepted(self, jwt_service: JWTService) -> None:
        """A content cookie matching the request product_id is accepted."""
        uid = uuid4()
        token, _ = jwt_service.create_content_token(uid, product_id="vex", ttl=timedelta(hours=1))

        # Use a dict subclass so both .get() and [] assignment work.
        class DictState(dict):  # type: ignore[type-arg]
            pass

        state = DictState({"product_id": "vex"})
        mock_connection = MagicMock()
        mock_connection.headers.get.return_value = None
        mock_connection.cookies.get.return_value = token
        mock_connection.state = state
        mock_connection.app.state.get.return_value = jwt_service

        import asyncio

        asyncio.run(content_auth_guard(mock_connection, MagicMock()))
        assert state["user_id"] == uid
        assert isinstance(state["auth_user"], AuthenticatedUser)
