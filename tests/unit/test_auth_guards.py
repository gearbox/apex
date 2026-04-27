"""Tests for route authorization — verifies auth guards are applied
and user_id is correctly extracted from JWT tokens.

Uses Litestar's test client to exercise the full request pipeline
including guards, dependency injection, and route handlers.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock
from uuid import UUID, uuid4

import pytest
from litestar import Litestar, get, post
from litestar.di import Provide
from litestar.status_codes import (
    HTTP_200_OK,
    HTTP_201_CREATED,
    HTTP_401_UNAUTHORIZED,
)
from litestar.testing import TestClient

from src.api.dependencies.auth import get_current_user_id
from src.api.security import JWTConfig, JWTService, auth_guard
from src.api.security.guards import AuthenticatedUser, extract_token_from_header

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

TEST_SECRET = "test_secret_key_for_testing_only_256bits_long"


@pytest.fixture
def jwt_config() -> JWTConfig:
    return JWTConfig(secret_key=TEST_SECRET)


@pytest.fixture
def jwt_service(jwt_config: JWTConfig) -> JWTService:
    return JWTService(jwt_config)


@pytest.fixture
def test_user_id() -> UUID:
    return uuid4()


@pytest.fixture
def auth_header(jwt_service: JWTService, test_user_id: UUID) -> dict[str, str]:
    """Generate a valid Authorization header for the test user."""
    token, _ = jwt_service.create_access_token(test_user_id)
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def expired_auth_header() -> dict[str, str]:
    """Generate an expired token header."""
    expired_config = JWTConfig(secret_key=TEST_SECRET, access_token_expire_minutes=-1)
    expired_service = JWTService(expired_config)
    token, _ = expired_service.create_access_token(uuid4())
    return {"Authorization": f"Bearer {token}"}


# ---------------------------------------------------------------------------
# Unit Tests: extract_token_from_header
# ---------------------------------------------------------------------------


class TestExtractTokenFromHeader:
    """Tests for the token extraction utility."""

    def test_valid_bearer_token(self) -> None:
        assert extract_token_from_header("Bearer abc123") == "abc123"

    def test_case_insensitive_bearer(self) -> None:
        assert extract_token_from_header("bearer abc123") == "abc123"

    def test_none_header(self) -> None:
        assert extract_token_from_header(None) is None

    def test_empty_header(self) -> None:
        assert extract_token_from_header("") is None

    def test_missing_token(self) -> None:
        assert extract_token_from_header("Bearer") is None

    def test_wrong_scheme(self) -> None:
        assert extract_token_from_header("Basic abc123") is None

    def test_too_many_parts(self) -> None:
        assert extract_token_from_header("Bearer abc 123") is None


# ---------------------------------------------------------------------------
# Unit Tests: AuthenticatedUser
# ---------------------------------------------------------------------------


class TestAuthenticatedUser:
    """Tests for AuthenticatedUser value object."""

    def test_user_id(self) -> None:
        uid = uuid4()
        auth_user = AuthenticatedUser(user_id=uid)
        assert auth_user.user_id == uid

    def test_user_not_loaded_raises(self) -> None:
        auth_user = AuthenticatedUser(user_id=uuid4())
        with pytest.raises(RuntimeError, match="User not loaded"):
            _ = auth_user.user

    def test_repr(self) -> None:
        uid = uuid4()
        auth_user = AuthenticatedUser(user_id=uid)
        assert str(uid) in repr(auth_user)


# ---------------------------------------------------------------------------
# Unit Tests: get_current_user_id dependency
# ---------------------------------------------------------------------------


class TestGetCurrentUserIdDependency:
    """Tests for the shared auth dependency function."""

    @pytest.mark.asyncio
    async def test_returns_user_id_from_state(self) -> None:
        """Verify get_current_user_id returns user_id from request state."""
        from types import SimpleNamespace
        from unittest.mock import MagicMock

        uid = uuid4()
        state_dict = {"user_id": uid}
        mock_request = MagicMock()
        # Litestar's State has a .get() method; simulate with SimpleNamespace
        mock_state = SimpleNamespace(**state_dict)
        mock_state.get = state_dict.get
        mock_request.state = mock_state

        result = await get_current_user_id(mock_request)
        assert result == uid

    @pytest.mark.asyncio
    async def test_raises_when_no_user_id(self) -> None:
        """Verify 401 raised when user_id is missing from state."""
        from types import SimpleNamespace
        from unittest.mock import MagicMock

        from litestar.exceptions import NotAuthorizedException

        empty_dict: dict[str, Any] = {}
        mock_request = MagicMock()
        mock_state = SimpleNamespace()
        mock_state.get = empty_dict.get
        mock_request.state = mock_state

        with pytest.raises(NotAuthorizedException):
            await get_current_user_id(mock_request)


# ---------------------------------------------------------------------------
# Integration Tests: Auth Guard with Litestar TestClient
# ---------------------------------------------------------------------------


def _create_test_app(jwt_service: JWTService) -> Litestar:
    """Create a minimal Litestar app with a guarded route for testing."""

    @get("/public")
    async def public_route() -> dict[str, str]:
        return {"status": "ok"}

    @get("/protected", guards=[auth_guard])
    async def protected_route(current_user_id: UUID) -> dict[str, str]:
        return {"user_id": str(current_user_id)}

    @post("/protected/action", guards=[auth_guard])
    async def protected_action(current_user_id: UUID) -> dict[str, str]:
        return {"user_id": str(current_user_id), "action": "done"}

    app = Litestar(
        route_handlers=[public_route, protected_route, protected_action],
        dependencies={"current_user_id": Provide(get_current_user_id)},
    )
    app.state["jwt_service"] = jwt_service
    return app


class TestAuthGuardIntegration:
    """Integration tests: auth guard works end-to-end through Litestar."""

    def test_public_route_no_auth_required(self, jwt_service: JWTService) -> None:
        app = _create_test_app(jwt_service)
        with TestClient(app=app) as client:
            resp = client.get("/public")
            assert resp.status_code == HTTP_200_OK
            assert resp.json() == {"status": "ok"}

    def test_protected_route_returns_401_without_token(self, jwt_service: JWTService) -> None:
        app = _create_test_app(jwt_service)
        with TestClient(app=app) as client:
            resp = client.get("/protected")
            assert resp.status_code == HTTP_401_UNAUTHORIZED

    def test_protected_route_returns_401_with_invalid_token(self, jwt_service: JWTService) -> None:
        app = _create_test_app(jwt_service)
        with TestClient(app=app) as client:
            resp = client.get("/protected", headers={"Authorization": "Bearer invalid.token"})
            assert resp.status_code == HTTP_401_UNAUTHORIZED

    def test_protected_route_returns_401_with_expired_token(
        self,
        jwt_service: JWTService,
        expired_auth_header: dict[str, str],
    ) -> None:
        app = _create_test_app(jwt_service)
        with TestClient(app=app) as client:
            resp = client.get("/protected", headers=expired_auth_header)
            assert resp.status_code == HTTP_401_UNAUTHORIZED

    def test_protected_route_returns_401_with_wrong_secret(self, jwt_service: JWTService) -> None:
        """Token signed with different secret is rejected."""
        other_service = JWTService(JWTConfig(secret_key="different_secret_for_wrong_key_test_32b"))
        token, _ = other_service.create_access_token(uuid4())

        app = _create_test_app(jwt_service)
        with TestClient(app=app) as client:
            resp = client.get("/protected", headers={"Authorization": f"Bearer {token}"})
            assert resp.status_code == HTTP_401_UNAUTHORIZED

    def test_protected_route_succeeds_with_valid_token(
        self,
        jwt_service: JWTService,
        test_user_id: UUID,
        auth_header: dict[str, str],
    ) -> None:
        app = _create_test_app(jwt_service)
        with TestClient(app=app) as client:
            resp = client.get("/protected", headers=auth_header)
            assert resp.status_code == HTTP_200_OK
            assert resp.json()["user_id"] == str(test_user_id)

    def test_post_protected_route_succeeds_with_valid_token(
        self,
        jwt_service: JWTService,
        test_user_id: UUID,
        auth_header: dict[str, str],
    ) -> None:
        app = _create_test_app(jwt_service)
        with TestClient(app=app) as client:
            resp = client.post("/protected/action", headers=auth_header)
            assert resp.status_code == HTTP_201_CREATED
            body = resp.json()
            assert body["user_id"] == str(test_user_id)
            assert body["action"] == "done"

    def test_user_id_matches_token_subject(self, jwt_service: JWTService) -> None:
        """Verify the extracted user_id matches the token's subject claim."""
        uid = uuid4()
        token, _ = jwt_service.create_access_token(uid)

        app = _create_test_app(jwt_service)
        with TestClient(app=app) as client:
            resp = client.get("/protected", headers={"Authorization": f"Bearer {token}"})
            assert resp.status_code == HTTP_200_OK
            assert resp.json()["user_id"] == str(uid)

    def test_different_users_get_different_ids(self, jwt_service: JWTService) -> None:
        user_a = uuid4()
        user_b = uuid4()
        token_a, _ = jwt_service.create_access_token(user_a)
        token_b, _ = jwt_service.create_access_token(user_b)

        app = _create_test_app(jwt_service)
        with TestClient(app=app) as client:
            resp_a = client.get("/protected", headers={"Authorization": f"Bearer {token_a}"})
            resp_b = client.get("/protected", headers={"Authorization": f"Bearer {token_b}"})

            assert resp_a.json()["user_id"] == str(user_a)
            assert resp_b.json()["user_id"] == str(user_b)
            assert resp_a.json()["user_id"] != resp_b.json()["user_id"]


# ---------------------------------------------------------------------------
# Controller-level tests: verify guards are applied on real controllers
# ---------------------------------------------------------------------------


class TestStorageControllerAuth:
    """Verify auth guards are properly applied to StorageController."""

    def _create_storage_app(self, jwt_service: JWTService) -> Litestar:
        from unittest.mock import AsyncMock

        from src.api.routes.storage import StorageController
        from src.api.services.user_content import UserContentService

        mock_content_service = AsyncMock(spec=UserContentService)
        mock_content_service.list_user_uploads.return_value = []
        mock_content_service.list_user_outputs.return_value = []
        mock_content_service.get_user_stats.return_value = {
            "upload_count": 0,
            "output_count": 0,
            "total_bytes": 0,
        }

        app = Litestar(
            route_handlers=[StorageController],
            dependencies={
                "user_content": Provide(lambda: mock_content_service, sync_to_thread=False),
            },
        )
        app.state["jwt_service"] = jwt_service
        app.state["mock_content_service"] = mock_content_service
        return app

    def test_upload_requires_auth(self, jwt_service: JWTService) -> None:
        app = self._create_storage_app(jwt_service)
        with TestClient(app=app) as client:
            resp = client.post("/v1/storage/upload")
            assert resp.status_code == HTTP_401_UNAUTHORIZED

    def test_list_uploads_requires_auth(self, jwt_service: JWTService) -> None:
        app = self._create_storage_app(jwt_service)
        with TestClient(app=app) as client:
            resp = client.get("/v1/storage/uploads")
            assert resp.status_code == HTTP_401_UNAUTHORIZED

    def test_list_outputs_requires_auth(self, jwt_service: JWTService) -> None:
        app = self._create_storage_app(jwt_service)
        with TestClient(app=app) as client:
            resp = client.get("/v1/storage/outputs")
            assert resp.status_code == HTTP_401_UNAUTHORIZED

    def test_stats_requires_auth(self, jwt_service: JWTService) -> None:
        app = self._create_storage_app(jwt_service)
        with TestClient(app=app) as client:
            resp = client.get("/v1/storage/stats")
            assert resp.status_code == HTTP_401_UNAUTHORIZED

    def test_get_upload_access_requires_auth(self, jwt_service: JWTService) -> None:
        app = self._create_storage_app(jwt_service)
        with TestClient(app=app) as client:
            resp = client.get(f"/v1/storage/uploads/{uuid4()}")
            assert resp.status_code == HTTP_401_UNAUTHORIZED

    def test_list_uploads_with_auth_returns_data(
        self,
        jwt_service: JWTService,
        test_user_id: UUID,
        auth_header: dict[str, str],
    ) -> None:
        app = self._create_storage_app(jwt_service)
        with TestClient(app=app) as client:
            resp = client.get("/v1/storage/uploads", headers=auth_header)
            assert resp.status_code == HTTP_200_OK
            body = resp.json()
            assert body["has_more"] is False
            assert body["items"] == []

        # Verify the service was called with the authenticated user's ID
        mock = app.state["mock_content_service"]
        mock.list_user_uploads.assert_called_once_with(
            test_user_id, limit=50, cursor_ts=None, cursor_id=None
        )

    def test_stats_with_auth_uses_authenticated_user_id(
        self,
        jwt_service: JWTService,
        test_user_id: UUID,
        auth_header: dict[str, str],
    ) -> None:
        app = self._create_storage_app(jwt_service)
        with TestClient(app=app) as client:
            resp = client.get("/v1/storage/stats", headers=auth_header)
            assert resp.status_code == HTTP_200_OK

        mock = app.state["mock_content_service"]
        mock.get_user_stats.assert_called_once_with(test_user_id)

    def test_storage_no_longer_accepts_user_id_query_param(
        self,
        jwt_service: JWTService,
        auth_header: dict[str, str],
    ) -> None:
        """Verify the old ?user_id=... query parameter is not accepted."""
        app = self._create_storage_app(jwt_service)
        other_user = uuid4()
        with TestClient(app=app) as client:
            resp = client.get(
                f"/v1/storage/stats?user_id={other_user}",
                headers=auth_header,
            )
            # Should succeed (extra query param ignored) but use auth user_id, not query
            assert resp.status_code == HTTP_200_OK

        mock = app.state["mock_content_service"]
        # The call should use the authenticated user, not the query param
        call_args = mock.get_user_stats.call_args
        assert call_args[0][0] != other_user


class TestGenerationControllersAuth:
    """Verify auth guards on generation controllers."""

    def _create_gen_app(self, jwt_service: JWTService) -> Litestar:
        from src.api.routes.generation import ImageController
        from src.api.routes.health import HealthController
        from src.api.services.health.registry import HealthCheckRegistry
        from src.api.services.health.service import HealthService

        health_service = HealthService(registry=HealthCheckRegistry())

        app = Litestar(
            route_handlers=[
                HealthController,
                ImageController,
            ],
            dependencies={
                "health_service": Provide(lambda: health_service, sync_to_thread=False),
            },
        )
        app.state["jwt_service"] = jwt_service
        return app

    def test_health_live_is_public(self, jwt_service: JWTService) -> None:
        """HealthController /live should NOT require auth."""
        app = self._create_gen_app(jwt_service)
        with TestClient(app=app) as client:
            resp = client.get("/health/live")
            assert resp.status_code == HTTP_200_OK

    def test_image_upload_requires_auth(self, jwt_service: JWTService) -> None:
        app = self._create_gen_app(jwt_service)
        with TestClient(app=app) as client:
            resp = client.post("/v1/images/upload")
            assert resp.status_code == HTTP_401_UNAUTHORIZED


# ---------------------------------------------------------------------------
# Test: No hardcoded placeholder user_id in source
# ---------------------------------------------------------------------------


class TestNoPlaceholderUserIds:
    """Verify that the hardcoded placeholder UUID has been removed from routes."""

    PLACEHOLDER = "00000000-0000-0000-0000-000000000001"

    def _read_source(self, module_path: str) -> str:
        import importlib
        import inspect

        module = importlib.import_module(module_path)
        return inspect.getsource(module)

    def test_storage_routes_no_placeholder(self) -> None:
        source = self._read_source("src.api.routes.storage")
        assert f'user_id = UUID("{self.PLACEHOLDER}")' not in source

    def test_generation_routes_no_placeholder(self) -> None:
        source = self._read_source("src.api.routes.generation")
        assert f'user_id = UUID("{self.PLACEHOLDER}")' not in source


# ---------------------------------------------------------------------------
# Test: Verify controller class-level guard declarations
# ---------------------------------------------------------------------------


class TestControllerGuardDeclarations:
    """Verify guards are declared at the controller class level."""

    def test_storage_controller_has_guard(self) -> None:
        from src.api.routes.storage import StorageController

        guards = StorageController.guards
        assert guards is not None
        assert auth_guard in guards

    def test_image_controller_has_guard(self) -> None:
        from src.api.routes.generation import ImageController

        guards = ImageController.guards
        assert guards is not None
        assert auth_guard in guards

    def test_health_controller_has_no_guard(self) -> None:
        from src.api.routes.health import HealthController

        guards = HealthController.__dict__.get("guards")
        assert guards is None or (isinstance(guards, list) and auth_guard not in guards)


# ---------------------------------------------------------------------------
# Test: UnifiedJobController auth
# ---------------------------------------------------------------------------


class TestUnifiedJobControllerAuth:
    """Verify auth guards are applied to UnifiedJobController."""

    def _create_unified_jobs_app(self, jwt_service: JWTService) -> Litestar:
        from unittest.mock import AsyncMock, MagicMock

        from sqlalchemy.ext.asyncio import AsyncSession

        from src.api.routes.jobs import UnifiedJobController
        from src.api.services.unified_jobs import UnifiedJobService

        mock_session = MagicMock(spec=AsyncSession)
        mock_session.commit = AsyncMock()
        mock_unified_job_service = MagicMock(spec=UnifiedJobService)

        app = Litestar(
            route_handlers=[UnifiedJobController],
            dependencies={
                "session": Provide(lambda: mock_session, sync_to_thread=False),
                "unified_job_service": Provide(
                    lambda: mock_unified_job_service, sync_to_thread=False
                ),
            },
        )
        app.state["jwt_service"] = jwt_service
        return app

    def test_list_jobs_requires_auth(self, jwt_service: JWTService) -> None:
        app = self._create_unified_jobs_app(jwt_service)
        with TestClient(app=app) as client:
            resp = client.get("/v1/jobs/")
            assert resp.status_code == HTTP_401_UNAUTHORIZED

    def test_get_job_requires_auth(self, jwt_service: JWTService) -> None:
        app = self._create_unified_jobs_app(jwt_service)
        with TestClient(app=app) as client:
            resp = client.get(f"/v1/jobs/{uuid4()}")
            assert resp.status_code == HTTP_401_UNAUTHORIZED


# ---------------------------------------------------------------------------
# Unit tests for uncovered guard branches
# ---------------------------------------------------------------------------


class TestAuthGuardUncoveredBranches:
    """Cover remaining branches of auth_guard and optional_auth_guard."""

    def _make_connection(
        self,
        jwt_service: JWTService | None = None,
        authorization: str | None = None,
        state_product_id: str | None = None,
    ) -> Any:
        from unittest.mock import MagicMock

        state: dict[str, Any] = {}
        if state_product_id is not None:
            state["product_id"] = state_product_id

        conn = MagicMock()
        conn.headers.get.return_value = authorization
        conn.state.__getitem__ = lambda self, k: state[k]
        conn.state.__setitem__ = lambda self, k, v: state.update({k: v})
        conn.state.get = state.get

        conn.app.state.get.return_value = None if jwt_service is None else jwt_service
        return conn

    @pytest.mark.asyncio
    async def test_authenticated_user_user_property_returns_user(self) -> None:
        from unittest.mock import MagicMock

        from src.api.security.guards import AuthenticatedUser

        mock_user = MagicMock()
        auth_user = AuthenticatedUser(user_id=uuid4(), user=mock_user)
        assert auth_user.user is mock_user

    @pytest.mark.asyncio
    async def test_auth_guard_raises_runtime_when_jwt_service_missing(self) -> None:
        from litestar.handlers import BaseRouteHandler

        from src.api.security.guards import auth_guard

        conn = self._make_connection(jwt_service=None, authorization="Bearer sometoken")

        with pytest.raises(RuntimeError, match="JWT service not configured"):
            await auth_guard(conn, MagicMock(spec=BaseRouteHandler))

    @pytest.mark.asyncio
    async def test_auth_guard_raises_on_invalid_uuid_in_sub(self, jwt_service: JWTService) -> None:
        from unittest.mock import MagicMock, patch

        from litestar.exceptions import NotAuthorizedException
        from litestar.handlers import BaseRouteHandler

        from src.api.security.guards import auth_guard

        fake_payload = MagicMock()
        fake_payload.sub = "not-a-uuid"
        fake_payload.product_id = None

        conn = self._make_connection(jwt_service=jwt_service, authorization="Bearer tok")
        with (
            patch.object(jwt_service, "decode_access_token", return_value=fake_payload),
            pytest.raises(NotAuthorizedException, match="Invalid token subject"),
        ):
            await auth_guard(conn, MagicMock(spec=BaseRouteHandler))

    @pytest.mark.asyncio
    async def test_auth_guard_raises_on_product_mismatch(
        self, jwt_service: JWTService, test_user_id: UUID
    ) -> None:
        from litestar.exceptions import NotAuthorizedException
        from litestar.handlers import BaseRouteHandler

        from src.api.security.guards import auth_guard

        token, _ = jwt_service.create_access_token(test_user_id, product_id="vex")
        conn = self._make_connection(
            jwt_service=jwt_service,
            authorization=f"Bearer {token}",
            state_product_id="synthara",
        )
        with pytest.raises(NotAuthorizedException, match="different product"):
            await auth_guard(conn, MagicMock(spec=BaseRouteHandler))


class TestOptionalAuthGuard:
    """Tests for optional_auth_guard."""

    def _make_connection(
        self,
        jwt_service: JWTService | None = None,
        authorization: str | None = None,
    ) -> Any:
        from unittest.mock import MagicMock

        state: dict[str, Any] = {}
        conn = MagicMock()
        conn.headers.get.return_value = authorization
        conn.state.__setitem__ = lambda self, k, v: state.update({k: v})
        conn.state.__getitem__ = lambda self, k: state[k]
        conn.state.get = state.get

        conn.app.state.get.return_value = jwt_service
        return conn

    @pytest.mark.asyncio
    async def test_sets_none_when_no_token(self) -> None:
        from unittest.mock import MagicMock

        from litestar.handlers import BaseRouteHandler

        from src.api.security.guards import optional_auth_guard

        state: dict[str, Any] = {}
        conn = MagicMock()
        conn.headers.get.return_value = None
        conn.state.__setitem__ = lambda self, k, v: state.update({k: v})

        await optional_auth_guard(conn, MagicMock(spec=BaseRouteHandler))

        assert state["user_id"] is None
        assert state["auth_user"] is None

    @pytest.mark.asyncio
    async def test_sets_none_when_jwt_service_missing(self) -> None:
        from unittest.mock import MagicMock

        from litestar.handlers import BaseRouteHandler

        from src.api.security.guards import optional_auth_guard

        state: dict[str, Any] = {}
        conn = MagicMock()
        conn.headers.get.return_value = "Bearer sometoken"
        conn.state.__setitem__ = lambda self, k, v: state.update({k: v})
        conn.app.state.get.return_value = None

        await optional_auth_guard(conn, MagicMock(spec=BaseRouteHandler))

        assert state["user_id"] is None
        assert state["auth_user"] is None

    @pytest.mark.asyncio
    async def test_sets_user_id_when_valid_token(self, jwt_service: JWTService) -> None:
        from unittest.mock import MagicMock

        from litestar.handlers import BaseRouteHandler

        from src.api.security.guards import optional_auth_guard

        uid = uuid4()
        token, _ = jwt_service.create_access_token(uid)
        state: dict[str, Any] = {}
        conn = MagicMock()
        conn.headers.get.return_value = f"Bearer {token}"
        conn.state.__setitem__ = lambda self, k, v: state.update({k: v})
        conn.app.state.get.return_value = jwt_service

        await optional_auth_guard(conn, MagicMock(spec=BaseRouteHandler))

        assert state["user_id"] == uid
        assert isinstance(state["auth_user"], AuthenticatedUser)
        assert state["auth_user"].user_id == uid

    @pytest.mark.asyncio
    async def test_sets_none_when_invalid_token(self, jwt_service: JWTService) -> None:
        from unittest.mock import MagicMock

        from litestar.handlers import BaseRouteHandler

        from src.api.security.guards import optional_auth_guard

        state: dict[str, Any] = {}
        conn = MagicMock()
        conn.headers.get.return_value = "Bearer bad.token.here"
        conn.state.__setitem__ = lambda self, k, v: state.update({k: v})
        conn.app.state.get.return_value = jwt_service

        await optional_auth_guard(conn, MagicMock(spec=BaseRouteHandler))

        assert state["user_id"] is None
        assert state["auth_user"] is None
