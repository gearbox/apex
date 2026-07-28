"""Tests for route authorization — verifies auth guards are applied
and user_id is correctly extracted from JWT tokens.

Uses Litestar's test client to exercise the full request pipeline
including guards, dependency injection, and route handlers.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock
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

from src.api.dependencies.auth import get_current_token_payload, get_current_user_id
from src.api.security import JWTConfig, JWTService, auth_guard
from src.api.security.guards import AuthenticatedUser, extract_token_from_header
from src.api.services.token_revocation import TokenRevocationService

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

TEST_SECRET = "test_secret_key_for_testing_only_256bits_long"


def _no_op_token_revocation() -> TokenRevocationService:
    """A TokenRevocationService with no Redis client — never reports revoked.

    Used to wire app.state["token_revocation"] in guard tests that don't
    exercise revocation itself, mirroring pre-#142 behavior.
    """
    return TokenRevocationService(None, max_token_ttl_seconds=0)


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
# Unit Tests: get_current_token_payload dependency (R2, be-push-subscription-race-fix)
# ---------------------------------------------------------------------------


class TestGetCurrentTokenPayloadDependency:
    """Tests for the shared token_payload dependency function."""

    @pytest.mark.asyncio
    async def test_returns_payload_from_state(self, jwt_service: JWTService) -> None:
        from types import SimpleNamespace
        from unittest.mock import MagicMock

        token, _ = jwt_service.create_access_token(uuid4())
        payload = jwt_service.decode_access_token(token)
        state_dict = {"token_payload": payload}
        mock_request = MagicMock()
        mock_state = SimpleNamespace(**state_dict)
        mock_state.get = state_dict.get
        mock_request.state = mock_state

        result = await get_current_token_payload(mock_request)
        assert result is payload

    @pytest.mark.asyncio
    async def test_raises_when_no_token_payload(self) -> None:
        """Verify 401 raised when token_payload is missing from state —
        mirrors get_current_user_id's contract."""
        from types import SimpleNamespace
        from unittest.mock import MagicMock

        from litestar.exceptions import NotAuthorizedException

        empty_dict: dict[str, Any] = {}
        mock_request = MagicMock()
        mock_state = SimpleNamespace()
        mock_state.get = empty_dict.get
        mock_request.state = mock_state

        with pytest.raises(NotAuthorizedException):
            await get_current_token_payload(mock_request)


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
    app.state["token_revocation"] = _no_op_token_revocation()
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
        app.state["token_revocation"] = _no_op_token_revocation()
        app.state["mock_content_service"] = mock_content_service
        return app

    def test_upload_requires_auth(self, jwt_service: JWTService) -> None:
        app = self._create_storage_app(jwt_service)
        with TestClient(app=app) as client:
            resp = client.post("/v1/storage/upload")
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

    def test_health_live_is_public(self, jwt_service: JWTService) -> None:
        """HealthController /live should NOT require auth."""
        from src.api.routes.health import HealthController
        from src.api.services.health.registry import HealthCheckRegistry
        from src.api.services.health.service import HealthService

        health_service = HealthService(registry=HealthCheckRegistry())
        app = Litestar(
            route_handlers=[HealthController],
            dependencies={
                "health_service": Provide(lambda: health_service, sync_to_thread=False),
            },
        )
        app.state["jwt_service"] = jwt_service
        with TestClient(app=app) as client:
            resp = client.get("/health/live")
            assert resp.status_code == HTTP_200_OK


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
        source = self._read_source("src.api.routes.unified_generation")
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
        conn.state.__getitem__ = lambda self, k: state[k]  # noqa: ARG005
        conn.state.__setitem__ = lambda self, k, v: state.update({k: v})  # noqa: ARG005
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
            pytest.raises(NotAuthorizedException, match="Invalid or expired token"),
        ):
            await auth_guard(conn, MagicMock(spec=BaseRouteHandler))

    @pytest.mark.asyncio
    async def test_auth_guard_rejects_unscoped_token_on_product_request(
        self, jwt_service: JWTService, test_user_id: UUID
    ) -> None:
        """Token with product_id=None is rejected when the request has a product scope."""
        from unittest.mock import MagicMock, patch

        from litestar.exceptions import NotAuthorizedException
        from litestar.handlers import BaseRouteHandler

        from src.api.security.guards import auth_guard

        fake_payload = MagicMock()
        fake_payload.sub = str(test_user_id)
        fake_payload.product_id = None

        conn = self._make_connection(
            jwt_service=jwt_service,
            authorization="Bearer tok",
            state_product_id="vex",
        )
        with (
            patch.object(jwt_service, "decode_access_token", return_value=fake_payload),
            pytest.raises(NotAuthorizedException, match="not scoped to the requested product"),
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


class TestAuthGuardRevocation:
    """auth_guard's revocation integration (D5): ordering and state-on-reject."""

    def _make_connection(
        self,
        jwt_service: JWTService,
        token_revocation: Any,
        authorization: str,
        state_product_id: str | None = None,
    ) -> Any:
        from unittest.mock import MagicMock

        state: dict[str, Any] = {}
        if state_product_id is not None:
            state["product_id"] = state_product_id

        conn = MagicMock()
        conn.headers.get.return_value = authorization
        conn.state.__getitem__ = lambda self, k: state[k]  # noqa: ARG005
        conn.state.__setitem__ = lambda self, k, v: state.update({k: v})  # noqa: ARG005
        conn.state.get = state.get

        app_state = {"jwt_service": jwt_service, "token_revocation": token_revocation}
        conn.app.state.get = app_state.get
        return conn, state

    @pytest.mark.asyncio
    async def test_product_mismatch_short_circuits_before_revocation_check(
        self, jwt_service: JWTService
    ) -> None:
        """_enforce_product is a cheap local check — it must run before the
        Redis round-trip in is_revoked(), not after."""
        from litestar.exceptions import NotAuthorizedException
        from litestar.handlers import BaseRouteHandler

        from src.api.security.guards import auth_guard

        token, _ = jwt_service.create_access_token(uuid4(), product_id="vex")
        token_revocation = AsyncMock()
        conn, _state = self._make_connection(
            jwt_service,
            token_revocation,
            authorization=f"Bearer {token}",
            state_product_id="synthara",
        )

        with pytest.raises(NotAuthorizedException, match="different product"):
            await auth_guard(conn, MagicMock(spec=BaseRouteHandler))

        token_revocation.is_revoked.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_revoked_token_rejected_and_state_never_set(
        self, jwt_service: JWTService
    ) -> None:
        from litestar.exceptions import NotAuthorizedException
        from litestar.handlers import BaseRouteHandler

        from src.api.security.guards import auth_guard

        token, _ = jwt_service.create_access_token(uuid4(), product_id="vex")
        token_revocation = AsyncMock()
        token_revocation.is_revoked.return_value = True
        conn, state = self._make_connection(
            jwt_service,
            token_revocation,
            authorization=f"Bearer {token}",
            state_product_id="vex",
        )

        with pytest.raises(NotAuthorizedException, match="Invalid or expired token"):
            await auth_guard(conn, MagicMock(spec=BaseRouteHandler))

        token_revocation.is_revoked.assert_awaited_once()
        assert "user_id" not in state
        assert "auth_user" not in state
        assert "token_payload" not in state

    @pytest.mark.asyncio
    async def test_non_revoked_token_sets_state(self, jwt_service: JWTService) -> None:
        from litestar.handlers import BaseRouteHandler

        from src.api.security.guards import auth_guard

        uid = uuid4()
        token, _ = jwt_service.create_access_token(uid, product_id="vex")
        token_revocation = AsyncMock()
        token_revocation.is_revoked.return_value = False
        conn, state = self._make_connection(
            jwt_service,
            token_revocation,
            authorization=f"Bearer {token}",
            state_product_id="vex",
        )

        await auth_guard(conn, MagicMock(spec=BaseRouteHandler))

        assert state["user_id"] == uid
        token_revocation.is_revoked.assert_awaited_once()
        # R2 (be-push-subscription-race-fix) — the decoded TokenPayload must
        # also land in state so a handler can re-run is_revoked() itself.
        assert state["token_payload"].sub == str(uid)
        assert state["token_payload"] is token_revocation.is_revoked.call_args.args[0]

    @pytest.mark.asyncio
    async def test_missing_token_revocation_service_raises_runtime_error(
        self, jwt_service: JWTService
    ) -> None:
        """Mirrors the jwt_service-missing case — a misconfigured app.state,
        not a runtime condition to degrade on."""
        from litestar.handlers import BaseRouteHandler

        from src.api.security.guards import auth_guard

        token, _ = jwt_service.create_access_token(uuid4(), product_id="vex")
        conn, _state = self._make_connection(
            jwt_service,
            token_revocation=None,
            authorization=f"Bearer {token}",
            state_product_id="vex",
        )

        with pytest.raises(RuntimeError, match="Token revocation service not configured"):
            await auth_guard(conn, MagicMock(spec=BaseRouteHandler))


_UNSET = object()


class TestOptionalAuthGuard:
    """Tests for optional_auth_guard."""

    def _make_connection(
        self,
        jwt_service: JWTService | None = None,
        authorization: str | None = None,
        product_id: str | None = None,
        token_revocation: Any = _UNSET,
    ) -> Any:
        from unittest.mock import MagicMock

        state: dict[str, Any] = {}
        if product_id is not None:
            state["product_id"] = product_id
        conn = MagicMock()
        conn.headers.get.return_value = authorization
        conn.state.__setitem__ = lambda self, k, v: state.update({k: v})  # noqa: ARG005
        conn.state.__getitem__ = lambda self, k: state[k]  # noqa: ARG005
        conn.state.get = state.get

        # Default: a no-op revocation service, mirroring pre-#142 behavior
        # for tests that don't exercise revocation. Pass token_revocation=None
        # explicitly to simulate the service being entirely absent from
        # app.state (the A2 contract — must not raise).
        resolved_token_revocation = (
            _no_op_token_revocation() if token_revocation is _UNSET else token_revocation
        )
        app_state = {"jwt_service": jwt_service, "token_revocation": resolved_token_revocation}
        conn.app.state.get = app_state.get
        return conn

    @pytest.mark.asyncio
    async def test_sets_none_when_no_token(self) -> None:
        from litestar.handlers import BaseRouteHandler

        from src.api.security.guards import optional_auth_guard

        conn = self._make_connection()

        await optional_auth_guard(conn, MagicMock(spec=BaseRouteHandler))

        assert conn.state.get("user_id") is None
        assert conn.state.get("auth_user") is None
        assert conn.state.get("token_payload") is None

    @pytest.mark.asyncio
    async def test_sets_none_when_jwt_service_missing(self) -> None:
        from litestar.handlers import BaseRouteHandler

        from src.api.security.guards import optional_auth_guard

        conn = self._make_connection(jwt_service=None, authorization="Bearer sometoken")

        await optional_auth_guard(conn, MagicMock(spec=BaseRouteHandler))

        assert conn.state.get("user_id") is None
        assert conn.state.get("auth_user") is None

    @pytest.mark.asyncio
    async def test_sets_user_id_when_valid_token(self, jwt_service: JWTService) -> None:
        from litestar.handlers import BaseRouteHandler

        from src.api.security.guards import optional_auth_guard

        uid = uuid4()
        token, _ = jwt_service.create_access_token(uid)
        conn = self._make_connection(jwt_service=jwt_service, authorization=f"Bearer {token}")

        await optional_auth_guard(conn, MagicMock(spec=BaseRouteHandler))

        assert conn.state.get("user_id") == uid
        assert isinstance(conn.state.get("auth_user"), AuthenticatedUser)
        assert conn.state.get("auth_user").user_id == uid
        # R2 (be-push-subscription-race-fix) — token_payload mirrors auth_guard.
        assert conn.state.get("token_payload").sub == str(uid)

    @pytest.mark.asyncio
    async def test_sets_none_when_invalid_token(self, jwt_service: JWTService) -> None:
        from litestar.handlers import BaseRouteHandler

        from src.api.security.guards import optional_auth_guard

        conn = self._make_connection(jwt_service=jwt_service, authorization="Bearer bad.token.here")

        await optional_auth_guard(conn, MagicMock(spec=BaseRouteHandler))

        assert conn.state.get("user_id") is None
        assert conn.state.get("auth_user") is None

    @pytest.mark.asyncio
    async def test_sets_user_id_when_token_product_matches_request(
        self, jwt_service: JWTService
    ) -> None:
        """A token scoped to the same product as the request authenticates normally."""
        from litestar.handlers import BaseRouteHandler

        from src.api.security.guards import optional_auth_guard

        uid = uuid4()
        token, _ = jwt_service.create_access_token(uid, product_id="vex")
        conn = self._make_connection(
            jwt_service=jwt_service, authorization=f"Bearer {token}", product_id="vex"
        )

        await optional_auth_guard(conn, MagicMock(spec=BaseRouteHandler))

        assert conn.state.get("user_id") == uid
        assert conn.state.get("auth_user").user_id == uid

    @pytest.mark.asyncio
    async def test_degrades_to_anonymous_when_token_product_mismatches_request(
        self, jwt_service: JWTService
    ) -> None:
        """A token scoped to a *different* product than the request degrades to
        anonymous (no user_id set) rather than raising 401 — unlike auth_guard,
        an optional guard must not turn a page that doesn't require
        authentication into one that 401s for a stale other-product token."""
        from litestar.handlers import BaseRouteHandler

        from src.api.security.guards import optional_auth_guard

        uid = uuid4()
        token, _ = jwt_service.create_access_token(uid, product_id="synthara")
        conn = self._make_connection(
            jwt_service=jwt_service, authorization=f"Bearer {token}", product_id="vex"
        )

        await optional_auth_guard(conn, MagicMock(spec=BaseRouteHandler))

        assert conn.state.get("user_id") is None
        assert conn.state.get("auth_user") is None

    @pytest.mark.asyncio
    async def test_degrades_to_anonymous_when_token_has_no_product_claim(
        self, jwt_service: JWTService
    ) -> None:
        """A token with no product_id claim at all (malformed/legacy) also
        degrades to anonymous on a product-scoped request."""
        from litestar.handlers import BaseRouteHandler

        from src.api.security.guards import optional_auth_guard

        uid = uuid4()
        token, _ = jwt_service.create_access_token(uid)  # no product_id
        conn = self._make_connection(
            jwt_service=jwt_service, authorization=f"Bearer {token}", product_id="vex"
        )

        await optional_auth_guard(conn, MagicMock(spec=BaseRouteHandler))

        assert conn.state.get("user_id") is None
        assert conn.state.get("auth_user") is None

    @pytest.mark.asyncio
    async def test_degrades_to_anonymous_when_token_is_revoked(
        self, jwt_service: JWTService
    ) -> None:
        """R3 (issue #142) — a revoked token (logout-all, password
        change/reset, deactivation, reuse detection) must degrade to
        anonymous on this guard too, exactly like a product mismatch does.
        Previously only auth_guard/content_auth_guard checked revocation,
        so a revoked token still authenticated on GET /v1/providers."""
        from litestar.handlers import BaseRouteHandler

        from src.api.security.guards import optional_auth_guard

        uid = uuid4()
        token, _ = jwt_service.create_access_token(uid, product_id="vex")
        token_revocation = AsyncMock()
        token_revocation.is_revoked.return_value = True
        conn = self._make_connection(
            jwt_service=jwt_service,
            authorization=f"Bearer {token}",
            product_id="vex",
            token_revocation=token_revocation,
        )

        await optional_auth_guard(conn, MagicMock(spec=BaseRouteHandler))

        assert conn.state.get("user_id") is None
        assert conn.state.get("auth_user") is None
        token_revocation.is_revoked.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_sets_user_id_when_token_not_revoked(self, jwt_service: JWTService) -> None:
        """A non-revoked, product-matching token still authenticates normally."""
        from litestar.handlers import BaseRouteHandler

        from src.api.security.guards import optional_auth_guard

        uid = uuid4()
        token, _ = jwt_service.create_access_token(uid, product_id="vex")
        token_revocation = AsyncMock()
        token_revocation.is_revoked.return_value = False
        conn = self._make_connection(
            jwt_service=jwt_service,
            authorization=f"Bearer {token}",
            product_id="vex",
            token_revocation=token_revocation,
        )

        await optional_auth_guard(conn, MagicMock(spec=BaseRouteHandler))

        assert conn.state.get("user_id") == uid
        token_revocation.is_revoked.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_succeeds_when_token_revocation_service_absent(
        self, jwt_service: JWTService
    ) -> None:
        """A2 contract — unlike auth_guard's _get_token_revocation(), a
        missing token_revocation in app.state must NOT raise here. It
        degrades the revocation check to a no-op, so a valid token still
        authenticates normally rather than 500ing an anonymous-capable
        route."""
        from litestar.handlers import BaseRouteHandler

        from src.api.security.guards import optional_auth_guard

        uid = uuid4()
        token, _ = jwt_service.create_access_token(uid, product_id="vex")
        conn = self._make_connection(
            jwt_service=jwt_service,
            authorization=f"Bearer {token}",
            product_id="vex",
            token_revocation=None,
        )

        await optional_auth_guard(conn, MagicMock(spec=BaseRouteHandler))

        assert conn.state.get("user_id") == uid
        assert conn.state.get("auth_user").user_id == uid
