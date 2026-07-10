"""Real-request smoke tests for controllers otherwise only tested via `.fn()`.

Companion to `tests/unit/test_route_handlers.py` (which documents "Tests call
Handler.fn(self, ...) directly to exercise handler logic") and
`tests/unit/test_unified_generation_endpoint.py` (same pattern, `MagicMock` in
place of a real `ProductConfig`). Calling `Handler.fn(...)` invokes the plain
Python function directly — it never goes through Litestar's per-request
signature model, so it can't catch a `ProductConfig`-shaped bug like the one
in commit 2873e8e (see `tests/unit/api/test_product_config_signature_resolution.py`
for the full mechanism writeup): a dataclass/struct field whose type only
resolves under `TYPE_CHECKING` raises `NameError` inside `msgspec.convert()`
the first time Litestar actually converts a real value into that shape, which
only happens on an actual request.

These tests register the *real* controller class into a minimal `Litestar`
app (mocking only the service/session dependencies, not the framework
plumbing) and drive it through `TestClient`/`AsyncTestClient` with a real JWT
so `auth_guard`, the DI graph, the signature model, and response
serialization all run for real — same pattern as
`TestStorageControllerAuth`/`TestUnifiedJobControllerAuth` in
`tests/unit/test_auth_guards.py`, extended to `UserController` and
`UnifiedGenerationController`, which previously had no real-request coverage
at all, plus a genuine (not `MagicMock`) `ProductConfig` on the generation
endpoint specifically because that's the exact shape that broke in
production.
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID, uuid4

import pytest
from litestar import Litestar
from litestar.di import Provide
from litestar.status_codes import HTTP_200_OK, HTTP_201_CREATED, HTTP_401_UNAUTHORIZED
from litestar.testing import TestClient
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.routes.unified_generation import UnifiedGenerationController
from src.api.routes.user import UserController
from src.api.schemas.jobs import JobCreatedResponse
from src.api.schemas.user import UserProfileResponse
from src.api.security.jwt import JWTConfig, JWTService
from src.api.services.generation.service import GenerationService
from src.api.services.idempotency import IdempotencyService
from src.api.services.user import UserService
from src.core.enums import GenerationType, JobStatus, ModelType
from src.core.product_registry import VEX_CONFIG

pytestmark = pytest.mark.unit

TEST_SECRET = "test_secret_key_for_testing_only_256bits_long"


@pytest.fixture
def jwt_service() -> JWTService:
    return JWTService(JWTConfig(secret_key=TEST_SECRET))


@pytest.fixture
def test_user_id() -> UUID:
    return uuid4()


@pytest.fixture
def auth_header(jwt_service: JWTService, test_user_id: UUID) -> dict[str, str]:
    token, _ = jwt_service.create_access_token(test_user_id, product_id="vex")
    return {"Authorization": f"Bearer {token}"}


# ---------------------------------------------------------------------------
# UserController — GET /v1/users/me
# ---------------------------------------------------------------------------


class TestUserControllerSmoke:
    """UserController had zero real-request coverage before this file."""

    def _make_profile_response(self, user_id: UUID) -> UserProfileResponse:
        now = datetime.now(UTC)
        return UserProfileResponse(
            id=str(user_id),
            email="user@example.com",
            display_name="Test User",
            subscription_tier="free",
            locale="en",
            role="user",
            is_active=True,
            created_at=now,
            updated_at=now,
            age_verified=False,
        )

    def _make_app(self, user_service: AsyncMock) -> Litestar:
        return Litestar(
            route_handlers=[UserController],
            dependencies={
                "user_service": Provide(lambda: user_service, sync_to_thread=False),
            },
        )

    def test_get_profile_returns_real_serialized_response(
        self, jwt_service: JWTService, test_user_id: UUID, auth_header: dict[str, str]
    ) -> None:
        mock_service = AsyncMock(spec=UserService)
        mock_service.get_profile = AsyncMock(return_value=self._make_profile_response(test_user_id))
        app = self._make_app(mock_service)
        app.state["jwt_service"] = jwt_service

        with TestClient(app=app) as client:
            resp = client.get("/v1/users/me", headers=auth_header)

        assert resp.status_code == HTTP_200_OK
        body = resp.json()
        assert body["id"] == str(test_user_id)
        assert body["email"] == "user@example.com"
        mock_service.get_profile.assert_called_once_with(test_user_id)

    def test_get_profile_requires_auth(self, jwt_service: JWTService) -> None:
        mock_service = AsyncMock(spec=UserService)
        app = self._make_app(mock_service)
        app.state["jwt_service"] = jwt_service

        with TestClient(app=app) as client:
            resp = client.get("/v1/users/me")

        assert resp.status_code == HTTP_401_UNAUTHORIZED
        mock_service.get_profile.assert_not_called()


# ---------------------------------------------------------------------------
# UnifiedGenerationController — POST /v1/generate/
# ---------------------------------------------------------------------------


class TestUnifiedGenerationControllerSmoke:
    """UnifiedGenerationController had zero real-request coverage before this
    file, and the only existing `.fn()`-based tests pass `product_config=
    MagicMock(slug="vex")` — a mock that trivially "resolves" and would never
    have caught commit 2873e8e. This uses the real `ProductConfig` dataclass
    from the product registry, run through Litestar's actual signature model.
    """

    def _make_job_created_response(self) -> JobCreatedResponse:
        return JobCreatedResponse(
            job_id=uuid4(),
            status=JobStatus.QUEUED,
            name="a cat",
            model=ModelType.AISHA_IMAGE.value,
            generation_type=GenerationType.T2I,
            created_at=datetime.now(UTC),
            message="Poll job status for results.",
            tokens_charged=10,
            balance_remaining=90,
        )

    def _make_app(
        self,
        *,
        generation_service: AsyncMock,
        idempotency_service: AsyncMock,
        session: AsyncMock,
    ) -> Litestar:
        return Litestar(
            route_handlers=[UnifiedGenerationController],
            dependencies={
                "generation_service": Provide(lambda: generation_service, sync_to_thread=False),
                "idempotency_service": Provide(lambda: idempotency_service, sync_to_thread=False),
                "session": Provide(lambda: session, sync_to_thread=False),
                "product_config": Provide(lambda: VEX_CONFIG, sync_to_thread=False),
                "product_id": Provide(lambda: "vex", sync_to_thread=False),
            },
        )

    def test_generate_returns_real_serialized_response(
        self, jwt_service: JWTService, test_user_id: UUID, auth_header: dict[str, str]
    ) -> None:
        job_response = self._make_job_created_response()

        mock_generation = AsyncMock(spec=GenerationService)
        mock_generation.generate = AsyncMock(return_value=job_response)

        mock_idempotency = AsyncMock(spec=IdempotencyService)
        mock_idempotency.check = AsyncMock(return_value=uuid4())
        mock_idempotency.complete = AsyncMock()

        mock_session = MagicMock(spec=AsyncSession)
        mock_session.commit = AsyncMock()

        app = self._make_app(
            generation_service=mock_generation,
            idempotency_service=mock_idempotency,
            session=mock_session,
        )
        app.state["jwt_service"] = jwt_service

        with TestClient(app=app) as client:
            resp = client.post(
                "/v1/generate/",
                json={"prompt": "a cat", "generation_type": "t2i", "model": "aisha-image"},
                headers={**auth_header, "Idempotency-Key": "smoke-test-key-1"},
            )

        assert resp.status_code == HTTP_201_CREATED
        body = resp.json()
        assert body["job_id"] == str(job_response.job_id)
        assert body["status"] == "queued"
        assert body["model"] == "aisha-image"

        mock_generation.generate.assert_called_once()
        call_kwargs = mock_generation.generate.call_args.kwargs
        assert call_kwargs["user_id"] == test_user_id
        assert call_kwargs["product_config"] is VEX_CONFIG
        mock_session.commit.assert_awaited_once()

    def test_generate_requires_auth(
        self,
        jwt_service: JWTService,
    ) -> None:
        mock_generation = AsyncMock(spec=GenerationService)
        mock_idempotency = AsyncMock(spec=IdempotencyService)
        mock_session = MagicMock(spec=AsyncSession)

        app = self._make_app(
            generation_service=mock_generation,
            idempotency_service=mock_idempotency,
            session=mock_session,
        )
        app.state["jwt_service"] = jwt_service

        with TestClient(app=app) as client:
            resp = client.post(
                "/v1/generate/",
                json={"prompt": "a cat", "generation_type": "t2i", "model": "aisha-image"},
                headers={"Idempotency-Key": "smoke-test-key-2"},
            )

        assert resp.status_code == HTTP_401_UNAUTHORIZED
        mock_generation.generate.assert_not_called()
