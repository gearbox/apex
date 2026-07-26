"""HTTP-level contract tests for LibraryProjectController (S3 remediation).

Companion to ``tests/unit/test_library_project_routes.py`` (which calls
``.fn()`` directly and therefore bypasses Litestar's exception-handler
pipeline). These tests drive the real ASGI request path via
``create_test_client`` — with the app's actual conflict-exception handler
wired in — so the 409/404 status codes AND their OpenAPI declarations
(added by the S3 remediation) are both verified for real, not asserted
against a hand-maintained response map. Mirrors
``test_library_tag_routes_http.py``.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest
from litestar.datastructures import State
from litestar.di import Provide
from litestar.testing import create_test_client
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.app import library_project_name_conflict_handler
from src.api.routes.library_project import LibraryProjectController
from src.api.schemas.library import LibraryProject
from src.api.security.jwt import JWTConfig, JWTService
from src.api.services.library_project import LibraryProjectNameConflictError, LibraryProjectService
from src.api.services.token_revocation import TokenRevocationService

if TYPE_CHECKING:
    from contextlib import AbstractContextManager

    from litestar import Litestar
    from litestar.testing import TestClient

pytestmark = pytest.mark.unit

TEST_SECRET = "test_secret_key_for_testing_only_256bits_long"


def _project(**overrides: object) -> LibraryProject:
    now = datetime.now(UTC)
    defaults: dict[str, object] = {
        "id": uuid4(),
        "name": "Vacation Photos",
        "description": None,
        "created_at": now,
        "updated_at": now,
    } | overrides
    return LibraryProject(**defaults)  # type: ignore[arg-type]


@pytest.fixture
def jwt_service() -> JWTService:
    return JWTService(JWTConfig(secret_key=TEST_SECRET))


@pytest.fixture
def auth_header(jwt_service: JWTService) -> dict[str, str]:
    token, _ = jwt_service.create_access_token(uuid4(), product_id="vex")
    return {"Authorization": f"Bearer {token}"}


def _make_client(
    service: AsyncMock, jwt_service: JWTService
) -> AbstractContextManager[TestClient[Litestar]]:
    session = MagicMock(spec=AsyncSession)
    return create_test_client(
        route_handlers=[LibraryProjectController],
        exception_handlers={LibraryProjectNameConflictError: library_project_name_conflict_handler},
        dependencies={
            "library_project_service": Provide(lambda: service, sync_to_thread=False),
            "session": Provide(lambda: session, sync_to_thread=False),
            "product_id": Provide(lambda: "vex", sync_to_thread=False),
        },
        state=State(
            {
                "jwt_service": jwt_service,
                "token_revocation": TokenRevocationService(None, max_token_ttl_seconds=0),
            }
        ),
    )


class TestCreateProjectHttp:
    def test_duplicate_name_returns_409(
        self, jwt_service: JWTService, auth_header: dict[str, str]
    ) -> None:
        service = AsyncMock(spec=LibraryProjectService)
        service.create.side_effect = LibraryProjectNameConflictError("Vacation Photos")

        with _make_client(service, jwt_service) as client:
            resp = client.post(
                "/v1/library/projects",
                json={"name": "Vacation Photos"},
                headers=auth_header,
            )

        assert resp.status_code == 409
        assert resp.json()["error"] == "project_name_conflict"

    def test_openapi_declares_409(self, jwt_service: JWTService) -> None:
        service = AsyncMock(spec=LibraryProjectService)

        with _make_client(service, jwt_service) as client:
            responses = client.app.openapi_schema.paths["/v1/library/projects"].post.responses

        assert "409" in responses


class TestPatchProjectHttp:
    def test_duplicate_name_returns_409(
        self, jwt_service: JWTService, auth_header: dict[str, str]
    ) -> None:
        service = AsyncMock(spec=LibraryProjectService)
        service.patch.side_effect = LibraryProjectNameConflictError("Vacation Photos")
        project_id = uuid4()

        with _make_client(service, jwt_service) as client:
            resp = client.patch(
                f"/v1/library/projects/{project_id}",
                json={"name": "Vacation Photos"},
                headers=auth_header,
            )

        assert resp.status_code == 409

    def test_unknown_id_returns_404(
        self, jwt_service: JWTService, auth_header: dict[str, str]
    ) -> None:
        service = AsyncMock(spec=LibraryProjectService)
        service.patch.return_value = None
        project_id = uuid4()

        with _make_client(service, jwt_service) as client:
            resp = client.patch(
                f"/v1/library/projects/{project_id}",
                json={"name": "Renamed"},
                headers=auth_header,
            )

        assert resp.status_code == 404

    def test_openapi_declares_404_and_409(self, jwt_service: JWTService) -> None:
        service = AsyncMock(spec=LibraryProjectService)

        with _make_client(service, jwt_service) as client:
            responses = client.app.openapi_schema.paths[
                "/v1/library/projects/{project_id}"
            ].patch.responses

        assert "404" in responses
        assert "409" in responses


class TestGetProjectHttp:
    def test_unknown_id_returns_404(
        self, jwt_service: JWTService, auth_header: dict[str, str]
    ) -> None:
        service = AsyncMock(spec=LibraryProjectService)
        service.get.return_value = None

        with _make_client(service, jwt_service) as client:
            resp = client.get(f"/v1/library/projects/{uuid4()}", headers=auth_header)

        assert resp.status_code == 404

    def test_openapi_declares_404(self, jwt_service: JWTService) -> None:
        service = AsyncMock(spec=LibraryProjectService)

        with _make_client(service, jwt_service) as client:
            responses = client.app.openapi_schema.paths[
                "/v1/library/projects/{project_id}"
            ].get.responses

        assert "404" in responses


class TestDeleteProjectHttp:
    def test_unknown_id_returns_404(
        self, jwt_service: JWTService, auth_header: dict[str, str]
    ) -> None:
        service = AsyncMock(spec=LibraryProjectService)
        service.delete.return_value = False

        with _make_client(service, jwt_service) as client:
            resp = client.delete(f"/v1/library/projects/{uuid4()}", headers=auth_header)

        assert resp.status_code == 404

    def test_openapi_declares_404(self, jwt_service: JWTService) -> None:
        service = AsyncMock(spec=LibraryProjectService)

        with _make_client(service, jwt_service) as client:
            responses = client.app.openapi_schema.paths[
                "/v1/library/projects/{project_id}"
            ].delete.responses

        assert "404" in responses
