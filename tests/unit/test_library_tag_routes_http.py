"""HTTP-level contract tests for LibraryTagController (S3 remediation).

Companion to ``tests/unit/test_library_tag_routes.py`` (which calls
``.fn()`` directly and therefore bypasses Litestar's exception-handler
pipeline). These tests drive the real ASGI request path via
``create_test_client`` — with the app's actual conflict-exception handler
wired in — so the 409/404 status codes AND their OpenAPI declarations
(added by the S3 remediation) are both verified for real, not asserted
against a hand-maintained response map.
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

from src.api.app import library_tag_name_conflict_handler
from src.api.routes.library_tag import LibraryTagController
from src.api.schemas.library import LibraryTag
from src.api.security.jwt import JWTConfig, JWTService
from src.api.services.library_tag import LibraryTagNameConflictError, LibraryTagService
from src.api.services.token_revocation import TokenRevocationService

if TYPE_CHECKING:
    from contextlib import AbstractContextManager

    from litestar import Litestar
    from litestar.testing import TestClient

pytestmark = pytest.mark.unit

TEST_SECRET = "test_secret_key_for_testing_only_256bits_long"


def _tag(**overrides: object) -> LibraryTag:
    now = datetime.now(UTC)
    defaults: dict[str, object] = {
        "id": uuid4(),
        "name": "sunset",
        "created_at": now,
        "updated_at": now,
    } | overrides
    return LibraryTag(**defaults)  # type: ignore[arg-type]


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
        route_handlers=[LibraryTagController],
        exception_handlers={LibraryTagNameConflictError: library_tag_name_conflict_handler},
        dependencies={
            "library_tag_service": Provide(lambda: service, sync_to_thread=False),
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


class TestCreateTagHttp:
    def test_duplicate_name_returns_409(
        self, jwt_service: JWTService, auth_header: dict[str, str]
    ) -> None:
        service = AsyncMock(spec=LibraryTagService)
        service.create.side_effect = LibraryTagNameConflictError("sunset")

        with _make_client(service, jwt_service) as client:
            resp = client.post("/v1/library/tags", json={"name": "sunset"}, headers=auth_header)

        assert resp.status_code == 409
        assert resp.json()["error"] == "tag_name_conflict"

    def test_openapi_declares_409(self, jwt_service: JWTService) -> None:
        service = AsyncMock(spec=LibraryTagService)

        with _make_client(service, jwt_service) as client:
            responses = client.app.openapi_schema.paths["/v1/library/tags"].post.responses

        assert "409" in responses


class TestPatchTagHttp:
    def test_duplicate_name_returns_409(
        self, jwt_service: JWTService, auth_header: dict[str, str]
    ) -> None:
        service = AsyncMock(spec=LibraryTagService)
        service.patch.side_effect = LibraryTagNameConflictError("sunset")
        tag_id = uuid4()

        with _make_client(service, jwt_service) as client:
            resp = client.patch(
                f"/v1/library/tags/{tag_id}", json={"name": "sunset"}, headers=auth_header
            )

        assert resp.status_code == 409

    def test_unknown_id_returns_404(
        self, jwt_service: JWTService, auth_header: dict[str, str]
    ) -> None:
        service = AsyncMock(spec=LibraryTagService)
        service.patch.return_value = None
        tag_id = uuid4()

        with _make_client(service, jwt_service) as client:
            resp = client.patch(
                f"/v1/library/tags/{tag_id}", json={"name": "renamed"}, headers=auth_header
            )

        assert resp.status_code == 404

    def test_openapi_declares_404_and_409(self, jwt_service: JWTService) -> None:
        service = AsyncMock(spec=LibraryTagService)

        with _make_client(service, jwt_service) as client:
            responses = client.app.openapi_schema.paths["/v1/library/tags/{tag_id}"].patch.responses

        assert "404" in responses
        assert "409" in responses


class TestGetTagHttp:
    def test_unknown_id_returns_404(
        self, jwt_service: JWTService, auth_header: dict[str, str]
    ) -> None:
        service = AsyncMock(spec=LibraryTagService)
        service.get.return_value = None

        with _make_client(service, jwt_service) as client:
            resp = client.get(f"/v1/library/tags/{uuid4()}", headers=auth_header)

        assert resp.status_code == 404

    def test_openapi_declares_404(self, jwt_service: JWTService) -> None:
        service = AsyncMock(spec=LibraryTagService)

        with _make_client(service, jwt_service) as client:
            responses = client.app.openapi_schema.paths["/v1/library/tags/{tag_id}"].get.responses

        assert "404" in responses


class TestDeleteTagHttp:
    def test_unknown_id_returns_404(
        self, jwt_service: JWTService, auth_header: dict[str, str]
    ) -> None:
        service = AsyncMock(spec=LibraryTagService)
        service.delete.return_value = False

        with _make_client(service, jwt_service) as client:
            resp = client.delete(f"/v1/library/tags/{uuid4()}", headers=auth_header)

        assert resp.status_code == 404

    def test_openapi_declares_404(self, jwt_service: JWTService) -> None:
        service = AsyncMock(spec=LibraryTagService)

        with _make_client(service, jwt_service) as client:
            responses = client.app.openapi_schema.paths[
                "/v1/library/tags/{tag_id}"
            ].delete.responses

        assert "404" in responses
