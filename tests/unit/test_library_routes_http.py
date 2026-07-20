"""HTTP-level contract tests for ``GET /v1/library``.

Companion to ``tests/unit/test_library_routes.py``, which calls
``LibraryController.list_assets.fn(...)`` directly and therefore bypasses
Litestar's kwarg-resolution / signature model entirely. That blind spot let a
reserved-kwarg collision (a handler argument literally named ``query``, which
Litestar special-cases to receive the raw query-params ``MultiDict``) ship to
master: every real request to ``GET /v1/library`` 400'd, while the ``.fn()``
tests stayed green because they call the plain Python function with keyword
arguments, never going through the connection kwargs the ASGI request path
actually resolves.

These tests build a minimal real ``Litestar`` app (mirroring the pattern in
``tests/unit/test_route_handler_request_smoke.py`` /
``tests/unit/test_auth_guards.py::TestStorageControllerAuth``) and drive it
through ``create_test_client`` with a real JWT, so `auth_guard`, DI, the
signature model, and OpenAPI generation all run for real.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest
from litestar.datastructures import State
from litestar.di import Provide
from litestar.status_codes import HTTP_200_OK, HTTP_400_BAD_REQUEST
from litestar.testing import create_test_client
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.routes.library import LibraryController
from src.api.schemas.pagination import CursorPage
from src.api.security.jwt import JWTConfig, JWTService
from src.api.services.library import LibraryService
from src.core.enums import LibrarySort
from src.core.library_ref import LibraryAssetSource
from src.core.product_registry import VEX_CONFIG

if TYPE_CHECKING:
    from contextlib import AbstractContextManager

    from litestar import Litestar
    from litestar.testing import TestClient

pytestmark = pytest.mark.unit

TEST_SECRET = "test_secret_key_for_testing_only_256bits_long"

# The full documented query-param surface for GET /v1/library — a future
# param that's silently dropped from the OpenAPI schema (the exact failure
# mode of this regression) fails this list loudly instead of vanishing.
_DOCUMENTED_PARAMS = {
    "limit",
    "cursor",
    "source",
    "media_type",
    "model",
    "favorite",
    "project_id",
    "expiring",
    "query",
    "created_from",
    "created_to",
    "sort",
}


@pytest.fixture
def jwt_service() -> JWTService:
    return JWTService(JWTConfig(secret_key=TEST_SECRET))


@pytest.fixture
def auth_header(jwt_service: JWTService) -> dict[str, str]:
    token, _ = jwt_service.create_access_token(uuid4(), product_id="vex")
    return {"Authorization": f"Bearer {token}"}


def _make_service(page: CursorPage[object] | None = None) -> AsyncMock:
    service = AsyncMock(spec=LibraryService)
    service.list_assets.return_value = page or CursorPage(
        items=[], limit=30, has_more=False, next_cursor=None
    )
    return service


def _make_client(
    service: AsyncMock, jwt_service: JWTService
) -> AbstractContextManager[TestClient[Litestar]]:
    session = MagicMock(spec=AsyncSession)
    return create_test_client(
        route_handlers=[LibraryController],
        dependencies={
            "library_service": Provide(lambda: service, sync_to_thread=False),
            "session": Provide(lambda: session, sync_to_thread=False),
            "product_config": Provide(lambda: VEX_CONFIG, sync_to_thread=False),
            "product_id": Provide(lambda: "vex", sync_to_thread=False),
        },
        state=State({"jwt_service": jwt_service}),
    )


class TestListAssetsHttp:
    """Real-request coverage for GET /v1/library — the regression's blind spot."""

    def test_no_params_returns_200_and_calls_service_with_query_none(
        self, jwt_service: JWTService, auth_header: dict[str, str]
    ) -> None:
        service = _make_service()

        with _make_client(service, jwt_service) as client:
            resp = client.get("/v1/library", headers=auth_header)

        assert resp.status_code == HTTP_200_OK
        service.list_assets.assert_awaited_once()
        assert service.list_assets.call_args.kwargs["query"] is None

    def test_full_param_surface_binds_and_forwards_to_service(
        self, jwt_service: JWTService, auth_header: dict[str, str]
    ) -> None:
        service = _make_service()

        with _make_client(service, jwt_service) as client:
            resp = client.get(
                "/v1/library?query=hello&source=upload&sort=oldest",
                headers=auth_header,
            )

        assert resp.status_code == HTTP_200_OK
        kwargs = service.list_assets.call_args.kwargs
        assert kwargs["query"] == "hello"
        assert kwargs["source"] == LibraryAssetSource.UPLOAD
        assert kwargs["sort"] == LibrarySort.OLDEST

    def test_query_over_max_length_returns_400(
        self, jwt_service: JWTService, auth_header: dict[str, str]
    ) -> None:
        service = _make_service()

        with _make_client(service, jwt_service) as client:
            resp = client.get(f"/v1/library?query={'a' * 201}", headers=auth_header)

        assert resp.status_code == HTTP_400_BAD_REQUEST
        service.list_assets.assert_not_awaited()

    def test_openapi_schema_exposes_full_documented_param_surface(
        self, jwt_service: JWTService
    ) -> None:
        """The exact regression: `query` silently vanished from the OpenAPI
        schema because Litestar excludes reserved kwargs from it. This
        assertion would have caught it, and guards every other documented
        param against the same fate.
        """
        service = _make_service()

        with _make_client(service, jwt_service) as client:
            schema = client.app.openapi_schema
            parameters = schema.paths["/v1/library"].get.parameters or []

        param_names = {p.name for p in parameters}
        assert param_names >= _DOCUMENTED_PARAMS

        query_param = next(p for p in parameters if p.name == "query")
        assert query_param.schema.max_length == 200
