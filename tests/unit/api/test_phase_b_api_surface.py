"""Phase B API surface: server-side frame extraction and presigned-URL routes are gone.

The server-side frame-extraction subsystem (``/v1/frames/*``) and every
browser-facing presigned/duplicate storage route were removed. These tests pin
the resulting contract against the **full app's** OpenAPI schema, and guard the
Phase A client-side extraction surfaces that must survive the cleanup.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from unittest.mock import patch
from uuid import UUID, uuid4

import httpx
import pytest
from litestar.status_codes import HTTP_404_NOT_FOUND

from src.api.security import JWTConfig, JWTService
from src.core.config import get_settings
from tests.legal_support import TEST_LEGAL_DIGEST

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Generator

pytestmark = pytest.mark.unit

_JWT_SECRET = "ci-unit-test-secret-key-32-bytes-long"
_PRODUCT_HEADERS = {"X-Product-Id": "vex"}
_REMOVED_SCHEMAS = ("FrameJobResponse", "ImageAccessResponse", "OutputListItem")
_REMOVED_PROPERTIES = frozenset({"presigned_url", "expires_in_seconds"})


@pytest.fixture(scope="module")
def _app_env() -> Generator[None]:
    """Env for ``create_app()``; module-scoped so the app is built (and its lifespan
    entered) once, not per test.
    """
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("JWT_SECRET_KEY", _JWT_SECRET)
        mp.setenv("ENABLE_DOCS", "true")
        # Avoid the aisha poller's startup domain-guard — irrelevant to this test.
        mp.setenv("AISHA_POLLER_ENABLED", "false")
        get_settings.cache_clear()
        yield
        get_settings.cache_clear()


def _build_app() -> Any:
    """``create_app()`` without reconfiguring the global structlog processors.

    ``tests/unit/api/conftest.py`` suppresses ``configure_logging`` with a function-scoped
    autouse fixture, which is not yet active while a module-scoped fixture builds the app,
    so the patch is repeated here. Typed ``Any`` because httpx's ``ASGITransport`` is typed
    for a bare ASGI callable, not ``Litestar``.
    """
    from src.api.app import create_app

    with patch("src.api.app.configure_logging"):
        return create_app()


@pytest.fixture(scope="module")
def openapi(_app_env: None) -> dict[str, Any]:
    """OpenAPI document of the full app. Generated without entering the lifespan."""
    schema: dict[str, Any] = _build_app().openapi_schema.to_schema()
    return schema


@pytest.fixture
async def client(_app_env: None) -> AsyncIterator[httpx.AsyncClient]:
    """In-process client that never sends ASGI lifespan events.

    A removed route 404s in routing, so no service container is needed — and the real
    lifespan reaches the network (bundle sync, R2 probe) and shares a temp cache dir,
    which makes parallel xdist workers collide.
    """
    transport = httpx.ASGITransport(app=_build_app())
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as c:
        yield c


def _auth_headers(user_id: UUID) -> dict[str, str]:
    token, _ = JWTService(JWTConfig(secret_key=_JWT_SECRET)).create_access_token(
        user_id, product_id="vex", legal_digest=TEST_LEGAL_DIGEST
    )
    return {**_PRODUCT_HEADERS, "Authorization": f"Bearer {token}"}


class TestRemovedRoutesAreAbsentFromOpenAPI:
    """I1 + I2: removed surfaces are not documented."""

    def test_no_frames_paths(self, openapi: dict[str, Any]) -> None:
        assert [p for p in openapi["paths"] if p.startswith("/v1/frames")] == []

    def test_storage_exposes_only_upload_and_stats(self, openapi: dict[str, Any]) -> None:
        storage = {
            path: sorted(methods)
            for path, methods in openapi["paths"].items()
            if path.startswith("/v1/storage")
        }
        assert storage == {"/v1/storage/upload": ["post"], "/v1/storage/stats": ["get"]}

    def test_no_schema_exposes_presigned_url_fields(self, openapi: dict[str, Any]) -> None:
        offenders = [
            (name, prop)
            for name, schema in openapi["components"]["schemas"].items()
            for prop in schema.get("properties", {})
            if prop in _REMOVED_PROPERTIES
        ]
        assert offenders == []

    @pytest.mark.parametrize("name", _REMOVED_SCHEMAS)
    def test_removed_schema_is_absent(self, openapi: dict[str, Any], name: str) -> None:
        assert name not in openapi["components"]["schemas"]


class TestPhaseASurfacesIntact:
    """I3: the client-side extraction contract survives the cleanup."""

    def test_upload_form_keeps_frame_lineage_fields(self, openapi: dict[str, Any]) -> None:
        properties = openapi["components"]["schemas"]["UploadForm"]["properties"]
        assert "source_asset_ref" in properties
        assert "source_timestamp_ms" in properties

    def test_media_original_keeps_duration_ms(self, openapi: dict[str, Any]) -> None:
        assert "duration_ms" in openapi["components"]["schemas"]["MediaOriginal"]["properties"]


class TestRemovedRoutesReturn404:
    """I4: hard cutover — no 410 stubs, the routes simply do not exist."""

    @pytest.mark.parametrize(
        "path",
        [
            "/v1/frames/jobs/{id}",
            "/v1/storage/uploads/{id}",
            "/v1/storage/uploads/{id}/download",
            "/v1/storage/outputs/{id}",
            "/v1/storage/outputs/{id}/download",
            "/v1/storage/jobs/{id}/outputs",
        ],
    )
    async def test_authenticated_request_returns_404(
        self, client: httpx.AsyncClient, path: str
    ) -> None:
        resp = await client.get(
            path.format(id=uuid4()),
            headers=_auth_headers(uuid4()),
        )
        assert resp.status_code == HTTP_404_NOT_FOUND

    async def test_storage_output_list_returns_404(self, client: httpx.AsyncClient) -> None:
        resp = await client.get("/v1/storage/outputs", headers=_auth_headers(uuid4()))
        assert resp.status_code == HTTP_404_NOT_FOUND


class TestFrameWorkerWiringRemoved:
    """I5: no frame worker is constructed, started or registered."""

    def test_service_container_has_no_frame_worker(self) -> None:
        from src.api.dependencies import common

        assert not hasattr(common._services, "frame_extraction_worker")
        assert not hasattr(common, "get_frame_extraction_service")

    def test_no_frame_extraction_dependency_is_registered(self) -> None:
        from src.api.dependencies.common import dependencies

        assert "frame_extraction_service" not in dependencies
