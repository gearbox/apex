"""End-to-end HTTP tests for the content proxy's Range/conditional-GET support.

Exercises the *real* ContentProxyController (guards, route handlers,
_stream_from_r2) through a Litestar TestClient, with ContentProxyService and
R2StorageService replaced by lightweight in-memory doubles — no DB, no R2,
but real HTTP request/response handling and real header semantics.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import timedelta
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock
from uuid import UUID, uuid4

import pytest
from litestar import Litestar
from litestar.di import Provide
from litestar.middleware.base import AbstractMiddleware
from litestar.status_codes import (
    HTTP_200_OK,
    HTTP_206_PARTIAL_CONTENT,
    HTTP_304_NOT_MODIFIED,
    HTTP_404_NOT_FOUND,
    HTTP_416_REQUESTED_RANGE_NOT_SATISFIABLE,
)
from litestar.testing import TestClient
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.routes.content import ContentProxyController
from src.api.security.jwt import JWTConfig, JWTService
from src.api.services.content_proxy import ContentNotFoundError, ContentProxyService
from src.api.services.storage.r2 import ObjectStream, R2StorageService
from src.api.services.token_revocation import TokenRevocationService

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from litestar.types import Receive, Scope, Send

TEST_SECRET = "test_secret_key_for_testing_only_256bits_long"
PRODUCT_ID = "vex"
COOKIE_NAME = "apex_content"
PAYLOAD = bytes(range(100))  # 100 distinct bytes — makes slice equality unambiguous
ETAG = "content-etag-abc"


class _StubR2(R2StorageService):
    """In-memory stand-in for R2StorageService.stream_object.

    Slices PAYLOAD per the forwarded Range param, mirroring R2's real
    behavior, and records every call so tests can assert R2 was (or
    wasn't) touched. Subclasses the real service (rather than duck-typing)
    so Litestar's signature model — which validates injected DI values
    against their declared type — accepts it via isinstance; __init__ is
    deliberately not chained since it requires real R2 credentials.
    """

    def __init__(self, payload: bytes = PAYLOAD, content_type: str = "image/png") -> None:
        self.payload = payload
        self.content_type = content_type
        self.calls: list[str | None] = []

    @asynccontextmanager
    async def stream_object(
        self, storage_key: str, *, range_header: str | None = None
    ) -> AsyncIterator[ObjectStream]:
        del storage_key
        self.calls.append(range_header)
        size = len(self.payload)

        if range_header is None:
            body = self.payload
            content_range = None
        else:
            spec = range_header.removeprefix("bytes=")
            start_s, end_s = spec.split("-")
            start, end = int(start_s), int(end_s)
            body = self.payload[start : end + 1]
            content_range = f"bytes {start}-{end}/{size}"

        async def _chunks() -> AsyncIterator[bytes]:
            yield body

        yield ObjectStream(
            chunks=_chunks(),
            content_type=self.content_type,
            content_length=len(body),
            content_range=content_range,
        )


class _StubContentProxy(ContentProxyService):
    """In-memory stand-in for ContentProxyService — no DB lookups.

    Subclasses the real service so isinstance-based DI signature
    validation accepts it (see _StubR2 for the same rationale); __init__
    is deliberately not chained since it requires real storage/settings.
    """

    def __init__(
        self, *, size_bytes: int = len(PAYLOAD), etag: str = ETAG, ttl: int = 3600
    ) -> None:
        self._size_bytes = size_bytes
        self._etag = etag
        self._ttl = ttl

    async def resolve_output(
        self, output_id: UUID, *, user_id: UUID, product_id: str, session: AsyncSession
    ) -> tuple[str, str, int]:
        del output_id, user_id, product_id, session
        return "users/u/outputs/j/f.png", self._etag, self._size_bytes

    async def resolve_upload(
        self, image_id: UUID, *, user_id: UUID, product_id: str, session: AsyncSession
    ) -> tuple[str, str, int]:
        del image_id, user_id, product_id, session
        return "users/u/uploads/f.png", self._etag, self._size_bytes


class _NotFoundContentProxy(ContentProxyService):
    """Always raises — simulates foreign/nonexistent content."""

    def __init__(self) -> None:
        self._ttl = 3600

    async def resolve_output(
        self, output_id: UUID, *, user_id: UUID, product_id: str, session: AsyncSession
    ) -> tuple[str, str, int]:
        del output_id, user_id, product_id, session
        raise ContentNotFoundError("not found")

    async def resolve_upload(
        self, image_id: UUID, *, user_id: UUID, product_id: str, session: AsyncSession
    ) -> tuple[str, str, int]:
        del image_id, user_id, product_id, session
        raise ContentNotFoundError("not found")


def _make_app(content_proxy: Any, r2_storage: Any, jwt_service: JWTService) -> Litestar:
    class FakeProductMiddleware(AbstractMiddleware):
        async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
            if scope["type"] in ("http", "websocket"):
                scope.setdefault("state", {})
                scope["state"]["product_id"] = PRODUCT_ID
            await self.app(scope, receive, send)

    app = Litestar(
        route_handlers=[ContentProxyController],
        middleware=[FakeProductMiddleware],
        dependencies={
            "product_id": Provide(lambda: PRODUCT_ID, sync_to_thread=False),
            "content_proxy": Provide(lambda: content_proxy, sync_to_thread=False),
            "r2_storage": Provide(lambda: r2_storage, sync_to_thread=False),
            "session": Provide(lambda: AsyncMock(spec=AsyncSession), sync_to_thread=False),
        },
    )
    app.state["jwt_service"] = jwt_service
    app.state["token_revocation"] = TokenRevocationService(None, max_token_ttl_seconds=0)
    return app


@pytest.fixture
def jwt_service() -> JWTService:
    return JWTService(JWTConfig(secret_key=TEST_SECRET))


@pytest.fixture
def bearer_headers(jwt_service: JWTService) -> dict[str, str]:
    token, _ = jwt_service.create_access_token(uuid4(), product_id=PRODUCT_ID)
    return {"Authorization": f"Bearer {token}"}


class TestFullBodyResponse:
    def test_200_advertises_accept_ranges_and_full_body(
        self, jwt_service: JWTService, bearer_headers: dict[str, str]
    ) -> None:
        r2 = _StubR2()
        app = _make_app(_StubContentProxy(), r2, jwt_service)
        with TestClient(app=app) as client:
            resp = client.get(f"/v1/content/outputs/{uuid4()}", headers=bearer_headers)

        assert resp.status_code == HTTP_200_OK
        assert resp.headers["Accept-Ranges"] == "bytes"
        assert resp.headers["Content-Length"] == str(len(PAYLOAD))
        assert resp.content == PAYLOAD
        assert r2.calls == [None]


class TestPartialContentResponse:
    def test_206_body_slice_equality(
        self, jwt_service: JWTService, bearer_headers: dict[str, str]
    ) -> None:
        r2 = _StubR2()
        app = _make_app(_StubContentProxy(), r2, jwt_service)
        with TestClient(app=app) as client:
            resp = client.get(
                f"/v1/content/outputs/{uuid4()}",
                headers={**bearer_headers, "Range": "bytes=10-19"},
            )

        assert resp.status_code == HTTP_206_PARTIAL_CONTENT
        assert resp.headers["Content-Range"] == f"bytes 10-19/{len(PAYLOAD)}"
        assert resp.headers["Content-Length"] == "10"
        assert resp.content == PAYLOAD[10:20]
        assert r2.calls == ["bytes=10-19"]

    def test_206_suffix_range(
        self, jwt_service: JWTService, bearer_headers: dict[str, str]
    ) -> None:
        r2 = _StubR2()
        app = _make_app(_StubContentProxy(), r2, jwt_service)
        with TestClient(app=app) as client:
            resp = client.get(
                f"/v1/content/uploads/{uuid4()}",
                headers={**bearer_headers, "Range": "bytes=-10"},
            )

        assert resp.status_code == HTTP_206_PARTIAL_CONTENT
        assert resp.content == PAYLOAD[-10:]


class TestUnsatisfiableRange:
    def test_416_no_r2_round_trip(
        self, jwt_service: JWTService, bearer_headers: dict[str, str]
    ) -> None:
        r2 = _StubR2()
        app = _make_app(_StubContentProxy(), r2, jwt_service)
        with TestClient(app=app) as client:
            resp = client.get(
                f"/v1/content/outputs/{uuid4()}",
                headers={**bearer_headers, "Range": "bytes=9999-10999"},
            )

        assert resp.status_code == HTTP_416_REQUESTED_RANGE_NOT_SATISFIABLE
        assert resp.headers["Content-Range"] == f"bytes */{len(PAYLOAD)}"
        assert r2.calls == []  # rejected before any R2 traffic


class TestConditionalGet:
    def test_304_on_matching_etag_no_r2_round_trip(
        self, jwt_service: JWTService, bearer_headers: dict[str, str]
    ) -> None:
        r2 = _StubR2()
        app = _make_app(_StubContentProxy(), r2, jwt_service)
        with TestClient(app=app) as client:
            resp = client.get(
                f"/v1/content/outputs/{uuid4()}",
                headers={**bearer_headers, "If-None-Match": f'"{ETAG}"'},
            )

        assert resp.status_code == HTTP_304_NOT_MODIFIED
        assert resp.headers["ETag"] == f'"{ETAG}"'
        assert resp.content == b""
        assert r2.calls == []

    def test_non_matching_etag_serves_full_body(
        self, jwt_service: JWTService, bearer_headers: dict[str, str]
    ) -> None:
        r2 = _StubR2()
        app = _make_app(_StubContentProxy(), r2, jwt_service)
        with TestClient(app=app) as client:
            resp = client.get(
                f"/v1/content/outputs/{uuid4()}",
                headers={**bearer_headers, "If-None-Match": '"stale-etag"'},
            )

        assert resp.status_code == HTTP_200_OK
        assert resp.content == PAYLOAD


class TestNotFound:
    def test_404_unchanged_for_foreign_content(
        self, jwt_service: JWTService, bearer_headers: dict[str, str]
    ) -> None:
        r2 = _StubR2()
        app = _make_app(_NotFoundContentProxy(), r2, jwt_service)
        with TestClient(app=app) as client:
            resp = client.get(f"/v1/content/outputs/{uuid4()}", headers=bearer_headers)

        assert resp.status_code == HTTP_404_NOT_FOUND
        assert r2.calls == []


class TestAuthPaths:
    def test_bearer_auth_path_streams_content(
        self, jwt_service: JWTService, bearer_headers: dict[str, str]
    ) -> None:
        r2 = _StubR2()
        app = _make_app(_StubContentProxy(), r2, jwt_service)
        with TestClient(app=app) as client:
            resp = client.get(f"/v1/content/outputs/{uuid4()}", headers=bearer_headers)
        assert resp.status_code == HTTP_200_OK
        assert resp.content == PAYLOAD

    def test_cookie_auth_path_streams_content(self, jwt_service: JWTService) -> None:
        token, _ = jwt_service.create_content_token(
            uuid4(), product_id=PRODUCT_ID, ttl=timedelta(hours=1)
        )
        r2 = _StubR2()
        app = _make_app(_StubContentProxy(), r2, jwt_service)
        with TestClient(app=app) as client:
            resp = client.get(f"/v1/content/uploads/{uuid4()}", cookies={COOKIE_NAME: token})
        assert resp.status_code == HTTP_200_OK
        assert resp.content == PAYLOAD

    def test_no_credentials_returns_401(self, jwt_service: JWTService) -> None:
        r2 = _StubR2()
        app = _make_app(_StubContentProxy(), r2, jwt_service)
        with TestClient(app=app) as client:
            resp = client.get(f"/v1/content/outputs/{uuid4()}")
        assert resp.status_code == 401
