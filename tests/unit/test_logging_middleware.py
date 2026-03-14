"""Tests for RequestLoggingMiddleware."""

from __future__ import annotations

import uuid
from collections.abc import Generator
from typing import Any, cast
from unittest.mock import AsyncMock

import pytest
import structlog
from litestar.types import Scope
from structlog.contextvars import clear_contextvars
from structlog.testing import capture_logs

from src.api.middleware.logging import RequestLoggingMiddleware

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_scope(
    path: str = "/test",
    method: str = "GET",
    headers: list[tuple[bytes, bytes]] | None = None,
    client: tuple[str, int] | None = ("127.0.0.1", 12345),
) -> Scope:
    return cast(
        Scope,
        {
            "type": "http",
            "method": method,
            "path": path,
            "headers": headers or [],
            "client": client,
        },
    )


async def _noop_receive() -> Any:
    return {}


async def _noop_send(message: Any) -> None:  # noqa: ARG001
    pass


@pytest.fixture(autouse=True)
def _reset_context() -> Generator[None]:
    """Clear structlog context vars between tests."""
    yield
    clear_contextvars()


# ---------------------------------------------------------------------------
# RequestLoggingMiddleware
# ---------------------------------------------------------------------------


class TestRequestLoggingMiddleware:
    def _build_middleware(self) -> tuple[RequestLoggingMiddleware, AsyncMock]:
        app_mock = AsyncMock()
        mw = RequestLoggingMiddleware(app=app_mock)  # type: ignore[arg-type]
        return mw, app_mock

    @pytest.mark.anyio
    async def test_request_started_and_finished_are_logged(self) -> None:
        mw, _ = self._build_middleware()
        scope = _make_scope()

        with capture_logs() as cap:
            await mw(scope, _noop_receive, _noop_send)

        events = [r["event"] for r in cap]
        assert "request.started" in events
        assert "request.finished" in events

    @pytest.mark.anyio
    async def test_duration_ms_is_present_in_finished_event(self) -> None:
        mw, _ = self._build_middleware()
        scope = _make_scope()

        with capture_logs() as cap:
            await mw(scope, _noop_receive, _noop_send)

        finished = next(r for r in cap if r["event"] == "request.finished")
        assert "duration_ms" in finished
        assert isinstance(finished["duration_ms"], float)

    @pytest.mark.anyio
    async def test_provided_x_request_id_is_used(self) -> None:
        mw, _ = self._build_middleware()
        rid = "my-custom-request-id"
        scope = _make_scope(headers=[(b"x-request-id", rid.encode())])

        with capture_logs() as cap:
            await mw(scope, _noop_receive, _noop_send)

        started = next(r for r in cap if r["event"] == "request.started")
        assert started.get("request_id") == rid

    @pytest.mark.anyio
    async def test_uuid_is_generated_when_header_absent(self) -> None:
        mw, _ = self._build_middleware()
        scope = _make_scope()

        with capture_logs() as cap:
            await mw(scope, _noop_receive, _noop_send)

        started = next(r for r in cap if r["event"] == "request.started")
        rid = started.get("request_id")
        assert rid is not None
        # Verify it's a valid UUID
        uuid.UUID(str(rid))

    @pytest.mark.anyio
    async def test_non_http_scope_passes_through_without_logging(self) -> None:
        mw, app_mock = self._build_middleware()
        scope = cast(Scope, {"type": "websocket", "path": "/ws"})

        with capture_logs() as cap:
            await mw(scope, _noop_receive, _noop_send)

        assert cap == []
        app_mock.assert_awaited_once_with(scope, _noop_receive, _noop_send)

    @pytest.mark.anyio
    async def test_context_is_cleared_after_request(self) -> None:
        mw, _ = self._build_middleware()
        scope = _make_scope()

        await mw(scope, _noop_receive, _noop_send)

        # After the request, context vars should be cleared
        with capture_logs() as cap:
            structlog.get_logger().info("post.request")

        assert "request_id" not in cap[0]

    @pytest.mark.anyio
    async def test_context_cleared_even_on_downstream_exception(self) -> None:
        app_mock = AsyncMock(side_effect=RuntimeError("boom"))
        mw = RequestLoggingMiddleware(app=app_mock)  # type: ignore[arg-type]
        scope = _make_scope()

        with pytest.raises(RuntimeError, match="boom"), capture_logs():
            await mw(scope, _noop_receive, _noop_send)

        # Context must be cleared despite the exception
        with capture_logs() as cap:
            structlog.get_logger().info("after.exception")

        assert "request_id" not in cap[0]
