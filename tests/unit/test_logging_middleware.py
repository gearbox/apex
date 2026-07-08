"""Tests for RequestLoggingMiddleware."""

from __future__ import annotations

import uuid
from collections.abc import Generator
from typing import Any, cast
from unittest.mock import AsyncMock

import pytest
import structlog
from litestar.types import Scope
from structlog.contextvars import clear_contextvars, merge_contextvars
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


async def _noop_send(message: Any) -> None:
    pass


class _CaptureSend:
    """Capture ASGI send messages for assertion."""

    def __init__(self) -> None:
        self.messages: list[dict[str, Any]] = []

    async def __call__(self, message: Any) -> None:
        self.messages.append(message)


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

        with capture_logs(processors=[merge_contextvars]) as cap:
            await mw(scope, _noop_receive, _noop_send)

        started = next(r for r in cap if r["event"] == "request.started")
        assert started.get("request_id") == rid

    @pytest.mark.anyio
    async def test_uuid_is_generated_when_header_absent(self) -> None:
        mw, _ = self._build_middleware()
        scope = _make_scope()

        with capture_logs(processors=[merge_contextvars]) as cap:
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
    async def test_context_preserved_on_downstream_exception(self) -> None:
        """On exception, contextvars are preserved for exception handler logging.

        Litestar's ExceptionHandlerMiddleware runs AFTER our middleware's
        except block, so the exception handler needs request_id in context.
        Cleanup happens at the start of the next request (clear_contextvars
        at the top of __call__).
        """
        app_mock = AsyncMock(side_effect=RuntimeError("boom"))
        mw = RequestLoggingMiddleware(app=app_mock)  # type: ignore[arg-type]
        scope = _make_scope()

        with pytest.raises(RuntimeError, match="boom"), capture_logs():
            await mw(scope, _noop_receive, _noop_send)

        # Context is intentionally preserved for exception handlers
        with capture_logs(processors=[merge_contextvars]) as cap:
            structlog.get_logger().info("after.exception")

        assert "request_id" in cap[0]

    @pytest.mark.anyio
    async def test_response_contains_x_request_id_header_when_generated(self) -> None:
        """Generated request_id must appear in the response X-Request-Id header."""
        scope = _make_scope()
        capture = _CaptureSend()

        async def fake_app(_scope: Any, _receive: Any, send: Any) -> None:
            await send({"type": "http.response.start", "status": 200, "headers": []})
            await send({"type": "http.response.body", "body": b""})

        mw = RequestLoggingMiddleware(app=fake_app)  # type: ignore[arg-type]

        with capture_logs():
            await mw(scope, _noop_receive, capture)

        start_msg = next(m for m in capture.messages if m["type"] == "http.response.start")
        header_dict = dict(start_msg["headers"])
        assert b"x-request-id" in header_dict
        # Verify it's a valid UUID (generated by new_id)
        uuid.UUID(header_dict[b"x-request-id"].decode())

    @pytest.mark.anyio
    async def test_response_echoes_client_provided_x_request_id(self) -> None:
        """Client-provided X-Request-Id must be echoed back in the response."""
        rid = "client-trace-id-12345"
        scope = _make_scope(headers=[(b"x-request-id", rid.encode())])
        capture = _CaptureSend()

        async def fake_app(_scope: Any, _receive: Any, send: Any) -> None:
            await send({"type": "http.response.start", "status": 200, "headers": []})
            await send({"type": "http.response.body", "body": b""})

        mw = RequestLoggingMiddleware(app=fake_app)  # type: ignore[arg-type]

        with capture_logs():
            await mw(scope, _noop_receive, capture)

        start_msg = next(m for m in capture.messages if m["type"] == "http.response.start")
        header_dict = dict(start_msg["headers"])
        assert header_dict[b"x-request-id"] == rid.encode()

    @pytest.mark.anyio
    async def test_request_id_stored_in_scope_state(self) -> None:
        """request_id must be accessible via scope['state']['request_id']."""
        scope = _make_scope()
        captured_state: dict[str, Any] = {}

        async def state_capturing_app(scope: Any, _receive: Any, _send: Any) -> None:
            captured_state.update(scope.get("state", {}))

        mw = RequestLoggingMiddleware(app=state_capturing_app)  # type: ignore[arg-type]

        with capture_logs():
            await mw(scope, _noop_receive, _noop_send)

        assert "request_id" in captured_state
        # Verify it's a valid UUID
        uuid.UUID(captured_state["request_id"])

    @pytest.mark.anyio
    async def test_client_request_id_stored_in_scope_state(self) -> None:
        """Client-provided X-Request-Id must be stored in scope state."""
        rid = "my-external-trace-id"
        scope = _make_scope(headers=[(b"x-request-id", rid.encode())])
        captured_state: dict[str, Any] = {}

        async def state_capturing_app(scope: Any, _receive: Any, _send: Any) -> None:
            captured_state.update(scope.get("state", {}))

        mw = RequestLoggingMiddleware(app=state_capturing_app)  # type: ignore[arg-type]

        with capture_logs():
            await mw(scope, _noop_receive, _noop_send)

        assert captured_state["request_id"] == rid

    @pytest.mark.anyio
    async def test_response_header_preserves_existing_headers(self) -> None:
        """X-Request-Id injection must not clobber existing response headers."""
        scope = _make_scope()
        capture = _CaptureSend()

        async def fake_app(_scope: Any, _receive: Any, send: Any) -> None:
            await send(
                {
                    "type": "http.response.start",
                    "status": 200,
                    "headers": [(b"content-type", b"application/json")],
                }
            )
            await send({"type": "http.response.body", "body": b"{}"})

        mw = RequestLoggingMiddleware(app=fake_app)  # type: ignore[arg-type]

        with capture_logs():
            await mw(scope, _noop_receive, capture)

        start_msg = next(m for m in capture.messages if m["type"] == "http.response.start")
        header_dict = dict(start_msg["headers"])
        assert b"content-type" in header_dict
        assert b"x-request-id" in header_dict

    # ------------------------------------------------------------------
    # Fix 1: status_code propagation in request.finished
    # ------------------------------------------------------------------

    @pytest.mark.anyio
    async def test_status_code_200_in_finished_event(self) -> None:
        """request.finished must carry status_code=200 for a normal response."""
        scope = _make_scope()

        async def fake_app(_scope: Any, _receive: Any, send: Any) -> None:
            await send({"type": "http.response.start", "status": 200, "headers": []})
            await send({"type": "http.response.body", "body": b""})

        mw = RequestLoggingMiddleware(app=fake_app)  # type: ignore[arg-type]

        with capture_logs() as cap:
            await mw(scope, _noop_receive, _CaptureSend())

        finished = next(r for r in cap if r["event"] == "request.finished")
        assert finished["status_code"] == 200
        assert finished["log_level"] == "info"

    @pytest.mark.anyio
    async def test_status_code_401_returned_response_in_finished_event(self) -> None:
        """Regression: returned Response(401) must appear in request.finished, not just raised exceptions."""
        scope = _make_scope()

        async def fake_app(_scope: Any, _receive: Any, send: Any) -> None:
            # Simulates a handler that returns _error(...) instead of raising HTTPException
            await send({"type": "http.response.start", "status": 401, "headers": []})
            await send({"type": "http.response.body", "body": b'{"error": "unauthorized"}'})

        mw = RequestLoggingMiddleware(app=fake_app)  # type: ignore[arg-type]

        with capture_logs() as cap:
            await mw(scope, _noop_receive, _CaptureSend())

        finished = next(r for r in cap if r["event"] == "request.finished")
        assert finished["status_code"] == 401
        assert finished["log_level"] == "info"

    @pytest.mark.anyio
    async def test_status_code_500_logs_warning(self) -> None:
        """5xx responses must emit request.finished at warning level."""
        scope = _make_scope()

        async def fake_app(_scope: Any, _receive: Any, send: Any) -> None:
            await send({"type": "http.response.start", "status": 500, "headers": []})
            await send({"type": "http.response.body", "body": b""})

        mw = RequestLoggingMiddleware(app=fake_app)  # type: ignore[arg-type]

        with capture_logs() as cap:
            await mw(scope, _noop_receive, _CaptureSend())

        finished = next(r for r in cap if r["event"] == "request.finished")
        assert finished["status_code"] == 500
        assert finished["log_level"] == "warning"

    @pytest.mark.anyio
    async def test_status_code_none_when_no_response_start(self) -> None:
        """If the app never sends http.response.start, status_code is None (e.g. client disconnect)."""
        scope = _make_scope()

        async def fake_app(_scope: Any, _receive: Any, _send: Any) -> None:
            pass  # never sends http.response.start

        mw = RequestLoggingMiddleware(app=fake_app)  # type: ignore[arg-type]

        with capture_logs() as cap:
            await mw(scope, _noop_receive, _CaptureSend())

        finished = next(r for r in cap if r["event"] == "request.finished")
        assert finished["status_code"] is None
        assert finished["log_level"] == "info"
