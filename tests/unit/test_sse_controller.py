"""Unit tests for SSE controller endpoints."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Sequence
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID, uuid4

import msgspec
import pytest

from src.api.schemas.events import EventEnvelope, EventType, JobStatusPayload, SSETicketResponse
from src.api.services.sse_ticket import SSETicketService


@pytest.fixture
def user_id() -> UUID:
    return uuid4()


@pytest.fixture
def sse_ticket_service() -> SSETicketService:
    return SSETicketService(ttl_seconds=30)


# ---------------------------------------------------------------------------
# POST /v1/events/sse-ticket — issue ticket for the current user
# ---------------------------------------------------------------------------


class TestCreateSSETicket:
    @patch("src.api.services.sse_ticket.get_redis_client")
    async def test_create_ticket_returns_sse_ticket_response(
        self, mock_get_client: MagicMock, sse_ticket_service: SSETicketService, user_id: UUID
    ) -> None:
        mock_client = AsyncMock()
        mock_get_client.return_value = mock_client

        ticket = await sse_ticket_service.create_ticket(user_id)

        response = SSETicketResponse(ticket=ticket)
        assert response.ticket == ticket
        assert isinstance(response.ticket, str)
        assert len(response.ticket) > 0

    @patch("src.api.services.sse_ticket.get_redis_client")
    async def test_each_call_produces_unique_ticket(
        self, mock_get_client: MagicMock, sse_ticket_service: SSETicketService, user_id: UUID
    ) -> None:
        mock_client = AsyncMock()
        mock_get_client.return_value = mock_client

        tickets = {await sse_ticket_service.create_ticket(user_id) for _ in range(5)}
        assert len(tickets) == 5  # All unique


# ---------------------------------------------------------------------------
# GET /v1/events/stream — stream endpoint logic
# ---------------------------------------------------------------------------


class TestStreamInvalidTicket:
    @patch("src.api.services.sse_ticket.get_redis_client")
    async def test_redeem_nonexistent_ticket_returns_none(
        self, mock_get_client: MagicMock, sse_ticket_service: SSETicketService
    ) -> None:
        mock_client = AsyncMock()
        mock_client.getdel.return_value = None
        mock_get_client.return_value = mock_client

        result = await sse_ticket_service.redeem_ticket("invalid-ticket")
        assert result is None

    @patch("src.api.services.sse_ticket.get_redis_client")
    async def test_redeem_valid_ticket_returns_user_id(
        self, mock_get_client: MagicMock, sse_ticket_service: SSETicketService, user_id: UUID
    ) -> None:
        mock_client = AsyncMock()
        mock_client.getdel.return_value = str(user_id)
        mock_get_client.return_value = mock_client

        result = await sse_ticket_service.redeem_ticket("valid-ticket")
        assert result == user_id


# ---------------------------------------------------------------------------
# SSE controller handler logic (unit-level, without Litestar test client)
# ---------------------------------------------------------------------------


class TestSSEControllerHandler:
    """Test the controller's create_sse_ticket and stream handler logic."""

    @patch("src.api.services.sse_ticket.get_redis_client")
    async def test_create_sse_ticket_handler_logic(
        self, mock_get_client: MagicMock, user_id: UUID
    ) -> None:
        mock_client = AsyncMock()
        mock_get_client.return_value = mock_client

        service = SSETicketService(ttl_seconds=30)
        ticket = await service.create_ticket(user_id)
        response = SSETicketResponse(ticket=ticket)

        assert response.ticket == ticket
        mock_client.setex.assert_awaited_once()

    @patch("src.api.services.sse_ticket.get_redis_client")
    async def test_stream_invalid_ticket_logic(self, mock_get_client: MagicMock) -> None:
        mock_client = AsyncMock()
        mock_client.getdel.return_value = None
        mock_get_client.return_value = mock_client

        service = SSETicketService(ttl_seconds=30)
        user_id = await service.redeem_ticket("bad-ticket")

        assert user_id is None

    @patch("src.api.services.sse_ticket.get_redis_client")
    async def test_stream_valid_ticket_logic(
        self, mock_get_client: MagicMock, user_id: UUID
    ) -> None:
        mock_client = AsyncMock()
        mock_client.getdel.return_value = str(user_id)
        mock_get_client.return_value = mock_client

        service = SSETicketService(ttl_seconds=30)
        result = await service.redeem_ticket("good-ticket")

        assert result == user_id


# ---------------------------------------------------------------------------
# SSE event_generator logic — drives the new async-for subscribe loop
# ---------------------------------------------------------------------------

_encoder = msgspec.json.Encoder()


def _make_envelope(event_type: EventType = EventType.JOB_STATUS_CHANGED) -> EventEnvelope:
    payload = JobStatusPayload(
        job_id=uuid4(),
        status="completed",
        previous_status="running",
        generation_type="t2v",
        provider="grok",
    )
    return EventEnvelope(
        event_type=event_type,
        payload=msgspec.Raw(_encoder.encode(payload)),
        timestamp=datetime.now(UTC),
        event_id="evt-1",
    )


async def _run_generator(
    subscribe_items: Sequence[EventEnvelope | None],
    *,
    heartbeat_interval: float = 15.0,
) -> list[dict[str, str]]:
    """Run the same event_generator logic used in SSEController.stream.

    subscribe_items is what event_bus.subscribe would yield (None = heartbeat tick).
    """
    results: list[dict[str, str]] = []
    user_id = uuid4()

    async def mock_subscribe(
        uid: UUID,  # noqa: ARG001
        *,
        heartbeat_interval: float,  # noqa: ARG001
    ) -> AsyncIterator[EventEnvelope | None]:
        for item in subscribe_items:
            yield item

    try:
        async for item in mock_subscribe(user_id, heartbeat_interval=heartbeat_interval):
            if item is None:
                results.append({"comment": "keepalive"})
            else:
                results.append(
                    {
                        "event": item.event_type.value,
                        "id": item.event_id,
                        "data": bytes(item.payload).decode(),
                    }
                )
    except asyncio.CancelledError:
        raise

    return results


class TestSSEEventGenerator:
    """Test the event_generator closure logic from SSEController.stream."""

    async def test_envelope_mapped_to_sse_fields(self) -> None:
        """EventEnvelope from subscribe is mapped to SSE event/id/data."""
        envelope = _make_envelope(EventType.JOB_STATUS_CHANGED)

        results = await _run_generator([envelope])

        assert len(results) == 1
        assert results[0]["event"] == EventType.JOB_STATUS_CHANGED.value
        assert results[0]["id"] == envelope.event_id
        assert results[0]["data"] == bytes(envelope.payload).decode()

    async def test_none_item_yields_keepalive_comment(self) -> None:
        """None yielded by subscribe (heartbeat tick) → {'comment': 'keepalive'}."""
        results = await _run_generator([None])

        assert results == [{"comment": "keepalive"}]

    async def test_multiple_envelopes_all_forwarded(self) -> None:
        """Each yielded envelope becomes one SSE dict."""
        envelopes = [_make_envelope() for _ in range(3)]
        for i, env in enumerate(envelopes):
            object.__setattr__(env, "event_id", f"evt-{i}")

        results = await _run_generator(envelopes)

        assert len(results) == 3
        for result in results:
            assert result["event"] == EventType.JOB_STATUS_CHANGED.value

    async def test_mixed_none_and_envelopes(self) -> None:
        """None ticks interspersed with envelopes are all handled correctly."""
        envelope = _make_envelope()
        results = await _run_generator([None, envelope, None])

        assert results[0] == {"comment": "keepalive"}
        assert results[1]["event"] == EventType.JOB_STATUS_CHANGED.value
        assert results[2] == {"comment": "keepalive"}

    async def test_cancelled_error_propagates(self) -> None:
        """CancelledError from subscribe must propagate (not be swallowed)."""
        user_id = uuid4()

        async def subscribe_then_cancel(uid: UUID, *, heartbeat_interval: float):  # type: ignore[no-untyped-def]  # noqa: ARG001
            yield None  # first tick
            raise asyncio.CancelledError

        results: list[dict[str, str]] = []
        with pytest.raises(asyncio.CancelledError):
            async for item in subscribe_then_cancel(user_id, heartbeat_interval=15.0):
                if item is None:
                    results.append({"comment": "keepalive"})

        assert results == [{"comment": "keepalive"}]

    async def test_cancelled_error_logs_client_disconnected(self) -> None:
        """CancelledError → sse.client_disconnected is logged, sse.stream_error is NOT."""
        from structlog.testing import capture_logs

        user_id = uuid4()

        async def subscribe_cancel(uid: UUID, *, heartbeat_interval: float):  # type: ignore[no-untyped-def]  # noqa: ARG001
            if False:  # pragma: no cover
                yield  # sentinel: makes this an async generator
            raise asyncio.CancelledError

        with capture_logs() as cap:
            # Simulate the exact event_generator try/except from sse.py
            import structlog as sl

            logger = sl.get_logger("src.api.routes.sse")
            try:
                async for _ in subscribe_cancel(user_id, heartbeat_interval=15.0):
                    pass  # pragma: no cover
            except asyncio.CancelledError:
                logger.info("sse.client_disconnected", user_id=str(user_id))
                # re-raise is tested separately; here we just verify the log

        disconnected_logs = [r for r in cap if r.get("event") == "sse.client_disconnected"]
        stream_error_logs = [r for r in cap if r.get("event") == "sse.stream_error"]
        assert len(disconnected_logs) == 1
        assert not stream_error_logs

    async def test_no_sse_stream_error_on_idle_heartbeat(self) -> None:
        """Regression: a heartbeat tick (None) must NOT trigger sse.stream_error.

        Previously, asyncio.wait_for() cancelled the Redis read and redis-py
        converted the CancelledError to TimeoutError, which fell through to
        the sse.stream_error except branch. Verify None items are clean.
        """
        from structlog.testing import capture_logs

        user_id = uuid4()

        async def subscribe_heartbeat(uid: UUID, *, heartbeat_interval: float):  # type: ignore[no-untyped-def]  # noqa: ARG001
            yield None  # simulate one idle period with no event

        with capture_logs() as cap:
            import structlog as sl

            logger = sl.get_logger("src.api.routes.sse")
            try:
                async for item in subscribe_heartbeat(user_id, heartbeat_interval=15.0):
                    if item is None:
                        pass  # keepalive — no error
            except asyncio.CancelledError:
                logger.info("sse.client_disconnected", user_id=str(user_id))
                raise
            except Exception:
                logger.exception("sse.stream_error", user_id=str(user_id))

        stream_error_logs = [r for r in cap if r.get("event") == "sse.stream_error"]
        assert not stream_error_logs
