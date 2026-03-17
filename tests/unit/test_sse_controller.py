"""Unit tests for SSE controller endpoints."""

from __future__ import annotations

import asyncio
import contextlib
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
    async def test_create_sse_ticket_handler_logic(self, mock_get_client: MagicMock, user_id: UUID) -> None:
        """Simulates what the create_sse_ticket handler does."""
        mock_client = AsyncMock()
        mock_get_client.return_value = mock_client

        service = SSETicketService(ttl_seconds=30)
        ticket = await service.create_ticket(user_id)
        response = SSETicketResponse(ticket=ticket)

        assert response.ticket == ticket
        mock_client.setex.assert_awaited_once()

    @patch("src.api.services.sse_ticket.get_redis_client")
    async def test_stream_invalid_ticket_logic(self, mock_get_client: MagicMock) -> None:
        """Simulates what the stream handler does when ticket is invalid."""
        mock_client = AsyncMock()
        mock_client.getdel.return_value = None
        mock_get_client.return_value = mock_client

        service = SSETicketService(ttl_seconds=30)
        user_id = await service.redeem_ticket("bad-ticket")

        # Handler returns 401 when user_id is None
        assert user_id is None

    @patch("src.api.services.sse_ticket.get_redis_client")
    async def test_stream_valid_ticket_logic(self, mock_get_client: MagicMock, user_id: UUID) -> None:
        """Simulates what the stream handler does with a valid ticket."""
        mock_client = AsyncMock()
        mock_client.getdel.return_value = str(user_id)
        mock_get_client.return_value = mock_client

        service = SSETicketService(ttl_seconds=30)
        result = await service.redeem_ticket("good-ticket")

        # Handler proceeds to open SSE stream when user_id is not None
        assert result == user_id


# ---------------------------------------------------------------------------
# SSE event_generator logic — envelope→SSE mapping and heartbeat
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


async def _run_generator(  # type: ignore[no-untyped-def]
    subscribe_fn,
    *,
    heartbeat_interval: float = 15.0,
) -> list[dict[str, str]]:
    """Run the same event_generator logic used in SSEController.stream."""
    results: list[dict[str, str]] = []
    user_id = uuid4()

    sub = subscribe_fn(user_id)
    sub_iter = sub.__aiter__()

    with contextlib.suppress(asyncio.CancelledError):
        while True:
            try:
                envelope: EventEnvelope = await asyncio.wait_for(
                    sub_iter.__anext__(),
                    timeout=heartbeat_interval,
                )
                results.append(
                    {
                        "event": envelope.event_type.value,
                        "id": envelope.event_id,
                        "data": bytes(envelope.payload).decode(),
                    }
                )
            except asyncio.TimeoutError:
                results.append({"comment": "keepalive"})
                break  # stop after first keepalive so the test terminates
            except StopAsyncIteration:
                break

    return results


class TestSSEEventGenerator:
    """Test the event_generator closure logic from SSEController.stream."""

    async def test_envelope_mapped_to_sse_fields(self) -> None:
        """EventEnvelope from EventBus.subscribe is mapped to SSE event/id/data."""
        envelope = _make_envelope(EventType.JOB_STATUS_CHANGED)

        async def mock_subscribe(user_id):  # type: ignore[no-untyped-def]
            yield envelope

        results = await _run_generator(mock_subscribe)

        assert len(results) == 1
        assert results[0]["event"] == EventType.JOB_STATUS_CHANGED.value
        assert results[0]["id"] == envelope.event_id
        assert results[0]["data"] == bytes(envelope.payload).decode()

    async def test_multiple_envelopes_all_forwarded(self) -> None:
        """Each yielded envelope becomes one SSE dict."""
        envelopes = [_make_envelope() for _ in range(3)]
        for i, env in enumerate(envelopes):
            object.__setattr__(env, "event_id", f"evt-{i}")

        async def mock_subscribe(user_id):  # type: ignore[no-untyped-def]
            for env in envelopes:
                yield env

        results = await _run_generator(mock_subscribe)

        assert len(results) == 3
        for i, result in enumerate(results):
            assert result["event"] == EventType.JOB_STATUS_CHANGED.value

    async def test_timeout_yields_keepalive_comment(self) -> None:
        """When wait_for times out, a keepalive comment is produced."""

        async def never_yields(user_id):  # type: ignore[no-untyped-def]
            # An async generator that never yields — causes wait_for to time out
            await asyncio.sleep(10)
            if False:  # pragma: no cover
                yield

        results = await _run_generator(never_yields, heartbeat_interval=0.01)

        assert results == [{"comment": "keepalive"}]
