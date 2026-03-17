"""Unit tests for SSE controller endpoints."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest

from src.api.schemas.events import SSETicketResponse
from src.api.services.sse_ticket import SSETicketService


@pytest.fixture
def user_id():  # type: ignore[no-untyped-def]
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
        self, mock_get_client, sse_ticket_service: SSETicketService, user_id
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
        self, mock_get_client, sse_ticket_service: SSETicketService, user_id
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
        self, mock_get_client, sse_ticket_service: SSETicketService
    ) -> None:
        mock_client = AsyncMock()
        mock_client.getdel.return_value = None
        mock_get_client.return_value = mock_client

        result = await sse_ticket_service.redeem_ticket("invalid-ticket")
        assert result is None

    @patch("src.api.services.sse_ticket.get_redis_client")
    async def test_redeem_valid_ticket_returns_user_id(
        self, mock_get_client, sse_ticket_service: SSETicketService, user_id
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
    async def test_create_sse_ticket_handler_logic(self, mock_get_client, user_id) -> None:
        """Simulates what the create_sse_ticket handler does."""
        mock_client = AsyncMock()
        mock_get_client.return_value = mock_client

        service = SSETicketService(ttl_seconds=30)
        ticket = await service.create_ticket(user_id)
        response = SSETicketResponse(ticket=ticket)

        assert response.ticket == ticket
        mock_client.setex.assert_awaited_once()

    @patch("src.api.services.sse_ticket.get_redis_client")
    async def test_stream_invalid_ticket_logic(self, mock_get_client) -> None:
        """Simulates what the stream handler does when ticket is invalid."""
        mock_client = AsyncMock()
        mock_client.getdel.return_value = None
        mock_get_client.return_value = mock_client

        service = SSETicketService(ttl_seconds=30)
        user_id = await service.redeem_ticket("bad-ticket")

        # Handler returns 401 when user_id is None
        assert user_id is None

    @patch("src.api.services.sse_ticket.get_redis_client")
    async def test_stream_valid_ticket_logic(self, mock_get_client, user_id) -> None:
        """Simulates what the stream handler does with a valid ticket."""
        mock_client = AsyncMock()
        mock_client.getdel.return_value = str(user_id)
        mock_get_client.return_value = mock_client

        service = SSETicketService(ttl_seconds=30)
        result = await service.redeem_ticket("good-ticket")

        # Handler proceeds to open SSE stream when user_id is not None
        assert result == user_id
