"""Unit tests for SSETicketService — mocked Redis client."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch
from uuid import UUID, uuid4

import pytest

from src.api.services.sse_ticket import SSE_TICKET_PREFIX, SSETicketService


@pytest.fixture
def service() -> SSETicketService:
    return SSETicketService(ttl_seconds=30)


@pytest.fixture
def user_id() -> UUID:
    return uuid4()


class TestCreateTicket:
    @patch("src.api.services.sse_ticket.get_redis_client")
    async def test_create_ticket_stores_user_id_with_ttl(
        self, mock_get_client, service: SSETicketService, user_id: UUID
    ) -> None:
        mock_client = AsyncMock()
        mock_get_client.return_value = mock_client

        ticket = await service.create_ticket(user_id)

        assert ticket  # Non-empty string
        mock_client.setex.assert_awaited_once()
        key_arg, ttl_arg, value_arg = mock_client.setex.call_args[0]
        assert key_arg == f"{SSE_TICKET_PREFIX}{ticket}"
        assert ttl_arg == 30
        assert value_arg == str(user_id)

    @patch("src.api.services.sse_ticket.get_redis_client")
    async def test_create_ticket_does_not_close_shared_pool(
        self, mock_get_client, service: SSETicketService, user_id: UUID
    ) -> None:
        mock_client = AsyncMock()
        mock_get_client.return_value = mock_client

        await service.create_ticket(user_id)

        # aclose must NOT be called — closing the client would tear down the shared pool
        mock_client.aclose.assert_not_awaited()

    @patch("src.api.services.sse_ticket.get_redis_client")
    async def test_create_ticket_returns_unique_tickets(
        self, mock_get_client, service: SSETicketService, user_id: UUID
    ) -> None:
        mock_client = AsyncMock()
        mock_get_client.return_value = mock_client

        ticket1 = await service.create_ticket(user_id)
        ticket2 = await service.create_ticket(user_id)

        assert ticket1 != ticket2


class TestRedeemTicket:
    @patch("src.api.services.sse_ticket.get_redis_client")
    async def test_redeem_valid_ticket_returns_user_id(
        self, mock_get_client, service: SSETicketService, user_id: UUID
    ) -> None:
        mock_client = AsyncMock()
        mock_client.getdel.return_value = str(user_id)
        mock_get_client.return_value = mock_client

        ticket = "some-ticket-value"
        result = await service.redeem_ticket(ticket)

        assert result == user_id
        mock_client.getdel.assert_awaited_once_with(f"{SSE_TICKET_PREFIX}{ticket}")

    @patch("src.api.services.sse_ticket.get_redis_client")
    async def test_redeem_unknown_ticket_returns_none(
        self, mock_get_client, service: SSETicketService
    ) -> None:
        mock_client = AsyncMock()
        mock_client.getdel.return_value = None
        mock_get_client.return_value = mock_client

        result = await service.redeem_ticket("nonexistent-ticket")

        assert result is None

    @patch("src.api.services.sse_ticket.get_redis_client")
    async def test_redeem_ticket_does_not_close_shared_pool(
        self, mock_get_client, service: SSETicketService, user_id: UUID
    ) -> None:
        mock_client = AsyncMock()
        mock_client.getdel.return_value = str(user_id)
        mock_get_client.return_value = mock_client

        await service.redeem_ticket("ticket")

        # aclose must NOT be called — closing the client would tear down the shared pool
        mock_client.aclose.assert_not_awaited()

    @patch("src.api.services.sse_ticket.get_redis_client")
    async def test_redeem_uses_getdel_for_atomic_one_time_use(
        self, mock_get_client, service: SSETicketService, user_id: UUID
    ) -> None:
        """GETDEL is atomic — ensures ticket cannot be reused."""
        call_count = 0

        async def getdel_side_effect(key: str) -> str | None:  # noqa: ARG001
            nonlocal call_count
            call_count += 1
            # First call: return value (ticket valid); second call: None (already deleted)
            return str(user_id) if call_count == 1 else None

        mock_client = AsyncMock()
        mock_client.getdel.side_effect = getdel_side_effect
        mock_get_client.return_value = mock_client

        ticket = "one-time-ticket"
        first = await service.redeem_ticket(ticket)
        second = await service.redeem_ticket(ticket)

        assert first == user_id
        assert second is None

    @patch("src.api.services.sse_ticket.get_redis_client")
    async def test_create_then_redeem_roundtrip(
        self, mock_get_client, service: SSETicketService, user_id: UUID
    ) -> None:
        """Integration-style: create a ticket and verify the stored key/value."""
        stored: dict[str, str] = {}

        async def setex_side_effect(key: str, ttl: int, value: str) -> None:  # noqa: ARG001
            stored[key] = value

        async def getdel_side_effect(key: str) -> str | None:
            return stored.pop(key, None)

        mock_client = AsyncMock()
        mock_client.setex.side_effect = setex_side_effect
        mock_client.getdel.side_effect = getdel_side_effect
        mock_get_client.return_value = mock_client

        ticket = await service.create_ticket(user_id)
        result = await service.redeem_ticket(ticket)
        assert result == user_id

        # Second redemption returns None
        second = await service.redeem_ticket(ticket)
        assert second is None
