"""Short-lived one-time tickets for SSE authentication.

Flow:
  1. Client calls POST /v1/events/sse-ticket with Bearer JWT → gets ticket
  2. Client opens EventSource(/v1/events/stream?ticket=<ticket>)
  3. Server redeems ticket (atomic GET+DELETE) → extracts user_id
  4. Ticket expires after TTL or first use (whichever comes first)
"""

from __future__ import annotations

from uuid import UUID

import structlog

from src.core.redis import get_redis_client
from src.core.uid import new_id

logger = structlog.get_logger(__name__)

SSE_TICKET_PREFIX = "sse_ticket:"


class SSETicketService:
    """Manages one-time SSE connection tickets stored in Redis."""

    def __init__(self, ttl_seconds: int = 30) -> None:
        self._ttl = ttl_seconds

    async def create_ticket(self, user_id: UUID) -> str:
        """Create a one-time ticket for SSE connection.

        Returns:
            Opaque ticket string.
        """
        ticket = str(new_id())
        key = f"{SSE_TICKET_PREFIX}{ticket}"

        client = get_redis_client()
        try:
            await client.setex(key, self._ttl, str(user_id))
        finally:
            await client.aclose()

        logger.debug("sse_ticket.created", user_id=str(user_id), ttl=self._ttl)
        return ticket

    async def redeem_ticket(self, ticket: str) -> UUID | None:
        """Redeem a ticket, returning the user_id if valid.

        Atomic GET+DELETE ensures one-time use.

        Returns:
            User UUID if ticket is valid, None otherwise.
        """
        key = f"{SSE_TICKET_PREFIX}{ticket}"

        client = get_redis_client()
        try:
            # GETDEL is atomic — gets value and deletes key in one round-trip
            value = await client.getdel(key)
        finally:
            await client.aclose()

        if value is None:
            logger.debug("sse_ticket.invalid_or_expired", ticket=ticket[:8])
            return None

        user_id = UUID(value)
        logger.debug("sse_ticket.redeemed", user_id=str(user_id))
        return user_id
