"""SSE (Server-Sent Events) endpoints for real-time updates."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, Sequence
from uuid import UUID

import msgspec
import structlog
from litestar import Controller, Response, get, post
from litestar.di import Provide
from litestar.response import ServerSentEvent
from litestar.status_codes import HTTP_401_UNAUTHORIZED

from src.api.dependencies.auth import get_current_user_id
from src.api.schemas.errors import ErrorEnvelope
from src.api.schemas.events import (
    EventEnvelope,
    SSETicketResponse,
)
from src.api.security import auth_guard
from src.api.services.event_bus import EventBus
from src.api.services.sse_ticket import SSETicketService
from src.core.config import get_settings

logger = structlog.get_logger(__name__)

_encoder = msgspec.json.Encoder()


class SSEController(Controller):
    """Server-Sent Events endpoints."""

    path = "/v1/events"
    tags: Sequence[str] | None = ["Events"]

    @post(
        "/sse-ticket",
        guards=[auth_guard],
        dependencies={"current_user_id": Provide(get_current_user_id)},
    )
    async def create_sse_ticket(
        self,
        current_user_id: UUID,
        sse_ticket_service: SSETicketService,
    ) -> SSETicketResponse:
        """Issue a short-lived one-time ticket for SSE connection.

        The ticket is valid for ~30 seconds and single-use.
        Pass it as `?ticket=<value>` when opening the EventSource.
        """
        ticket = await sse_ticket_service.create_ticket(current_user_id)
        return SSETicketResponse(ticket=ticket)

    @get("/stream")
    async def stream(
        self,
        ticket: str,
        sse_ticket_service: SSETicketService,
        event_bus: EventBus,
    ) -> ServerSentEvent | Response[ErrorEnvelope]:
        """SSE stream of real-time events for the authenticated user.

        Auth: pass `?ticket=<ticket>` obtained from POST /v1/events/sse-ticket.
        No Bearer header required (EventSource limitation).
        """
        user_id = await sse_ticket_service.redeem_ticket(ticket)
        if user_id is None:
            return Response(
                content=ErrorEnvelope(
                    error="invalid_ticket",
                    message="SSE ticket is invalid, expired, or already used.",
                    status_code=HTTP_401_UNAUTHORIZED,
                ),
                status_code=HTTP_401_UNAUTHORIZED,
            )

        settings = get_settings()

        async def event_generator() -> AsyncGenerator[dict[str, str]]:
            """Yield SSE-formatted events from the EventBus."""
            heartbeat_interval = settings.sse_heartbeat_interval

            try:
                sub = event_bus.subscribe(user_id)
                sub_iter = sub.__aiter__()

                while True:
                    try:
                        envelope: EventEnvelope = await asyncio.wait_for(
                            sub_iter.__anext__(),
                            timeout=float(heartbeat_interval),
                        )
                        # Forward the pre-encoded inner payload directly
                        yield {
                            "event": envelope.event_type.value,
                            "id": envelope.event_id,
                            "data": bytes(envelope.payload).decode(),
                        }
                    except TimeoutError:
                        # Send SSE comment as keepalive
                        yield {"comment": "keepalive"}
                    except StopAsyncIteration:
                        break

            except asyncio.CancelledError:
                logger.info("sse.client_disconnected", user_id=str(user_id))
            except Exception:
                logger.exception("sse.stream_error", user_id=str(user_id))

        return ServerSentEvent(content=event_generator())
