"""Web Push subscription endpoints.

GET    /v1/push/vapid-public-key   — public key for pushManager.subscribe()
POST   /v1/push/subscriptions      — register/upsert a browser subscription
DELETE /v1/push/subscriptions      — unregister a subscription (idempotent)
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from uuid import UUID

from litestar import Controller, Response, delete, get, post
from litestar.di import Provide
from litestar.status_codes import HTTP_204_NO_CONTENT, HTTP_503_SERVICE_UNAVAILABLE
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.dependencies.auth import get_current_user_id
from src.api.schemas.errors import ErrorEnvelope
from src.api.schemas.push import (
    PushSubscriptionDeleteRequest,
    PushSubscriptionRequest,
    PushSubscriptionResponse,
    VapidPublicKeyResponse,
)
from src.api.security import auth_guard
from src.api.services.push import PushService
from src.core.config import get_settings

if TYPE_CHECKING:
    from collections.abc import Sequence


def _push_disabled_response() -> Response[ErrorEnvelope]:
    return Response(
        content=ErrorEnvelope(
            error="service_unavailable",
            message="Push notifications are not available.",
            status_code=HTTP_503_SERVICE_UNAVAILABLE,
        ),
        status_code=HTTP_503_SERVICE_UNAVAILABLE,
    )


class PushController(Controller):
    """Web Push subscription management."""

    path = "/v1/push"
    tags: Sequence[str] | None = ("Push",)
    dependencies = {"current_user_id": Provide(get_current_user_id)}  # noqa: RUF012

    @get("/vapid-public-key", guards=[auth_guard])
    async def get_vapid_public_key(
        self,
        current_user_id: UUID,  # noqa: ARG002
    ) -> VapidPublicKeyResponse | Response[ErrorEnvelope]:
        """Return the VAPID public key for ``pushManager.subscribe()``."""
        settings = get_settings()
        if not settings.push_enabled or settings.vapid_public_key is None:
            return _push_disabled_response()
        return VapidPublicKeyResponse(public_key=settings.vapid_public_key)

    @post("/subscriptions", guards=[auth_guard])
    async def create_subscription(
        self,
        data: PushSubscriptionRequest,
        current_user_id: UUID,
        product_id: str,
        session: AsyncSession,
        push_service: PushService,
    ) -> PushSubscriptionResponse:
        """Register or upsert a browser push subscription for the current user.

        ``push_service`` resolution itself raises 503 when push is disabled
        (see ``get_push_service``), so no separate gate is needed here.
        """
        subscription = await push_service.upsert_subscription(
            user_id=current_user_id,
            product_id=product_id,
            endpoint=data.endpoint,
            p256dh=data.keys.p256dh,
            auth=data.keys.auth,
            user_agent=data.user_agent,
            session=session,
        )
        await session.commit()
        return PushSubscriptionResponse(
            id=subscription.id,
            endpoint=subscription.endpoint,
            created_at=subscription.created_at,
        )

    @delete(
        "/subscriptions",
        status_code=HTTP_204_NO_CONTENT,
        guards=[auth_guard],
    )
    async def delete_subscription(
        self,
        data: PushSubscriptionDeleteRequest,
        current_user_id: UUID,
        session: AsyncSession,
        push_service: PushService,
    ) -> None:
        """Unregister a push subscription. Idempotent — 204 even if not found.

        ``push_service`` resolution itself raises 503 when push is disabled
        (see ``get_push_service``), so no separate gate is needed here.
        """
        await push_service.delete_subscription(
            user_id=current_user_id,
            endpoint=data.endpoint,
            session=session,
        )
        await session.commit()
