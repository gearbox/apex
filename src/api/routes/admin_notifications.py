"""Admin ops-notification preference + Telegram link routes.

All endpoints require ADMIN or SUPERADMIN role, except reading another
admin's preferences which is superadmin-only.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from uuid import UUID

import structlog
from litestar import Controller, delete, get, post, put
from litestar.di import Provide
from litestar.exceptions import HTTPException, ValidationException
from litestar.status_codes import HTTP_503_SERVICE_UNAVAILABLE
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.dependencies.auth import get_current_admin_user, get_current_superadmin_user
from src.api.schemas.admin_notifications import (
    NotificationClassInfo,
    NotificationPreferenceItem,
    NotificationPreferencesResponse,
    NotificationPreferencesUpdateRequest,
    TelegramLinkInviteResponse,
    TelegramLinkStatusResponse,
)
from src.api.security import auth_guard
from src.api.services.admin_notifications import AdminNotificationService, TelegramLinkError
from src.db.models import User

if TYPE_CHECKING:
    from collections.abc import Sequence

    from src.db.models.admin_notifications import AdminNotificationPreference

logger = structlog.get_logger(__name__)


def _to_preferences_response(
    prefs: Sequence[AdminNotificationPreference],
) -> NotificationPreferencesResponse:
    return NotificationPreferencesResponse(
        items=[
            NotificationPreferenceItem(
                notification_class=p.notification_class,
                min_interval_seconds=p.min_interval_seconds,
            )
            for p in prefs
        ]
    )


class AdminNotificationController(Controller):
    """Admin ops-notification preferences and Telegram linking."""

    path = "/v1/admin/notifications"
    tags: Sequence[str] | None = ("Admin Notifications",)
    guards = [auth_guard]  # noqa: RUF012
    dependencies = {  # noqa: RUF012
        "admin": Provide(get_current_admin_user),
        "superadmin": Provide(get_current_superadmin_user),
    }

    @get("/classes")
    async def get_classes(
        self,
        admin: User,  # noqa: ARG002
        admin_notification_service: AdminNotificationService,
    ) -> list[NotificationClassInfo]:
        return admin_notification_service.get_class_catalog()

    @get("/preferences")
    async def get_preferences(
        self,
        admin: User,
        session: AsyncSession,
        admin_notification_service: AdminNotificationService,
    ) -> NotificationPreferencesResponse:
        prefs = await admin_notification_service.get_preferences(admin.id, session=session)
        return _to_preferences_response(prefs)

    @put("/preferences")
    async def replace_preferences(
        self,
        admin: User,
        data: NotificationPreferencesUpdateRequest,
        session: AsyncSession,
        product_id: str,
        admin_notification_service: AdminNotificationService,
    ) -> NotificationPreferencesResponse:
        try:
            await admin_notification_service.replace_preferences(
                admin.id, product_id, data.items, session=session
            )
            await session.commit()
        except ValueError as exc:
            raise ValidationException(detail=str(exc)) from exc

        prefs = await admin_notification_service.get_preferences(admin.id, session=session)
        return _to_preferences_response(prefs)

    @get("/preferences/{user_id:uuid}")
    async def get_preferences_for(
        self,
        superadmin: User,  # noqa: ARG002
        user_id: UUID,
        session: AsyncSession,
        admin_notification_service: AdminNotificationService,
    ) -> NotificationPreferencesResponse:
        prefs = await admin_notification_service.get_preferences_for(user_id, session=session)
        return _to_preferences_response(prefs)

    @get("/telegram")
    async def get_telegram_status(
        self,
        admin: User,
        session: AsyncSession,
        admin_notification_service: AdminNotificationService,
    ) -> TelegramLinkStatusResponse:
        linked, linked_at, chat_id_last4 = await admin_notification_service.get_link_status(
            admin.id, session=session
        )
        return TelegramLinkStatusResponse(
            linked=linked, linked_at=linked_at, chat_id_last4=chat_id_last4
        )

    @post("/telegram/link")
    async def create_telegram_link(
        self,
        admin: User,
        session: AsyncSession,
        product_id: str,
        admin_notification_service: AdminNotificationService,
    ) -> TelegramLinkInviteResponse:
        try:
            invite = await admin_notification_service.create_link_token(
                admin.id, product_id, session=session
            )
            await session.commit()
        except TelegramLinkError as exc:
            raise HTTPException(status_code=HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc

        return TelegramLinkInviteResponse(deep_link=invite.deep_link, expires_at=invite.expires_at)

    @delete("/telegram", status_code=200)
    async def delete_telegram_link(
        self,
        admin: User,
        session: AsyncSession,
        product_id: str,
        admin_notification_service: AdminNotificationService,
    ) -> dict[str, str]:
        await admin_notification_service.unlink(admin.id, product_id, session=session)
        await session.commit()
        return {"message": "Telegram link removed"}
