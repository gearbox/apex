"""Service for admin ops-notification preferences and the Telegram link flow.

Request-scoped, never commits — Litestar DI transaction ownership, same as
every other request-scoped service (repos flush, routes commit).
"""

from __future__ import annotations

import secrets
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

import structlog

from src.api.services.admin_management import AdminManagementError
from src.core.enums import PLATFORM_SCOPED_NOTIFICATION_CLASSES, NotificationClass
from src.core.uid import new_id
from src.db.models.admin import AdminAuditLog
from src.db.repositories.admin import AdminRepository
from src.db.repositories.admin_notifications import AdminNotificationRepository
from src.db.repositories.user import UserRepository

if TYPE_CHECKING:
    from collections.abc import Sequence
    from uuid import UUID

    from sqlalchemy.ext.asyncio import AsyncSession

    from src.api.schemas.admin_notifications import (
        NotificationClassInfo,
        NotificationPreferenceItem,
    )
    from src.api.services.telegram.sender import TelegramSender
    from src.db.models.admin_notifications import AdminNotificationPreference

logger = structlog.get_logger(__name__)

_MAX_MIN_INTERVAL_SECONDS = 86400
# String(64) column — leave headroom under the wire limit for the token itself.
_LINK_TOKEN_BYTES = 32

_CATALOG_DESCRIPTIONS: dict[NotificationClass, str] = {
    NotificationClass.USER_REGISTERED: "A new user registered on your product.",
    NotificationClass.GENERATION_CREATED: "A new generation job was submitted.",
    NotificationClass.GPU_NODE_STARTED: "A new GPU node finished provisioning.",
    NotificationClass.GENERATION_FAILED: "A generation job failed.",
    NotificationClass.HEALTH_DEGRADED: "A platform subsystem became degraded or unhealthy.",
    NotificationClass.HEALTH_RESTORED: "A platform subsystem recovered.",
    NotificationClass.TOKEN_REVOCATION_FAILED: (
        "A bulk access-token revocation failed to write to Redis — a user's "
        "existing access tokens/content cookies remain valid until they expire."
    ),
}
# Fail at import time, not at GET /v1/admin/notifications/classes request
# time: get_class_catalog() indexes this dict by every NotificationClass
# member, so a member added without its description here would otherwise
# surface as a 500 on that endpoint instead of a startup failure.
if set(_CATALOG_DESCRIPTIONS) != set(NotificationClass):
    raise RuntimeError("_CATALOG_DESCRIPTIONS is missing an entry for a NotificationClass member")


class AdminNotificationError(Exception):
    """Base error for admin notification operations."""


class TelegramLinkError(AdminNotificationError):
    """Telegram is not configured, or the Telegram API call failed."""


@dataclass(frozen=True)
class LinkInvite:
    deep_link: str
    expires_at: datetime


class AdminNotificationService:
    """Manages notification-class preferences and the Telegram deep-link flow."""

    def __init__(
        self,
        *,
        sender: TelegramSender | None,
        link_token_ttl_seconds: int,
    ) -> None:
        self._sender = sender
        self._link_token_ttl_seconds = link_token_ttl_seconds
        self._bot_username: str | None = None

    # ------------------------------------------------------------------
    # Class catalog
    # ------------------------------------------------------------------

    def get_class_catalog(self) -> list[NotificationClassInfo]:
        from src.api.schemas.admin_notifications import NotificationClassInfo

        return [
            NotificationClassInfo(
                notification_class=cls.value,
                scope="platform" if cls in PLATFORM_SCOPED_NOTIFICATION_CLASSES else "product",
                description=_CATALOG_DESCRIPTIONS[cls],
            )
            for cls in NotificationClass
        ]

    # ------------------------------------------------------------------
    # Preferences
    # ------------------------------------------------------------------

    async def get_preferences(
        self, user_id: UUID, *, session: AsyncSession
    ) -> Sequence[AdminNotificationPreference]:
        return await AdminNotificationRepository(session).list_preferences(user_id)

    async def get_preferences_for(
        self, target_user_id: UUID, product_id: str, *, session: AsyncSession
    ) -> Sequence[AdminNotificationPreference]:
        """Superadmin-only read-only view of another admin's preferences.

        Raises:
            AdminManagementError: Target user not found, or belongs to a
                different product — same convention as
                ``AdminManagementService._get_target_user``.
        """
        user = await UserRepository(session).get_active_user(target_user_id)
        if user is None or user.product_id != product_id:
            raise AdminManagementError(f"User {target_user_id} not found in product {product_id}")
        return await AdminNotificationRepository(session).list_preferences(target_user_id)

    async def replace_preferences(
        self,
        user_id: UUID,
        product_id: str,
        items: Sequence[NotificationPreferenceItem],
        *,
        session: AsyncSession,
    ) -> None:
        """Validate and full-set-replace a user's notification preferences.

        Raises:
            ValueError: An unknown notification_class, a duplicate
                notification_class, or min_interval_seconds outside [0, 86400].
        """
        valid_classes = {cls.value for cls in NotificationClass}
        seen_classes: set[str] = set()
        prefs: list[tuple[str, int]] = []
        for item in items:
            if item.notification_class not in valid_classes:
                raise ValueError(f"Unknown notification_class '{item.notification_class}'")
            if item.notification_class in seen_classes:
                raise ValueError(f"Duplicate notification_class '{item.notification_class}'")
            seen_classes.add(item.notification_class)
            if not 0 <= item.min_interval_seconds <= _MAX_MIN_INTERVAL_SECONDS:
                raise ValueError(
                    "min_interval_seconds must be in [0, 86400] "
                    f"(got {item.min_interval_seconds} for '{item.notification_class}')"
                )
            prefs.append((item.notification_class, item.min_interval_seconds))

        repo = AdminNotificationRepository(session)
        await repo.replace_preferences(user_id, product_id, prefs)

        admin_repo = AdminRepository(session)
        await admin_repo.write_audit(
            AdminAuditLog(
                id=new_id(),
                actor_id=user_id,
                target_user_id=user_id,
                product_id=product_id,
                action="notification_prefs.update",
                detail=f"Subscribed classes: {sorted(c for c, _ in prefs)}",
                source="api",
            )
        )
        logger.info(
            "admin_notifications.preferences_updated",
            user_id=str(user_id),
            classes=sorted(c for c, _ in prefs),
        )

    # ------------------------------------------------------------------
    # Telegram link
    # ------------------------------------------------------------------

    async def get_link_status(
        self, user_id: UUID, *, session: AsyncSession
    ) -> tuple[bool, datetime | None, str | None]:
        link = await AdminNotificationRepository(session).get_link(user_id)
        if link is None or link.chat_id is None:
            return False, None, None
        chat_id_last4 = str(link.chat_id)[-4:]
        return True, link.linked_at, chat_id_last4

    async def create_link_token(
        self, user_id: UUID, product_id: str, *, session: AsyncSession
    ) -> LinkInvite:
        """Issue (or rotate) a one-time deep-link token for the Telegram bot.

        Raises:
            TelegramLinkError: Telegram is not configured, or ``getMe`` failed.
        """
        if self._sender is None:
            raise TelegramLinkError("Telegram notifications are not configured")

        bot_username = await self._get_bot_username()

        token = secrets.token_urlsafe(_LINK_TOKEN_BYTES)
        expires_at = datetime.now(UTC) + timedelta(seconds=self._link_token_ttl_seconds)

        repo = AdminNotificationRepository(session)
        await repo.upsert_link_token(user_id, product_id, token, expires_at)

        admin_repo = AdminRepository(session)
        await admin_repo.write_audit(
            AdminAuditLog(
                id=new_id(),
                actor_id=user_id,
                target_user_id=user_id,
                product_id=product_id,
                action="telegram.link_requested",
                detail="Telegram link token issued",
                source="api",
            )
        )
        logger.info("admin_notifications.link_token_issued", user_id=str(user_id))

        return LinkInvite(
            deep_link=f"https://t.me/{bot_username}?start={token}",
            expires_at=expires_at,
        )

    async def unlink(self, user_id: UUID, product_id: str, *, session: AsyncSession) -> None:
        """Delete the Telegram link, if one exists.

        Idempotent — deleting a non-existent link is a no-op and writes no
        audit row, since no state actually changed.
        """
        repo = AdminNotificationRepository(session)
        deleted = await repo.delete_link(user_id)
        if not deleted:
            return

        admin_repo = AdminRepository(session)
        await admin_repo.write_audit(
            AdminAuditLog(
                id=new_id(),
                actor_id=user_id,
                target_user_id=user_id,
                product_id=product_id,
                action="telegram.unlinked",
                detail="Telegram link removed",
                source="api",
            )
        )
        logger.info("admin_notifications.unlinked", user_id=str(user_id))

    async def _get_bot_username(self) -> str:
        if self._bot_username is not None:
            return self._bot_username
        if self._sender is None:
            raise TelegramLinkError("Telegram notifications are not configured")
        try:
            self._bot_username = await self._sender.get_me()
        except Exception as exc:
            raise TelegramLinkError("Failed to resolve the Telegram bot username") from exc
        return self._bot_username
