"""Repository for admin notification preferences and Telegram links."""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, cast

import structlog
from sqlalchemy import delete, select

from src.core.enums import UserRole
from src.core.uid import new_id
from src.db.models.admin_notifications import AdminNotificationPreference, AdminTelegramLink
from src.db.models.user import User

if TYPE_CHECKING:
    from collections.abc import Sequence
    from datetime import datetime
    from uuid import UUID

    from sqlalchemy.engine import CursorResult
    from sqlalchemy.ext.asyncio import AsyncSession

logger = structlog.get_logger(__name__)

_ADMIN_ROLES = (UserRole.ADMIN.value, UserRole.SUPERADMIN.value)


@dataclasses.dataclass(frozen=True)
class RecipientRow:
    """One admin recipient eligible for a notification class, resolved at delivery time."""

    user_id: UUID
    product_id: str
    chat_id: int
    min_interval_seconds: int


class AdminNotificationRepository:
    """Data access for admin notification preferences and Telegram links.

    No commits/flushes on writes that don't require an immediately-visible
    id — callers own the transaction, per house convention.
    """

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    # --- Preferences ---

    async def list_preferences(self, user_id: UUID) -> Sequence[AdminNotificationPreference]:
        result = await self._session.execute(
            select(AdminNotificationPreference).where(
                AdminNotificationPreference.user_id == user_id
            )
        )
        return result.scalars().all()

    async def replace_preferences(
        self,
        user_id: UUID,
        product_id: str,
        prefs: Sequence[tuple[str, int]],
    ) -> None:
        """Full-set replace: delete all existing rows for the user, then insert.

        Idempotent — re-running with the same set produces the same rows.
        """
        await self._session.execute(
            delete(AdminNotificationPreference).where(
                AdminNotificationPreference.user_id == user_id
            )
        )
        for notification_class, min_interval_seconds in prefs:
            self._session.add(
                AdminNotificationPreference(
                    id=new_id(),
                    user_id=user_id,
                    product_id=product_id,
                    notification_class=notification_class,
                    min_interval_seconds=min_interval_seconds,
                )
            )
        await self._session.flush()

    # --- Telegram link ---

    async def get_link(self, user_id: UUID) -> AdminTelegramLink | None:
        result = await self._session.execute(
            select(AdminTelegramLink).where(AdminTelegramLink.user_id == user_id)
        )
        return result.scalar_one_or_none()

    async def upsert_link_token(
        self,
        user_id: UUID,
        product_id: str,
        token: str,
        expires_at: datetime,
    ) -> AdminTelegramLink:
        """Create the link row, or rotate the token on an existing one.

        Rotating leaves ``chat_id``/``linked_at`` untouched — requesting a
        new link while already linked is just a way to re-link (e.g. after
        losing access to the chat), not an implicit unlink.
        """
        existing = await self.get_link(user_id)
        if existing is not None:
            existing.link_token = token
            existing.token_expires_at = expires_at
            await self._session.flush()
            return existing

        link = AdminTelegramLink(
            id=new_id(),
            user_id=user_id,
            product_id=product_id,
            link_token=token,
            token_expires_at=expires_at,
        )
        self._session.add(link)
        await self._session.flush()
        return link

    async def confirm_link_by_token(
        self,
        token: str,
        chat_id: int,
        now: datetime,
    ) -> AdminTelegramLink | None:
        """Match an unexpired token, set chat_id/linked_at, clear the token.

        The token is cleared atomically in the same UPDATE that sets
        ``chat_id`` — single-use, so a replayed webhook update can never
        re-confirm a stale token.

        Returns None on unknown/expired token.
        """
        result = await self._session.execute(
            select(AdminTelegramLink).where(
                AdminTelegramLink.link_token == token,
                AdminTelegramLink.token_expires_at.is_not(None),
                AdminTelegramLink.token_expires_at > now,
            )
        )
        link = result.scalar_one_or_none()
        if link is None:
            return None

        link.chat_id = chat_id
        link.linked_at = now
        link.link_token = None
        link.token_expires_at = None
        await self._session.flush()
        return link

    async def delete_link(self, user_id: UUID) -> bool:
        result = cast(
            "CursorResult[tuple[()]]",
            await self._session.execute(
                delete(AdminTelegramLink).where(AdminTelegramLink.user_id == user_id)
            ),
        )
        return (result.rowcount or 0) > 0

    # --- Delivery-time recipient resolution ---

    async def list_recipients_for_class(self, notification_class: str) -> Sequence[RecipientRow]:
        """Resolve linked, active admin/superadmin recipients for a notification class.

        Role and is_active are checked at delivery time (joined against the
        live `users` row), not cached on the preference row — a demoted or
        deactivated admin stops receiving without any preference cleanup.
        """
        result = await self._session.execute(
            select(
                AdminNotificationPreference.user_id,
                AdminNotificationPreference.product_id,
                AdminTelegramLink.chat_id,
                AdminNotificationPreference.min_interval_seconds,
            )
            .join(
                AdminTelegramLink,
                AdminTelegramLink.user_id == AdminNotificationPreference.user_id,
            )
            .join(User, User.id == AdminNotificationPreference.user_id)
            .where(
                AdminNotificationPreference.notification_class == notification_class,
                AdminTelegramLink.chat_id.is_not(None),
                User.role.in_(_ADMIN_ROLES),
                User.is_active.is_(True),
            )
        )
        return [
            RecipientRow(
                user_id=row.user_id,
                product_id=row.product_id,
                chat_id=row.chat_id,
                min_interval_seconds=row.min_interval_seconds,
            )
            for row in result.all()
        ]
