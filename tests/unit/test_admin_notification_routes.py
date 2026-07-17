"""Tests for AdminNotificationController handlers.

Exercises handler business logic directly by calling the underlying
coroutine (bypassing Litestar's handler wrapping), same convention as
test_admin_management_routes.py. Guard/DI enforcement (403 for non-admins)
is Litestar's own dependency-resolution machinery, not re-tested here.
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest
from litestar.exceptions import HTTPException, ValidationException

from src.api.routes.admin_notifications import AdminNotificationController
from src.api.schemas.admin_notifications import (
    NotificationClassInfo,
    NotificationPreferenceItem,
    NotificationPreferencesUpdateRequest,
)
from src.api.services.admin_notifications import LinkInvite, TelegramLinkError
from src.core.enums import NotificationClass


def _make_admin(user_id=None) -> MagicMock:  # type: ignore[no-untyped-def]
    user = MagicMock()
    user.id = user_id or uuid4()
    return user


class TestGetClasses:
    async def test_returns_catalog_from_service(self) -> None:
        raw = AdminNotificationController.get_classes.fn  # type: ignore[attr-defined]
        service = MagicMock()
        catalog = [
            NotificationClassInfo(
                notification_class="user.registered", scope="product", description="x"
            )
        ]
        service.get_class_catalog.return_value = catalog

        result = await raw(MagicMock(), admin=_make_admin(), admin_notification_service=service)

        assert result == catalog


class TestGetPreferences:
    async def test_wraps_repo_rows_into_response(self) -> None:
        raw = AdminNotificationController.get_preferences.fn  # type: ignore[attr-defined]
        admin = _make_admin()
        service = AsyncMock()
        row = MagicMock()
        row.notification_class = "user.registered"
        row.min_interval_seconds = 30
        service.get_preferences = AsyncMock(return_value=[row])

        result = await raw(
            MagicMock(), admin=admin, session=AsyncMock(), admin_notification_service=service
        )

        assert result.items == [
            NotificationPreferenceItem(
                notification_class="user.registered", min_interval_seconds=30
            )
        ]


class TestReplacePreferences:
    async def test_success_commits_and_returns_updated_set(self) -> None:
        raw = AdminNotificationController.replace_preferences.fn  # type: ignore[attr-defined]
        admin = _make_admin()
        session = AsyncMock()
        service = AsyncMock()
        service.replace_preferences = AsyncMock()
        service.get_preferences = AsyncMock(return_value=[])
        data = NotificationPreferencesUpdateRequest(
            items=[
                NotificationPreferenceItem(
                    notification_class=NotificationClass.USER_REGISTERED.value
                )
            ]
        )

        await raw(
            MagicMock(),
            admin=admin,
            data=data,
            session=session,
            product_id="vex",
            admin_notification_service=service,
        )

        service.replace_preferences.assert_awaited_once_with(
            admin.id, "vex", data.items, session=session
        )
        session.commit.assert_awaited_once()

    async def test_value_error_maps_to_validation_exception(self) -> None:
        raw = AdminNotificationController.replace_preferences.fn  # type: ignore[attr-defined]
        service = AsyncMock()
        service.replace_preferences = AsyncMock(
            side_effect=ValueError("Unknown notification_class 'x'")
        )
        data = NotificationPreferencesUpdateRequest(items=[])

        with pytest.raises(ValidationException, match="Unknown notification_class"):
            await raw(
                MagicMock(),
                admin=_make_admin(),
                data=data,
                session=AsyncMock(),
                product_id="vex",
                admin_notification_service=service,
            )


class TestGetPreferencesFor:
    async def test_superadmin_can_view_target_users_preferences(self) -> None:
        raw = AdminNotificationController.get_preferences_for.fn  # type: ignore[attr-defined]
        target_id = uuid4()
        session = AsyncMock()
        service = AsyncMock()
        service.get_preferences_for = AsyncMock(return_value=[])

        result = await raw(
            MagicMock(),
            superadmin=_make_admin(),
            user_id=target_id,
            session=session,
            admin_notification_service=service,
        )

        service.get_preferences_for.assert_awaited_once_with(target_id, session=session)
        assert result.items == []


class TestTelegramLink:
    async def test_create_link_success(self) -> None:
        raw = AdminNotificationController.create_telegram_link.fn  # type: ignore[attr-defined]
        service = AsyncMock()
        service.create_link_token = AsyncMock(
            return_value=LinkInvite(
                deep_link="https://t.me/bot?start=tok", expires_at=datetime.now(UTC)
            )
        )
        session = AsyncMock()

        result = await raw(
            MagicMock(),
            admin=_make_admin(),
            session=session,
            product_id="vex",
            admin_notification_service=service,
        )

        assert result.deep_link == "https://t.me/bot?start=tok"
        session.commit.assert_awaited_once()

    async def test_create_link_telegram_error_maps_to_503(self) -> None:
        raw = AdminNotificationController.create_telegram_link.fn  # type: ignore[attr-defined]
        service = AsyncMock()
        service.create_link_token = AsyncMock(side_effect=TelegramLinkError("not configured"))

        with pytest.raises(HTTPException) as excinfo:
            await raw(
                MagicMock(),
                admin=_make_admin(),
                session=AsyncMock(),
                product_id="vex",
                admin_notification_service=service,
            )
        assert excinfo.value.status_code == 503

    async def test_delete_link_commits(self) -> None:
        raw = AdminNotificationController.delete_telegram_link.fn  # type: ignore[attr-defined]
        service = AsyncMock()
        session = AsyncMock()

        result = await raw(
            MagicMock(),
            admin=_make_admin(),
            session=session,
            product_id="vex",
            admin_notification_service=service,
        )

        service.unlink.assert_awaited_once()
        session.commit.assert_awaited_once()
        assert result == {"message": "Telegram link removed"}
