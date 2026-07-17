"""Unit tests for AdminNotificationService."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest

from src.api.services.admin_notifications import AdminNotificationService, TelegramLinkError
from src.core.enums import PLATFORM_SCOPED_NOTIFICATION_CLASSES, NotificationClass


class TestClassCatalog:
    def test_catalog_has_all_six_classes_with_correct_scope(self) -> None:
        service = AdminNotificationService(sender=None, link_token_ttl_seconds=900)

        catalog = service.get_class_catalog()

        assert {c.notification_class for c in catalog} == {c.value for c in NotificationClass}
        for entry in catalog:
            expected_scope = (
                "platform"
                if NotificationClass(entry.notification_class)
                in PLATFORM_SCOPED_NOTIFICATION_CLASSES
                else "product"
            )
            assert entry.scope == expected_scope


class TestReplacePreferences:
    async def test_valid_items_replace_and_audit(self) -> None:
        service = AdminNotificationService(sender=None, link_token_ttl_seconds=900)
        user_id = uuid4()
        session = AsyncMock()

        from src.api.schemas.admin_notifications import NotificationPreferenceItem

        items = [
            NotificationPreferenceItem(
                notification_class=NotificationClass.USER_REGISTERED.value, min_interval_seconds=30
            )
        ]

        with (
            patch(
                "src.api.services.admin_notifications.AdminNotificationRepository"
            ) as mock_repo_cls,
            patch("src.api.services.admin_notifications.AdminRepository") as mock_admin_repo_cls,
        ):
            mock_repo = AsyncMock()
            mock_repo_cls.return_value = mock_repo
            mock_admin_repo = AsyncMock()
            mock_admin_repo_cls.return_value = mock_admin_repo

            await service.replace_preferences(user_id, "vex", items, session=session)

        mock_repo.replace_preferences.assert_awaited_once_with(
            user_id, "vex", [(NotificationClass.USER_REGISTERED.value, 30)]
        )
        mock_admin_repo.write_audit.assert_awaited_once()

    async def test_unknown_class_raises_value_error(self) -> None:
        service = AdminNotificationService(sender=None, link_token_ttl_seconds=900)
        from src.api.schemas.admin_notifications import NotificationPreferenceItem

        items = [NotificationPreferenceItem(notification_class="not.a.real.class")]

        with pytest.raises(ValueError, match="Unknown notification_class"):
            await service.replace_preferences(uuid4(), "vex", items, session=AsyncMock())

    async def test_duplicate_class_raises_value_error(self) -> None:
        service = AdminNotificationService(sender=None, link_token_ttl_seconds=900)
        from src.api.schemas.admin_notifications import NotificationPreferenceItem

        items = [
            NotificationPreferenceItem(
                notification_class=NotificationClass.USER_REGISTERED.value,
                min_interval_seconds=10,
            ),
            NotificationPreferenceItem(
                notification_class=NotificationClass.USER_REGISTERED.value,
                min_interval_seconds=20,
            ),
        ]

        with pytest.raises(ValueError, match="Duplicate notification_class"):
            await service.replace_preferences(uuid4(), "vex", items, session=AsyncMock())

    @pytest.mark.parametrize("interval", [-1, 86401])
    async def test_interval_out_of_bounds_raises_value_error(self, interval: int) -> None:
        service = AdminNotificationService(sender=None, link_token_ttl_seconds=900)
        from src.api.schemas.admin_notifications import NotificationPreferenceItem

        items = [
            NotificationPreferenceItem(
                notification_class=NotificationClass.USER_REGISTERED.value,
                min_interval_seconds=interval,
            )
        ]

        with pytest.raises(ValueError, match="min_interval_seconds"):
            await service.replace_preferences(uuid4(), "vex", items, session=AsyncMock())

    async def test_empty_set_is_idempotent_full_clear(self) -> None:
        service = AdminNotificationService(sender=None, link_token_ttl_seconds=900)
        session = AsyncMock()

        with (
            patch(
                "src.api.services.admin_notifications.AdminNotificationRepository"
            ) as mock_repo_cls,
            patch("src.api.services.admin_notifications.AdminRepository") as mock_admin_repo_cls,
        ):
            mock_repo = AsyncMock()
            mock_repo_cls.return_value = mock_repo
            mock_admin_repo_cls.return_value = AsyncMock()

            await service.replace_preferences(uuid4(), "vex", [], session=session)

        mock_repo.replace_preferences.assert_awaited_once()
        args, _ = mock_repo.replace_preferences.call_args
        assert args[2] == []


class TestTelegramLink:
    async def test_create_link_token_without_sender_raises_telegram_link_error(self) -> None:
        service = AdminNotificationService(sender=None, link_token_ttl_seconds=900)

        with pytest.raises(TelegramLinkError):
            await service.create_link_token(uuid4(), "vex", session=AsyncMock())

    async def test_create_link_token_caches_bot_username_across_calls(self) -> None:
        sender = AsyncMock()
        sender.get_me = AsyncMock(return_value="my_bot")
        service = AdminNotificationService(sender=sender, link_token_ttl_seconds=900)
        session = AsyncMock()

        with (
            patch(
                "src.api.services.admin_notifications.AdminNotificationRepository"
            ) as mock_repo_cls,
            patch("src.api.services.admin_notifications.AdminRepository") as mock_admin_repo_cls,
        ):
            mock_repo_cls.return_value = AsyncMock()
            mock_admin_repo_cls.return_value = AsyncMock()

            invite1 = await service.create_link_token(uuid4(), "vex", session=session)
            invite2 = await service.create_link_token(uuid4(), "vex", session=session)

        assert invite1.deep_link.startswith("https://t.me/my_bot?start=")
        assert invite2.deep_link.startswith("https://t.me/my_bot?start=")
        sender.get_me.assert_awaited_once()  # cached — only fetched once

    async def test_get_me_failure_raises_telegram_link_error(self) -> None:
        sender = AsyncMock()
        sender.get_me = AsyncMock(side_effect=RuntimeError("network down"))
        service = AdminNotificationService(sender=sender, link_token_ttl_seconds=900)

        with pytest.raises(TelegramLinkError):
            await service.create_link_token(uuid4(), "vex", session=AsyncMock())

    async def test_unlink_deletes_and_audits(self) -> None:
        service = AdminNotificationService(sender=None, link_token_ttl_seconds=900)
        user_id = uuid4()

        with (
            patch(
                "src.api.services.admin_notifications.AdminNotificationRepository"
            ) as mock_repo_cls,
            patch("src.api.services.admin_notifications.AdminRepository") as mock_admin_repo_cls,
        ):
            mock_repo = AsyncMock()
            mock_repo.delete_link = AsyncMock(return_value=True)
            mock_repo_cls.return_value = mock_repo
            mock_admin_repo = AsyncMock()
            mock_admin_repo_cls.return_value = mock_admin_repo

            await service.unlink(user_id, "vex", session=AsyncMock())

        mock_repo.delete_link.assert_awaited_once_with(user_id)
        mock_admin_repo.write_audit.assert_awaited_once()

    async def test_unlink_with_no_existing_link_writes_no_audit(self) -> None:
        service = AdminNotificationService(sender=None, link_token_ttl_seconds=900)
        user_id = uuid4()

        with (
            patch(
                "src.api.services.admin_notifications.AdminNotificationRepository"
            ) as mock_repo_cls,
            patch("src.api.services.admin_notifications.AdminRepository") as mock_admin_repo_cls,
        ):
            mock_repo = AsyncMock()
            mock_repo.delete_link = AsyncMock(return_value=False)
            mock_repo_cls.return_value = mock_repo
            mock_admin_repo = AsyncMock()
            mock_admin_repo_cls.return_value = mock_admin_repo

            await service.unlink(user_id, "vex", session=AsyncMock())

        mock_repo.delete_link.assert_awaited_once_with(user_id)
        mock_admin_repo.write_audit.assert_not_awaited()
