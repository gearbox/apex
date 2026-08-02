"""Unit tests for AdminNotificationService."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from src.api.services.admin_management import AdminManagementError
from src.api.services.admin_notifications import AdminNotificationService, TelegramLinkError
from src.core.enums import PLATFORM_SCOPED_NOTIFICATION_CLASSES, NotificationClass


class TestClassCatalog:
    def test_catalog_has_every_class_with_correct_scope(self) -> None:
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

    def test_token_revocation_failed_is_platform_scoped(self) -> None:
        """N2 (description present) + N3 (frozenset membership) in one assertion."""
        service = AdminNotificationService(sender=None, link_token_ttl_seconds=900)

        catalog = service.get_class_catalog()

        entry = next(
            c
            for c in catalog
            if c.notification_class == NotificationClass.TOKEN_REVOCATION_FAILED.value
        )
        assert entry.scope == "platform"

    def test_push_subscriptions_cleanup_failed_is_platform_scoped(self) -> None:
        """M1: same wiring gap as TOKEN_REVOCATION_FAILED — a catalog entry +
        frozenset membership is required or this event never reaches an operator."""
        service = AdminNotificationService(sender=None, link_token_ttl_seconds=900)

        catalog = service.get_class_catalog()

        entry = next(
            c
            for c in catalog
            if c.notification_class == NotificationClass.PUSH_SUBSCRIPTIONS_CLEANUP_FAILED.value
        )
        assert entry.scope == "platform"

    def test_provider_authentication_failed_is_platform_scoped(self) -> None:
        service = AdminNotificationService(sender=None, link_token_ttl_seconds=900)

        catalog = service.get_class_catalog()

        entry = next(
            c
            for c in catalog
            if c.notification_class == NotificationClass.PROVIDER_AUTHENTICATION_FAILED.value
        )
        assert entry.scope == "platform"


class TestGetPreferences:
    async def test_returns_repo_rows(self) -> None:
        service = AdminNotificationService(sender=None, link_token_ttl_seconds=900)
        user_id = uuid4()
        rows = [object()]

        with patch(
            "src.api.services.admin_notifications.AdminNotificationRepository"
        ) as mock_repo_cls:
            mock_repo = AsyncMock()
            mock_repo.list_preferences = AsyncMock(return_value=rows)
            mock_repo_cls.return_value = mock_repo

            result = await service.get_preferences(user_id, session=AsyncMock())

        mock_repo.list_preferences.assert_awaited_once_with(user_id)
        assert result is rows


class TestGetPreferencesFor:
    async def test_returns_rows_when_target_in_same_product(self) -> None:
        service = AdminNotificationService(sender=None, link_token_ttl_seconds=900)
        target_id = uuid4()
        rows = [object()]
        target_user = MagicMock()
        target_user.product_id = "vex"

        with (
            patch("src.api.services.admin_notifications.UserRepository") as mock_user_repo_cls,
            patch(
                "src.api.services.admin_notifications.AdminNotificationRepository"
            ) as mock_notif_repo_cls,
        ):
            mock_user_repo = AsyncMock()
            mock_user_repo.get_active_user = AsyncMock(return_value=target_user)
            mock_user_repo_cls.return_value = mock_user_repo
            mock_notif_repo = AsyncMock()
            mock_notif_repo.list_preferences = AsyncMock(return_value=rows)
            mock_notif_repo_cls.return_value = mock_notif_repo

            result = await service.get_preferences_for(target_id, "vex", session=AsyncMock())

        assert result is rows

    async def test_raises_when_target_user_not_found(self) -> None:
        service = AdminNotificationService(sender=None, link_token_ttl_seconds=900)

        with patch("src.api.services.admin_notifications.UserRepository") as mock_user_repo_cls:
            mock_user_repo = AsyncMock()
            mock_user_repo.get_active_user = AsyncMock(return_value=None)
            mock_user_repo_cls.return_value = mock_user_repo

            with pytest.raises(AdminManagementError, match="not found in product"):
                await service.get_preferences_for(uuid4(), "vex", session=AsyncMock())

    async def test_raises_when_target_user_in_different_product(self) -> None:
        service = AdminNotificationService(sender=None, link_token_ttl_seconds=900)
        target_user = MagicMock()
        target_user.product_id = "synthara"

        with patch("src.api.services.admin_notifications.UserRepository") as mock_user_repo_cls:
            mock_user_repo = AsyncMock()
            mock_user_repo.get_active_user = AsyncMock(return_value=target_user)
            mock_user_repo_cls.return_value = mock_user_repo

            with pytest.raises(AdminManagementError, match="not found in product"):
                await service.get_preferences_for(uuid4(), "vex", session=AsyncMock())


class TestGetLinkStatus:
    async def test_no_link_row_returns_unlinked(self) -> None:
        service = AdminNotificationService(sender=None, link_token_ttl_seconds=900)

        with patch(
            "src.api.services.admin_notifications.AdminNotificationRepository"
        ) as mock_repo_cls:
            mock_repo = AsyncMock()
            mock_repo.get_link = AsyncMock(return_value=None)
            mock_repo_cls.return_value = mock_repo

            linked, linked_at, chat_id_last4 = await service.get_link_status(
                uuid4(), session=AsyncMock()
            )

        assert (linked, linked_at, chat_id_last4) == (False, None, None)

    async def test_link_row_without_chat_id_returns_unlinked(self) -> None:
        service = AdminNotificationService(sender=None, link_token_ttl_seconds=900)
        link = MagicMock()
        link.chat_id = None

        with patch(
            "src.api.services.admin_notifications.AdminNotificationRepository"
        ) as mock_repo_cls:
            mock_repo = AsyncMock()
            mock_repo.get_link = AsyncMock(return_value=link)
            mock_repo_cls.return_value = mock_repo

            linked, linked_at, chat_id_last4 = await service.get_link_status(
                uuid4(), session=AsyncMock()
            )

        assert (linked, linked_at, chat_id_last4) == (False, None, None)

    async def test_confirmed_link_returns_last4_of_chat_id(self) -> None:
        service = AdminNotificationService(sender=None, link_token_ttl_seconds=900)
        link = MagicMock()
        link.chat_id = 123456789
        link.linked_at = datetime(2026, 1, 1, tzinfo=UTC)

        with patch(
            "src.api.services.admin_notifications.AdminNotificationRepository"
        ) as mock_repo_cls:
            mock_repo = AsyncMock()
            mock_repo.get_link = AsyncMock(return_value=link)
            mock_repo_cls.return_value = mock_repo

            linked, linked_at, chat_id_last4 = await service.get_link_status(
                uuid4(), session=AsyncMock()
            )

        assert linked is True
        assert linked_at == link.linked_at
        assert chat_id_last4 == "6789"


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

    async def test_get_bot_username_without_sender_raises_telegram_link_error(self) -> None:
        service = AdminNotificationService(sender=None, link_token_ttl_seconds=900)

        with pytest.raises(TelegramLinkError, match="not configured"):
            await service._get_bot_username()

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
