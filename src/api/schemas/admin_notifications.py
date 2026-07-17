"""Admin notification preference + Telegram link API schemas."""

from __future__ import annotations

from datetime import datetime

import msgspec


class NotificationClassInfo(msgspec.Struct, kw_only=True):
    """One entry in the static notification-class catalog."""

    notification_class: str
    scope: str  # "product" | "platform"
    description: str


class NotificationPreferenceItem(msgspec.Struct, kw_only=True):
    notification_class: str
    min_interval_seconds: int = 0


class NotificationPreferencesResponse(msgspec.Struct, kw_only=True):
    items: list[NotificationPreferenceItem]


class NotificationPreferencesUpdateRequest(
    msgspec.Struct, forbid_unknown_fields=True, kw_only=True
):
    """Full-set replacement — omitted classes are unsubscribed."""

    items: list[NotificationPreferenceItem]


class TelegramLinkStatusResponse(msgspec.Struct, kw_only=True):
    linked: bool
    linked_at: datetime | None = None
    chat_id_last4: str | None = None


class TelegramLinkInviteResponse(msgspec.Struct, kw_only=True):
    deep_link: str
    expires_at: datetime
