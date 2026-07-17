"""Admin notification preference + Telegram link models."""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from sqlalchemy import (
    BigInteger,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    UniqueConstraint,
    text,
)
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import Mapped, mapped_column

from src.db.models.base import Base


class AdminNotificationPreference(Base):
    """An admin's subscription to one notification class, with an optional throttle.

    Row-presence = subscribed (no ``enabled`` flag) — unsubscribing deletes
    the row. ``PUT /v1/admin/notifications/preferences`` replaces the full
    set per user in one transaction (delete-and-insert), so toggling a class
    off is just omitting it from the next PUT.
    """

    __tablename__ = "admin_notification_preferences"

    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True)
    user_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    product_id: Mapped[str] = mapped_column(String(32), nullable=False)
    notification_class: Mapped[str] = mapped_column(String(50), nullable=False)
    min_interval_seconds: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=text("CURRENT_TIMESTAMP"),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=text("CURRENT_TIMESTAMP"),
        onupdate=text("clock_timestamp()"),
    )

    __table_args__ = (
        UniqueConstraint(
            "user_id",
            "product_id",
            "notification_class",
            name="uq_admin_notif_pref_user_product_class",
        ),
        Index("ix_admin_notif_pref_class", "notification_class"),
    )

    def __repr__(self) -> str:
        return f"<AdminNotificationPreference {self.id} user={self.user_id} class={self.notification_class}>"


class AdminTelegramLink(Base):
    """An admin's linked Telegram chat, established via the deep-link flow.

    ``chat_id`` is nullable — a row can exist mid-flow (a token issued but
    not yet confirmed via ``/start``). ``link_token`` is single-use: cleared
    atomically by ``confirm_link_by_token`` in the same UPDATE that sets
    ``chat_id``.
    """

    __tablename__ = "admin_telegram_links"

    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True)
    user_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
    )
    product_id: Mapped[str] = mapped_column(String(32), nullable=False)
    # Telegram chat ids exceed the range of a 32-bit int.
    chat_id: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    link_token: Mapped[str | None] = mapped_column(String(64), nullable=True, unique=True)
    token_expires_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    linked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=text("CURRENT_TIMESTAMP"),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=text("CURRENT_TIMESTAMP"),
        onupdate=text("clock_timestamp()"),
    )

    def __repr__(self) -> str:
        return (
            f"<AdminTelegramLink {self.id} user={self.user_id} linked={self.chat_id is not None}>"
        )
