"""User and authentication database models."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING
from uuid import UUID

from sqlalchemy import Boolean, DateTime, ForeignKey, Index, String, Text, text
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.core.enums import SubscriptionTier
from src.db.models.storage import Base

if TYPE_CHECKING:
    from src.db.models.storage import GenerationJob, GenerationOutput, UserImage


class User(Base):
    """User account model.

    Stores user credentials and profile information.
    Supports soft deletion via is_active flag.
    """

    __tablename__ = "users"

    id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        primary_key=True,
    )
    email: Mapped[str] = mapped_column(
        String(255),
        unique=True,
        nullable=False,
        index=True,
    )
    password_hash: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )
    display_name: Mapped[str | None] = mapped_column(
        String(100),
        nullable=True,
    )
    subscription_tier: Mapped[SubscriptionTier] = mapped_column(
        String(20),
        nullable=False,
        default=SubscriptionTier.FREE.value,
    )

    # Account status
    is_active: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=True,
        index=True,
    )

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=text("CURRENT_TIMESTAMP"),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=text("CURRENT_TIMESTAMP"),
        onupdate=datetime.utcnow,
    )

    # Relationships
    refresh_tokens: Mapped[list[RefreshToken]] = relationship(
        "RefreshToken",
        back_populates="user",
        cascade="all, delete-orphan",
    )
    generation_jobs: Mapped[list[GenerationJob]] = relationship(
        "GenerationJob",
        back_populates="user",
        cascade="all, delete-orphan",
        foreign_keys="GenerationJob.user_id",
    )
    user_images: Mapped[list[UserImage]] = relationship(
        "UserImage",
        back_populates="user",
        cascade="all, delete-orphan",
        foreign_keys="UserImage.user_id",
    )
    generation_outputs: Mapped[list[GenerationOutput]] = relationship(
        "GenerationOutput",
        back_populates="user",
        cascade="all, delete-orphan",
        foreign_keys="GenerationOutput.user_id",
    )

    __table_args__ = (Index("ix_users_email_active", "email", "is_active"),)

    def __repr__(self) -> str:
        return f"<User {self.id} email={self.email} tier={self.subscription_tier}>"


class RefreshToken(Base):
    """Refresh token for JWT rotation.

    Each refresh token is single-use. When used, it's revoked and
    a new token is issued. This enables token rotation and
    detection of token theft.
    """

    __tablename__ = "refresh_tokens"

    id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        primary_key=True,
    )
    user_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    token_hash: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        unique=True,
        index=True,
    )
    # Token family for rotation tracking
    # All tokens in a family share this ID; if a revoked token is reused,
    # we revoke the entire family (potential token theft)
    family_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        nullable=False,
        index=True,
    )

    # Device/client info for audit
    user_agent: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )
    ip_address: Mapped[str | None] = mapped_column(
        String(45),  # IPv6 max length
        nullable=True,
    )

    # Status
    is_revoked: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
    )

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=text("CURRENT_TIMESTAMP"),
    )
    expires_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        index=True,
    )
    revoked_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    # Relationships
    user: Mapped[User] = relationship(
        "User",
        back_populates="refresh_tokens",
    )

    __table_args__ = (
        Index("ix_refresh_tokens_user_valid", "user_id", "is_revoked", "expires_at"),
        Index("ix_refresh_tokens_cleanup", "expires_at", "is_revoked"),
    )

    def __repr__(self) -> str:
        return f"<RefreshToken {self.id} user={self.user_id} revoked={self.is_revoked}>"
