"""Database models for time-limited authentication tokens.

Two token types:
- ``EmailVerificationToken`` — single-use, 24 h, verifies a user's email address
- ``PasswordResetToken``     — single-use, 30 min, authorises a password change

Both store only the SHA-256 **hash** of the token, never the raw token.
The raw token is generated at request time, returned once to the caller,
and must be included in the verification/reset URL.

Token lifecycle (both types):
1. Token generated (``secrets.token_urlsafe(32)``)
2. Hash stored in DB with ``expires_at`` and ``used_at = NULL``
3. User clicks link → API hashes the token, looks up the DB record
4. If found, not expired, not used → ``used_at`` is set (single-use)
5. Expired/used tokens are cleaned up by a periodic task
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING
from uuid import UUID

from sqlalchemy import (
    DateTime,
    ForeignKey,
    Index,
    String,
    text,
)
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.db.models.base import Base

if TYPE_CHECKING:
    from .user import User


class EmailVerificationToken(Base):
    """Single-use token for verifying a user's email address.

    One active (unused, non-expired) token is kept per user.
    When a new token is requested the previous ones are invalidated
    in the service layer (``used_at = now()``).
    """

    __tablename__ = "email_verification_tokens"

    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True)

    user_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # SHA-256 hex digest of the raw token — never store raw tokens
    token_hash: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        unique=True,
        index=True,
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
    used_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    # Relationship
    user: Mapped[User] = relationship("User", back_populates="email_verification_tokens")

    __table_args__ = (
        Index("ix_email_verification_tokens_user_active", "user_id", "used_at"),
    )

    @property
    def is_used(self) -> bool:
        """Return True if this token has already been consumed."""
        return self.used_at is not None

    def __repr__(self) -> str:
        return f"<EmailVerificationToken user={self.user_id} used={self.is_used}>"


class PasswordResetToken(Base):
    """Single-use token authorising a password reset.

    Shorter lifetime than verification tokens (30 min vs 24 h) because
    a password reset is a higher-privilege action.
    """

    __tablename__ = "password_reset_tokens"

    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True)

    user_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # SHA-256 hex digest of the raw token
    token_hash: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        unique=True,
        index=True,
    )

    # Originating IP for audit purposes
    ip_address: Mapped[str | None] = mapped_column(String(45), nullable=True)

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
    used_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    # Relationship
    user: Mapped[User] = relationship("User", back_populates="password_reset_tokens")

    __table_args__ = (
        Index("ix_password_reset_tokens_user_active", "user_id", "used_at"),
    )

    @property
    def is_used(self) -> bool:
        """Return True if this token has already been consumed."""
        return self.used_at is not None

    def __repr__(self) -> str:
        return f"<PasswordResetToken user={self.user_id} used={self.is_used}>"
