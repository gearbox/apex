"""Repository for time-limited authentication tokens.

Handles DB operations for:
- Email verification tokens
- Password reset tokens

All tokens are stored as SHA-256 hashes. Raw tokens are never persisted.
"""

from __future__ import annotations

import hashlib
import secrets
from datetime import UTC, datetime, timedelta
from uuid import UUID, uuid4

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.models.auth_tokens import EmailVerificationToken, PasswordResetToken

# Token lifetimes
_VERIFICATION_EXPIRE_HOURS = 24
_RESET_EXPIRE_MINUTES = 30


def _hash_token(raw_token: str) -> str:
    """Return SHA-256 hex digest of a raw token string."""
    return hashlib.sha256(raw_token.encode()).hexdigest()


def generate_raw_token() -> str:
    """Generate a cryptographically secure URL-safe token.

    Returns:
        43-character URL-safe base64 token (256 bits of entropy).
    """
    return secrets.token_urlsafe(32)


class AuthTokenRepository:
    """Data access for email verification and password reset tokens.

    Args:
        session: Async SQLAlchemy session (request-scoped).
    """

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    # -------------------------------------------------------------------------
    # Email verification
    # -------------------------------------------------------------------------

    async def create_verification_token(self, user_id: UUID) -> str:
        """Invalidate any existing tokens and create a fresh verification token.

        Args:
            user_id: The user to create a token for.

        Returns:
            The raw (unhashed) token to embed in the verification URL.
        """
        # Invalidate all previous unused tokens for this user
        await self._session.execute(
            update(EmailVerificationToken)
            .where(EmailVerificationToken.user_id == user_id)
            .where(EmailVerificationToken.used_at.is_(None))
            .values(used_at=datetime.now(UTC))
        )

        raw_token = generate_raw_token()
        token = EmailVerificationToken(
            id=uuid4(),
            user_id=user_id,
            token_hash=_hash_token(raw_token),
            expires_at=datetime.now(UTC) + timedelta(hours=_VERIFICATION_EXPIRE_HOURS),
        )
        self._session.add(token)
        await self._session.flush()
        return raw_token

    async def consume_verification_token(self, raw_token: str) -> UUID | None:
        """Validate and consume an email verification token.

        Marks the token as used (single-use). Ignores expired or already-used tokens.

        Args:
            raw_token: The raw token from the verification URL.

        Returns:
            The ``user_id`` associated with the token, or ``None`` if invalid.
        """
        token_hash = _hash_token(raw_token)
        now = datetime.now(UTC)

        result = await self._session.execute(
            select(EmailVerificationToken)
            .where(EmailVerificationToken.token_hash == token_hash)
            .where(EmailVerificationToken.used_at.is_(None))
            .where(EmailVerificationToken.expires_at > now)
        )
        token = result.scalar_one_or_none()

        if token is None:
            return None

        token.used_at = now
        await self._session.flush()
        return token.user_id

    async def cleanup_expired_email_verification_tokens(self) -> int:
        """Delete expired/used email verification tokens. Returns count deleted."""
        now = datetime.now(UTC)
        result = await self._session.execute(
            select(EmailVerificationToken).where(
                (EmailVerificationToken.expires_at < now)
                | (EmailVerificationToken.used_at.is_not(None))
            )
        )
        tokens_to_delete = result.scalars().all()
        count = len(tokens_to_delete)

        for token in tokens_to_delete:
            await self._session.delete(token)

        await self._session.flush()
        return count

    # -------------------------------------------------------------------------
    # Password reset
    # -------------------------------------------------------------------------

    async def create_reset_token(
        self,
        user_id: UUID,
        *,
        ip_address: str | None = None,
    ) -> str:
        """Invalidate any existing reset tokens and create a new one.

        Args:
            user_id: The user requesting the reset.
            ip_address: Originating IP address for audit trail.

        Returns:
            The raw (unhashed) token to embed in the reset URL.
        """
        # Invalidate any active reset tokens for this user
        await self._session.execute(
            update(PasswordResetToken)
            .where(PasswordResetToken.user_id == user_id)
            .where(PasswordResetToken.used_at.is_(None))
            .values(used_at=datetime.now(UTC))
        )

        raw_token = generate_raw_token()
        token = PasswordResetToken(
            id=uuid4(),
            user_id=user_id,
            token_hash=_hash_token(raw_token),
            ip_address=ip_address,
            expires_at=datetime.now(UTC) + timedelta(minutes=_RESET_EXPIRE_MINUTES),
        )
        self._session.add(token)
        await self._session.flush()
        return raw_token

    async def consume_reset_token(self, raw_token: str) -> UUID | None:
        """Validate and consume a password reset token.

        Args:
            raw_token: The raw token from the reset URL.

        Returns:
            The ``user_id`` associated with the token, or ``None`` if invalid/expired.
        """
        token_hash = _hash_token(raw_token)
        now = datetime.now(UTC)

        result = await self._session.execute(
            select(PasswordResetToken)
            .where(PasswordResetToken.token_hash == token_hash)
            .where(PasswordResetToken.used_at.is_(None))
            .where(PasswordResetToken.expires_at > now)
        )
        token = result.scalar_one_or_none()

        if token is None:
            return None

        token.used_at = now
        await self._session.flush()
        return token.user_id

    async def cleanup_expired_password_reset_tokens(self) -> int:
        """Delete expired/used password reset tokens. Returns count deleted."""
        now = datetime.now(UTC)
        result = await self._session.execute(
            select(PasswordResetToken).where(
                (PasswordResetToken.expires_at < now) | (PasswordResetToken.used_at.is_not(None))
            )
        )
        tokens_to_delete = result.scalars().all()
        count = len(tokens_to_delete)

        for token in tokens_to_delete:
            await self._session.delete(token)

        await self._session.flush()
        return count
