"""Authentication service for user registration and login."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING
from uuid import UUID, uuid4

from src.api.security import (
    JWTService,
    PasswordService,
    generate_token,
    hash_token,
)
from src.db.repositories import UserRepository

if TYPE_CHECKING:
    from src.db.models import User

logger = logging.getLogger(__name__)


class AuthError(Exception):
    """Base authentication error."""

    pass


class InvalidCredentialsError(AuthError):
    """Invalid email or password."""

    pass


class EmailAlreadyExistsError(AuthError):
    """Email is already registered."""

    pass


class UserNotFoundError(AuthError):
    """User not found."""

    pass


class UserInactiveError(AuthError):
    """User account is deactivated."""

    pass


class InvalidRefreshTokenError(AuthError):
    """Refresh token is invalid, expired, or revoked."""

    pass


class TokenReuseDetectedError(AuthError):
    """Potential token theft detected - revoked token was reused."""

    pass


@dataclass
class TokenPair:
    """Access and refresh token pair."""

    access_token: str
    refresh_token: str
    expires_at: datetime
    expires_in: int


class AuthService:
    """Authentication service.

    Handles user registration, login, token refresh, and logout.
    Implements secure token rotation with family tracking for
    detecting potential token theft.
    """

    def __init__(
        self,
        repository: UserRepository,
        jwt_service: JWTService,
        password_service: PasswordService,
    ) -> None:
        """Initialize auth service.

        Args:
            repository: User repository.
            jwt_service: JWT token service.
            password_service: Password hashing service.
        """
        self._repo = repository
        self._jwt = jwt_service
        self._password = password_service

    async def register(
        self,
        *,
        email: str,
        password: str,
        display_name: str | None = None,
    ) -> tuple[User, TokenPair]:
        """Register a new user.

        Args:
            email: User email.
            password: Plain text password.
            display_name: Optional display name.

        Returns:
            Tuple of (User, TokenPair).

        Raises:
            EmailAlreadyExistsError: If email is taken.
        """
        # Check for existing email
        if await self._repo.email_exists(email):
            raise EmailAlreadyExistsError(f"Email {email} is already registered")

        # Create user
        user_id = uuid4()
        password_hash = self._password.hash(password)

        user = await self._repo.create_user(
            id=user_id,
            email=email,
            password_hash=password_hash,
            display_name=display_name,
        )

        logger.info(f"User registered: {user_id} ({email})")

        # Generate tokens
        tokens = await self._create_token_pair(user_id)

        return user, tokens

    async def login(
        self,
        *,
        email: str,
        password: str,
        user_agent: str | None = None,
        ip_address: str | None = None,
    ) -> tuple[User, TokenPair]:
        """Authenticate user and return tokens.

        Args:
            email: User email.
            password: Plain text password.
            user_agent: Client user agent (optional).
            ip_address: Client IP address (optional).

        Returns:
            Tuple of (User, TokenPair).

        Raises:
            InvalidCredentialsError: If email or password is wrong.
            UserInactiveError: If user account is deactivated.
        """
        user = await self._repo.get_user_by_email(email)

        if user is None:
            # Prevent timing attacks
            self._password.hash("dummy_password")
            raise InvalidCredentialsError("Invalid email or password")

        if not user.is_active:
            raise UserInactiveError("User account is deactivated")

        if not self._password.verify(user.password_hash, password):
            raise InvalidCredentialsError("Invalid email or password")

        # Check if password needs rehash (parameter upgrade)
        if self._password.needs_rehash(user.password_hash):
            new_hash = self._password.hash(password)
            await self._repo.update_user(user.id, password_hash=new_hash)
            logger.info(f"Rehashed password for user {user.id}")

        logger.info(f"User logged in: {user.id}")

        tokens = await self._create_token_pair(
            user.id,
            user_agent=user_agent,
            ip_address=ip_address,
        )

        return user, tokens

    async def refresh_tokens(
        self,
        refresh_token: str,
        *,
        user_agent: str | None = None,
        ip_address: str | None = None,
    ) -> TokenPair:
        """Refresh access token using refresh token.

        Implements token rotation: the old refresh token is revoked
        and a new one is issued in the same family.

        Args:
            refresh_token: Current refresh token.
            user_agent: Client user agent (optional).
            ip_address: Client IP address (optional).

        Returns:
            New TokenPair.

        Raises:
            InvalidRefreshTokenError: If token is invalid/expired/revoked.
            TokenReuseDetectedError: If revoked token was reused.
        """
        token_hash = hash_token(refresh_token)
        stored_token = await self._repo.get_refresh_token_by_hash(token_hash)

        if stored_token is None:
            raise InvalidRefreshTokenError("Invalid refresh token")

        # Check if token was already revoked (potential theft)
        if stored_token.is_revoked:
            # Revoke entire token family as precaution
            revoked_count = await self._repo.revoke_token_family(stored_token.family_id)
            logger.warning(
                f"Token reuse detected for user {stored_token.user_id}. "
                f"Revoked {revoked_count} tokens in family {stored_token.family_id}"
            )
            raise TokenReuseDetectedError(
                "Security alert: This refresh token was already used. "
                "All sessions have been invalidated."
            )

        # Check expiration
        if stored_token.expires_at < datetime.now(timezone.utc):
            raise InvalidRefreshTokenError("Refresh token has expired")

        # Verify user is still active
        user = await self._repo.get_active_user(stored_token.user_id)
        if user is None:
            raise UserInactiveError("User account is deactivated")

        # Revoke current token
        await self._repo.revoke_refresh_token(stored_token.id)

        # Issue new token pair in same family
        tokens = await self._create_token_pair(
            stored_token.user_id,
            family_id=stored_token.family_id,
            user_agent=user_agent,
            ip_address=ip_address,
        )

        logger.debug(f"Tokens refreshed for user {stored_token.user_id}")

        return tokens

    async def logout(self, refresh_token: str) -> bool:
        """Logout by revoking the refresh token.

        Args:
            refresh_token: Refresh token to revoke.

        Returns:
            True if token was revoked, False if not found.
        """
        token_hash = hash_token(refresh_token)
        stored_token = await self._repo.get_refresh_token_by_hash(token_hash)

        if stored_token is None:
            return False

        await self._repo.revoke_refresh_token(stored_token.id)
        logger.debug(f"User {stored_token.user_id} logged out")

        return True

    async def logout_all(self, user_id: UUID) -> int:
        """Logout from all devices by revoking all tokens.

        Args:
            user_id: User ID.

        Returns:
            Number of tokens revoked.
        """
        count = await self._repo.revoke_all_user_tokens(user_id)
        logger.info(f"Revoked {count} tokens for user {user_id}")
        return count

    async def _create_token_pair(
        self,
        user_id: UUID,
        *,
        family_id: UUID | None = None,
        user_agent: str | None = None,
        ip_address: str | None = None,
    ) -> TokenPair:
        """Create new access and refresh token pair.

        Args:
            user_id: User ID.
            family_id: Token family ID (for rotation).
            user_agent: Client user agent.
            ip_address: Client IP address.

        Returns:
            TokenPair with both tokens.
        """
        # Create access token
        access_token, expires_at = self._jwt.create_access_token(user_id)
        expires_in = int(self._jwt.access_token_lifetime.total_seconds())

        # Create refresh token
        refresh_token = generate_token(32)
        refresh_token_hash = hash_token(refresh_token)
        refresh_expires_at = datetime.now(timezone.utc) + self._jwt.refresh_token_lifetime

        await self._repo.create_refresh_token(
            id=uuid4(),
            user_id=user_id,
            token_hash=refresh_token_hash,
            family_id=family_id or uuid4(),
            expires_at=refresh_expires_at,
            user_agent=user_agent,
            ip_address=ip_address,
        )

        return TokenPair(
            access_token=access_token,
            refresh_token=refresh_token,
            expires_at=expires_at,
            expires_in=expires_in,
        )
