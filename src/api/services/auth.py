"""Authentication service for user registration and login."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING
from uuid import UUID

import structlog

from src.api.security import (
    JWTService,
    PasswordService,
    generate_token,
    hash_token,
)
from src.core.uid import new_id
from src.db.repositories import UserRepository
from src.db.repositories.billing import BillingRepository

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from src.api.services.email_verification import EmailVerificationService
    from src.db.models import User

logger = structlog.get_logger(__name__)


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
        session: AsyncSession | None = None,
        email_verification_service: EmailVerificationService | None = None,
    ) -> None:
        """Initialize auth service.

        Args:
            repository: User repository.
            jwt_service: JWT token service.
            password_service: Password hashing service.
            session: Database session (for billing account creation).
            email_verification_service: Optional — when provided, sends a
                verification email immediately after successful registration.
                Pass ``None`` to skip email sending (e.g. in tests).
        """
        self._repo = repository
        self._jwt = jwt_service
        self._password = password_service
        self._session = session
        self._email_verification = email_verification_service

    async def register(
        self,
        *,
        email: str,
        password: str,
        product_id: str,
        display_name: str | None = None,
    ) -> tuple[User, TokenPair]:
        """Register a new user.

        Args:
            email: User email.
            password: Plain text password.
            product_id: Product the user is registering on.
            display_name: Optional display name.

        Returns:
            Tuple of (User, TokenPair).

        Raises:
            EmailAlreadyExistsError: If email is taken on this product.
        """
        # Check for existing email within the same product
        if await self._repo.email_exists(email, product_id=product_id):
            raise EmailAlreadyExistsError(f"Email {email} is already registered")

        # Create user
        user_id = new_id()
        password_hash = self._password.hash(password)

        user = await self._repo.create_user(
            id=user_id,
            email=email,
            password_hash=password_hash,
            product_id=product_id,
            display_name=display_name,
        )

        # Create personal token account in the same transaction
        if self._session is not None:
            billing_repo = BillingRepository(self._session)
            await billing_repo.create_personal_account(
                id=new_id(), user_id=user_id, product_id=product_id
            )
            logger.info("billing.account_created", user_id=str(user_id))

        logger.info("user.registered", user_id=str(user_id))

        # Generate tokens
        tokens = await self._create_token_pair(user_id, product_id=product_id)

        # Send verification email — non-blocking failure: if the email provider
        # is down we still complete registration and let the user resend manually.
        if self._email_verification is not None and self._session is not None:
            try:
                await self._email_verification.send_verification_email(
                    user_id, session=self._session
                )
            except Exception:
                logger.exception(
                    "auth.email_verification_failed",
                    user_id=str(user_id),
                    email=email,
                )

        return user, tokens

    async def login(
        self,
        *,
        email: str,
        password: str,
        product_id: str,
        user_agent: str | None = None,
        ip_address: str | None = None,
    ) -> tuple[User, TokenPair]:
        """Authenticate user and return tokens.

        Args:
            email: User email.
            password: Plain text password.
            product_id: Product the user is logging into.
            user_agent: Client user agent (optional).
            ip_address: Client IP address (optional).

        Returns:
            Tuple of (User, TokenPair).

        Raises:
            InvalidCredentialsError: If email or password is wrong, or user
                does not exist on this product.
            UserInactiveError: If user account is deactivated.
        """
        user = await self._repo.get_active_user_by_email(email, product_id=product_id)

        if user is None:
            # Prevent timing attacks — covers both not-found and inactive accounts
            self._password.hash("dummy_password")
            raise InvalidCredentialsError("Invalid email or password")

        if not self._password.verify(user.password_hash, password):
            raise InvalidCredentialsError("Invalid email or password")

        # Check if password needs rehash (parameter upgrade)
        if self._password.needs_rehash(user.password_hash):
            new_hash = self._password.hash(password)
            await self._repo.update_user(user.id, password_hash=new_hash)
            logger.info("user.password_rehashed", user_id=str(user.id))

        logger.info("user.logged_in", user_id=str(user.id))

        tokens = await self._create_token_pair(
            user.id,
            product_id=product_id,
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
                "auth.token_reuse_detected",
                user_id=str(stored_token.user_id),
                revoked=revoked_count,
                family=str(stored_token.family_id),
            )
            raise TokenReuseDetectedError(
                "Security alert: This refresh token was already used. "
                "All sessions have been invalidated."
            )

        # Check expiration
        if stored_token.expires_at < datetime.now(UTC):
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
            product_id=stored_token.product_id,
            family_id=stored_token.family_id,
            user_agent=user_agent,
            ip_address=ip_address,
        )

        logger.debug("auth.token_refreshed", user_id=str(stored_token.user_id))

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
        logger.debug("auth.logged_out", user_id=str(stored_token.user_id))

        return True

    async def logout_all(self, user_id: UUID) -> int:
        """Logout from all devices by revoking all tokens.

        Args:
            user_id: User ID.

        Returns:
            Number of tokens revoked.
        """
        count = await self._repo.revoke_all_user_tokens(user_id)
        logger.info("auth.tokens_revoked", user_id=str(user_id), count=count)
        return count

    async def _create_token_pair(
        self,
        user_id: UUID,
        *,
        product_id: str | None = None,
        family_id: UUID | None = None,
        user_agent: str | None = None,
        ip_address: str | None = None,
    ) -> TokenPair:
        """Create new access and refresh token pair.

        Args:
            user_id: User ID.
            product_id: Product scope to embed in the JWT.
            family_id: Token family ID (for rotation).
            user_agent: Client user agent.
            ip_address: Client IP address.

        Returns:
            TokenPair with both tokens.
        """
        # Create access token
        access_token, expires_at = self._jwt.create_access_token(user_id, product_id=product_id)
        expires_in = int(self._jwt.access_token_lifetime.total_seconds())

        # Create refresh token
        refresh_token = generate_token(32)
        refresh_token_hash = hash_token(refresh_token)
        refresh_expires_at = datetime.now(UTC) + self._jwt.refresh_token_lifetime

        await self._repo.create_refresh_token(
            id=new_id(),
            user_id=user_id,
            token_hash=refresh_token_hash,
            family_id=family_id or new_id(),
            expires_at=refresh_expires_at,
            product_id=product_id or "vex",
            user_agent=user_agent,
            ip_address=ip_address,
        )

        return TokenPair(
            access_token=access_token,
            refresh_token=refresh_token,
            expires_at=expires_at,
            expires_in=expires_in,
        )
