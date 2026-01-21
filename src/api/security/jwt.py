"""JWT token handling for authentication."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import UUID

import jwt

from .password import generate_token


@dataclass
class TokenPayload:
    """JWT access token payload."""

    sub: str  # Subject (user_id)
    exp: int  # Expiration timestamp
    iat: int  # Issued at timestamp
    jti: str  # JWT ID (for tracking/revocation)
    type: str = "access"  # Token type


@dataclass(frozen=True)
class JWTConfig:
    """JWT configuration."""

    secret_key: str
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 15
    refresh_token_expire_days: int = 7
    issuer: str | None = None
    audience: str | None = None


class JWTService:
    """JWT token creation and validation.

    Handles creation and validation of access tokens.
    Refresh tokens are stored in the database, not as JWTs.
    """

    def __init__(self, config: JWTConfig) -> None:
        """Initialize JWT service.

        Args:
            config: JWT configuration.
        """
        self._config = config

    @property
    def access_token_lifetime(self) -> timedelta:
        """Access token lifetime."""
        return timedelta(minutes=self._config.access_token_expire_minutes)

    @property
    def refresh_token_lifetime(self) -> timedelta:
        """Refresh token lifetime."""
        return timedelta(days=self._config.refresh_token_expire_days)

    def create_access_token(
        self,
        user_id: UUID,
        *,
        jti: str | None = None,
        extra_claims: dict[str, Any] | None = None,
    ) -> tuple[str, datetime]:
        """Create a new access token.

        Args:
            user_id: User ID to encode in token.
            jti: Optional JWT ID for tracking.
            extra_claims: Optional additional claims.

        Returns:
            Tuple of (token string, expiration datetime).
        """
        now = datetime.now(timezone.utc)
        expires_at = now + self.access_token_lifetime

        payload: dict[str, Any] = {
            "sub": str(user_id),
            "exp": int(expires_at.timestamp()),
            "iat": int(now.timestamp()),
            "jti": jti or generate_token(16),
            "type": "access",
        }

        if self._config.issuer:
            payload["iss"] = self._config.issuer
        if self._config.audience:
            payload["aud"] = self._config.audience
        if extra_claims:
            payload |= extra_claims

        token = jwt.encode(
            payload,
            self._config.secret_key,
            algorithm=self._config.algorithm,
        )

        return token, expires_at

    def decode_access_token(self, token: str) -> TokenPayload | None:
        """Decode and validate an access token.

        Args:
            token: JWT token string.

        Returns:
            TokenPayload if valid, None otherwise.
        """
        try:
            options = {"require": ["sub", "exp", "iat", "jti"]}

            payload = jwt.decode(
                token,
                self._config.secret_key,
                algorithms=[self._config.algorithm],
                options=options,
                issuer=self._config.issuer,
                audience=self._config.audience,
            )

            # Verify token type
            if payload.get("type") != "access":
                return None

            return TokenPayload(
                sub=payload["sub"],
                exp=payload["exp"],
                iat=payload["iat"],
                jti=payload["jti"],
                type=payload.get("type", "access"),
            )

        except jwt.ExpiredSignatureError:
            return None
        except jwt.InvalidTokenError:
            return None

    def get_user_id_from_token(self, token: str) -> UUID | None:
        """Extract user ID from a valid token.

        Args:
            token: JWT token string.

        Returns:
            User UUID if valid, None otherwise.
        """
        payload = self.decode_access_token(token)
        if payload is None:
            return None
        try:
            return UUID(payload.sub)
        except ValueError:
            return None


class InvalidTokenError(Exception):
    """Raised when token is invalid or expired."""

    pass


class TokenExpiredError(InvalidTokenError):
    """Raised when token has expired."""

    pass
