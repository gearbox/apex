"""Authentication schemas using msgspec."""

from __future__ import annotations

from datetime import datetime
from typing import Annotated

import msgspec

# -----------------------------------------------------------------------------
# Request schemas
# -----------------------------------------------------------------------------


class RegisterRequest(msgspec.Struct, kw_only=True):
    """User registration request."""

    email: str
    password: str
    display_name: str | None = None


class LoginRequest(msgspec.Struct, kw_only=True):
    """User login request."""

    email: str
    password: str


class RefreshTokenRequest(msgspec.Struct, kw_only=True):
    """Token refresh request."""

    refresh_token: str


class PKCEAuthRequest(msgspec.Struct, kw_only=True):
    """PKCE authorization request.

    Used for OAuth2 PKCE flow initiation.
    """

    code_challenge: str
    code_challenge_method: str = "S256"
    state: str | None = None


class PKCETokenRequest(msgspec.Struct, kw_only=True):
    """PKCE token exchange request."""

    code: str
    code_verifier: str


class VerifyEmailRequest(msgspec.Struct, forbid_unknown_fields=True, kw_only=True):
    """Token from the verification link query string."""

    token: Annotated[str, msgspec.Meta(min_length=20, max_length=100)]


class ForgotPasswordRequest(msgspec.Struct, forbid_unknown_fields=True, kw_only=True):
    """Email address to send the reset link to."""

    email: Annotated[str, msgspec.Meta(min_length=3, max_length=255)]


class ResetPasswordRequest(msgspec.Struct, forbid_unknown_fields=True, kw_only=True):
    """Token + new password to apply."""

    token: Annotated[str, msgspec.Meta(min_length=20, max_length=100)]
    new_password: Annotated[str, msgspec.Meta(min_length=8, max_length=128)]


# -----------------------------------------------------------------------------
# Response schemas
# -----------------------------------------------------------------------------


class TokenResponse(msgspec.Struct, kw_only=True):
    """Authentication token response."""

    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int  # Seconds until access token expires
    expires_at: datetime  # Absolute expiration time


class AccessTokenResponse(msgspec.Struct, kw_only=True):
    """Access token only response (for refresh)."""

    access_token: str
    token_type: str = "bearer"
    expires_in: int
    expires_at: datetime


class AuthUserResponse(msgspec.Struct, kw_only=True):
    """Authenticated user info response."""

    id: str
    email: str
    display_name: str | None
    subscription_tier: str
    created_at: datetime


class MessageResponse(msgspec.Struct, kw_only=True):
    """Generic message response."""

    message: str


