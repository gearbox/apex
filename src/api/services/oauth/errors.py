"""OAuth flow errors — one class per wire ``OAuthErrorCode``."""

from __future__ import annotations

from typing import ClassVar

from src.core.enums import OAuthErrorCode


class OAuthError(Exception):
    """Base for every error that ends an OAuth flow with a wire error code."""

    code: ClassVar[OAuthErrorCode]

    def __init__(self, message: str | None = None) -> None:
        super().__init__(message or self.code.value)


class OAuthCancelledError(OAuthError):
    """The user declined consent at the provider (``error=access_denied``)."""

    code = OAuthErrorCode.OAUTH_CANCELLED


class OAuthFailedError(OAuthError):
    """Token endpoint, JWKS, or id_token verification failure."""

    code = OAuthErrorCode.OAUTH_FAILED


class FlowExpiredError(OAuthError):
    """State unknown/consumed, or binding/product/provider mismatch at the callback."""

    code = OAuthErrorCode.FLOW_EXPIRED


class EmailUnverifiedError(OAuthError):
    """The provider did not assert ``email_verified: true``."""

    code = OAuthErrorCode.EMAIL_UNVERIFIED


class AccountExistsUnverifiedError(OAuthError):
    """A same-email local account exists but its email was never verified (no auto-link)."""

    code = OAuthErrorCode.ACCOUNT_EXISTS_UNVERIFIED


class AccountInactiveError(OAuthError):
    """The linked account is deactivated."""

    code = OAuthErrorCode.ACCOUNT_INACTIVE


class IdentityConflictError(OAuthError):
    """The account is already linked to a different subject of this provider."""

    code = OAuthErrorCode.IDENTITY_CONFLICT


class InvalidHandoffError(OAuthError):
    """Handoff code unknown/consumed/expired, or binding/product mismatch."""

    code = OAuthErrorCode.INVALID_HANDOFF


class InvalidSignupTicketError(OAuthError):
    """Signup ticket unknown/consumed/expired, or binding/product mismatch."""

    code = OAuthErrorCode.INVALID_SIGNUP_TICKET


class OAuthProviderNotEnabledError(Exception):
    """The provider is not allowed or not configured for the product (404 — not a flow error)."""
