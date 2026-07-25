"""Content cookie helpers for auth-gated media access."""

from __future__ import annotations

from datetime import timedelta
from typing import TYPE_CHECKING

from litestar.datastructures import Cookie

from src.core.config import get_settings

if TYPE_CHECKING:
    from datetime import datetime
    from uuid import UUID

    from src.api.security.jwt import JWTService
    from src.core.config import Settings
    from src.core.product import ProductConfig


def content_cookie_max_age(settings: Settings) -> int:
    """Content cookie Set-Cookie Max-Age, in seconds.

    Just the configured TTL converted to seconds — the advertised absolute
    expiry is no longer computed here. See mint_content_cookie for why: it
    comes from create_content_token's returned `exp` instead, so there is
    one clock read for the value that actually governs auth, not two.

    Args:
        settings: Application settings supplying the configured TTL.

    Returns:
        Max-Age in seconds.
    """
    return settings.content_cookie_ttl_hours * 3600


def effective_cookie_domain(settings: Settings, product_config: ProductConfig) -> str | None:
    """Return the cookie Domain to use, or None for a host-only cookie.

    Host-only in dev (when Secure is also dropped) so the content cookie is
    actually stored over http://localhost; the product's registrable domain
    in production. Keyed off the same flag as the Secure attribute so the two
    never diverge.
    """
    return product_config.cookie_domain if settings.content_cookie_secure else None


def build_content_cookie(
    token: str,
    *,
    domain: str | None,
    secure: bool,
    max_age: int,
) -> Cookie:
    """Build a Set-Cookie for the content authentication cookie.

    Args:
        token: Signed content JWT to store.
        domain: Registrable domain (e.g. "vex-domain.com"), or None for localhost.
        secure: Whether to set the Secure attribute.
        max_age: Cookie lifetime in seconds.

    Returns:
        Litestar Cookie ready to attach to a Response.
    """
    settings = get_settings()
    return Cookie(
        key=settings.content_cookie_name,
        value=token,
        httponly=True,
        secure=secure,
        samesite="lax",
        path="/v1/content",
        domain=domain,
        max_age=max_age,
    )


def clear_content_cookie(*, domain: str | None, secure: bool) -> Cookie:
    """Build a Set-Cookie that expires the content authentication cookie.

    Args:
        domain: Same domain used when the cookie was set.
        secure: Same Secure attribute used when the cookie was set.

    Returns:
        Litestar Cookie with Max-Age=0 to clear the cookie.
    """
    settings = get_settings()
    return Cookie(
        key=settings.content_cookie_name,
        value="",
        httponly=True,
        secure=secure,
        samesite="lax",
        path="/v1/content",
        domain=domain,
        max_age=0,
    )


def mint_content_cookie(
    *,
    user_id: UUID,
    product_id: str,
    jwt_service: JWTService,
    settings: Settings,
    product_config: ProductConfig,
) -> tuple[Cookie, datetime]:
    """Mint a content token and its Set-Cookie, independent of any Response.

    Lets callers know the expiry *before* constructing a response body (e.g.
    to populate TokenResponse.content_cookie_expires_at / ContentCookieResponse.expires_at
    in the same struct literal, rather than mutating a response after the fact).

    Truth flow: config → max_age → JWT `exp` → advertised `expires_at`. The
    returned expires_at *is* jwt_service.create_content_token's returned
    expiry — the same object, not a parallel `now() + max_age` computation —
    so the advertised value can never diverge from the `exp` that actually
    governs the guard.

    Args:
        user_id: Authenticated user's UUID.
        product_id: Product slug for the token scope.
        jwt_service: JWT service used to sign the token.
        settings: Application settings (TTL, secure flag, cookie name).
        product_config: Product config supplying the cookie domain.

    Returns:
        Tuple of (cookie, expires_at) — expires_at is the token's `exp` claim.
    """
    max_age = content_cookie_max_age(settings)
    content_token, expires_at = jwt_service.create_content_token(
        user_id,
        product_id=product_id,
        ttl=timedelta(seconds=max_age),
    )
    cookie = build_content_cookie(
        content_token,
        domain=effective_cookie_domain(settings, product_config),
        secure=settings.content_cookie_secure,
        max_age=max_age,
    )
    return cookie, expires_at
