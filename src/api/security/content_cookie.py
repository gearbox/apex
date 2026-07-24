"""Content cookie helpers for auth-gated media access."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

from litestar.datastructures import Cookie

from src.core.config import get_settings

if TYPE_CHECKING:
    from uuid import UUID

    from litestar import Response

    from src.api.security.jwt import JWTService
    from src.core.config import Settings
    from src.core.product import ProductConfig


def content_cookie_lifetime(settings: Settings) -> tuple[int, datetime]:
    """Single source of truth for the content cookie's lifetime.

    Returns both the Set-Cookie Max-Age (seconds) and the absolute expiry
    (UTC) derived from the same instant, so build_content_cookie's Max-Age
    and any advertised expiry (ContentCookieResponse.expires_at,
    TokenResponse.content_cookie_expires_at) can never diverge.

    Args:
        settings: Application settings supplying the configured TTL.

    Returns:
        Tuple of (max_age_seconds, expires_at).
    """
    max_age = settings.content_cookie_ttl_hours * 3600
    expires_at = datetime.now(UTC) + timedelta(seconds=max_age)
    return max_age, expires_at


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

    Args:
        user_id: Authenticated user's UUID.
        product_id: Product slug for the token scope.
        jwt_service: JWT service used to sign the token.
        settings: Application settings (TTL, secure flag, cookie name).
        product_config: Product config supplying the cookie domain.

    Returns:
        Tuple of (cookie, expires_at) — expires_at is the cookie's absolute
        expiry (UTC), derived from the same content_cookie_lifetime() call
        used to build the cookie's Max-Age.
    """
    max_age, expires_at = content_cookie_lifetime(settings)
    content_token, _ = jwt_service.create_content_token(
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


def attach_content_cookie(
    response: Response[Any],
    *,
    user_id: UUID,
    product_id: str,
    jwt_service: JWTService,
    settings: Settings,
    product_config: ProductConfig,
) -> datetime:
    """Mint a content token and append the Set-Cookie to the response.

    Args:
        response: Litestar Response to append the cookie to.
        user_id: Authenticated user's UUID.
        product_id: Product slug for the token scope.
        jwt_service: JWT service used to sign the token.
        settings: Application settings (TTL, secure flag, cookie name).
        product_config: Product config supplying the cookie domain.

    Returns:
        The cookie's absolute expiry (UTC) — callers advertise this via
        ContentCookieResponse.expires_at / TokenResponse.content_cookie_expires_at.
    """
    cookie, expires_at = mint_content_cookie(
        user_id=user_id,
        product_id=product_id,
        jwt_service=jwt_service,
        settings=settings,
        product_config=product_config,
    )
    response.cookies.append(cookie)
    return expires_at
