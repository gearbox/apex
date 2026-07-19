"""Content cookie helpers for auth-gated media access."""

from __future__ import annotations

from datetime import timedelta
from typing import TYPE_CHECKING, Any

from litestar.datastructures import Cookie

from src.core.config import get_settings

if TYPE_CHECKING:
    from uuid import UUID

    from litestar import Response

    from src.api.security.jwt import JWTService
    from src.core.config import Settings
    from src.core.product import ProductConfig


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


def attach_content_cookie(
    response: Response[Any],
    *,
    user_id: UUID,
    product_id: str,
    jwt_service: JWTService,
    settings: Settings,
    product_config: ProductConfig,
) -> None:
    """Mint a content token and append the Set-Cookie to the response.

    Args:
        response: Litestar Response to append the cookie to.
        user_id: Authenticated user's UUID.
        product_id: Product slug for the token scope.
        jwt_service: JWT service used to sign the token.
        settings: Application settings (TTL, secure flag, cookie name).
        product_config: Product config supplying the cookie domain.
    """
    content_token, _ = jwt_service.create_content_token(
        user_id,
        product_id=product_id,
        ttl=timedelta(hours=settings.content_cookie_ttl_hours),
    )
    response.cookies.append(
        build_content_cookie(
            content_token,
            domain=effective_cookie_domain(settings, product_config),
            secure=settings.content_cookie_secure,
            max_age=settings.content_cookie_ttl_hours * 3600,
        )
    )
