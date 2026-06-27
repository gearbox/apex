"""Content cookie helpers for auth-gated media access."""

from __future__ import annotations

from litestar.datastructures import Cookie

from src.core.config import get_settings


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
        domain: Registrable domain (e.g. "vex.pics"), or None for localhost.
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
