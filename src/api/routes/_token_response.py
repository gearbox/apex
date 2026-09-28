"""Shared builder for every endpoint that hands out a session (``TokenResponse``)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from litestar import Response

from src.api.schemas.auth import TokenResponse
from src.api.security.content_cookie import mint_content_cookie

if TYPE_CHECKING:
    from collections.abc import Sequence
    from uuid import UUID

    from litestar.datastructures import Cookie

    from src.api.schemas.errors import ErrorEnvelope
    from src.api.security.jwt import JWTService
    from src.api.services.auth import TokenPair
    from src.core.config import Settings
    from src.core.product import ProductConfig


def build_token_response(
    *,
    user_id: UUID,
    product_id: str,
    tokens: TokenPair,
    jwt_service: JWTService,
    settings: Settings,
    product_config: ProductConfig,
    status_code: int,
    extra_cookies: Sequence[Cookie] = (),
) -> Response[TokenResponse | ErrorEnvelope]:
    """Mint the content cookie and wrap the token pair in the standard response.

    Used by register, login, refresh, and the OAuth exchange / complete-signup
    endpoints so all five return byte-identical bodies and cookies.

    Args:
        user_id: The authenticated user (content-cookie subject).
        product_id: Product scope for the content token.
        tokens: The freshly minted access/refresh pair.
        jwt_service: Signs the content token.
        settings: Content-cookie TTL / secure flag.
        product_config: Supplies the content-cookie domain.
        status_code: 200 or 201.
        extra_cookies: Additional Set-Cookies (e.g. clearing ``apex_oauth_tx``).
    """
    content_cookie, content_cookie_expires_at = mint_content_cookie(
        user_id=user_id,
        product_id=product_id,
        jwt_service=jwt_service,
        settings=settings,
        product_config=product_config,
    )
    return Response(
        content=TokenResponse(
            access_token=tokens.access_token,
            refresh_token=tokens.refresh_token,
            expires_in=tokens.expires_in,
            expires_at=tokens.expires_at,
            content_cookie_expires_at=content_cookie_expires_at,
        ),
        status_code=status_code,
        cookies=[content_cookie, *extra_cookies],
    )
