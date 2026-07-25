"""Authentication guards for Litestar routes."""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING, Any
from uuid import UUID

import structlog
from litestar.exceptions import NotAuthorizedException

from src.core.config import get_settings

if TYPE_CHECKING:
    from collections.abc import Callable

    from litestar.connection import ASGIConnection
    from litestar.handlers import BaseRouteHandler

    from src.api.security.jwt import JWTService, TokenPayload
    from src.api.services.token_revocation import TokenRevocationService
    from src.db.models import User

logger = structlog.get_logger(__name__)


class AuthenticatedUser:
    """Represents an authenticated user in request scope.

    This is injected into route handlers that require authentication.
    """

    def __init__(self, user_id: UUID, user: User | None = None) -> None:
        """Initialize authenticated user.

        Args:
            user_id: User's UUID from token.
            user: Full user model (loaded lazily if needed).
        """
        self.user_id = user_id
        self._user = user

    @property
    def user(self) -> User:
        """Get full user model.

        Raises:
            RuntimeError: If user not loaded.
        """
        if self._user is None:
            raise RuntimeError("User not loaded. Use get_current_user dependency.")
        return self._user

    def __repr__(self) -> str:
        return f"<AuthenticatedUser {self.user_id}>"


def extract_token_from_header(authorization: str | None) -> str | None:
    """Extract bearer token from Authorization header.

    Args:
        authorization: Authorization header value.

    Returns:
        Token string if valid bearer token, None otherwise.
    """
    if not authorization:
        return None

    parts = authorization.split()
    return None if len(parts) != 2 or parts[0].lower() != "bearer" else parts[1]


def _identity_from_token(
    raw: str | None,
    decode: Callable[[str], TokenPayload | None],
) -> tuple[UUID, TokenPayload] | None:
    """Decode a raw token string into (user_id, payload).

    Returns the full TokenPayload (not just product_id) so callers can also
    run it through TokenRevocationService.is_revoked(), which needs `sub`,
    `iat`, and `jti`. Returns None if the token is absent, invalid, or has an
    unparseable subject.
    """
    if not raw:
        return None
    payload = decode(raw)
    if payload is None:
        return None
    try:
        return UUID(payload.sub), payload
    except ValueError:
        return None


def _enforce_product(
    connection: ASGIConnection[Any, Any, Any, Any],
    token_product_id: str | None,
) -> None:
    """Raise 401 if the token's product_id is absent or mismatches the request product.

    A None token_product_id is always rejected on product-scoped requests — all
    validly issued tokens carry a product_id, so None means malformed/legacy token.
    """
    request_product_id: str | None = None
    with contextlib.suppress(Exception):
        request_product_id = connection.state.get("product_id")
    if request_product_id is None:
        return
    if token_product_id is None:
        raise NotAuthorizedException(detail="Token is not scoped to the requested product")
    if token_product_id != request_product_id:
        raise NotAuthorizedException(detail="Token was issued for a different product")


def _get_token_revocation(
    connection: ASGIConnection[Any, Any, Any, Any],
) -> TokenRevocationService:
    """Fetch the TokenRevocationService wired into app.state by the lifespan.

    Mirrors the jwt_service lookup pattern — always present once the app has
    started, so a miss means the app wasn't wired correctly, not a runtime
    condition to degrade on.
    """
    token_revocation: TokenRevocationService | None = connection.app.state.get("token_revocation")
    if token_revocation is None:
        raise RuntimeError("Token revocation service not configured")
    return token_revocation


async def auth_guard(connection: ASGIConnection[Any, Any, Any, Any], _: BaseRouteHandler) -> None:
    """Guard that requires valid JWT authentication.

    Extracts and validates JWT from Authorization header.
    Sets user_id in connection state for downstream handlers.

    Args:
        connection: ASGI connection.
        _: Route handler (unused).

    Raises:
        NotAuthorizedException: If authentication fails.
    """
    authorization = connection.headers.get("authorization")
    token = extract_token_from_header(authorization)

    if not token:
        raise NotAuthorizedException(detail="Missing authorization header")

    jwt_service: JWTService | None = connection.app.state.get("jwt_service")
    if jwt_service is None:
        raise RuntimeError("JWT service not configured")

    identity = _identity_from_token(token, jwt_service.decode_access_token)
    if identity is None:
        raise NotAuthorizedException(detail="Invalid or expired token")

    user_id, payload = identity
    # Cheap, local check first; the Redis round-trip only runs for tokens
    # that already passed product scoping.
    _enforce_product(connection, payload.product_id)

    token_revocation = _get_token_revocation(connection)
    if await token_revocation.is_revoked(payload):
        logger.info("auth.token_revoked", user_id=str(user_id))
        raise NotAuthorizedException(detail="Invalid or expired token")

    connection.state["user_id"] = user_id
    connection.state["auth_user"] = AuthenticatedUser(user_id=user_id)


async def content_auth_guard(
    connection: ASGIConnection[Any, Any, Any, Any], _: BaseRouteHandler
) -> None:
    """Guard that accepts either a Bearer access token or a content cookie.

    Used exclusively on the content GET proxy routes so that browser media
    elements (img, video) can authenticate via the first-party SameSite=Lax
    cookie without needing an Authorization header.

    Args:
        connection: ASGI connection.
        _: Route handler (unused).

    Raises:
        NotAuthorizedException: If neither credential is valid.
    """
    settings = get_settings()
    jwt_service: JWTService | None = connection.app.state.get("jwt_service")
    if jwt_service is None:
        raise RuntimeError("JWT service not configured")

    # 1. Try Bearer access token (for SSR / API clients)
    raw_bearer = extract_token_from_header(connection.headers.get("authorization"))
    identity = _identity_from_token(raw_bearer, jwt_service.decode_access_token)

    # 2. Fall back to content cookie
    if identity is None:
        raw_cookie = connection.cookies.get(settings.content_cookie_name)
        identity = _identity_from_token(raw_cookie, jwt_service.decode_content_token)

    if identity is None:
        raise NotAuthorizedException(detail="Missing or invalid credentials")

    user_id, payload = identity

    # 3. Product check — token must belong to the resolved request product
    _enforce_product(connection, payload.product_id)

    # 4. Revocation check — covers both access tokens and content cookies:
    # logout_all's epoch rejects any token (of either type) issued before
    # it, and single-device logout's jti denylist rejects the specific
    # access token that was retired.
    token_revocation = _get_token_revocation(connection)
    if await token_revocation.is_revoked(payload):
        raise NotAuthorizedException(detail="Missing or invalid credentials")

    # 5. Mirror auth_guard state — downstream DI reads from here
    connection.state["user_id"] = user_id
    connection.state["auth_user"] = AuthenticatedUser(user_id=user_id)


async def optional_auth_guard(
    connection: ASGIConnection[Any, Any, Any, Any], _: BaseRouteHandler
) -> None:
    """Guard that optionally extracts JWT authentication.

    Does not raise if no token is provided. Routes through the same
    ``_identity_from_token`` / ``_enforce_product`` path as ``auth_guard``, so
    a token's product claim is checked here too — but unlike ``auth_guard``,
    a product mismatch degrades to anonymous instead of raising 401. An
    optional guard should degrade, not reject: a stale other-product token
    sitting in a shared browser (e.g. a prior vex-domain.com login while visiting a
    synthara.app page) must not break access to a page that doesn't require
    authentication in the first place.

    Sets user_id/auth_user in connection state if authenticated.

    Contract — read before attaching this guard to a new route:
        (a) This guard MUST only be attached to routes whose response is
            identical, or safely narrower, when the caller is anonymous.
            It is not a "try to authenticate" convenience — it is a promise
            that anonymous access to this route is intentional and safe.
        (b) A product-mismatched token always degrades to anonymous here,
            never raises 401. Do not rely on this guard to reject
            cross-product tokens — that is ``auth_guard``'s job.
        (c) Attaching this guard to a route that returns user-scoped or
            sensitive data merely because a token happened to be present is
            a security bug: an anonymous caller could get the same response
            by omitting the token, so the "optional" auth buys nothing and
            the route silently depends on callers behaving honestly.
        (d) A revoked token (logout-all, password change/reset, deactivation,
            or refresh-token reuse detection — issue #142) also degrades to
            anonymous here, never raises. Unlike ``auth_guard``/
            ``content_auth_guard``, this guard tolerates a missing
            ``token_revocation`` app-state entry (treats it as "not
            revoked") rather than raising — see A2: reusing
            ``_get_token_revocation()``'s raise-on-missing behavior here
            would break this guard's degrade-to-anonymous contract on an
            anonymous-capable route.
        Routes using this guard are enumerated and pinned in
        ``tests/unit/security/test_optional_auth_guard_usage.py`` — adding a
        new route requires updating that allowlist, forcing a conscious
        review of (a)-(c) for the new route.

    Args:
        connection: ASGI connection.
        _: Route handler (unused).
    """
    authorization = connection.headers.get("authorization")
    token = extract_token_from_header(authorization)

    jwt_service: JWTService | None = connection.app.state.get("jwt_service")
    identity: tuple[UUID, TokenPayload] | None = None
    if jwt_service is not None:
        identity = _identity_from_token(token, jwt_service.decode_access_token)

    if identity is not None:
        payload = identity[1]
        try:
            _enforce_product(connection, payload.product_id)
        except NotAuthorizedException:
            logger.debug("optional_auth.product_mismatch_treated_as_anonymous")
            identity = None

    if identity is not None:
        # Tolerant lookup (A2) — unlike auth_guard/content_auth_guard's
        # _get_token_revocation(), a missing service here must not raise:
        # this guard's whole contract is to degrade to anonymous, never
        # 500 an anonymous-capable route.
        token_revocation: TokenRevocationService | None = connection.app.state.get(
            "token_revocation"
        )
        if token_revocation is not None and await token_revocation.is_revoked(identity[1]):
            logger.debug("optional_auth.revoked_treated_as_anonymous")
            identity = None

    if identity is not None:
        user_id = identity[0]
        connection.state["user_id"] = user_id
        connection.state["auth_user"] = AuthenticatedUser(user_id=user_id)
    else:
        connection.state["user_id"] = None
        connection.state["auth_user"] = None
