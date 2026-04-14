"""Authentication guards for Litestar routes."""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING, Any
from uuid import UUID

import structlog
from litestar.connection import ASGIConnection
from litestar.exceptions import NotAuthorizedException
from litestar.handlers import BaseRouteHandler

if TYPE_CHECKING:
    from src.api.security.jwt import JWTService
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

    # Get JWT service from app state
    jwt_service: JWTService | None = connection.app.state.get("jwt_service")
    if jwt_service is None:
        raise RuntimeError("JWT service not configured")

    token_payload = jwt_service.decode_access_token(token)
    if token_payload is None:
        raise NotAuthorizedException(detail="Invalid or expired token")

    try:
        user_id = UUID(token_payload.sub)
    except ValueError as exc:
        raise NotAuthorizedException(detail="Invalid token subject") from exc

    # Validate product_id claim matches request product context
    request_product_id: str | None = None
    with contextlib.suppress(Exception):
        request_product_id = connection.state.get("product_id")

    if request_product_id is not None:
        token_product_id = token_payload.product_id
        if token_product_id != request_product_id:
            raise NotAuthorizedException(detail="Token was issued for a different product")

    # Store user_id in connection state for dependency injection
    connection.state["user_id"] = user_id
    connection.state["auth_user"] = AuthenticatedUser(user_id=user_id)


async def optional_auth_guard(
    connection: ASGIConnection[Any, Any, Any, Any], _: BaseRouteHandler
) -> None:
    """Guard that optionally extracts JWT authentication.

    Does not raise if no token provided, but validates if present.
    Sets user_id in connection state if authenticated.

    Args:
        connection: ASGI connection.
        _: Route handler (unused).
    """
    authorization = connection.headers.get("authorization")
    token = extract_token_from_header(authorization)

    if not token:
        connection.state["user_id"] = None
        connection.state["auth_user"] = None
        return

    jwt_service: JWTService | None = connection.app.state.get("jwt_service")
    if jwt_service is None:
        connection.state["user_id"] = None
        connection.state["auth_user"] = None
        return

    if user_id := jwt_service.get_user_id_from_token(token):
        connection.state["user_id"] = user_id
        connection.state["auth_user"] = AuthenticatedUser(user_id=user_id)
    else:
        connection.state["user_id"] = None
        connection.state["auth_user"] = None
