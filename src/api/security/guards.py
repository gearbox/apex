"""Authentication guards for Litestar routes."""

from __future__ import annotations

from typing import TYPE_CHECKING
from uuid import UUID

from litestar.connection import ASGIConnection
from litestar.exceptions import NotAuthorizedException
from litestar.handlers import BaseRouteHandler

if TYPE_CHECKING:
    from src.api.security.jwt import JWTService
    from src.db.models import User


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


async def auth_guard(connection: ASGIConnection, _: BaseRouteHandler) -> None:
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

    user_id = jwt_service.get_user_id_from_token(token)
    if user_id is None:
        raise NotAuthorizedException(detail="Invalid or expired token")

    # Store user_id in connection state for dependency injection
    connection.state["user_id"] = user_id
    connection.state["auth_user"] = AuthenticatedUser(user_id=user_id)


async def optional_auth_guard(connection: ASGIConnection, _: BaseRouteHandler) -> None:
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
