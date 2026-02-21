"""Shared authentication dependencies for route handlers.

Provides reusable dependency functions that extract authentication
context from request state (set by auth_guard).
"""

from __future__ import annotations

from uuid import UUID

from litestar import Request
from litestar.exceptions import NotAuthorizedException


async def get_current_user_id(request: Request) -> UUID:
    """Extract current user ID from request state.

    Must be used in routes protected by auth_guard, which populates
    request.state["user_id"] after validating the JWT token.

    Args:
        request: Litestar request.

    Returns:
        Authenticated user's UUID.

    Raises:
        NotAuthorizedException: If not authenticated.
    """
    user_id = request.state.get("user_id")
    if user_id is None:
        raise NotAuthorizedException(detail="Not authenticated")
    return user_id
