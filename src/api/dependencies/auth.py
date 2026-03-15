"""Shared authentication dependencies for route handlers.

Provides reusable dependency functions that extract authentication
context from request state (set by auth_guard).
"""

from __future__ import annotations

from typing import Any
from uuid import UUID

from litestar import Request
from litestar.exceptions import NotAuthorizedException
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.enums import UserRole
from src.db.models import User
from src.db.repositories import UserRepository


async def get_current_user_id(request: Request[Any, Any, Any]) -> UUID:
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
    return UUID(str(user_id))


async def get_current_admin_user(request: Request[Any, Any, Any], session: AsyncSession) -> User:
    """Load and verify the current user is an admin.

    Uses the request-scoped session and UserRepository to load the
    user from DB and check the is_admin flag. Replaces the old
    admin_guard which created a separate session outside DI.

    Args:
        request: Litestar request.
        session: Request-scoped database session.

    Returns:
        Verified admin User model.

    Raises:
        NotAuthorizedException: If not authenticated or not admin.
    """
    user_id = request.state.get("user_id")
    if user_id is None:
        raise NotAuthorizedException(detail="Not authenticated")

    repo = UserRepository(session)
    user = await repo.get_active_user(user_id)

    if user is None or user.role != UserRole.ADMIN:
        raise NotAuthorizedException(detail="Admin access required")

    return user


async def get_optional_user_id(request: Request[Any, Any, Any]) -> UUID | None:
    """Extract current user ID from request state, or None if unauthenticated.

    Must be used in routes protected by optional_auth_guard, which populates
    request.state["user_id"] after optionally validating the JWT token.

    Returns:
        Authenticated user's UUID, or None if no token was provided.
    """
    user_id = request.state.get("user_id")
    return None if user_id is None else UUID(str(user_id))
