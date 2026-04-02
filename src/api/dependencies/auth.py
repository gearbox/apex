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
    """Load and verify the current user has admin-level access (ADMIN or SUPERADMIN).

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

    if user is None or user.role not in (UserRole.ADMIN, UserRole.SUPERADMIN):
        raise NotAuthorizedException(detail="Admin access required")

    return user


async def get_current_superadmin_user(
    request: Request[Any, Any, Any], session: AsyncSession
) -> User:
    """Load and verify the current user is a SUPERADMIN.

    Used for role management and permission grant endpoints.

    Args:
        request: Litestar request.
        session: Request-scoped database session.

    Returns:
        Verified superadmin User model.

    Raises:
        NotAuthorizedException: If not authenticated or not superadmin.
    """
    user_id = request.state.get("user_id")
    if user_id is None:
        raise NotAuthorizedException(detail="Not authenticated")

    repo = UserRepository(session)
    user = await repo.get_active_user(user_id)

    if user is None or user.role != UserRole.SUPERADMIN:
        raise NotAuthorizedException(detail="Superadmin access required")

    return user


async def get_billing_adjust_admin(request: Request[Any, Any, Any], session: AsyncSession) -> User:
    """Load and verify the current user can perform billing adjustments.

    Allowed: SUPERADMIN (inherent) or ADMIN with 'billing_adjust' permission.

    Args:
        request: Litestar request.
        session: Request-scoped database session.

    Returns:
        Verified User model with billing adjust access.

    Raises:
        NotAuthorizedException: If not authenticated or lacking permission.
    """
    user_id = request.state.get("user_id")
    if user_id is None:
        raise NotAuthorizedException(detail="Not authenticated")

    repo = UserRepository(session)
    user = await repo.get_active_user(user_id)

    if user is None or user.role not in (UserRole.ADMIN, UserRole.SUPERADMIN):
        raise NotAuthorizedException(detail="Admin access required")

    # Superadmins have inherent billing adjust permission
    if user.role == UserRole.SUPERADMIN:
        return user

    # Admins need explicit permission grant
    from src.db.repositories.admin import AdminRepository

    admin_repo = AdminRepository(session)
    product_id: str = request.state.get("product_id") or user.product_id
    has_perm = await admin_repo.has_permission(user.id, "billing_adjust", product_id)
    if not has_perm:
        raise NotAuthorizedException(
            detail="Billing adjustment permission required. Contact a superadmin."
        )

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
