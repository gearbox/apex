"""Shared authentication dependencies for route handlers.

Provides reusable dependency functions that extract authentication
context from request state (set by auth_guard).
"""

from __future__ import annotations

from collections.abc import Collection
from typing import Any
from uuid import UUID

from litestar import Request
from litestar.exceptions import NotAuthorizedException
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.enums import AdminPermission, UserRole
from src.db.models import User
from src.db.repositories import UserRepository


async def _get_current_active_user(
    request: Request[Any, Any, Any],
    session: AsyncSession,
) -> User:
    """Load the authenticated active user from request state.

    Shared by all admin-level auth dependencies.

    Raises:
        NotAuthorizedException: If not authenticated or user not found/inactive.
    """
    user_id = request.state.get("user_id")
    if user_id is None:
        raise NotAuthorizedException(detail="Not authenticated")

    repo = UserRepository(session)
    user = await repo.get_active_user(user_id)
    if user is None:
        raise NotAuthorizedException(detail="Not authenticated")

    return user


def _require_role(
    user: User,
    allowed_roles: Collection[UserRole],
    detail: str,
) -> User:
    """Assert the user's role is in the allowed set.

    Normalizes user.role to a UserRole enum before comparison to handle
    cases where the ORM returns a plain string. An unrecognized role value
    is treated as an authorization failure rather than raising ValueError.

    Raises:
        NotAuthorizedException: If role not in allowed_roles or unrecognized.
    """
    try:
        role = user.role if isinstance(user.role, UserRole) else UserRole(user.role)
    except ValueError as exc:
        raise NotAuthorizedException(detail=detail) from exc
    if role not in allowed_roles:
        raise NotAuthorizedException(detail=detail)
    return user


async def ensure_billing_adjust_permission(
    user: User,
    session: AsyncSession,
    product_id: str,
) -> None:
    """Verify the user can perform billing adjustments.

    SUPERADMIN: inherent access (no permission grant needed).
    ADMIN: requires explicit 'billing_adjust' permission grant.

    This is the single source of truth for billing-adjust authorization.
    Called by get_billing_adjust_admin (DI) and can be called directly
    by handlers that already have the admin_user injected.

    Raises:
        NotAuthorizedException: If user lacks permission.
    """
    role_value = user.role if isinstance(user.role, str) else user.role.value
    if role_value == UserRole.SUPERADMIN.value:
        return

    from src.db.repositories.admin import AdminRepository

    admin_repo = AdminRepository(session)
    has_perm = await admin_repo.has_permission(
        user.id, AdminPermission.BILLING_ADJUST.value, product_id
    )
    if not has_perm:
        raise NotAuthorizedException(
            detail="Billing adjustment permission required. Contact a superadmin."
        )


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
    """Load and verify the current user has admin-level access (ADMIN or SUPERADMIN)."""
    user = await _get_current_active_user(request, session)
    return _require_role(
        user,
        allowed_roles=(UserRole.ADMIN, UserRole.SUPERADMIN),
        detail="Admin access required",
    )


async def get_current_superadmin_user(
    request: Request[Any, Any, Any], session: AsyncSession
) -> User:
    """Load and verify the current user is a SUPERADMIN."""
    user = await _get_current_active_user(request, session)
    return _require_role(
        user,
        allowed_roles=(UserRole.SUPERADMIN,),
        detail="Superadmin access required",
    )


async def get_billing_adjust_admin(request: Request[Any, Any, Any], session: AsyncSession) -> User:
    """Load and verify the user can perform billing adjustments.

    SUPERADMIN inherent; ADMIN requires billing_adjust permission grant.
    """
    user = await _get_current_active_user(request, session)
    _require_role(
        user,
        allowed_roles=(UserRole.ADMIN, UserRole.SUPERADMIN),
        detail="Admin access required",
    )
    product_id: str = request.state.get("product_id") or user.product_id
    await ensure_billing_adjust_permission(user, session, product_id)
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
