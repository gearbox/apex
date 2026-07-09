"""Admin management routes — role grants/revokes, permissions, audit log.

All endpoints require SUPERADMIN role (via get_current_superadmin_user).
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from uuid import UUID

import structlog
from litestar import Controller, get, post
from litestar.di import Provide
from litestar.exceptions import NotFoundException, PermissionDeniedException, ValidationException
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.dependencies.auth import get_current_superadmin_user
from src.api.schemas.admin import (
    AdminRoleResponse,
    AuditLogEntry,
    GrantPermissionRequest,
    GrantRoleRequest,
)
from src.api.schemas.pagination import CursorPage, decode_cursor, encode_cursor
from src.api.security import auth_guard
from src.api.services.admin_management import (
    AdminManagementError,
    AdminManagementService,
    InvalidRoleTransitionError,
    LastSuperadminError,
    SelfModificationError,
)
from src.db.models import User

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = structlog.get_logger(__name__)


class AdminManagementController(Controller):
    """Superadmin-only endpoints for managing admin roles and permissions."""

    path = "/v1/admin/manage"
    tags: Sequence[str] | None = ("Admin Management",)
    guards = [auth_guard]  # noqa: RUF012
    dependencies = {  # noqa: RUF012
        "superadmin": Provide(get_current_superadmin_user),
        "admin_mgmt": Provide(AdminManagementService, sync_to_thread=False),
    }

    @get("/admins")
    async def list_admins(
        self,
        superadmin: User,
        session: AsyncSession,
        product_id: str,
        admin_mgmt: AdminManagementService,
    ) -> list[AdminRoleResponse]:
        """List all admin and superadmin users for the current product."""
        logger.info("admin_mgmt.listing_admins", superadmin_id=str(superadmin.id))
        admins_with_perms = await admin_mgmt.list_admins_with_permissions(
            product_id, session=session
        )
        return [
            AdminRoleResponse(
                id=u.id,
                email=u.email,
                display_name=u.display_name,
                role=u.role.value if hasattr(u.role, "value") else u.role,
                permissions=perms,
                is_active=u.is_active,
                created_at=u.created_at,
                updated_at=u.updated_at,
            )
            for u, perms in admins_with_perms
        ]

    @post("/roles/{user_id:uuid}/grant")
    async def grant_role(
        self,
        superadmin: User,
        user_id: UUID,
        data: GrantRoleRequest,
        session: AsyncSession,
        product_id: str,
        admin_mgmt: AdminManagementService,
    ) -> dict[str, str]:
        """Grant admin or superadmin role to a user."""
        try:
            await admin_mgmt.grant_role(
                actor_id=superadmin.id,
                target_user_id=user_id,
                new_role=data.role,
                product_id=product_id,
                source="api",
                session=session,
            )
            await session.commit()
        except SelfModificationError as exc:
            raise PermissionDeniedException(detail=str(exc)) from exc
        except InvalidRoleTransitionError as exc:
            raise ValidationException(detail=str(exc)) from exc
        except AdminManagementError as exc:
            raise NotFoundException(detail=str(exc)) from exc
        else:
            return {"message": f"Role '{data.role.value}' granted to user {user_id}"}

    @post("/roles/{user_id:uuid}/revoke")
    async def revoke_role(
        self,
        superadmin: User,
        user_id: UUID,
        session: AsyncSession,
        product_id: str,
        admin_mgmt: AdminManagementService,
    ) -> dict[str, str]:
        """Revoke admin role, demoting user back to USER."""
        try:
            await admin_mgmt.revoke_role(
                actor_id=superadmin.id,
                target_user_id=user_id,
                product_id=product_id,
                source="api",
                session=session,
            )
            await session.commit()
        except SelfModificationError as exc:
            raise PermissionDeniedException(detail=str(exc)) from exc
        except LastSuperadminError as exc:
            raise ValidationException(detail=str(exc)) from exc
        except InvalidRoleTransitionError as exc:
            raise ValidationException(detail=str(exc)) from exc
        except AdminManagementError as exc:
            raise NotFoundException(detail=str(exc)) from exc
        else:
            return {"message": f"Admin role revoked from user {user_id}"}

    @post("/permissions/{user_id:uuid}/grant")
    async def grant_permission(
        self,
        superadmin: User,
        user_id: UUID,
        data: GrantPermissionRequest,
        session: AsyncSession,
        product_id: str,
        admin_mgmt: AdminManagementService,
    ) -> dict[str, str]:
        """Grant a specific permission to an admin user."""
        try:
            await admin_mgmt.grant_permission(
                actor_id=superadmin.id,
                target_user_id=user_id,
                permission=data.permission,
                product_id=product_id,
                source="api",
                session=session,
            )
            await session.commit()
        except InvalidRoleTransitionError as exc:
            raise ValidationException(detail=str(exc)) from exc
        except AdminManagementError as exc:
            raise NotFoundException(detail=str(exc)) from exc
        else:
            return {"message": f"Permission '{data.permission.value}' granted to user {user_id}"}

    @post("/permissions/{user_id:uuid}/revoke")
    async def revoke_permission(
        self,
        superadmin: User,
        user_id: UUID,
        data: GrantPermissionRequest,
        session: AsyncSession,
        product_id: str,
        admin_mgmt: AdminManagementService,
    ) -> dict[str, str]:
        """Revoke a specific permission from a user."""
        await admin_mgmt.revoke_permission(
            actor_id=superadmin.id,
            target_user_id=user_id,
            permission=data.permission,
            product_id=product_id,
            source="api",
            session=session,
        )
        await session.commit()
        return {"message": f"Permission '{data.permission.value}' revoked from user {user_id}"}

    @get("/audit")
    async def get_audit_log(
        self,
        superadmin: User,
        session: AsyncSession,
        product_id: str,
        admin_mgmt: AdminManagementService,
        target_user_id: UUID | None = None,
        limit: int = 50,
        cursor: str | None = None,
    ) -> CursorPage[AuditLogEntry]:
        """Retrieve admin audit log entries (cursor-paginated, newest first)."""
        logger.info(
            "admin_mgmt.viewing_audit_log",
            superadmin_id=str(superadmin.id),
            product_id=product_id,
        )
        cursor_ts = None
        cursor_id = None
        if cursor is not None:
            cursor_ts, cursor_id = decode_cursor(cursor)

        entries = await admin_mgmt.get_audit_log(
            product_id,
            target_user_id=target_user_id,
            limit=limit,
            cursor_ts=cursor_ts,
            cursor_id=cursor_id,
            session=session,
        )

        has_more = len(entries) > limit
        if has_more:
            entries = entries[:limit]

        items = [
            AuditLogEntry(
                id=e.id,
                actor_id=e.actor_id,
                target_user_id=e.target_user_id,
                action=e.action,
                detail=e.detail,
                source=e.source,
                created_at=e.created_at,
            )
            for e in entries
        ]

        next_cursor: str | None = None
        if has_more and entries:
            last = entries[-1]
            next_cursor = encode_cursor(last.created_at, last.id)

        return CursorPage(
            items=items,
            limit=limit,
            has_more=has_more,
            next_cursor=next_cursor,
        )
