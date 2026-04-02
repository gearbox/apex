"""Repository for admin permission and audit operations."""

from __future__ import annotations

from collections.abc import Sequence
from typing import cast
from uuid import UUID

import structlog
from sqlalchemy import CursorResult, delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.models.admin import AdminAuditLog, AdminPermissionGrant

logger = structlog.get_logger(__name__)


class AdminRepository:
    """Data access for admin permissions and audit log."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    # --- Permissions ---

    async def grant_permission(self, grant: AdminPermissionGrant) -> AdminPermissionGrant:
        """Insert a permission grant. Caller must flush/commit."""
        self._session.add(grant)
        await self._session.flush()
        return grant

    async def revoke_permission(self, user_id: UUID, permission: str, product_id: str) -> bool:
        """Delete a permission grant. Returns True if a row was deleted."""
        result = cast(
            CursorResult[tuple[()]],
            await self._session.execute(
                delete(AdminPermissionGrant).where(
                    AdminPermissionGrant.user_id == user_id,
                    AdminPermissionGrant.permission == permission,
                    AdminPermissionGrant.product_id == product_id,
                )
            ),
        )
        return (result.rowcount or 0) > 0

    async def get_permissions(
        self, user_id: UUID, product_id: str
    ) -> Sequence[AdminPermissionGrant]:
        """Get all permissions for a user in a product."""
        result = await self._session.execute(
            select(AdminPermissionGrant).where(
                AdminPermissionGrant.user_id == user_id,
                AdminPermissionGrant.product_id == product_id,
            )
        )
        return result.scalars().all()

    async def has_permission(self, user_id: UUID, permission: str, product_id: str) -> bool:
        """Check if a user has a specific permission."""
        result = await self._session.execute(
            select(AdminPermissionGrant.id)
            .where(
                AdminPermissionGrant.user_id == user_id,
                AdminPermissionGrant.permission == permission,
                AdminPermissionGrant.product_id == product_id,
            )
            .limit(1)
        )
        return result.scalar_one_or_none() is not None

    async def delete_all_permissions(self, user_id: UUID, product_id: str) -> int:
        """Delete all permissions for a user in a product. Returns count deleted."""
        result = cast(
            CursorResult[tuple[()]],
            await self._session.execute(
                delete(AdminPermissionGrant).where(
                    AdminPermissionGrant.user_id == user_id,
                    AdminPermissionGrant.product_id == product_id,
                )
            ),
        )
        return result.rowcount or 0

    # --- Audit log ---

    async def write_audit(self, entry: AdminAuditLog) -> AdminAuditLog:
        """Append an audit log entry. Caller must flush/commit."""
        self._session.add(entry)
        await self._session.flush()
        return entry

    async def get_audit_log(
        self,
        product_id: str,
        *,
        target_user_id: UUID | None = None,
        limit: int = 50,
    ) -> Sequence[AdminAuditLog]:
        """Retrieve audit log entries, newest first."""
        stmt = select(AdminAuditLog).where(AdminAuditLog.product_id == product_id)
        if target_user_id is not None:
            stmt = stmt.where(AdminAuditLog.target_user_id == target_user_id)
        stmt = stmt.order_by(AdminAuditLog.created_at.desc()).limit(limit)
        result = await self._session.execute(stmt)
        return result.scalars().all()
