"""Service for admin role and permission management."""

from __future__ import annotations

from uuid import UUID

import structlog
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.enums import AdminPermission, UserRole
from src.core.uid import new_id
from src.db.models.admin import AdminAuditLog, AdminPermissionGrant
from src.db.models.user import User
from src.db.repositories.admin import AdminRepository
from src.db.repositories.user import UserRepository

logger = structlog.get_logger(__name__)


class AdminManagementError(Exception):
    """Base error for admin management operations."""


class LastSuperadminError(AdminManagementError):
    """Cannot revoke the last superadmin for a product."""


class SelfModificationError(AdminManagementError):
    """Cannot modify own role via this endpoint."""


class InvalidRoleTransitionError(AdminManagementError):
    """Invalid role transition (e.g. granting SYSTEM)."""


class AdminManagementService:
    """Business logic for admin role grants, revocations, and permission management."""

    # Roles that count toward "admin" for access to admin endpoints
    ADMIN_ROLES: frozenset[str] = frozenset({UserRole.ADMIN.value, UserRole.SUPERADMIN.value})

    # Roles that can be granted/revoked via the admin management API
    GRANTABLE_ROLES: frozenset[str] = frozenset({UserRole.ADMIN.value, UserRole.SUPERADMIN.value})

    async def list_admins(
        self,
        product_id: str,
        *,
        session: AsyncSession,
    ) -> list[User]:
        """List all users with admin or superadmin role for a product.

        Single query with role IN filter, ordered superadmins first.
        """
        repo = UserRepository(session)
        users = await repo.list_users_by_roles(
            product_id=product_id,
            roles=[UserRole.SUPERADMIN.value, UserRole.ADMIN.value],
            limit=500,
        )
        return list(users)

    async def list_admins_with_permissions(
        self,
        product_id: str,
        *,
        session: AsyncSession,
    ) -> list[tuple[User, list[str]]]:
        """List admins with their permissions. Two queries total (users + permissions batch)."""
        users = await self.list_admins(product_id, session=session)
        if not users:
            return []

        admin_repo = AdminRepository(session)
        user_ids = [u.id for u in users]
        permissions_map = await admin_repo.get_permissions_batch(user_ids, product_id)

        return [(u, permissions_map.get(u.id, [])) for u in users]

    async def grant_role(
        self,
        *,
        actor_id: UUID,
        target_user_id: UUID,
        new_role: UserRole,
        product_id: str,
        source: str = "api",
        session: AsyncSession,
    ) -> None:
        """Grant an admin or superadmin role to a user.

        Invariants:
        - actor cannot modify own role
        - new_role must be ADMIN or SUPERADMIN (never SYSTEM or USER)
        - target user must exist, be active, and belong to the same product
        """
        if actor_id == target_user_id:
            raise SelfModificationError("Cannot modify your own role")

        if new_role.value not in self.GRANTABLE_ROLES:
            raise InvalidRoleTransitionError(
                f"Cannot grant role '{new_role.value}' via admin management"
            )

        user_repo = UserRepository(session)
        target = await user_repo.get_active_user(target_user_id)
        if target is None or target.product_id != product_id:
            raise AdminManagementError(f"User {target_user_id} not found in product {product_id}")

        old_role = target.role if isinstance(target.role, str) else target.role.value

        await user_repo.update_user_admin(target_user_id, role=new_role.value)

        admin_repo = AdminRepository(session)
        await admin_repo.write_audit(
            AdminAuditLog(
                id=new_id(),
                actor_id=actor_id,
                target_user_id=target_user_id,
                product_id=product_id,
                action="role.grant",
                detail=f"Role changed from '{old_role}' to '{new_role.value}'",
                source=source,
            )
        )

        logger.info(
            "admin.role.granted",
            actor_id=str(actor_id),
            target_user_id=str(target_user_id),
            old_role=old_role,
            new_role=new_role.value,
            product_id=product_id,
            source=source,
        )

    async def revoke_role(
        self,
        *,
        actor_id: UUID,
        target_user_id: UUID,
        product_id: str,
        source: str = "api",
        session: AsyncSession,
    ) -> None:
        """Revoke admin/superadmin role, demoting user back to USER.

        Invariants:
        - actor cannot revoke own role
        - cannot revoke the last superadmin for a product
        - also revokes all permissions for the user in this product
        """
        if actor_id == target_user_id:
            raise SelfModificationError("Cannot revoke your own role")

        user_repo = UserRepository(session)
        target = await user_repo.get_active_user(target_user_id)
        if target is None or target.product_id != product_id:
            raise AdminManagementError(f"User {target_user_id} not found in product {product_id}")

        old_role = target.role if isinstance(target.role, str) else target.role.value

        if old_role not in self.ADMIN_ROLES:
            raise InvalidRoleTransitionError(
                f"User {target_user_id} does not have an admin role (current: {old_role})"
            )

        # Lockout guard + role update (atomic for superadmins)
        if old_role == UserRole.SUPERADMIN.value:
            demoted = await user_repo.revoke_superadmin_if_not_last(target_user_id, product_id)
            if not demoted:
                raise LastSuperadminError(
                    f"Cannot revoke the last superadmin for product '{product_id}'. "
                    "Grant superadmin to another user first."
                )
        else:
            # Non-superadmin admin — no lockout concern, just demote
            await user_repo.update_user_admin(target_user_id, role=UserRole.USER.value)

        admin_repo = AdminRepository(session)
        revoked_count = await admin_repo.delete_all_permissions(target_user_id, product_id)

        await admin_repo.write_audit(
            AdminAuditLog(
                id=new_id(),
                actor_id=actor_id,
                target_user_id=target_user_id,
                product_id=product_id,
                action="role.revoke",
                detail=(
                    f"Role changed from '{old_role}' to 'user'. "
                    f"{revoked_count} permission(s) revoked."
                ),
                source=source,
            )
        )

        logger.info(
            "admin.role.revoked",
            actor_id=str(actor_id),
            target_user_id=str(target_user_id),
            old_role=old_role,
            product_id=product_id,
            permissions_revoked=revoked_count,
            source=source,
        )

    async def force_revoke_role(
        self,
        *,
        actor_id: UUID,
        target_user_id: UUID,
        product_id: str,
        source: str = "cli",
        session: AsyncSession,
    ) -> None:
        """Force-revoke an admin/superadmin role, bypassing the last-superadmin guard.

        This is the break-glass escape hatch — intended for CLI recovery only.
        Demotes to USER and revokes all permissions unconditionally.
        """
        user_repo = UserRepository(session)
        target = await user_repo.get_active_user(target_user_id)
        if target is None or target.product_id != product_id:
            raise AdminManagementError(f"User {target_user_id} not found in product {product_id}")

        old_role = target.role if isinstance(target.role, str) else target.role.value

        if old_role not in self.ADMIN_ROLES:
            raise InvalidRoleTransitionError(
                f"User {target_user_id} does not have an admin role (current: {old_role})"
            )

        # Direct update — no lockout guard
        await user_repo.update_user_admin(target_user_id, role=UserRole.USER.value)

        admin_repo = AdminRepository(session)
        revoked_count = await admin_repo.delete_all_permissions(target_user_id, product_id)

        await admin_repo.write_audit(
            AdminAuditLog(
                id=new_id(),
                actor_id=actor_id,
                target_user_id=target_user_id,
                product_id=product_id,
                action="role.revoke",
                detail=(
                    f"FORCED: Role changed from '{old_role}' to 'user'. "
                    f"{revoked_count} permission(s) revoked."
                ),
                source=source,
            )
        )

        logger.warning(
            "admin.role.force_revoked",
            actor_id=str(actor_id),
            target_user_id=str(target_user_id),
            old_role=old_role,
            product_id=product_id,
            permissions_revoked=revoked_count,
            source=source,
        )

    async def grant_permission(
        self,
        *,
        actor_id: UUID,
        target_user_id: UUID,
        permission: AdminPermission,
        product_id: str,
        source: str = "api",
        session: AsyncSession,
    ) -> None:
        """Grant a specific permission to an admin user.

        Target must have ADMIN or SUPERADMIN role.
        """
        user_repo = UserRepository(session)
        target = await user_repo.get_active_user(target_user_id)
        if target is None or target.product_id != product_id:
            raise AdminManagementError(f"User {target_user_id} not found in product {product_id}")

        target_role = target.role if isinstance(target.role, str) else target.role.value
        if target_role not in self.ADMIN_ROLES:
            raise InvalidRoleTransitionError(
                f"Cannot grant permission to non-admin user (role: {target_role})"
            )

        admin_repo = AdminRepository(session)
        already = await admin_repo.has_permission(target_user_id, permission.value, product_id)
        if already:
            return  # idempotent

        await admin_repo.grant_permission(
            AdminPermissionGrant(
                id=new_id(),
                user_id=target_user_id,
                permission=permission.value,
                product_id=product_id,
                granted_by=actor_id,
            )
        )

        await admin_repo.write_audit(
            AdminAuditLog(
                id=new_id(),
                actor_id=actor_id,
                target_user_id=target_user_id,
                product_id=product_id,
                action="permission.grant",
                detail=f"Granted permission '{permission.value}'",
                source=source,
            )
        )

        logger.info(
            "admin.permission.granted",
            actor_id=str(actor_id),
            target_user_id=str(target_user_id),
            permission=permission.value,
            product_id=product_id,
            source=source,
        )

    async def revoke_permission(
        self,
        *,
        actor_id: UUID,
        target_user_id: UUID,
        permission: AdminPermission,
        product_id: str,
        source: str = "api",
        session: AsyncSession,
    ) -> None:
        """Revoke a specific permission from a user."""
        admin_repo = AdminRepository(session)
        deleted = await admin_repo.revoke_permission(target_user_id, permission.value, product_id)
        if not deleted:
            return  # idempotent — permission wasn't there

        await admin_repo.write_audit(
            AdminAuditLog(
                id=new_id(),
                actor_id=actor_id,
                target_user_id=target_user_id,
                product_id=product_id,
                action="permission.revoke",
                detail=f"Revoked permission '{permission.value}'",
                source=source,
            )
        )

        logger.info(
            "admin.permission.revoked",
            actor_id=str(actor_id),
            target_user_id=str(target_user_id),
            permission=permission.value,
            product_id=product_id,
            source=source,
        )

    async def get_user_permissions(
        self,
        user_id: UUID,
        product_id: str,
        *,
        session: AsyncSession,
    ) -> list[str]:
        """Get all permissions for a user."""
        admin_repo = AdminRepository(session)
        grants = await admin_repo.get_permissions(user_id, product_id)
        return [g.permission for g in grants]

    async def get_audit_log(
        self,
        product_id: str,
        *,
        target_user_id: UUID | None = None,
        limit: int = 50,
        session: AsyncSession,
    ) -> list[AdminAuditLog]:
        """Retrieve audit log entries."""
        admin_repo = AdminRepository(session)
        entries = await admin_repo.get_audit_log(
            product_id,
            target_user_id=target_user_id,
            limit=limit,
        )
        return list(entries)
