"""Service for admin role and permission management."""

from __future__ import annotations

from uuid import UUID

import structlog
from sqlalchemy.exc import IntegrityError
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

    @staticmethod
    def _role_value(user: User) -> str:
        """Normalize user.role to a plain string regardless of ORM return type."""
        role = user.role
        return role if isinstance(role, str) else role.value

    async def _get_target_user(
        self,
        user_id: UUID,
        product_id: str,
        *,
        session: AsyncSession,
    ) -> User:
        """Load an active user and verify product membership.

        Raises:
            AdminManagementError: If user not found or wrong product.
        """
        user_repo = UserRepository(session)
        user = await user_repo.get_active_user(user_id)
        if user is None or user.product_id != product_id:
            raise AdminManagementError(f"User {user_id} not found in product {product_id}")
        return user

    async def _demote_admin_to_user(
        self,
        *,
        target: User,
        product_id: str,
        session: AsyncSession,
        enforce_last_superadmin_guard: bool,
    ) -> tuple[str, int]:
        """Demote an admin/superadmin to USER and revoke all permissions.

        Args:
            target: The user to demote (must have admin/superadmin role).
            product_id: Product scope.
            session: Database session.
            enforce_last_superadmin_guard: If True and target is the last superadmin,
                raises LastSuperadminError instead of demoting.

        Returns:
            Tuple of (old_role_value, permissions_revoked_count).

        Raises:
            InvalidRoleTransitionError: If target has no admin role.
            LastSuperadminError: If guard is enforced and target is last superadmin.
        """
        old_role = self._role_value(target)
        user_repo = UserRepository(session)

        if old_role not in self.ADMIN_ROLES:
            raise InvalidRoleTransitionError(
                f"User {target.id} does not have an admin role (current: {old_role})"
            )

        if enforce_last_superadmin_guard and old_role == UserRole.SUPERADMIN.value:
            demoted = await user_repo.revoke_superadmin_if_not_last(target.id, product_id)
            if not demoted:
                raise LastSuperadminError(
                    f"Cannot revoke the last superadmin for product '{product_id}'. "
                    "Grant superadmin to another user first."
                )
        else:
            await user_repo.update_user_admin(target.id, role=UserRole.USER.value)

        admin_repo = AdminRepository(session)
        revoked_count = await admin_repo.delete_all_permissions(target.id, product_id)
        return old_role, revoked_count

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
        """Grant an admin or superadmin role to a user."""
        if actor_id == target_user_id:
            raise SelfModificationError("Cannot modify your own role")

        if new_role.value not in self.GRANTABLE_ROLES:
            raise InvalidRoleTransitionError(
                f"Cannot grant role '{new_role.value}' via admin management"
            )

        target = await self._get_target_user(target_user_id, product_id, session=session)
        old_role = self._role_value(target)

        user_repo = UserRepository(session)
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
        """Revoke admin/superadmin role, demoting user back to USER."""
        if actor_id == target_user_id:
            raise SelfModificationError("Cannot revoke your own role")

        target = await self._get_target_user(target_user_id, product_id, session=session)
        old_role, revoked_count = await self._demote_admin_to_user(
            target=target,
            product_id=product_id,
            session=session,
            enforce_last_superadmin_guard=True,
        )

        admin_repo = AdminRepository(session)
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
        target = await self._get_target_user(target_user_id, product_id, session=session)
        old_role, revoked_count = await self._demote_admin_to_user(
            target=target,
            product_id=product_id,
            session=session,
            enforce_last_superadmin_guard=False,
        )

        admin_repo = AdminRepository(session)
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
        target = await self._get_target_user(target_user_id, product_id, session=session)
        target_role = self._role_value(target)
        if target_role not in self.ADMIN_ROLES:
            raise InvalidRoleTransitionError(
                f"Cannot grant permission to non-admin user (role: {target_role})"
            )

        admin_repo = AdminRepository(session)
        already = await admin_repo.has_permission(target_user_id, permission.value, product_id)
        if already:
            return  # idempotent — fast path

        try:
            async with session.begin_nested():
                await admin_repo.grant_permission(
                    AdminPermissionGrant(
                        id=new_id(),
                        user_id=target_user_id,
                        permission=permission.value,
                        product_id=product_id,
                        granted_by=actor_id,
                    )
                )
        except IntegrityError:
            # Concurrent grant won the race — the SAVEPOINT was rolled back,
            # but the outer transaction remains valid. Treat as idempotent success.
            logger.info(
                "admin.permission.grant_race",
                target_user_id=str(target_user_id),
                permission=permission.value,
                product_id=product_id,
            )
            return

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
