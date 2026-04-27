"""Tests for AdminManagementService business logic."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from sqlalchemy.exc import IntegrityError

from src.api.services.admin_management import (
    AdminManagementError,
    AdminManagementService,
    InvalidRoleTransitionError,
    LastSuperadminError,
    SelfModificationError,
)
from src.core.enums import AdminPermission, UserRole


def _make_user(
    user_id=None,
    role: str = "user",
    product_id: str = "vex",
    is_active: bool = True,
) -> MagicMock:
    user = MagicMock()
    user.id = user_id or uuid4()
    user.role = role
    user.product_id = product_id
    user.is_active = is_active
    return user


@pytest.fixture
def service() -> AdminManagementService:
    return AdminManagementService()


@pytest.fixture
def mock_session() -> AsyncMock:
    session = AsyncMock()
    session.begin_nested = MagicMock(
        return_value=AsyncMock(__aenter__=AsyncMock(), __aexit__=AsyncMock(return_value=False))
    )
    return session


# ---------------------------------------------------------------------------
# grant_role
# ---------------------------------------------------------------------------


class TestGrantRole:
    async def test_grant_admin_role_success(
        self, service: AdminManagementService, mock_session: AsyncMock
    ) -> None:
        actor_id = uuid4()
        target_id = uuid4()
        target = _make_user(user_id=target_id, role="user", product_id="vex")

        with (
            patch("src.api.services.admin_management.UserRepository") as MockUserRepo,
            patch("src.api.services.admin_management.AdminRepository") as MockAdminRepo,
            patch("src.api.services.admin_management.new_id", return_value=uuid4()),
        ):
            user_repo = MockUserRepo.return_value
            user_repo.get_active_user = AsyncMock(return_value=target)
            user_repo.update_user_admin = AsyncMock(return_value=target)

            admin_repo = MockAdminRepo.return_value
            admin_repo.write_audit = AsyncMock()

            await service.grant_role(
                actor_id=actor_id,
                target_user_id=target_id,
                new_role=UserRole.ADMIN,
                product_id="vex",
                session=mock_session,
            )

        user_repo.update_user_admin.assert_awaited_once_with(target_id, role="admin")
        admin_repo.write_audit.assert_awaited_once()

    async def test_grant_superadmin_role_success(
        self, service: AdminManagementService, mock_session: AsyncMock
    ) -> None:
        actor_id = uuid4()
        target_id = uuid4()
        target = _make_user(user_id=target_id, role="admin", product_id="vex")

        with (
            patch("src.api.services.admin_management.UserRepository") as MockUserRepo,
            patch("src.api.services.admin_management.AdminRepository") as MockAdminRepo,
            patch("src.api.services.admin_management.new_id", return_value=uuid4()),
        ):
            user_repo = MockUserRepo.return_value
            user_repo.get_active_user = AsyncMock(return_value=target)
            user_repo.update_user_admin = AsyncMock(return_value=target)

            admin_repo = MockAdminRepo.return_value
            admin_repo.write_audit = AsyncMock()

            await service.grant_role(
                actor_id=actor_id,
                target_user_id=target_id,
                new_role=UserRole.SUPERADMIN,
                product_id="vex",
                session=mock_session,
            )

        user_repo.update_user_admin.assert_awaited_once_with(target_id, role="superadmin")

    async def test_grant_role_rejects_self_modification(
        self, service: AdminManagementService, mock_session: AsyncMock
    ) -> None:
        actor_id = uuid4()
        with pytest.raises(SelfModificationError):
            await service.grant_role(
                actor_id=actor_id,
                target_user_id=actor_id,
                new_role=UserRole.ADMIN,
                product_id="vex",
                session=mock_session,
            )

    async def test_grant_role_rejects_system_role(
        self, service: AdminManagementService, mock_session: AsyncMock
    ) -> None:
        with pytest.raises(InvalidRoleTransitionError):
            await service.grant_role(
                actor_id=uuid4(),
                target_user_id=uuid4(),
                new_role=UserRole.SYSTEM,
                product_id="vex",
                session=mock_session,
            )

    async def test_grant_role_rejects_user_role(
        self, service: AdminManagementService, mock_session: AsyncMock
    ) -> None:
        with pytest.raises(InvalidRoleTransitionError):
            await service.grant_role(
                actor_id=uuid4(),
                target_user_id=uuid4(),
                new_role=UserRole.USER,
                product_id="vex",
                session=mock_session,
            )

    async def test_grant_role_target_not_found(
        self, service: AdminManagementService, mock_session: AsyncMock
    ) -> None:
        with patch("src.api.services.admin_management.UserRepository") as MockUserRepo:
            MockUserRepo.return_value.get_active_user = AsyncMock(return_value=None)
            with pytest.raises(AdminManagementError):
                await service.grant_role(
                    actor_id=uuid4(),
                    target_user_id=uuid4(),
                    new_role=UserRole.ADMIN,
                    product_id="vex",
                    session=mock_session,
                )

    async def test_grant_role_target_wrong_product(
        self, service: AdminManagementService, mock_session: AsyncMock
    ) -> None:
        target = _make_user(role="user", product_id="synthara")
        with patch("src.api.services.admin_management.UserRepository") as MockUserRepo:
            MockUserRepo.return_value.get_active_user = AsyncMock(return_value=target)
            with pytest.raises(AdminManagementError):
                await service.grant_role(
                    actor_id=uuid4(),
                    target_user_id=target.id,
                    new_role=UserRole.ADMIN,
                    product_id="vex",
                    session=mock_session,
                )

    async def test_grant_role_writes_audit_entry(
        self, service: AdminManagementService, mock_session: AsyncMock
    ) -> None:
        actor_id = uuid4()
        target = _make_user(role="user", product_id="vex")

        with (
            patch("src.api.services.admin_management.UserRepository") as MockUserRepo,
            patch("src.api.services.admin_management.AdminRepository") as MockAdminRepo,
            patch("src.api.services.admin_management.new_id", return_value=uuid4()),
        ):
            MockUserRepo.return_value.get_active_user = AsyncMock(return_value=target)
            MockUserRepo.return_value.update_user_admin = AsyncMock(return_value=target)
            admin_repo = MockAdminRepo.return_value
            admin_repo.write_audit = AsyncMock()

            await service.grant_role(
                actor_id=actor_id,
                target_user_id=target.id,
                new_role=UserRole.ADMIN,
                product_id="vex",
                session=mock_session,
            )

        assert admin_repo.write_audit.await_count == 1
        audit_entry = admin_repo.write_audit.call_args[0][0]
        assert audit_entry.action == "role.grant"
        assert audit_entry.actor_id == actor_id
        assert audit_entry.source == "api"


# ---------------------------------------------------------------------------
# revoke_role
# ---------------------------------------------------------------------------


class TestRevokeRole:
    async def test_revoke_role_success(
        self, service: AdminManagementService, mock_session: AsyncMock
    ) -> None:
        actor_id = uuid4()
        target_id = uuid4()
        target = _make_user(user_id=target_id, role="admin", product_id="vex")

        with (
            patch("src.api.services.admin_management.UserRepository") as MockUserRepo,
            patch("src.api.services.admin_management.AdminRepository") as MockAdminRepo,
            patch("src.api.services.admin_management.new_id", return_value=uuid4()),
        ):
            user_repo = MockUserRepo.return_value
            user_repo.get_active_user = AsyncMock(return_value=target)
            user_repo.update_user_admin = AsyncMock(return_value=target)

            admin_repo = MockAdminRepo.return_value
            admin_repo.delete_all_permissions = AsyncMock(return_value=2)
            admin_repo.write_audit = AsyncMock()

            await service.revoke_role(
                actor_id=actor_id,
                target_user_id=target_id,
                product_id="vex",
                session=mock_session,
            )

        user_repo.update_user_admin.assert_awaited_once_with(target_id, role="user")
        admin_repo.delete_all_permissions.assert_awaited_once()
        admin_repo.write_audit.assert_awaited_once()

    async def test_revoke_role_rejects_self_modification(
        self, service: AdminManagementService, mock_session: AsyncMock
    ) -> None:
        actor_id = uuid4()
        with pytest.raises(SelfModificationError):
            await service.revoke_role(
                actor_id=actor_id,
                target_user_id=actor_id,
                product_id="vex",
                session=mock_session,
            )

    async def test_revoke_role_rejects_non_admin_target(
        self, service: AdminManagementService, mock_session: AsyncMock
    ) -> None:
        target = _make_user(role="user", product_id="vex")
        with patch("src.api.services.admin_management.UserRepository") as MockUserRepo:
            MockUserRepo.return_value.get_active_user = AsyncMock(return_value=target)
            with pytest.raises(InvalidRoleTransitionError):
                await service.revoke_role(
                    actor_id=uuid4(),
                    target_user_id=target.id,
                    product_id="vex",
                    session=mock_session,
                )

    async def test_revoke_role_blocks_last_superadmin(
        self, service: AdminManagementService, mock_session: AsyncMock
    ) -> None:
        target = _make_user(role="superadmin", product_id="vex")
        with patch("src.api.services.admin_management.UserRepository") as MockUserRepo:
            user_repo = MockUserRepo.return_value
            user_repo.get_active_user = AsyncMock(return_value=target)
            user_repo.revoke_superadmin_if_not_last = AsyncMock(return_value=False)

            with pytest.raises(LastSuperadminError):
                await service.revoke_role(
                    actor_id=uuid4(),
                    target_user_id=target.id,
                    product_id="vex",
                    session=mock_session,
                )

    async def test_revoke_superadmin_role_success(
        self, service: AdminManagementService, mock_session: AsyncMock
    ) -> None:
        actor_id = uuid4()
        target = _make_user(role="superadmin", product_id="vex")

        with (
            patch("src.api.services.admin_management.UserRepository") as MockUserRepo,
            patch("src.api.services.admin_management.AdminRepository") as MockAdminRepo,
            patch("src.api.services.admin_management.new_id", return_value=uuid4()),
        ):
            user_repo = MockUserRepo.return_value
            user_repo.get_active_user = AsyncMock(return_value=target)
            user_repo.revoke_superadmin_if_not_last = AsyncMock(return_value=True)

            admin_repo = MockAdminRepo.return_value
            admin_repo.delete_all_permissions = AsyncMock(return_value=0)
            admin_repo.write_audit = AsyncMock()

            await service.revoke_role(
                actor_id=actor_id,
                target_user_id=target.id,
                product_id="vex",
                session=mock_session,
            )

        user_repo.revoke_superadmin_if_not_last.assert_awaited_once_with(target.id, "vex")
        admin_repo.delete_all_permissions.assert_awaited_once()

    async def test_revoke_role_also_revokes_permissions(
        self, service: AdminManagementService, mock_session: AsyncMock
    ) -> None:
        actor_id = uuid4()
        target = _make_user(role="admin", product_id="vex")

        with (
            patch("src.api.services.admin_management.UserRepository") as MockUserRepo,
            patch("src.api.services.admin_management.AdminRepository") as MockAdminRepo,
            patch("src.api.services.admin_management.new_id", return_value=uuid4()),
        ):
            MockUserRepo.return_value.get_active_user = AsyncMock(return_value=target)
            MockUserRepo.return_value.update_user_admin = AsyncMock(return_value=target)
            admin_repo = MockAdminRepo.return_value
            admin_repo.delete_all_permissions = AsyncMock(return_value=1)
            admin_repo.write_audit = AsyncMock()

            await service.revoke_role(
                actor_id=actor_id,
                target_user_id=target.id,
                product_id="vex",
                session=mock_session,
            )

        admin_repo.delete_all_permissions.assert_awaited_once_with(target.id, "vex")

    async def test_revoke_role_writes_audit_entry(
        self, service: AdminManagementService, mock_session: AsyncMock
    ) -> None:
        actor_id = uuid4()
        target = _make_user(role="admin", product_id="vex")

        with (
            patch("src.api.services.admin_management.UserRepository") as MockUserRepo,
            patch("src.api.services.admin_management.AdminRepository") as MockAdminRepo,
            patch("src.api.services.admin_management.new_id", return_value=uuid4()),
        ):
            MockUserRepo.return_value.get_active_user = AsyncMock(return_value=target)
            MockUserRepo.return_value.update_user_admin = AsyncMock(return_value=target)
            admin_repo = MockAdminRepo.return_value
            admin_repo.delete_all_permissions = AsyncMock(return_value=0)
            admin_repo.write_audit = AsyncMock()

            await service.revoke_role(
                actor_id=actor_id,
                target_user_id=target.id,
                product_id="vex",
                session=mock_session,
            )

        audit_entry = admin_repo.write_audit.call_args[0][0]
        assert audit_entry.action == "role.revoke"
        assert audit_entry.actor_id == actor_id


# ---------------------------------------------------------------------------
# grant_permission
# ---------------------------------------------------------------------------


class TestGrantPermission:
    async def test_grant_permission_success(
        self, service: AdminManagementService, mock_session: AsyncMock
    ) -> None:
        actor_id = uuid4()
        target = _make_user(role="admin", product_id="vex")

        with (
            patch("src.api.services.admin_management.UserRepository") as MockUserRepo,
            patch("src.api.services.admin_management.AdminRepository") as MockAdminRepo,
            patch("src.api.services.admin_management.new_id", return_value=uuid4()),
        ):
            MockUserRepo.return_value.get_active_user = AsyncMock(return_value=target)
            admin_repo = MockAdminRepo.return_value
            admin_repo.has_permission = AsyncMock(return_value=False)
            admin_repo.grant_permission = AsyncMock()
            admin_repo.write_audit = AsyncMock()

            await service.grant_permission(
                actor_id=actor_id,
                target_user_id=target.id,
                permission=AdminPermission.BILLING_ADJUST,
                product_id="vex",
                session=mock_session,
            )

        admin_repo.grant_permission.assert_awaited_once()
        admin_repo.write_audit.assert_awaited_once()

    async def test_grant_permission_idempotent(
        self, service: AdminManagementService, mock_session: AsyncMock
    ) -> None:
        target = _make_user(role="admin", product_id="vex")

        with (
            patch("src.api.services.admin_management.UserRepository") as MockUserRepo,
            patch("src.api.services.admin_management.AdminRepository") as MockAdminRepo,
        ):
            MockUserRepo.return_value.get_active_user = AsyncMock(return_value=target)
            admin_repo = MockAdminRepo.return_value
            admin_repo.has_permission = AsyncMock(return_value=True)  # already granted
            admin_repo.grant_permission = AsyncMock()

            await service.grant_permission(
                actor_id=uuid4(),
                target_user_id=target.id,
                permission=AdminPermission.BILLING_ADJUST,
                product_id="vex",
                session=mock_session,
            )

        admin_repo.grant_permission.assert_not_awaited()

    async def test_grant_permission_rejects_non_admin_target(
        self, service: AdminManagementService, mock_session: AsyncMock
    ) -> None:
        target = _make_user(role="user", product_id="vex")
        with patch("src.api.services.admin_management.UserRepository") as MockUserRepo:
            MockUserRepo.return_value.get_active_user = AsyncMock(return_value=target)
            with pytest.raises(InvalidRoleTransitionError):
                await service.grant_permission(
                    actor_id=uuid4(),
                    target_user_id=target.id,
                    permission=AdminPermission.BILLING_ADJUST,
                    product_id="vex",
                    session=mock_session,
                )

    async def test_grant_permission_writes_audit_entry(
        self, service: AdminManagementService, mock_session: AsyncMock
    ) -> None:
        actor_id = uuid4()
        target = _make_user(role="admin", product_id="vex")

        with (
            patch("src.api.services.admin_management.UserRepository") as MockUserRepo,
            patch("src.api.services.admin_management.AdminRepository") as MockAdminRepo,
            patch("src.api.services.admin_management.new_id", return_value=uuid4()),
        ):
            MockUserRepo.return_value.get_active_user = AsyncMock(return_value=target)
            admin_repo = MockAdminRepo.return_value
            admin_repo.has_permission = AsyncMock(return_value=False)
            admin_repo.grant_permission = AsyncMock()
            admin_repo.write_audit = AsyncMock()

            await service.grant_permission(
                actor_id=actor_id,
                target_user_id=target.id,
                permission=AdminPermission.BILLING_ADJUST,
                product_id="vex",
                session=mock_session,
            )

        audit_entry = admin_repo.write_audit.call_args[0][0]
        assert audit_entry.action == "permission.grant"
        assert AdminPermission.BILLING_ADJUST.value in audit_entry.detail

    async def test_grant_permission_handles_concurrent_race(
        self, service: AdminManagementService, mock_session: AsyncMock
    ) -> None:
        """Concurrent grant that hits unique constraint is treated as idempotent no-op."""
        target = _make_user(role="admin", product_id="vex")

        # Mock begin_nested as an async context manager whose __aexit__
        # propagates the IntegrityError (simulating SAVEPOINT rollback).
        nested_ctx = MagicMock()
        nested_ctx.__aenter__ = AsyncMock(return_value=MagicMock())
        nested_ctx.__aexit__ = AsyncMock(return_value=False)  # propagate exception
        mock_session.begin_nested = MagicMock(return_value=nested_ctx)

        with (
            patch("src.api.services.admin_management.UserRepository") as MockUserRepo,
            patch("src.api.services.admin_management.AdminRepository") as MockAdminRepo,
            patch("src.api.services.admin_management.new_id", return_value=uuid4()),
        ):
            MockUserRepo.return_value.get_active_user = AsyncMock(return_value=target)
            admin_repo = MockAdminRepo.return_value
            admin_repo.has_permission = AsyncMock(return_value=False)  # pre-check passes
            admin_repo.grant_permission = AsyncMock(side_effect=IntegrityError("", {}, Exception()))
            admin_repo.write_audit = AsyncMock()

            # Should NOT raise — treated as idempotent success
            await service.grant_permission(
                actor_id=uuid4(),
                target_user_id=target.id,
                permission=AdminPermission.BILLING_ADJUST,
                product_id="vex",
                session=mock_session,
            )

        # No audit entry for the race loser
        admin_repo.write_audit.assert_not_awaited()
        # Session.rollback should NOT be called — SAVEPOINT handles it
        mock_session.rollback.assert_not_awaited()


# ---------------------------------------------------------------------------
# revoke_permission
# ---------------------------------------------------------------------------


class TestRevokePermission:
    async def test_revoke_permission_success(
        self, service: AdminManagementService, mock_session: AsyncMock
    ) -> None:
        actor_id = uuid4()
        target_id = uuid4()

        with (
            patch("src.api.services.admin_management.AdminRepository") as MockAdminRepo,
            patch("src.api.services.admin_management.new_id", return_value=uuid4()),
        ):
            admin_repo = MockAdminRepo.return_value
            admin_repo.revoke_permission = AsyncMock(return_value=True)
            admin_repo.write_audit = AsyncMock()

            await service.revoke_permission(
                actor_id=actor_id,
                target_user_id=target_id,
                permission=AdminPermission.BILLING_ADJUST,
                product_id="vex",
                session=mock_session,
            )

        admin_repo.write_audit.assert_awaited_once()

    async def test_revoke_permission_idempotent_when_missing(
        self, service: AdminManagementService, mock_session: AsyncMock
    ) -> None:
        with patch("src.api.services.admin_management.AdminRepository") as MockAdminRepo:
            admin_repo = MockAdminRepo.return_value
            admin_repo.revoke_permission = AsyncMock(return_value=False)  # wasn't there
            admin_repo.write_audit = AsyncMock()

            await service.revoke_permission(
                actor_id=uuid4(),
                target_user_id=uuid4(),
                permission=AdminPermission.BILLING_ADJUST,
                product_id="vex",
                session=mock_session,
            )

        admin_repo.write_audit.assert_not_awaited()

    async def test_revoke_permission_writes_audit_entry(
        self, service: AdminManagementService, mock_session: AsyncMock
    ) -> None:
        actor_id = uuid4()
        target_id = uuid4()

        with (
            patch("src.api.services.admin_management.AdminRepository") as MockAdminRepo,
            patch("src.api.services.admin_management.new_id", return_value=uuid4()),
        ):
            admin_repo = MockAdminRepo.return_value
            admin_repo.revoke_permission = AsyncMock(return_value=True)
            admin_repo.write_audit = AsyncMock()

            await service.revoke_permission(
                actor_id=actor_id,
                target_user_id=target_id,
                permission=AdminPermission.BILLING_ADJUST,
                product_id="vex",
                session=mock_session,
            )

        audit_entry = admin_repo.write_audit.call_args[0][0]
        assert audit_entry.action == "permission.revoke"
        assert audit_entry.actor_id == actor_id


# ---------------------------------------------------------------------------
# force_revoke_role
# ---------------------------------------------------------------------------


class TestForceRevokeRole:
    async def test_force_revoke_bypasses_last_superadmin_guard(
        self, service: AdminManagementService, mock_session: AsyncMock
    ) -> None:
        """force_revoke_role demotes even the last superadmin."""
        actor_id = uuid4()
        target = _make_user(role="superadmin", product_id="vex")

        with (
            patch("src.api.services.admin_management.UserRepository") as MockUserRepo,
            patch("src.api.services.admin_management.AdminRepository") as MockAdminRepo,
            patch("src.api.services.admin_management.new_id", return_value=uuid4()),
        ):
            user_repo = MockUserRepo.return_value
            user_repo.get_active_user = AsyncMock(return_value=target)
            user_repo.update_user_admin = AsyncMock(return_value=target)

            admin_repo = MockAdminRepo.return_value
            admin_repo.delete_all_permissions = AsyncMock(return_value=1)
            admin_repo.write_audit = AsyncMock()

            await service.force_revoke_role(
                actor_id=actor_id,
                target_user_id=target.id,
                product_id="vex",
                session=mock_session,
            )

        user_repo.update_user_admin.assert_awaited_once_with(target.id, role="user")
        admin_repo.delete_all_permissions.assert_awaited_once()
        audit_entry = admin_repo.write_audit.call_args[0][0]
        assert "FORCED" in audit_entry.detail

    async def test_force_revoke_rejects_non_admin_target(
        self, service: AdminManagementService, mock_session: AsyncMock
    ) -> None:
        target = _make_user(role="user", product_id="vex")
        with patch("src.api.services.admin_management.UserRepository") as MockUserRepo:
            MockUserRepo.return_value.get_active_user = AsyncMock(return_value=target)
            with pytest.raises(InvalidRoleTransitionError):
                await service.force_revoke_role(
                    actor_id=uuid4(),
                    target_user_id=target.id,
                    product_id="vex",
                    session=mock_session,
                )

    async def test_force_revoke_target_not_found(
        self, service: AdminManagementService, mock_session: AsyncMock
    ) -> None:
        with patch("src.api.services.admin_management.UserRepository") as MockUserRepo:
            MockUserRepo.return_value.get_active_user = AsyncMock(return_value=None)
            with pytest.raises(AdminManagementError):
                await service.force_revoke_role(
                    actor_id=uuid4(),
                    target_user_id=uuid4(),
                    product_id="vex",
                    session=mock_session,
                )


# ---------------------------------------------------------------------------
# list_admins / list_admins_with_permissions
# ---------------------------------------------------------------------------


class TestListAdmins:
    async def test_list_admins_returns_users(
        self, service: AdminManagementService, mock_session: AsyncMock
    ) -> None:
        users = [_make_user(role="admin"), _make_user(role="superadmin")]

        with patch("src.api.services.admin_management.UserRepository") as MockUserRepo:
            MockUserRepo.return_value.list_users_by_roles = AsyncMock(return_value=users)
            result = await service.list_admins("vex", session=mock_session)

        assert result == users

    async def test_list_admins_with_permissions_empty(
        self, service: AdminManagementService, mock_session: AsyncMock
    ) -> None:
        with patch("src.api.services.admin_management.UserRepository") as MockUserRepo:
            MockUserRepo.return_value.list_users_by_roles = AsyncMock(return_value=[])
            result = await service.list_admins_with_permissions("vex", session=mock_session)

        assert result == []

    async def test_list_admins_with_permissions_returns_pairs(
        self, service: AdminManagementService, mock_session: AsyncMock
    ) -> None:
        user1 = _make_user(role="admin")
        user2 = _make_user(role="superadmin")
        perms_map = {user1.id: ["read_users"], user2.id: []}

        with (
            patch("src.api.services.admin_management.UserRepository") as MockUserRepo,
            patch("src.api.services.admin_management.AdminRepository") as MockAdminRepo,
        ):
            MockUserRepo.return_value.list_users_by_roles = AsyncMock(return_value=[user1, user2])
            MockAdminRepo.return_value.get_permissions_batch = AsyncMock(return_value=perms_map)

            result = await service.list_admins_with_permissions("vex", session=mock_session)

        assert len(result) == 2
        assert result[0] == (user1, ["read_users"])
        assert result[1] == (user2, [])


# ---------------------------------------------------------------------------
# get_user_permissions / get_audit_log
# ---------------------------------------------------------------------------


class TestGetUserPermissionsAndAuditLog:
    async def test_get_user_permissions(
        self, service: AdminManagementService, mock_session: AsyncMock
    ) -> None:
        grant1 = MagicMock()
        grant1.permission = "read_users"
        grant2 = MagicMock()
        grant2.permission = "manage_billing"

        with patch("src.api.services.admin_management.AdminRepository") as MockAdminRepo:
            MockAdminRepo.return_value.get_permissions = AsyncMock(return_value=[grant1, grant2])
            result = await service.get_user_permissions(uuid4(), "vex", session=mock_session)

        assert result == ["read_users", "manage_billing"]

    async def test_get_audit_log(
        self, service: AdminManagementService, mock_session: AsyncMock
    ) -> None:
        entry1 = MagicMock()
        entry2 = MagicMock()

        with patch("src.api.services.admin_management.AdminRepository") as MockAdminRepo:
            MockAdminRepo.return_value.get_audit_log = AsyncMock(return_value=[entry1, entry2])
            result = await service.get_audit_log("vex", session=mock_session)

        assert result == [entry1, entry2]
