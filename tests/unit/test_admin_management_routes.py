"""Tests for the PATCH /v1/admin/users/{user_id} hardening.

These tests exercise the handler business logic directly by calling
the underlying coroutine (bypassing Litestar's handler wrapping).
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from litestar.exceptions import PermissionDeniedException, ValidationException

from src.core.enums import UserRole


def _make_user(role: str = "admin", user_id=None) -> MagicMock:
    user = MagicMock()
    user.id = user_id or uuid4()
    user.role = role
    user.product_id = "vex"
    return user


class TestPatchUserHardening:
    async def test_patch_user_rejects_superadmin_role_value(self) -> None:
        """Cannot set superadmin role via patch_user endpoint."""
        from src.api.routes.admin import AdminController
        from src.api.schemas.admin import AdminPatchUserRequest

        # Access the raw function underneath the Litestar handler decorator
        raw_handler = AdminController.patch_user.fn  # type: ignore[attr-defined]

        admin_user = _make_user(role="admin")
        session = AsyncMock()
        target_id = uuid4()
        data = AdminPatchUserRequest(role=UserRole.SUPERADMIN)

        with pytest.raises(ValidationException, match="superadmin"):
            await raw_handler(
                MagicMock(),  # self
                admin_user=admin_user,
                user_id=target_id,
                data=data,
                session=session,
            )

    async def test_patch_user_rejects_modification_of_superadmin_users(self) -> None:
        """Cannot patch a superadmin user via this endpoint."""
        from src.api.routes.admin import AdminController
        from src.api.schemas.admin import AdminPatchUserRequest

        raw_handler = AdminController.patch_user.fn  # type: ignore[attr-defined]

        admin_user = _make_user(role="admin")
        session = AsyncMock()
        target_id = uuid4()
        target_superadmin = _make_user(role="superadmin", user_id=target_id)
        data = AdminPatchUserRequest(is_active=False)

        with patch("src.api.routes.admin.UserRepository") as MockRepo:
            MockRepo.return_value.get_active_user = AsyncMock(return_value=target_superadmin)
            with pytest.raises(PermissionDeniedException, match="superadmin"):
                await raw_handler(
                    MagicMock(),  # self
                    admin_user=admin_user,
                    user_id=target_id,
                    data=data,
                    session=session,
                )
