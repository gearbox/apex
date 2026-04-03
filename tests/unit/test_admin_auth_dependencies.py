"""Tests for admin auth dependency functions."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest
from litestar.exceptions import NotAuthorizedException

from src.api.dependencies.auth import (
    get_billing_adjust_admin,
    get_current_admin_user,
    get_current_superadmin_user,
)
from src.core.enums import UserRole


def _make_request(user_id=None, product_id: str = "vex") -> MagicMock:
    request = MagicMock()
    request.state.get = MagicMock(
        side_effect=lambda key, default=None: {
            "user_id": user_id,
            "product_id": product_id,
        }.get(key, default)
    )
    return request


def _make_user(role: str, product_id: str = "vex") -> MagicMock:
    user = MagicMock()
    user.id = uuid4()
    user.role = role
    user.product_id = product_id
    return user


# ---------------------------------------------------------------------------
# get_current_admin_user
# ---------------------------------------------------------------------------


class TestGetCurrentAdminUser:
    async def test_accepts_admin(self) -> None:
        user = _make_user("admin")
        request = _make_request(user_id=user.id)
        session = AsyncMock()

        with __import__("unittest.mock", fromlist=["patch"]).patch(
            "src.api.dependencies.auth.UserRepository"
        ) as MockRepo:
            MockRepo.return_value.get_active_user = AsyncMock(return_value=user)
            result = await get_current_admin_user(request, session)

        assert result is user

    async def test_accepts_superadmin(self) -> None:
        user = _make_user("superadmin")
        request = _make_request(user_id=user.id)
        session = AsyncMock()

        with __import__("unittest.mock", fromlist=["patch"]).patch(
            "src.api.dependencies.auth.UserRepository"
        ) as MockRepo:
            MockRepo.return_value.get_active_user = AsyncMock(return_value=user)
            result = await get_current_admin_user(request, session)

        assert result is user

    async def test_rejects_user_role(self) -> None:
        user = _make_user("user")
        request = _make_request(user_id=user.id)
        session = AsyncMock()

        with __import__("unittest.mock", fromlist=["patch"]).patch(
            "src.api.dependencies.auth.UserRepository"
        ) as MockRepo:
            MockRepo.return_value.get_active_user = AsyncMock(return_value=user)
            with pytest.raises(NotAuthorizedException):
                await get_current_admin_user(request, session)

    async def test_rejects_system_role(self) -> None:
        user = _make_user("system")
        request = _make_request(user_id=user.id)
        session = AsyncMock()

        with __import__("unittest.mock", fromlist=["patch"]).patch(
            "src.api.dependencies.auth.UserRepository"
        ) as MockRepo:
            MockRepo.return_value.get_active_user = AsyncMock(return_value=user)
            with pytest.raises(NotAuthorizedException):
                await get_current_admin_user(request, session)

    async def test_rejects_unauthenticated(self) -> None:
        request = _make_request(user_id=None)
        session = AsyncMock()
        with pytest.raises(NotAuthorizedException):
            await get_current_admin_user(request, session)

    async def test_accepts_admin_as_plain_string(self) -> None:
        """Ensure _require_role handles user.role as a plain string (ORM edge case)."""
        user = _make_user("admin")  # sets user.role = "admin" (str, not UserRole)
        assert isinstance(user.role, str)
        assert not isinstance(user.role, UserRole)

        request = _make_request(user_id=user.id)
        session = AsyncMock()

        with __import__("unittest.mock", fromlist=["patch"]).patch(
            "src.api.dependencies.auth.UserRepository"
        ) as MockRepo:
            MockRepo.return_value.get_active_user = AsyncMock(return_value=user)
            result = await get_current_admin_user(request, session)

        assert result is user

    async def test_rejects_unknown_role_as_unauthorized(self) -> None:
        """An unrecognized role value should raise NotAuthorizedException, not ValueError."""
        user = _make_user("bogus_role_value")
        request = _make_request(user_id=user.id)
        session = AsyncMock()

        with __import__("unittest.mock", fromlist=["patch"]).patch(
            "src.api.dependencies.auth.UserRepository"
        ) as MockRepo:
            MockRepo.return_value.get_active_user = AsyncMock(return_value=user)
            with pytest.raises(NotAuthorizedException):
                await get_current_admin_user(request, session)


# ---------------------------------------------------------------------------
# get_current_superadmin_user
# ---------------------------------------------------------------------------


class TestGetCurrentSuperadminUser:
    async def test_accepts_superadmin(self) -> None:
        user = _make_user("superadmin")
        request = _make_request(user_id=user.id)
        session = AsyncMock()

        with __import__("unittest.mock", fromlist=["patch"]).patch(
            "src.api.dependencies.auth.UserRepository"
        ) as MockRepo:
            MockRepo.return_value.get_active_user = AsyncMock(return_value=user)
            result = await get_current_superadmin_user(request, session)

        assert result is user

    async def test_rejects_admin(self) -> None:
        user = _make_user("admin")
        request = _make_request(user_id=user.id)
        session = AsyncMock()

        with __import__("unittest.mock", fromlist=["patch"]).patch(
            "src.api.dependencies.auth.UserRepository"
        ) as MockRepo:
            MockRepo.return_value.get_active_user = AsyncMock(return_value=user)
            with pytest.raises(NotAuthorizedException):
                await get_current_superadmin_user(request, session)

    async def test_rejects_user(self) -> None:
        user = _make_user("user")
        request = _make_request(user_id=user.id)
        session = AsyncMock()

        with __import__("unittest.mock", fromlist=["patch"]).patch(
            "src.api.dependencies.auth.UserRepository"
        ) as MockRepo:
            MockRepo.return_value.get_active_user = AsyncMock(return_value=user)
            with pytest.raises(NotAuthorizedException):
                await get_current_superadmin_user(request, session)


# ---------------------------------------------------------------------------
# get_billing_adjust_admin
# ---------------------------------------------------------------------------


class TestGetBillingAdjustAdmin:
    async def test_accepts_superadmin_without_grant(self) -> None:
        user = _make_user("superadmin")
        request = _make_request(user_id=user.id)
        session = AsyncMock()

        with __import__("unittest.mock", fromlist=["patch"]).patch(
            "src.api.dependencies.auth.UserRepository"
        ) as MockRepo:
            MockRepo.return_value.get_active_user = AsyncMock(return_value=user)
            # No AdminRepository mock needed — superadmin short-circuits
            result = await get_billing_adjust_admin(request, session)

        assert result is user

    async def test_accepts_admin_with_grant(self) -> None:
        user = _make_user("admin")
        request = _make_request(user_id=user.id)
        session = AsyncMock()

        # AdminRepository is lazily imported inside the function — patch the module attribute
        with (
            __import__("unittest.mock", fromlist=["patch"]).patch(
                "src.api.dependencies.auth.UserRepository"
            ) as MockUserRepo,
            __import__("unittest.mock", fromlist=["patch"]).patch(
                "src.db.repositories.admin.AdminRepository"
            ) as MockAdminRepo,
        ):
            MockUserRepo.return_value.get_active_user = AsyncMock(return_value=user)
            MockAdminRepo.return_value.has_permission = AsyncMock(return_value=True)
            result = await get_billing_adjust_admin(request, session)

        assert result is user

    async def test_rejects_admin_without_grant(self) -> None:
        user = _make_user("admin")
        request = _make_request(user_id=user.id)
        session = AsyncMock()

        with (
            __import__("unittest.mock", fromlist=["patch"]).patch(
                "src.api.dependencies.auth.UserRepository"
            ) as MockUserRepo,
            __import__("unittest.mock", fromlist=["patch"]).patch(
                "src.db.repositories.admin.AdminRepository"
            ) as MockAdminRepo,
        ):
            MockUserRepo.return_value.get_active_user = AsyncMock(return_value=user)
            MockAdminRepo.return_value.has_permission = AsyncMock(return_value=False)
            with pytest.raises(NotAuthorizedException, match="Billing adjustment"):
                await get_billing_adjust_admin(request, session)

    async def test_rejects_user(self) -> None:
        user = _make_user("user")
        request = _make_request(user_id=user.id)
        session = AsyncMock()

        with __import__("unittest.mock", fromlist=["patch"]).patch(
            "src.api.dependencies.auth.UserRepository"
        ) as MockRepo:
            MockRepo.return_value.get_active_user = AsyncMock(return_value=user)
            with pytest.raises(NotAuthorizedException):
                await get_billing_adjust_admin(request, session)
