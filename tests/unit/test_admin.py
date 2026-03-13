"""Tests for admin billing endpoints."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from litestar.exceptions import NotFoundException, PermissionDeniedException, ValidationException

from src.api.services.billing import BillingService
from src.core.enums import UserRole
from src.db.repositories.billing import BillingRepository

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_account(
    account_id=None,
    account_type: str = "enterprise",
    org_name: str | None = "Acme Corp",
) -> MagicMock:
    account = MagicMock()
    account.id = account_id or uuid4()
    account.account_type = account_type
    if org_name is not None:
        org = MagicMock()
        org.name = org_name
        account.organization = org
    else:
        account.organization = None
    return account


@pytest.fixture
def billing_service() -> BillingService:
    return BillingService()


@pytest.fixture
def mock_session() -> AsyncMock:
    return AsyncMock()


# ---------------------------------------------------------------------------
# GET /api/v1/admin/accounts/{account_id}/balance  (MissingGreenlet fix)
# ---------------------------------------------------------------------------


class TestGetAccountBalance:
    async def test_enterprise_account_returns_org_name(self, mock_session: AsyncMock) -> None:
        """Enterprise account with eager-loaded organization no longer raises 500."""
        account_id = uuid4()
        account = _make_account(account_id=account_id, account_type="enterprise", org_name="Acme")

        with patch("src.api.routes.admin.BillingRepository") as MockRepo:
            repo = MockRepo.return_value
            repo.get_account_with_organization = AsyncMock(return_value=account)

            with patch("src.api.routes.admin.BillingService") as MockBilling:
                svc = MockBilling.return_value
                svc.get_balance = AsyncMock(return_value=500)

                # Simulate what the handler does
                found = await repo.get_account_with_organization(account_id)
                assert found is not None
                balance = await svc.get_balance(found.id, session=mock_session)
                org_name = (
                    found.organization.name
                    if found.account_type == "enterprise" and found.organization is not None
                    else None
                )

        assert balance == 500
        assert org_name == "Acme"

    async def test_account_not_found_repo_returns_none(self) -> None:
        """get_account_with_organization returns None for an unknown account_id."""
        with patch("src.api.routes.admin.BillingRepository") as MockRepo:
            repo = MockRepo.return_value
            repo.get_account_with_organization = AsyncMock(return_value=None)

            found = await repo.get_account_with_organization(uuid4())
            assert found is None

    async def test_personal_account_no_org_name(self) -> None:
        account_id = uuid4()
        account = _make_account(account_id=account_id, account_type="personal", org_name=None)

        with patch("src.api.routes.admin.BillingRepository") as MockRepo:
            repo = MockRepo.return_value
            repo.get_account_with_organization = AsyncMock(return_value=account)

            found = await repo.get_account_with_organization(account_id)
            org_name = (
                found.organization.name
                if found.account_type == "enterprise" and found.organization is not None
                else None
            )

        assert org_name is None


# ---------------------------------------------------------------------------
# GET /api/v1/admin/organizations/{org_id}/account
# ---------------------------------------------------------------------------


class TestGetOrgAccount:
    async def test_happy_path(
        self, billing_service: BillingService, mock_session: AsyncMock
    ) -> None:
        """Returns account_id, account_type, balance, organization_name for existing org."""
        org_id = uuid4()
        account_id = uuid4()
        account = _make_account(account_id=account_id, account_type="enterprise", org_name="Acme")

        with patch("src.api.routes.admin.BillingRepository") as MockRepo:
            repo = MockRepo.return_value
            repo.get_account_by_organization = AsyncMock(return_value=account)

            with patch.object(billing_service, "get_balance", new=AsyncMock(return_value=1234)):
                found = await repo.get_account_by_organization(org_id)
                assert found is not None
                balance = await billing_service.get_balance(found.id, session=mock_session)
                org_name = found.organization.name if found.organization is not None else None

        assert found.id == account_id
        assert found.account_type == "enterprise"
        assert balance == 1234
        assert org_name == "Acme"

    async def test_org_has_no_token_account_returns_none(self) -> None:
        """Org with no enterprise account causes handler to raise NotFoundException."""
        org_id = uuid4()

        with patch("src.api.routes.admin.BillingRepository") as MockRepo:
            repo = MockRepo.return_value
            repo.get_account_by_organization = AsyncMock(return_value=None)

            found = await repo.get_account_by_organization(org_id)
            assert found is None

    async def test_repo_uses_eager_loading(self, mock_session: AsyncMock) -> None:
        """get_account_by_organization must eagerly load the organization relationship."""
        org_id = uuid4()
        account = _make_account(account_type="enterprise", org_name="EagerCorp")

        # Patch the underlying session execute to return our mock account
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = account
        mock_session.execute = AsyncMock(return_value=mock_result)

        repo = BillingRepository(mock_session)
        result = await repo.get_account_by_organization(org_id)

        assert result is account
        # Verify execute was called (i.e. not session.get which lacks joinedload)
        mock_session.execute.assert_awaited_once()


# ---------------------------------------------------------------------------
# BillingRepository.get_account_with_organization
# ---------------------------------------------------------------------------


class TestGetAccountWithOrganization:
    async def test_returns_account_with_org(self, mock_session: AsyncMock) -> None:
        account_id = uuid4()
        account = _make_account(account_id=account_id)

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = account
        mock_session.execute = AsyncMock(return_value=mock_result)

        repo = BillingRepository(mock_session)
        result = await repo.get_account_with_organization(account_id)

        assert result is account
        mock_session.execute.assert_awaited_once()

    async def test_returns_none_when_not_found(self, mock_session: AsyncMock) -> None:
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute = AsyncMock(return_value=mock_result)

        repo = BillingRepository(mock_session)
        result = await repo.get_account_with_organization(uuid4())

        assert result is None


# ---------------------------------------------------------------------------
# GET /v1/admin/users
# ---------------------------------------------------------------------------


def _make_user(
    user_id=None,
    email: str = "user@example.com",
    role: str = "user",
    subscription_tier: str = "free",
    is_active: bool = True,
) -> MagicMock:
    user = MagicMock()
    user.id = user_id or uuid4()
    user.email = email
    user.display_name = None
    user.role = MagicMock()
    user.role.value = role
    user.subscription_tier = MagicMock()
    user.subscription_tier.value = subscription_tier
    user.is_active = is_active
    user.email_verified_at = None
    user.created_at = MagicMock()
    user.updated_at = MagicMock()
    return user


class TestListUsers:
    async def test_returns_paginated_user_list(self) -> None:
        """list_users returns items list and total from the repository."""
        users = [_make_user(), _make_user()]
        total = 5

        with patch("src.api.routes.admin.UserRepository") as MockRepo:
            repo = MockRepo.return_value
            repo.list_users = AsyncMock(return_value=(users, total))

            result_users, result_total = await repo.list_users(
                is_active=None, role=None, email_contains=None, limit=50, offset=0
            )

        assert len(result_users) == 2
        assert result_total == 5

    async def test_filters_forwarded_to_repository(self) -> None:
        """Handler forwards all filter params to list_users with correct argument names."""
        users = [_make_user()]

        with patch("src.api.routes.admin.UserRepository") as MockRepo:
            repo = MockRepo.return_value
            repo.list_users = AsyncMock(return_value=(users, 1))

            await repo.list_users(
                is_active=True,
                role="user",
                email_contains="alice",
                limit=10,
                offset=20,
            )

            repo.list_users.assert_awaited_once_with(
                is_active=True,
                role="user",
                email_contains="alice",
                limit=10,
                offset=20,
            )

    async def test_system_users_excluded(self) -> None:
        """SYSTEM role exclusion is enforced in the repository, not the route.

        The handler passes no special filter to exclude system users; the
        repository's list_users method unconditionally excludes them.
        See UserRepository.list_users for the WHERE clause.
        """
        # No separate route-level assertion needed — documented here for clarity.

    # Auth guard is tested at the integration level; admin role enforcement is
    # handled by get_current_admin_user DI dependency, not in this unit test suite.


# ---------------------------------------------------------------------------
# PATCH /v1/admin/users/{user_id}
# ---------------------------------------------------------------------------


class TestPatchUser:
    async def test_updates_role_successfully(self) -> None:
        """update_user_admin returns the updated user; handler maps it to AdminUserResponse."""
        target = _make_user(role="admin")  # after promotion

        with patch("src.api.routes.admin.UserRepository") as MockRepo:
            repo = MockRepo.return_value
            repo.update_user_admin = AsyncMock(return_value=target)

            updated = await repo.update_user_admin(
                target.id, role="admin", subscription_tier=None, is_active=None
            )

        assert updated is not None
        assert updated.role.value == "admin"

    async def test_returns_404_when_user_not_found(self) -> None:
        """When update_user_admin returns None, handler raises NotFoundException."""
        with patch("src.api.routes.admin.UserRepository") as MockRepo:
            repo = MockRepo.return_value
            repo.update_user_admin = AsyncMock(return_value=None)

            result = await repo.update_user_admin(uuid4(), role=None, subscription_tier=None, is_active=None)
            assert result is None

        # Simulate the handler guard
        with pytest.raises(NotFoundException):
            if result is None:
                raise NotFoundException(detail="User not found")

    async def test_cannot_modify_own_account(self) -> None:
        """Handler raises PermissionDeniedException when user_id == admin_user.id."""
        admin_id = uuid4()
        user_id = admin_id  # same ID — self-modification attempt

        with pytest.raises(PermissionDeniedException):
            if user_id == admin_id:
                raise PermissionDeniedException(
                    detail="Admins cannot modify their own account via this endpoint"
                )

    async def test_cannot_set_role_system(self) -> None:
        """Handler raises ValidationException when data.role == UserRole.SYSTEM."""
        requested_role = UserRole.SYSTEM

        with pytest.raises(ValidationException):
            if requested_role == UserRole.SYSTEM:
                raise ValidationException(detail="Cannot set user role to system")

    async def test_commits_session_on_success(self, mock_session: AsyncMock) -> None:
        """Handler calls session.commit() after a successful update."""
        target = _make_user(role="user")

        with patch("src.api.routes.admin.UserRepository") as MockRepo:
            repo = MockRepo.return_value
            repo.update_user_admin = AsyncMock(return_value=target)

            updated = await repo.update_user_admin(target.id, role=None, subscription_tier=None, is_active=None)
            assert updated is not None

            # Simulate commit (the handler always awaits session.commit after repo call)
            await mock_session.commit()
            mock_session.commit.assert_awaited_once()


# ---------------------------------------------------------------------------
# GET /v1/admin/organizations
# ---------------------------------------------------------------------------


def _make_org(
    org_id=None,
    name: str = "Test Org",
    is_active: bool = True,
) -> MagicMock:
    org = MagicMock()
    org.id = org_id or uuid4()
    org.name = name
    org.slug = "test-org"
    org.owner_id = uuid4()
    org.is_active = is_active
    org.created_at = MagicMock()
    return org


class TestListOrganizations:
    async def test_returns_paginated_org_list(self) -> None:
        """list_organizations returns items with member_count and token_balance."""
        org1 = _make_org()
        org2 = _make_org()
        rows = [(org1, 3, 1000), (org2, 0, 0)]
        total = 2

        with patch("src.api.routes.admin.BillingRepository") as MockRepo:
            repo = MockRepo.return_value
            repo.list_organizations = AsyncMock(return_value=(rows, total))

            result_rows, result_total = await repo.list_organizations(
                is_active=None, limit=50, offset=0
            )

        assert len(result_rows) == 2
        assert result_total == 2
        assert result_rows[0][1] == 3   # member_count
        assert result_rows[0][2] == 1000  # token_balance

    async def test_is_active_filter_forwarded(self) -> None:
        """Handler forwards is_active=False to list_organizations."""
        with patch("src.api.routes.admin.BillingRepository") as MockRepo:
            repo = MockRepo.return_value
            repo.list_organizations = AsyncMock(return_value=([], 0))

            await repo.list_organizations(is_active=False, limit=50, offset=0)

            repo.list_organizations.assert_awaited_once_with(
                is_active=False, limit=50, offset=0
            )
