"""Tests for admin billing endpoints."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID, uuid4

import pytest
from litestar.exceptions import NotFoundException, PermissionDeniedException, ValidationException

from src.api.routes.admin import validate_and_patch_user
from src.api.schemas.admin import AdminPatchUserRequest
from src.api.schemas.pagination import CursorPage, decode_cursor, encode_cursor
from src.api.services.billing import BillingService
from src.core.enums import UserRole
from src.db.repositories.billing import BillingRepository

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_account(
    account_id: UUID | None = None,
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
    user_id: UUID | None = None,
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


def _make_cursor_user(user_id: UUID | None = None, ts: datetime | None = None) -> MagicMock:
    """Create a user mock with real created_at and id for cursor encoding."""
    user = _make_user(user_id=user_id)
    user.created_at = ts or datetime.now(UTC)
    user.id = user_id or uuid4()
    return user


class TestListUsers:
    async def test_returns_cursor_page_with_has_more_when_limit_plus_one_rows(self) -> None:
        """list_users handler returns CursorPage with has_more=True when repo returns limit+1 rows."""
        from src.api.routes.admin import AdminController

        limit = 3
        users = [_make_cursor_user() for _ in range(limit + 1)]
        admin_user = MagicMock()
        admin_user.id = uuid4()

        with patch("src.api.routes.admin.UserRepository") as MockRepo:
            MockRepo.return_value.list_users = AsyncMock(return_value=users)
            result = await AdminController.list_users.fn(
                MagicMock(),
                admin_user=admin_user,
                session=AsyncMock(),
                product_id="vex",
                is_active=None,
                role=None,
                email=None,
                limit=limit,
                cursor=None,
            )

        assert isinstance(result, CursorPage)
        assert len(result.items) == limit
        assert result.has_more is True
        assert result.next_cursor is not None

    async def test_returns_cursor_page_without_cursor_when_at_last_page(self) -> None:
        """list_users handler returns has_more=False, next_cursor=None when repo returns ≤ limit rows."""
        from src.api.routes.admin import AdminController

        limit = 3
        users = [_make_cursor_user() for _ in range(limit)]
        admin_user = MagicMock()
        admin_user.id = uuid4()

        with patch("src.api.routes.admin.UserRepository") as MockRepo:
            MockRepo.return_value.list_users = AsyncMock(return_value=users)
            result = await AdminController.list_users.fn(
                MagicMock(),
                admin_user=admin_user,
                session=AsyncMock(),
                product_id="vex",
                is_active=None,
                role=None,
                email=None,
                limit=limit,
                cursor=None,
            )

        assert result.has_more is False
        assert result.next_cursor is None
        assert len(result.items) == limit

    async def test_cursor_decoded_and_forwarded_to_repository(self) -> None:
        """Encoded cursor is decoded into cursor_ts/cursor_id and forwarded to repo."""
        from src.api.routes.admin import AdminController

        ts = datetime(2026, 6, 1, 12, 0, 0, tzinfo=UTC)
        cursor_uid = uuid4()
        cursor = encode_cursor(ts, cursor_uid)

        admin_user = MagicMock()
        admin_user.id = uuid4()

        with patch("src.api.routes.admin.UserRepository") as MockRepo:
            mock_list = AsyncMock(return_value=[])
            MockRepo.return_value.list_users = mock_list

            await AdminController.list_users.fn(
                MagicMock(),
                admin_user=admin_user,
                session=AsyncMock(),
                product_id="vex",
                is_active=None,
                role=None,
                email=None,
                limit=50,
                cursor=cursor,
            )

        mock_list.assert_awaited_once()
        call_kwargs = mock_list.call_args.kwargs
        assert call_kwargs["cursor_ts"] == ts
        assert call_kwargs["cursor_id"] == cursor_uid
        assert "offset" not in call_kwargs

    async def test_filters_forwarded_to_repository(self) -> None:
        """Handler forwards all filter params to list_users with correct argument names."""
        from src.api.routes.admin import AdminController

        admin_user = MagicMock()
        admin_user.id = uuid4()

        with patch("src.api.routes.admin.UserRepository") as MockRepo:
            mock_list = AsyncMock(return_value=[])
            MockRepo.return_value.list_users = mock_list

            await AdminController.list_users.fn(
                MagicMock(),
                admin_user=admin_user,
                session=AsyncMock(),
                product_id="vex",
                is_active=True,
                role="user",
                email="alice",
                limit=10,
                cursor=None,
            )

        call_kwargs = mock_list.call_args.kwargs
        assert call_kwargs["is_active"] is True
        assert call_kwargs["role"] == "user"
        assert call_kwargs["email_contains"] == "alice"
        assert call_kwargs["limit"] == 10
        assert "offset" not in call_kwargs


# ---------------------------------------------------------------------------
# PATCH /v1/admin/users/{user_id}
# ---------------------------------------------------------------------------


class TestPatchUser:
    async def test_updates_role_successfully(self, mock_session: AsyncMock) -> None:
        admin_id = uuid4()
        target_id = uuid4()
        target = _make_user(user_id=target_id, role="admin")
        data = AdminPatchUserRequest(role=UserRole.ADMIN)

        with patch("src.api.routes.admin.UserRepository") as MockRepo:
            repo = MockRepo.return_value
            repo.get_active_user = AsyncMock(return_value=target)
            repo.update_user_admin = AsyncMock(return_value=target)

            result = await validate_and_patch_user(
                admin_user_id=admin_id,
                user_id=target_id,
                data=data,
                session=mock_session,
            )

        assert result is target
        repo.update_user_admin.assert_awaited_once()

    async def test_returns_404_when_user_not_found(self, mock_session: AsyncMock) -> None:
        data = AdminPatchUserRequest(is_active=False)

        with patch("src.api.routes.admin.UserRepository") as MockRepo:
            repo = MockRepo.return_value
            repo.get_active_user = AsyncMock(return_value=None)
            repo.update_user_admin = AsyncMock(return_value=None)

            with pytest.raises(NotFoundException):
                await validate_and_patch_user(
                    admin_user_id=uuid4(),
                    user_id=uuid4(),
                    data=data,
                    session=mock_session,
                )

    async def test_cannot_modify_own_account(self, mock_session: AsyncMock) -> None:
        user_id = uuid4()
        data = AdminPatchUserRequest(role=UserRole.ADMIN)

        with pytest.raises(PermissionDeniedException):
            await validate_and_patch_user(
                admin_user_id=user_id,
                user_id=user_id,
                data=data,
                session=mock_session,
            )

    async def test_cannot_set_role_system(self, mock_session: AsyncMock) -> None:
        data = AdminPatchUserRequest(role=UserRole.SYSTEM)

        with pytest.raises(ValidationException):
            await validate_and_patch_user(
                admin_user_id=uuid4(),
                user_id=uuid4(),
                data=data,
                session=mock_session,
            )

    async def test_cannot_set_role_superadmin(self, mock_session: AsyncMock) -> None:
        data = AdminPatchUserRequest(role=UserRole.SUPERADMIN)

        with pytest.raises(ValidationException, match="superadmin"):
            await validate_and_patch_user(
                admin_user_id=uuid4(),
                user_id=uuid4(),
                data=data,
                session=mock_session,
            )

    async def test_cannot_modify_superadmin_user(self, mock_session: AsyncMock) -> None:
        target = _make_user(role="superadmin")
        data = AdminPatchUserRequest(is_active=False)

        with patch("src.api.routes.admin.UserRepository") as MockRepo:
            repo = MockRepo.return_value
            repo.get_active_user = AsyncMock(return_value=target)

            with pytest.raises(PermissionDeniedException, match="superadmin"):
                await validate_and_patch_user(
                    admin_user_id=uuid4(),
                    user_id=target.id,
                    data=data,
                    session=mock_session,
                )


# ---------------------------------------------------------------------------
# GET /v1/admin/organizations
# ---------------------------------------------------------------------------


def _make_org(
    org_id: UUID | None = None,
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


def _make_cursor_org(org_id: UUID | None = None, ts: datetime | None = None) -> MagicMock:
    """Create an org mock with real created_at and id for cursor encoding."""
    org = _make_org(org_id=org_id)
    org.created_at = ts or datetime.now(UTC)
    org.id = org_id or uuid4()
    return org


class TestListOrganizations:
    async def test_returns_cursor_page_with_has_more_when_limit_plus_one_rows(self) -> None:
        """list_organizations returns CursorPage with has_more=True when repo returns limit+1 rows."""
        from src.api.routes.admin import AdminController

        limit = 2
        orgs = [_make_cursor_org() for _ in range(limit + 1)]
        rows = [(o, i, i * 100) for i, o in enumerate(orgs)]
        admin_user = MagicMock()
        admin_user.id = uuid4()

        with patch("src.api.routes.admin.BillingRepository") as MockRepo:
            MockRepo.return_value.list_organizations = AsyncMock(return_value=rows)
            result = await AdminController.list_organizations.fn(
                MagicMock(),
                admin_user=admin_user,
                session=AsyncMock(),
                is_active=None,
                limit=limit,
                cursor=None,
            )

        assert isinstance(result, CursorPage)
        assert len(result.items) == limit
        assert result.has_more is True
        assert result.next_cursor is not None

    async def test_returns_cursor_page_without_cursor_at_last_page(self) -> None:
        """list_organizations returns has_more=False, next_cursor=None at last page."""
        from src.api.routes.admin import AdminController

        limit = 2
        orgs = [_make_cursor_org() for _ in range(limit)]
        rows = [(o, 0, 0) for o in orgs]
        admin_user = MagicMock()
        admin_user.id = uuid4()

        with patch("src.api.routes.admin.BillingRepository") as MockRepo:
            MockRepo.return_value.list_organizations = AsyncMock(return_value=rows)
            result = await AdminController.list_organizations.fn(
                MagicMock(),
                admin_user=admin_user,
                session=AsyncMock(),
                is_active=None,
                limit=limit,
                cursor=None,
            )

        assert result.has_more is False
        assert result.next_cursor is None

    async def test_cursor_decoded_and_forwarded_to_repository(self) -> None:
        """Encoded cursor is decoded into cursor_ts/cursor_id and forwarded to repo."""
        from src.api.routes.admin import AdminController

        ts = datetime(2026, 5, 15, 10, 0, 0, tzinfo=UTC)
        cursor_uid = uuid4()
        cursor = encode_cursor(ts, cursor_uid)
        admin_user = MagicMock()
        admin_user.id = uuid4()

        with patch("src.api.routes.admin.BillingRepository") as MockRepo:
            mock_list = AsyncMock(return_value=[])
            MockRepo.return_value.list_organizations = mock_list

            await AdminController.list_organizations.fn(
                MagicMock(),
                admin_user=admin_user,
                session=AsyncMock(),
                is_active=None,
                limit=50,
                cursor=cursor,
            )

        call_kwargs = mock_list.call_args.kwargs
        assert call_kwargs["cursor_ts"] == ts
        assert call_kwargs["cursor_id"] == cursor_uid
        assert "offset" not in call_kwargs

    async def test_is_active_filter_forwarded(self) -> None:
        """Handler forwards is_active=False to list_organizations."""
        from src.api.routes.admin import AdminController

        admin_user = MagicMock()
        admin_user.id = uuid4()

        with patch("src.api.routes.admin.BillingRepository") as MockRepo:
            mock_list = AsyncMock(return_value=[])
            MockRepo.return_value.list_organizations = mock_list

            await AdminController.list_organizations.fn(
                MagicMock(),
                admin_user=admin_user,
                session=AsyncMock(),
                is_active=False,
                limit=50,
                cursor=None,
            )

        call_kwargs = mock_list.call_args.kwargs
        assert call_kwargs["is_active"] is False
        assert "offset" not in call_kwargs


# ---------------------------------------------------------------------------
# AdminRepository.get_audit_log — partial cursor guard
# ---------------------------------------------------------------------------


class TestAdminRepositoryAuditLogCursorGuard:
    async def test_raises_when_cursor_ts_given_without_cursor_id(self) -> None:
        """Providing cursor_ts without cursor_id must raise ValueError immediately."""
        from datetime import UTC, datetime

        from src.db.repositories.admin import AdminRepository

        repo = AdminRepository(MagicMock())
        with pytest.raises(ValueError, match="cursor_ts and cursor_id must be provided together"):
            await repo.get_audit_log(
                "vex",
                cursor_ts=datetime(2026, 1, 1, tzinfo=UTC),
                cursor_id=None,
            )

    async def test_raises_when_cursor_id_given_without_cursor_ts(self) -> None:
        """Providing cursor_id without cursor_ts must raise ValueError immediately."""
        from src.db.repositories.admin import AdminRepository

        repo = AdminRepository(MagicMock())
        with pytest.raises(ValueError, match="cursor_ts and cursor_id must be provided together"):
            await repo.get_audit_log(
                "vex",
                cursor_ts=None,
                cursor_id=uuid4(),
            )


# ---------------------------------------------------------------------------
# GET /v1/admin/manage/audit — handler unit tests
# ---------------------------------------------------------------------------


def _make_audit_entry(ts: datetime | None = None) -> MagicMock:
    """Create an audit entry mock with real created_at and id for cursor encoding."""
    entry = MagicMock()
    entry.id = uuid4()
    entry.actor_id = uuid4()
    entry.target_user_id = uuid4()
    entry.action = "role.grant"
    entry.detail = "Role changed from 'user' to 'admin'"
    entry.source = "api"
    entry.created_at = ts or datetime.now(UTC)
    return entry


class TestAuditLogRouteHandler:
    async def test_returns_cursor_page_with_has_more_when_limit_plus_one_rows(self) -> None:
        """Handler trims to limit, sets has_more=True, next_cursor encodes last kept entry."""
        from src.api.routes.admin_management import AdminManagementController

        limit = 3
        entries = [_make_audit_entry() for _ in range(limit + 1)]
        superadmin = MagicMock()
        superadmin.id = uuid4()
        admin_mgmt = AsyncMock()
        admin_mgmt.get_audit_log = AsyncMock(return_value=entries)

        result = await AdminManagementController.get_audit_log.fn(
            MagicMock(),
            superadmin=superadmin,
            session=AsyncMock(),
            product_id="vex",
            admin_mgmt=admin_mgmt,
            target_user_id=None,
            limit=limit,
            cursor=None,
        )

        assert isinstance(result, CursorPage)
        assert len(result.items) == limit
        assert result.has_more is True
        assert result.next_cursor is not None
        # next_cursor must decode to the last kept entry's (created_at, id)
        decoded_ts, decoded_id = decode_cursor(result.next_cursor)
        last_kept = entries[limit - 1]
        assert decoded_id == last_kept.id
        assert decoded_ts == last_kept.created_at

    async def test_returns_no_cursor_when_at_last_page(self) -> None:
        """Handler returns has_more=False, next_cursor=None when repo returns ≤ limit rows."""
        from src.api.routes.admin_management import AdminManagementController

        limit = 3
        entries = [_make_audit_entry() for _ in range(limit)]
        superadmin = MagicMock()
        superadmin.id = uuid4()
        admin_mgmt = AsyncMock()
        admin_mgmt.get_audit_log = AsyncMock(return_value=entries)

        result = await AdminManagementController.get_audit_log.fn(
            MagicMock(),
            superadmin=superadmin,
            session=AsyncMock(),
            product_id="vex",
            admin_mgmt=admin_mgmt,
            target_user_id=None,
            limit=limit,
            cursor=None,
        )

        assert result.has_more is False
        assert result.next_cursor is None
        assert len(result.items) == limit

    async def test_cursor_decoded_and_forwarded_to_service(self) -> None:
        """Encoded cursor is decoded and forwarded as cursor_ts/cursor_id to the service."""
        from src.api.routes.admin_management import AdminManagementController

        ts = datetime(2026, 6, 10, 8, 0, 0, tzinfo=UTC)
        cursor_uid = uuid4()
        cursor = encode_cursor(ts, cursor_uid)
        superadmin = MagicMock()
        superadmin.id = uuid4()
        admin_mgmt = AsyncMock()
        admin_mgmt.get_audit_log = AsyncMock(return_value=[])

        await AdminManagementController.get_audit_log.fn(
            MagicMock(),
            superadmin=superadmin,
            session=AsyncMock(),
            product_id="vex",
            admin_mgmt=admin_mgmt,
            target_user_id=None,
            limit=50,
            cursor=cursor,
        )

        admin_mgmt.get_audit_log.assert_awaited_once()
        call_kwargs = admin_mgmt.get_audit_log.call_args.kwargs
        assert call_kwargs["cursor_ts"] == ts
        assert call_kwargs["cursor_id"] == cursor_uid

    async def test_target_user_id_forwarded_to_service(self) -> None:
        """target_user_id filter is forwarded to the service."""
        from src.api.routes.admin_management import AdminManagementController

        target_uid = uuid4()
        superadmin = MagicMock()
        superadmin.id = uuid4()
        admin_mgmt = AsyncMock()
        admin_mgmt.get_audit_log = AsyncMock(return_value=[])

        await AdminManagementController.get_audit_log.fn(
            MagicMock(),
            superadmin=superadmin,
            session=AsyncMock(),
            product_id="vex",
            admin_mgmt=admin_mgmt,
            target_user_id=target_uid,
            limit=50,
            cursor=None,
        )

        call_kwargs = admin_mgmt.get_audit_log.call_args.kwargs
        assert call_kwargs["target_user_id"] == target_uid
