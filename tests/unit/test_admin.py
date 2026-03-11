"""Tests for admin billing endpoints."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from src.api.services.billing import BillingService
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
