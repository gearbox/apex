"""Tests for billing account preference feature."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from src.api.services.billing import BillingService
from src.api.services.billing_errors import (
    AccountNotFoundError,
    OrganizationPermissionError,
)
from src.core.enums import AccountType

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def billing_service() -> BillingService:
    return BillingService()


@pytest.fixture
def mock_session() -> AsyncMock:
    session = AsyncMock()
    session.flush = AsyncMock()
    session.commit = AsyncMock()
    return session


def _make_account(
    account_id=None,
    is_active: bool = True,
    account_type: str = "personal",
    organization=None,
) -> MagicMock:
    account = MagicMock()
    account.id = account_id or uuid4()
    account.is_active = is_active
    account.account_type = account_type
    account.organization = organization
    return account


def _make_membership(org_id=None) -> MagicMock:
    member = MagicMock()
    member.organization_id = org_id or uuid4()
    return member


# ---------------------------------------------------------------------------
# BillingService.set_billing_account_preference
# ---------------------------------------------------------------------------


class TestSetBillingAccountPreference:
    async def test_set_personal_always_valid(
        self, billing_service: BillingService, mock_session: AsyncMock
    ) -> None:
        user_id = uuid4()
        mock_user = MagicMock()

        with (
            patch("src.api.services.billing.BillingRepository"),
            patch("src.api.services.billing.UserRepository") as MockUserRepo,
        ):
            user_repo = MockUserRepo.return_value
            user_repo.set_preferred_billing_account = AsyncMock(return_value=mock_user)

            await billing_service.set_billing_account_preference(
                user_id, AccountType.PERSONAL, session=mock_session
            )

        user_repo.set_preferred_billing_account.assert_awaited_once_with(
            user_id, AccountType.PERSONAL.value
        )

    async def test_set_enterprise_with_membership(
        self, billing_service: BillingService, mock_session: AsyncMock
    ) -> None:
        user_id = uuid4()
        membership = _make_membership()
        mock_user = MagicMock()

        with (
            patch("src.api.services.billing.BillingRepository") as MockBillingRepo,
            patch("src.api.services.billing.UserRepository") as MockUserRepo,
        ):
            billing_repo = MockBillingRepo.return_value
            billing_repo.get_active_membership = AsyncMock(return_value=membership)

            user_repo = MockUserRepo.return_value
            user_repo.set_preferred_billing_account = AsyncMock(return_value=mock_user)

            await billing_service.set_billing_account_preference(
                user_id, AccountType.ENTERPRISE, session=mock_session
            )

        user_repo.set_preferred_billing_account.assert_awaited_once_with(
            user_id, AccountType.ENTERPRISE.value
        )

    async def test_set_enterprise_without_membership_raises(
        self, billing_service: BillingService, mock_session: AsyncMock
    ) -> None:
        user_id = uuid4()

        with (
            patch("src.api.services.billing.BillingRepository") as MockBillingRepo,
            patch("src.api.services.billing.UserRepository"),
        ):
            billing_repo = MockBillingRepo.return_value
            billing_repo.get_active_membership = AsyncMock(return_value=None)

            with pytest.raises(OrganizationPermissionError):
                await billing_service.set_billing_account_preference(
                    user_id, AccountType.ENTERPRISE, session=mock_session
                )

    async def test_user_not_found_raises(
        self, billing_service: BillingService, mock_session: AsyncMock
    ) -> None:
        user_id = uuid4()

        with (
            patch("src.api.services.billing.BillingRepository"),
            patch("src.api.services.billing.UserRepository") as MockUserRepo,
        ):
            user_repo = MockUserRepo.return_value
            user_repo.set_preferred_billing_account = AsyncMock(return_value=None)

            with pytest.raises(AccountNotFoundError):
                await billing_service.set_billing_account_preference(
                    user_id, AccountType.PERSONAL, session=mock_session
                )


# ---------------------------------------------------------------------------
# BillingService.resolve_account_for_user — preference logic
# ---------------------------------------------------------------------------


class TestResolveAccountForUser:
    async def test_personal_preference_returns_personal_account(
        self, billing_service: BillingService, mock_session: AsyncMock
    ) -> None:
        user_id = uuid4()
        personal_account = _make_account(account_type="personal")

        with (
            patch("src.api.services.billing.UserRepository") as MockUserRepo,
            patch("src.api.services.billing.BillingRepository") as MockBillingRepo,
        ):
            user_repo = MockUserRepo.return_value
            user_repo.get_preferred_billing_account = AsyncMock(return_value="personal")

            billing_repo = MockBillingRepo.return_value
            billing_repo.get_account_by_user = AsyncMock(return_value=personal_account)

            result = await billing_service.resolve_account_for_user(user_id, session=mock_session)

        assert result is personal_account
        billing_repo.get_active_membership.assert_not_called()

    async def test_enterprise_preference_returns_enterprise_account(
        self, billing_service: BillingService, mock_session: AsyncMock
    ) -> None:
        user_id = uuid4()
        membership = _make_membership()
        enterprise_account = _make_account(account_type="enterprise")

        with (
            patch("src.api.services.billing.UserRepository") as MockUserRepo,
            patch("src.api.services.billing.BillingRepository") as MockBillingRepo,
        ):
            user_repo = MockUserRepo.return_value
            user_repo.get_preferred_billing_account = AsyncMock(return_value="enterprise")

            billing_repo = MockBillingRepo.return_value
            billing_repo.get_active_membership = AsyncMock(return_value=membership)
            billing_repo.get_account_by_organization = AsyncMock(return_value=enterprise_account)

            result = await billing_service.resolve_account_for_user(user_id, session=mock_session)

        assert result is enterprise_account

    async def test_no_preference_defaults_to_enterprise_when_member(
        self, billing_service: BillingService, mock_session: AsyncMock
    ) -> None:
        user_id = uuid4()
        membership = _make_membership()
        enterprise_account = _make_account(account_type="enterprise")

        with (
            patch("src.api.services.billing.UserRepository") as MockUserRepo,
            patch("src.api.services.billing.BillingRepository") as MockBillingRepo,
        ):
            user_repo = MockUserRepo.return_value
            user_repo.get_preferred_billing_account = AsyncMock(return_value=None)

            billing_repo = MockBillingRepo.return_value
            billing_repo.get_active_membership = AsyncMock(return_value=membership)
            billing_repo.get_account_by_organization = AsyncMock(return_value=enterprise_account)

            result = await billing_service.resolve_account_for_user(user_id, session=mock_session)

        assert result is enterprise_account

    async def test_no_preference_defaults_to_personal_when_no_member(
        self, billing_service: BillingService, mock_session: AsyncMock
    ) -> None:
        user_id = uuid4()
        personal_account = _make_account(account_type="personal")

        with (
            patch("src.api.services.billing.UserRepository") as MockUserRepo,
            patch("src.api.services.billing.BillingRepository") as MockBillingRepo,
        ):
            user_repo = MockUserRepo.return_value
            user_repo.get_preferred_billing_account = AsyncMock(return_value=None)

            billing_repo = MockBillingRepo.return_value
            billing_repo.get_active_membership = AsyncMock(return_value=None)
            billing_repo.get_account_by_user = AsyncMock(return_value=personal_account)

            result = await billing_service.resolve_account_for_user(user_id, session=mock_session)

        assert result is personal_account

    async def test_enterprise_preference_no_membership_raises(
        self, billing_service: BillingService, mock_session: AsyncMock
    ) -> None:
        user_id = uuid4()

        with (
            patch("src.api.services.billing.UserRepository") as MockUserRepo,
            patch("src.api.services.billing.BillingRepository") as MockBillingRepo,
        ):
            user_repo = MockUserRepo.return_value
            user_repo.get_preferred_billing_account = AsyncMock(return_value="enterprise")

            billing_repo = MockBillingRepo.return_value
            billing_repo.get_active_membership = AsyncMock(return_value=None)

            with pytest.raises(AccountNotFoundError):
                await billing_service.resolve_account_for_user(user_id, session=mock_session)

    async def test_personal_preference_no_account_raises(
        self, billing_service: BillingService, mock_session: AsyncMock
    ) -> None:
        user_id = uuid4()

        with (
            patch("src.api.services.billing.UserRepository") as MockUserRepo,
            patch("src.api.services.billing.BillingRepository") as MockBillingRepo,
        ):
            user_repo = MockUserRepo.return_value
            user_repo.get_preferred_billing_account = AsyncMock(return_value="personal")

            billing_repo = MockBillingRepo.return_value
            billing_repo.get_account_by_user = AsyncMock(return_value=None)

            with pytest.raises(AccountNotFoundError):
                await billing_service.resolve_account_for_user(user_id, session=mock_session)


# ---------------------------------------------------------------------------
# BillingService.get_billing_account_preference
# ---------------------------------------------------------------------------


class TestGetBillingAccountPreference:
    async def test_returns_none_when_not_set(
        self, billing_service: BillingService, mock_session: AsyncMock
    ) -> None:
        user_id = uuid4()

        with patch("src.api.services.billing.UserRepository") as MockUserRepo:
            user_repo = MockUserRepo.return_value
            user_repo.get_preferred_billing_account = AsyncMock(return_value=None)

            result = await billing_service.get_billing_account_preference(
                user_id, session=mock_session
            )

        assert result is None

    async def test_returns_stored_preference(
        self, billing_service: BillingService, mock_session: AsyncMock
    ) -> None:
        user_id = uuid4()

        with patch("src.api.services.billing.UserRepository") as MockUserRepo:
            user_repo = MockUserRepo.return_value
            user_repo.get_preferred_billing_account = AsyncMock(return_value="personal")

            result = await billing_service.get_billing_account_preference(
                user_id, session=mock_session
            )

        assert result == "personal"
