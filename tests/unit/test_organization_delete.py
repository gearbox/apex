"""Tests for the delete organisation feature (service layer + HTTP route)."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID, uuid4

import pytest
from litestar import Litestar, Response
from litestar.di import Provide
from litestar.status_codes import (
    HTTP_200_OK,
    HTTP_401_UNAUTHORIZED,
    HTTP_403_FORBIDDEN,
    HTTP_404_NOT_FOUND,
    HTTP_409_CONFLICT,
)
from litestar.testing import TestClient
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.routes.organization import OrganizationController
from src.api.services.billing_errors import OrganizationBalanceError, OrganizationPermissionError
from src.api.services.organization import OrganizationService
from src.core.enums import OrgRole, TransactionType, UserRole

if TYPE_CHECKING:
    from src.api.security.jwt import JWTService


class _FakeAsyncSession(AsyncSession):
    """Minimal AsyncSession subclass for route testing.

    Bypasses the real __init__ so no SQLAlchemy engine is needed.
    commit / flush are overwritten with AsyncMocks by the fixture.
    """

    def __init__(self) -> None:
        pass  # intentionally skip AsyncSession.__init__


# ---------------------------------------------------------------------------
# Helpers / model factories
# ---------------------------------------------------------------------------


def _make_user(user_id: UUID | None = None, *, is_admin: bool = False) -> MagicMock:
    user = MagicMock()
    user.id = user_id or uuid4()
    user.role = UserRole.ADMIN if is_admin else UserRole.USER
    return user


def _make_membership(
    org_id: UUID | None = None,
    user_id: UUID | None = None,
    role: OrgRole = OrgRole.OWNER,
) -> MagicMock:
    member = MagicMock()
    member.organization_id = org_id or uuid4()
    member.user_id = user_id or uuid4()
    member.role = role.value
    return member


def _make_org(org_id: UUID | None = None, *, is_active: bool = True) -> MagicMock:
    org = MagicMock()
    org.id = org_id or uuid4()
    org.is_active = is_active
    org.name = "Test Org"
    org.slug = "test-org"
    org.owner_id = uuid4()
    org.created_at = None
    return org


def _make_account(account_id: UUID | None = None) -> MagicMock:
    account = MagicMock()
    account.id = account_id or uuid4()
    return account


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def organization_service() -> OrganizationService:
    return OrganizationService()


@pytest.fixture
def mock_session() -> _FakeAsyncSession:
    """Return a real AsyncSession subclass with mocked commit/flush.

    Using a real subclass (instead of AsyncMock) ensures Litestar's msgspec
    signature validator accepts the injected session value.
    """
    session = _FakeAsyncSession()
    session.commit = AsyncMock()  # type: ignore[method-assign]
    session.flush = AsyncMock()  # type: ignore[method-assign]
    return session


@pytest.fixture
def test_user_id() -> UUID:
    return uuid4()


@pytest.fixture
def auth_header(jwt_service: JWTService, test_user_id: UUID) -> dict[str, str]:
    token, _ = jwt_service.create_access_token(test_user_id)
    return {"Authorization": f"Bearer {token}"}


# ---------------------------------------------------------------------------
# Route app factory
# ---------------------------------------------------------------------------


def _build_org_service(
    *,
    get_organization_return: object = None,
    delete_organization_side_effect: BaseException | None = None,
    delete_organization_return: object = None,
) -> OrganizationService:
    """Create a real OrganizationService whose async methods are replaced with mocks.

    Using a real instance (instead of AsyncMock) ensures Litestar's msgspec
    signature validator accepts the injected value.
    """
    svc = OrganizationService()
    svc.get_organization = AsyncMock(return_value=get_organization_return)  # type: ignore[method-assign]
    if delete_organization_side_effect is not None:
        svc.delete_organization = AsyncMock(side_effect=delete_organization_side_effect)  # type: ignore[method-assign]
    else:
        svc.delete_organization = AsyncMock(return_value=delete_organization_return)  # type: ignore[method-assign]
    return svc


def _create_org_app(
    jwt_service: JWTService,
    org_service: OrganizationService,
    session: _FakeAsyncSession,
) -> Litestar:
    """Build a minimal Litestar app wrapping OrganizationController for route testing."""

    def _org_permission_handler(
        request: object,  # noqa: ARG001
        exc: OrganizationPermissionError,  # noqa: ARG001
    ) -> Response:
        return Response(content={"detail": "Forbidden"}, status_code=403)

    def _org_balance_handler(
        request: object,  # noqa: ARG001
        exc: OrganizationBalanceError,
    ) -> Response:
        return Response(
            content={"detail": str(exc), "balance": exc.balance},
            status_code=HTTP_409_CONFLICT,
        )

    app = Litestar(
        route_handlers=[OrganizationController],
        dependencies={
            "session": Provide(lambda: session, sync_to_thread=False),
            "organization_service": Provide(lambda: org_service, sync_to_thread=False),
            "billing_service": Provide(lambda: MagicMock(), sync_to_thread=False),  # noqa: PLW0108
        },
        exception_handlers={
            OrganizationPermissionError: _org_permission_handler,
            OrganizationBalanceError: _org_balance_handler,
        },
    )
    app.state["jwt_service"] = jwt_service
    return app


# ---------------------------------------------------------------------------
# Service unit tests
# ---------------------------------------------------------------------------


class TestDeleteOrganizationService:
    """Unit tests for OrganizationService.delete_organization."""

    async def test_owner_can_delete_with_zero_balance(
        self, organization_service: OrganizationService, mock_session: AsyncMock
    ) -> None:
        """Org owner can delete when the balance is zero."""
        org_id = uuid4()
        actor_id = uuid4()
        org = _make_org(org_id)
        membership = _make_membership(org_id=org_id, user_id=actor_id, role=OrgRole.OWNER)
        regular_user = _make_user(actor_id, is_admin=False)
        account = _make_account()

        with (
            patch("src.api.services.organization.BillingRepository") as MockBillingRepo,
            patch("src.api.services.organization.UserRepository") as MockUserRepo,
        ):
            billing_repo = MockBillingRepo.return_value
            billing_repo.get_membership = AsyncMock(return_value=membership)
            billing_repo.get_account_by_organization = AsyncMock(return_value=account)
            billing_repo.get_balance = AsyncMock(return_value=0)
            billing_repo.get_organization = AsyncMock(return_value=org)

            user_repo = MockUserRepo.return_value
            user_repo.get_active_user = AsyncMock(return_value=regular_user)

            await organization_service.delete_organization(org_id, actor_id, session=mock_session)

        assert org.is_active is False
        mock_session.flush.assert_awaited_once()
        billing_repo.create_transaction.assert_not_called()

    async def test_system_admin_can_delete_without_membership(
        self, organization_service: OrganizationService, mock_session: AsyncMock
    ) -> None:
        """System admin bypasses the owner/membership requirement."""
        org_id = uuid4()
        admin_id = uuid4()
        org = _make_org(org_id)
        admin_user = _make_user(admin_id, is_admin=True)
        account = _make_account()

        with (
            patch("src.api.services.organization.BillingRepository") as MockBillingRepo,
            patch("src.api.services.organization.UserRepository") as MockUserRepo,
        ):
            billing_repo = MockBillingRepo.return_value
            billing_repo.get_account_by_organization = AsyncMock(return_value=account)
            billing_repo.get_balance = AsyncMock(return_value=0)
            billing_repo.get_organization = AsyncMock(return_value=org)

            user_repo = MockUserRepo.return_value
            user_repo.get_active_user = AsyncMock(return_value=admin_user)

            await organization_service.delete_organization(org_id, admin_id, session=mock_session)

        billing_repo.get_membership.assert_not_called()
        assert org.is_active is False

    async def test_org_level_admin_member_cannot_delete(
        self, organization_service: OrganizationService, mock_session: AsyncMock
    ) -> None:
        """Org-level admin (not owner) is forbidden from deleting the organisation."""
        org_id = uuid4()
        actor_id = uuid4()
        admin_member = _make_membership(org_id=org_id, user_id=actor_id, role=OrgRole.ADMIN)
        regular_user = _make_user(actor_id, is_admin=False)

        with (
            patch("src.api.services.organization.BillingRepository") as MockBillingRepo,
            patch("src.api.services.organization.UserRepository") as MockUserRepo,
        ):
            billing_repo = MockBillingRepo.return_value
            billing_repo.get_membership = AsyncMock(return_value=admin_member)

            user_repo = MockUserRepo.return_value
            user_repo.get_active_user = AsyncMock(return_value=regular_user)

            with pytest.raises(OrganizationPermissionError):
                await organization_service.delete_organization(
                    org_id, actor_id, session=mock_session
                )

    async def test_non_member_cannot_delete(
        self, organization_service: OrganizationService, mock_session: AsyncMock
    ) -> None:
        """Authenticated user who is not a member cannot delete the organisation."""
        org_id = uuid4()
        actor_id = uuid4()
        regular_user = _make_user(actor_id, is_admin=False)

        with (
            patch("src.api.services.organization.BillingRepository") as MockBillingRepo,
            patch("src.api.services.organization.UserRepository") as MockUserRepo,
        ):
            billing_repo = MockBillingRepo.return_value
            billing_repo.get_membership = AsyncMock(return_value=None)

            user_repo = MockUserRepo.return_value
            user_repo.get_active_user = AsyncMock(return_value=regular_user)

            with pytest.raises(OrganizationPermissionError):
                await organization_service.delete_organization(
                    org_id, actor_id, session=mock_session
                )

    async def test_positive_balance_without_force_raises_balance_error(
        self, organization_service: OrganizationService, mock_session: AsyncMock
    ) -> None:
        """OrganizationBalanceError raised when balance > 0 and force_delete is False."""
        org_id = uuid4()
        actor_id = uuid4()
        membership = _make_membership(org_id=org_id, user_id=actor_id, role=OrgRole.OWNER)
        regular_user = _make_user(actor_id, is_admin=False)
        account = _make_account()

        with (
            patch("src.api.services.organization.BillingRepository") as MockBillingRepo,
            patch("src.api.services.organization.UserRepository") as MockUserRepo,
        ):
            billing_repo = MockBillingRepo.return_value
            billing_repo.get_membership = AsyncMock(return_value=membership)
            billing_repo.get_account_by_organization = AsyncMock(return_value=account)
            billing_repo.get_balance = AsyncMock(return_value=500)
            billing_repo.get_organization = AsyncMock(return_value=None)

            user_repo = MockUserRepo.return_value
            user_repo.get_active_user = AsyncMock(return_value=regular_user)

            with pytest.raises(OrganizationBalanceError) as exc_info:
                await organization_service.delete_organization(
                    org_id, actor_id, force_delete=False, session=mock_session
                )

        assert exc_info.value.balance == 500
        # Organisation must NOT be deactivated on error
        mock_session.flush.assert_not_awaited()

    async def test_force_delete_creates_adjustment_transaction_and_deactivates(
        self, organization_service: OrganizationService, mock_session: AsyncMock
    ) -> None:
        """force_delete=True zeroes the balance via ADMIN_ADJUSTMENT then soft-deletes."""
        org_id = uuid4()
        actor_id = uuid4()
        org = _make_org(org_id)
        membership = _make_membership(org_id=org_id, user_id=actor_id, role=OrgRole.OWNER)
        regular_user = _make_user(actor_id, is_admin=False)
        account = _make_account()
        mock_txn = MagicMock()

        with (
            patch("src.api.services.organization.BillingRepository") as MockBillingRepo,
            patch("src.api.services.organization.UserRepository") as MockUserRepo,
        ):
            billing_repo = MockBillingRepo.return_value
            billing_repo.get_membership = AsyncMock(return_value=membership)
            billing_repo.get_account_by_organization = AsyncMock(return_value=account)
            billing_repo.get_balance = AsyncMock(return_value=300)
            billing_repo.create_transaction = AsyncMock(return_value=mock_txn)
            billing_repo.get_organization = AsyncMock(return_value=org)

            user_repo = MockUserRepo.return_value
            user_repo.get_active_user = AsyncMock(return_value=regular_user)

            await organization_service.delete_organization(
                org_id, actor_id, force_delete=True, session=mock_session
            )

        billing_repo.create_transaction.assert_awaited_once()
        kwargs = billing_repo.create_transaction.call_args.kwargs
        assert kwargs["amount"] == -300
        assert kwargs["balance_after"] == 0
        assert kwargs["transaction_type"] == TransactionType.ADMIN_ADJUSTMENT.value
        assert kwargs["description"] == "Organization is force deleted by owner"
        assert kwargs["created_by"] == actor_id
        assert org.is_active is False

    async def test_force_delete_with_zero_balance_skips_transaction(
        self, organization_service: OrganizationService, mock_session: AsyncMock
    ) -> None:
        """force_delete=True with zero balance performs no adjustment transaction."""
        org_id = uuid4()
        actor_id = uuid4()
        org = _make_org(org_id)
        membership = _make_membership(org_id=org_id, user_id=actor_id, role=OrgRole.OWNER)
        regular_user = _make_user(actor_id, is_admin=False)
        account = _make_account()

        with (
            patch("src.api.services.organization.BillingRepository") as MockBillingRepo,
            patch("src.api.services.organization.UserRepository") as MockUserRepo,
        ):
            billing_repo = MockBillingRepo.return_value
            billing_repo.get_membership = AsyncMock(return_value=membership)
            billing_repo.get_account_by_organization = AsyncMock(return_value=account)
            billing_repo.get_balance = AsyncMock(return_value=0)
            billing_repo.get_organization = AsyncMock(return_value=org)

            user_repo = MockUserRepo.return_value
            user_repo.get_active_user = AsyncMock(return_value=regular_user)

            await organization_service.delete_organization(
                org_id, actor_id, force_delete=True, session=mock_session
            )

        billing_repo.create_transaction.assert_not_called()
        assert org.is_active is False

    async def test_no_token_account_proceeds_without_balance_check(
        self, organization_service: OrganizationService, mock_session: AsyncMock
    ) -> None:
        """If org has no enterprise token account, deletion proceeds without balance logic."""
        org_id = uuid4()
        actor_id = uuid4()
        org = _make_org(org_id)
        membership = _make_membership(org_id=org_id, user_id=actor_id, role=OrgRole.OWNER)
        regular_user = _make_user(actor_id, is_admin=False)

        with (
            patch("src.api.services.organization.BillingRepository") as MockBillingRepo,
            patch("src.api.services.organization.UserRepository") as MockUserRepo,
        ):
            billing_repo = MockBillingRepo.return_value
            billing_repo.get_membership = AsyncMock(return_value=membership)
            billing_repo.get_account_by_organization = AsyncMock(return_value=None)
            billing_repo.get_organization = AsyncMock(return_value=org)

            user_repo = MockUserRepo.return_value
            user_repo.get_active_user = AsyncMock(return_value=regular_user)

            await organization_service.delete_organization(org_id, actor_id, session=mock_session)

        billing_repo.get_balance.assert_not_called()
        billing_repo.create_transaction.assert_not_called()
        assert org.is_active is False

    async def test_balance_error_carries_correct_balance_value(
        self, organization_service: OrganizationService, mock_session: AsyncMock
    ) -> None:
        """The balance in OrganizationBalanceError matches the account's current balance."""
        org_id = uuid4()
        actor_id = uuid4()
        membership = _make_membership(org_id=org_id, user_id=actor_id, role=OrgRole.OWNER)
        regular_user = _make_user(actor_id, is_admin=False)
        account = _make_account()

        with (
            patch("src.api.services.organization.BillingRepository") as MockBillingRepo,
            patch("src.api.services.organization.UserRepository") as MockUserRepo,
        ):
            billing_repo = MockBillingRepo.return_value
            billing_repo.get_membership = AsyncMock(return_value=membership)
            billing_repo.get_account_by_organization = AsyncMock(return_value=account)
            billing_repo.get_balance = AsyncMock(return_value=1234)
            billing_repo.get_organization = AsyncMock(return_value=None)

            user_repo = MockUserRepo.return_value
            user_repo.get_active_user = AsyncMock(return_value=regular_user)

            with pytest.raises(OrganizationBalanceError) as exc_info:
                await organization_service.delete_organization(
                    org_id, actor_id, session=mock_session
                )

        assert exc_info.value.balance == 1234


# ---------------------------------------------------------------------------
# Route / HTTP integration tests
# ---------------------------------------------------------------------------


class TestDeleteOrganizationRoute:
    """HTTP integration tests for DELETE /api/v1/organizations/{org_id}."""

    def test_requires_authentication(
        self,
        jwt_service: JWTService,
        mock_session: _FakeAsyncSession,
    ) -> None:
        """No Authorization header → 401."""
        svc = _build_org_service()
        app = _create_org_app(jwt_service, svc, mock_session)

        with TestClient(app=app) as client:
            resp = client.delete(f"/v1/organizations/{uuid4()}")

        assert resp.status_code == HTTP_401_UNAUTHORIZED

    def test_returns_404_when_org_not_found(
        self,
        jwt_service: JWTService,
        mock_session: _FakeAsyncSession,
        auth_header: dict[str, str],
    ) -> None:
        """get_organization returns None → 404."""
        svc = _build_org_service(get_organization_return=None)
        app = _create_org_app(jwt_service, svc, mock_session)

        with TestClient(app=app) as client:
            resp = client.delete(f"/v1/organizations/{uuid4()}", headers=auth_header)

        assert resp.status_code == HTTP_404_NOT_FOUND

    def test_returns_403_when_permission_denied(
        self,
        jwt_service: JWTService,
        mock_session: _FakeAsyncSession,
        auth_header: dict[str, str],
    ) -> None:
        """OrganizationPermissionError from service → 403."""
        org = _make_org()
        svc = _build_org_service(
            get_organization_return=org,
            delete_organization_side_effect=OrganizationPermissionError("Insufficient permissions"),
        )
        app = _create_org_app(jwt_service, svc, mock_session)

        with TestClient(app=app) as client:
            resp = client.delete(f"/v1/organizations/{org.id}", headers=auth_header)

        assert resp.status_code == HTTP_403_FORBIDDEN

    def test_returns_409_when_balance_positive(
        self,
        jwt_service: JWTService,
        mock_session: _FakeAsyncSession,
        auth_header: dict[str, str],
    ) -> None:
        """OrganizationBalanceError from service → 409 with balance in body."""
        org = _make_org()
        svc = _build_org_service(
            get_organization_return=org,
            delete_organization_side_effect=OrganizationBalanceError(750),
        )
        app = _create_org_app(jwt_service, svc, mock_session)

        with TestClient(app=app) as client:
            resp = client.delete(f"/v1/organizations/{org.id}", headers=auth_header)

        assert resp.status_code == HTTP_409_CONFLICT
        assert resp.json()["balance"] == 750

    def test_successful_delete_returns_200_with_message(
        self,
        jwt_service: JWTService,
        mock_session: _FakeAsyncSession,
        auth_header: dict[str, str],
    ) -> None:
        """Happy path: org exists, no balance issues → 200."""
        org = _make_org()
        svc = _build_org_service(get_organization_return=org)
        app = _create_org_app(jwt_service, svc, mock_session)

        with TestClient(app=app) as client:
            resp = client.delete(f"/v1/organizations/{org.id}", headers=auth_header)

        assert resp.status_code == HTTP_200_OK
        assert resp.json() == {"message": "Organization deleted"}

    def test_force_delete_true_forwarded_to_service(
        self,
        jwt_service: JWTService,
        mock_session: _FakeAsyncSession,
        auth_header: dict[str, str],
    ) -> None:
        """?force_delete=true is parsed and passed to the service."""
        org = _make_org()
        svc = _build_org_service(get_organization_return=org)
        app = _create_org_app(jwt_service, svc, mock_session)

        with TestClient(app=app) as client:
            resp = client.delete(
                f"/v1/organizations/{org.id}?force_delete=true",
                headers=auth_header,
            )

        assert resp.status_code == HTTP_200_OK
        svc.delete_organization.assert_awaited_once()  # type: ignore[attr-defined]
        assert svc.delete_organization.call_args.kwargs["force_delete"] is True  # type: ignore[attr-defined]

    def test_force_delete_defaults_to_false(
        self,
        jwt_service: JWTService,
        mock_session: _FakeAsyncSession,
        auth_header: dict[str, str],
    ) -> None:
        """force_delete defaults to False when the query param is absent."""
        org = _make_org()
        svc = _build_org_service(get_organization_return=org)
        app = _create_org_app(jwt_service, svc, mock_session)

        with TestClient(app=app) as client:
            resp = client.delete(f"/v1/organizations/{org.id}", headers=auth_header)

        assert resp.status_code == HTTP_200_OK
        assert svc.delete_organization.call_args.kwargs["force_delete"] is False  # type: ignore[attr-defined]

    def test_session_committed_on_success(
        self,
        jwt_service: JWTService,
        mock_session: _FakeAsyncSession,
        auth_header: dict[str, str],
    ) -> None:
        """The database session is committed after a successful deletion."""
        org = _make_org()
        svc = _build_org_service(get_organization_return=org)
        app = _create_org_app(jwt_service, svc, mock_session)

        with TestClient(app=app) as client:
            client.delete(f"/v1/organizations/{org.id}", headers=auth_header)

        mock_session.commit.assert_awaited_once()  # type: ignore[attr-defined]

    def test_session_not_committed_when_org_not_found(
        self,
        jwt_service: JWTService,
        mock_session: _FakeAsyncSession,
        auth_header: dict[str, str],
    ) -> None:
        """Session is NOT committed when the route returns 404 early."""
        svc = _build_org_service(get_organization_return=None)
        app = _create_org_app(jwt_service, svc, mock_session)

        with TestClient(app=app) as client:
            client.delete(f"/v1/organizations/{uuid4()}", headers=auth_header)

        mock_session.commit.assert_not_awaited()  # type: ignore[attr-defined]
