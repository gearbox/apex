"""Direct handler tests for route controllers.

Tests call ``Handler.fn(self, ...)`` directly to exercise handler logic
without spinning up Litestar's HTTP layer.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import msgspec
import pytest
from litestar.exceptions import (
    NotAuthorizedException,
    NotFoundException,
    PermissionDeniedException,
    ValidationException,
)
from litestar.response import Response, ServerSentEvent, Stream
from litestar.status_codes import HTTP_200_OK, HTTP_400_BAD_REQUEST, HTTP_404_NOT_FOUND

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# UnifiedJobController (jobs.py)
# ---------------------------------------------------------------------------


class TestJobRouteHandlers:
    async def test_list_jobs_delegates_to_service(self) -> None:
        from src.api.routes.jobs import UnifiedJobController
        from src.api.schemas.pagination import CursorPage

        page = MagicMock(spec=CursorPage)
        unified_job_service = AsyncMock()
        unified_job_service.list_jobs = AsyncMock(return_value=page)

        result = await UnifiedJobController.list_jobs.fn(  # type: ignore[attr-defined]
            MagicMock(),
            current_user_id=uuid4(),
            session=AsyncMock(),
            unified_job_service=unified_job_service,
        )
        assert result is page

    async def test_get_job_returns_200_when_found(self) -> None:
        from src.api.routes.jobs import UnifiedJobController

        job = MagicMock()
        unified_job_service = AsyncMock()
        unified_job_service.get_job = AsyncMock(return_value=job)

        response = await UnifiedJobController.get_job.fn(  # type: ignore[attr-defined]
            MagicMock(),
            current_user_id=uuid4(),
            job_id=uuid4(),
            session=AsyncMock(),
            unified_job_service=unified_job_service,
        )
        assert response.status_code == HTTP_200_OK
        assert response.content is job

    async def test_get_job_returns_404_when_not_found(self) -> None:
        from src.api.routes.jobs import UnifiedJobController

        unified_job_service = AsyncMock()
        unified_job_service.get_job = AsyncMock(return_value=None)

        response = await UnifiedJobController.get_job.fn(  # type: ignore[attr-defined]
            MagicMock(),
            current_user_id=uuid4(),
            job_id=uuid4(),
            session=AsyncMock(),
            unified_job_service=unified_job_service,
        )
        assert response.status_code == HTTP_404_NOT_FOUND

    async def test_delete_job_succeeds(self) -> None:
        from src.api.routes.jobs import UnifiedJobController

        mock_job = MagicMock()
        session = AsyncMock()
        session.commit = AsyncMock()

        with patch("src.db.repositories.job.JobRepository") as repo_cls:
            repo = MagicMock()
            repo.soft_delete = AsyncMock(return_value=mock_job)
            repo_cls.return_value = repo

            await UnifiedJobController.delete_job.fn(  # type: ignore[attr-defined]
                MagicMock(),
                current_user_id=uuid4(),
                job_id=uuid4(),
                session=session,
            )

        session.commit.assert_awaited_once()

    async def test_delete_job_raises_404_when_not_found(self) -> None:
        from src.api.routes.jobs import UnifiedJobController

        session = AsyncMock()

        with patch("src.db.repositories.job.JobRepository") as repo_cls:
            repo = MagicMock()
            repo.soft_delete = AsyncMock(return_value=None)
            repo_cls.return_value = repo

            with pytest.raises(NotFoundException):
                await UnifiedJobController.delete_job.fn(  # type: ignore[attr-defined]
                    MagicMock(),
                    current_user_id=uuid4(),
                    job_id=uuid4(),
                    session=session,
                )


# ---------------------------------------------------------------------------
# UserController (user.py)
# ---------------------------------------------------------------------------


class TestUserRouteHandlers:
    def _make_profile(self) -> MagicMock:
        return MagicMock()

    async def test_get_profile_success(self) -> None:
        from src.api.routes.user import UserController

        profile = self._make_profile()
        user_service = AsyncMock()
        user_service.get_profile = AsyncMock(return_value=profile)

        response = await UserController.get_profile.fn(  # type: ignore[attr-defined]
            MagicMock(),
            current_user_id=uuid4(),
            user_service=user_service,
        )
        assert response.status_code == HTTP_200_OK

    async def test_get_profile_raises_404_on_not_found(self) -> None:
        from src.api.routes.user import UserController
        from src.api.services.user import UserNotFoundError

        user_service = AsyncMock()
        user_service.get_profile = AsyncMock(side_effect=UserNotFoundError("no user"))

        with pytest.raises(NotFoundException):
            await UserController.get_profile.fn(  # type: ignore[attr-defined]
                MagicMock(),
                current_user_id=uuid4(),
                user_service=user_service,
            )

    def _make_data(self, **overrides: object) -> MagicMock:
        data = MagicMock()
        data.display_name = overrides.get("display_name")
        data.email = overrides.get("email")
        data.locale = overrides.get("locale")
        data.age_confirmed = overrides.get("age_confirmed")
        data.date_of_birth = overrides.get("date_of_birth")
        return data

    async def test_update_profile_success(self) -> None:
        from src.api.routes.user import UserController

        profile = self._make_profile()
        user_service = AsyncMock()
        user_service.update_profile = AsyncMock(return_value=profile)

        response = await UserController.update_profile.fn(  # type: ignore[attr-defined]
            MagicMock(),
            current_user_id=uuid4(),
            data=self._make_data(display_name="Alice", email="alice@example.com"),
            user_service=user_service,
            product_config=MagicMock(),
        )
        assert response.status_code == HTTP_200_OK

    async def test_update_profile_raises_404_on_not_found(self) -> None:
        from src.api.routes.user import UserController
        from src.api.services.user import UserNotFoundError

        user_service = AsyncMock()
        user_service.update_profile = AsyncMock(side_effect=UserNotFoundError("no user"))

        with pytest.raises(NotFoundException):
            await UserController.update_profile.fn(  # type: ignore[attr-defined]
                MagicMock(),
                current_user_id=uuid4(),
                data=self._make_data(),
                user_service=user_service,
                product_config=MagicMock(),
            )

    async def test_update_profile_returns_400_on_email_exists(self) -> None:
        from src.api.routes.user import UserController
        from src.api.services.user import EmailAlreadyExistsError

        user_service = AsyncMock()
        user_service.update_profile = AsyncMock(side_effect=EmailAlreadyExistsError("email taken"))

        response = await UserController.update_profile.fn(  # type: ignore[attr-defined]
            MagicMock(),
            current_user_id=uuid4(),
            data=self._make_data(email="taken@example.com"),
            user_service=user_service,
            product_config=MagicMock(),
        )
        assert response.status_code == HTTP_400_BAD_REQUEST

    async def test_update_profile_returns_400_on_age_verification_error(self) -> None:
        from src.api.routes.user import UserController
        from src.api.services.age_verification import AgeVerificationError

        user_service = AsyncMock()
        user_service.update_profile = AsyncMock(
            side_effect=AgeVerificationError("age claim failed")
        )

        response = await UserController.update_profile.fn(  # type: ignore[attr-defined]
            MagicMock(),
            current_user_id=uuid4(),
            data=self._make_data(age_confirmed=False),
            user_service=user_service,
            product_config=MagicMock(),
        )
        assert response.status_code == HTTP_400_BAD_REQUEST

    async def test_change_password_success(self) -> None:
        from src.api.routes.user import UserController

        user_service = AsyncMock()
        user_service.change_password = AsyncMock()

        data = MagicMock()
        data.current_password = "old"
        data.new_password = "new"

        response = await UserController.change_password.fn(  # type: ignore[attr-defined]
            MagicMock(),
            current_user_id=uuid4(),
            data=data,
            user_service=user_service,
        )
        assert response.status_code == HTTP_200_OK
        assert response.headers["Clear-Site-Data"] == '"cache", "storage"'

    async def test_change_password_returns_400_on_invalid_password(self) -> None:
        from src.api.routes.user import UserController
        from src.api.services.user import InvalidPasswordError

        user_service = AsyncMock()
        user_service.change_password = AsyncMock(side_effect=InvalidPasswordError("wrong"))

        data = MagicMock()
        data.current_password = "wrong"
        data.new_password = "new"

        response = await UserController.change_password.fn(  # type: ignore[attr-defined]
            MagicMock(),
            current_user_id=uuid4(),
            data=data,
            user_service=user_service,
        )
        assert response.status_code == HTTP_400_BAD_REQUEST
        assert "Clear-Site-Data" not in response.headers

    async def test_change_password_raises_404_on_user_not_found(self) -> None:
        from src.api.routes.user import UserController
        from src.api.services.user import UserNotFoundError

        user_service = AsyncMock()
        user_service.change_password = AsyncMock(side_effect=UserNotFoundError("no user"))

        data = MagicMock()
        data.current_password = "old"
        data.new_password = "new"

        with pytest.raises(NotFoundException):
            await UserController.change_password.fn(  # type: ignore[attr-defined]
                MagicMock(),
                current_user_id=uuid4(),
                data=data,
                user_service=user_service,
            )

    async def test_delete_account_success(self) -> None:
        from src.api.routes.user import UserController

        user_service = AsyncMock()
        user_service.deactivate_account = AsyncMock(return_value=datetime.now(UTC))

        response = await UserController.delete_account.fn(  # type: ignore[attr-defined]
            MagicMock(),
            current_user_id=uuid4(),
            user_service=user_service,
        )
        assert response.status_code == HTTP_200_OK
        assert response.headers["Clear-Site-Data"] == '"cache", "storage"'

    async def test_delete_account_raises_404_on_user_not_found(self) -> None:
        from src.api.routes.user import UserController
        from src.api.services.user import UserNotFoundError

        user_service = AsyncMock()
        user_service.deactivate_account = AsyncMock(side_effect=UserNotFoundError("no user"))

        with pytest.raises(NotFoundException):
            await UserController.delete_account.fn(  # type: ignore[attr-defined]
                MagicMock(),
                current_user_id=uuid4(),
                user_service=user_service,
            )

    async def test_get_stats_success(self) -> None:
        from src.api.routes.user import UserController

        stats = MagicMock()
        user_service = AsyncMock()
        user_service.get_stats = AsyncMock(return_value=stats)

        response = await UserController.get_stats.fn(  # type: ignore[attr-defined]
            MagicMock(),
            current_user_id=uuid4(),
            user_service=user_service,
        )
        assert response.status_code == HTTP_200_OK

    async def test_get_stats_raises_404_on_user_not_found(self) -> None:
        from src.api.routes.user import UserController
        from src.api.services.user import UserNotFoundError

        user_service = AsyncMock()
        user_service.get_stats = AsyncMock(side_effect=UserNotFoundError("no user"))

        with pytest.raises(NotFoundException):
            await UserController.get_stats.fn(  # type: ignore[attr-defined]
                MagicMock(),
                current_user_id=uuid4(),
                user_service=user_service,
            )

    async def test_logout_all_returns_count(self) -> None:
        from src.api.routes.user import UserController

        auth_service = AsyncMock()
        auth_service.logout_all = AsyncMock(return_value=3)

        response = await UserController.logout_all.fn(  # type: ignore[attr-defined]
            MagicMock(),
            current_user_id=uuid4(),
            auth_service=auth_service,
        )
        assert response.status_code == HTTP_200_OK
        assert "3" in response.content.message
        assert response.headers["Clear-Site-Data"] == '"cache", "storage"'

    async def test_get_current_user_id_raises_when_missing(self) -> None:
        from src.api.routes.user import get_current_user_id

        request = MagicMock()
        request.state.get.return_value = None

        with pytest.raises(NotAuthorizedException):
            await get_current_user_id(request)


# ---------------------------------------------------------------------------
# ContentProxyController (content.py)
# ---------------------------------------------------------------------------


class TestContentRouteHandlers:
    @staticmethod
    def _make_request() -> MagicMock:
        request = MagicMock()
        request.headers.get.return_value = None
        return request

    async def test_proxy_output_success(self) -> None:
        from src.api.routes.content import ContentProxyController

        content_proxy = AsyncMock()
        content_proxy.resolve_output = AsyncMock(return_value=("key/file.png", "etag123", 1234))
        content_proxy.ttl = 3600

        self_mock = MagicMock()
        self_mock._stream_from_r2 = AsyncMock(return_value=MagicMock())

        await ContentProxyController.proxy_output.fn(  # type: ignore[attr-defined]
            self_mock,
            request=self._make_request(),
            current_user_id=uuid4(),
            product_id="vex",
            output_id=uuid4(),
            session=AsyncMock(),
            content_proxy=content_proxy,
            r2_storage=MagicMock(),
        )

    async def test_proxy_output_returns_404_on_not_found(self) -> None:
        from src.api.routes.content import ContentProxyController
        from src.api.services.content_proxy import ContentNotFoundError

        content_proxy = AsyncMock()
        content_proxy.resolve_output = AsyncMock(side_effect=ContentNotFoundError("no output"))

        response = await ContentProxyController.proxy_output.fn(  # type: ignore[attr-defined]
            MagicMock(),
            request=self._make_request(),
            current_user_id=uuid4(),
            product_id="vex",
            output_id=uuid4(),
            session=AsyncMock(),
            content_proxy=content_proxy,
            r2_storage=MagicMock(),
        )
        assert response.status_code == HTTP_404_NOT_FOUND

    async def test_proxy_upload_success(self) -> None:
        from src.api.routes.content import ContentProxyController

        content_proxy = AsyncMock()
        content_proxy.resolve_upload = AsyncMock(return_value=("key/img.jpg", "etag456", 5678))
        content_proxy.ttl = 3600

        self_mock = MagicMock()
        self_mock._stream_from_r2 = AsyncMock(return_value=MagicMock())

        await ContentProxyController.proxy_upload.fn(  # type: ignore[attr-defined]
            self_mock,
            request=self._make_request(),
            current_user_id=uuid4(),
            product_id="vex",
            image_id=uuid4(),
            session=AsyncMock(),
            content_proxy=content_proxy,
            r2_storage=MagicMock(),
        )

    async def test_proxy_upload_returns_404_on_not_found(self) -> None:
        from src.api.routes.content import ContentProxyController
        from src.api.services.content_proxy import ContentNotFoundError

        content_proxy = AsyncMock()
        content_proxy.resolve_upload = AsyncMock(side_effect=ContentNotFoundError("no upload"))

        response = await ContentProxyController.proxy_upload.fn(  # type: ignore[attr-defined]
            MagicMock(),
            request=self._make_request(),
            current_user_id=uuid4(),
            product_id="vex",
            image_id=uuid4(),
            session=AsyncMock(),
            content_proxy=content_proxy,
            r2_storage=MagicMock(),
        )
        assert response.status_code == HTTP_404_NOT_FOUND


# ---------------------------------------------------------------------------
# AdminManagementController (admin_management.py)
# ---------------------------------------------------------------------------


class TestAdminManagementRouteHandlers:
    def _make_superadmin(self) -> MagicMock:
        user = MagicMock()
        user.id = uuid4()
        return user

    def _make_mgmt_service(self) -> AsyncMock:
        svc = AsyncMock()
        svc.list_admins_with_permissions = AsyncMock(return_value=[])
        svc.grant_role = AsyncMock()
        svc.revoke_role = AsyncMock()
        svc.grant_permission = AsyncMock()
        svc.revoke_permission = AsyncMock()
        svc.get_audit_log = AsyncMock(return_value=[])
        return svc

    async def test_list_admins_returns_list(self) -> None:
        from src.api.routes.admin_management import AdminManagementController

        mgmt = self._make_mgmt_service()
        response = await AdminManagementController.list_admins.fn(  # type: ignore[attr-defined]
            MagicMock(),
            superadmin=self._make_superadmin(),
            session=AsyncMock(),
            product_id="vex",
            admin_mgmt=mgmt,
        )
        assert response == []

    async def test_grant_role_success(self) -> None:
        from src.api.routes.admin_management import AdminManagementController

        mgmt = self._make_mgmt_service()
        session = AsyncMock()
        session.commit = AsyncMock()
        data = MagicMock()
        data.role = MagicMock()
        data.role.value = "admin"

        result = await AdminManagementController.grant_role.fn(  # type: ignore[attr-defined]
            MagicMock(),
            superadmin=self._make_superadmin(),
            user_id=uuid4(),
            data=data,
            session=session,
            product_id="vex",
            admin_mgmt=mgmt,
        )
        assert "granted" in result["message"]

    async def test_grant_role_raises_permission_denied_on_self_modification(self) -> None:
        from src.api.routes.admin_management import AdminManagementController
        from src.api.services.admin_management import SelfModificationError

        mgmt = self._make_mgmt_service()
        mgmt.grant_role = AsyncMock(side_effect=SelfModificationError("self"))
        session = AsyncMock()
        data = MagicMock()
        data.role = MagicMock()
        data.role.value = "admin"

        with pytest.raises(PermissionDeniedException):
            await AdminManagementController.grant_role.fn(  # type: ignore[attr-defined]
                MagicMock(),
                superadmin=self._make_superadmin(),
                user_id=uuid4(),
                data=data,
                session=session,
                product_id="vex",
                admin_mgmt=mgmt,
            )

    async def test_grant_role_raises_validation_on_invalid_transition(self) -> None:
        from src.api.routes.admin_management import AdminManagementController
        from src.api.services.admin_management import InvalidRoleTransitionError

        mgmt = self._make_mgmt_service()
        mgmt.grant_role = AsyncMock(side_effect=InvalidRoleTransitionError("bad"))
        session = AsyncMock()
        data = MagicMock()
        data.role = MagicMock()
        data.role.value = "admin"

        with pytest.raises(ValidationException):
            await AdminManagementController.grant_role.fn(  # type: ignore[attr-defined]
                MagicMock(),
                superadmin=self._make_superadmin(),
                user_id=uuid4(),
                data=data,
                session=session,
                product_id="vex",
                admin_mgmt=mgmt,
            )

    async def test_grant_role_raises_not_found_on_mgmt_error(self) -> None:
        from src.api.routes.admin_management import AdminManagementController
        from src.api.services.admin_management import AdminManagementError

        mgmt = self._make_mgmt_service()
        mgmt.grant_role = AsyncMock(side_effect=AdminManagementError("not found"))
        session = AsyncMock()
        data = MagicMock()
        data.role = MagicMock()
        data.role.value = "admin"

        with pytest.raises(NotFoundException):
            await AdminManagementController.grant_role.fn(  # type: ignore[attr-defined]
                MagicMock(),
                superadmin=self._make_superadmin(),
                user_id=uuid4(),
                data=data,
                session=session,
                product_id="vex",
                admin_mgmt=mgmt,
            )

    async def test_revoke_role_success(self) -> None:
        from src.api.routes.admin_management import AdminManagementController

        mgmt = self._make_mgmt_service()
        session = AsyncMock()
        session.commit = AsyncMock()

        result = await AdminManagementController.revoke_role.fn(  # type: ignore[attr-defined]
            MagicMock(),
            superadmin=self._make_superadmin(),
            user_id=uuid4(),
            session=session,
            product_id="vex",
            admin_mgmt=mgmt,
        )
        assert "revoked" in result["message"]

    async def test_revoke_role_raises_validation_on_last_superadmin(self) -> None:
        from src.api.routes.admin_management import AdminManagementController
        from src.api.services.admin_management import LastSuperadminError

        mgmt = self._make_mgmt_service()
        mgmt.revoke_role = AsyncMock(side_effect=LastSuperadminError("last"))
        session = AsyncMock()

        with pytest.raises(ValidationException):
            await AdminManagementController.revoke_role.fn(  # type: ignore[attr-defined]
                MagicMock(),
                superadmin=self._make_superadmin(),
                user_id=uuid4(),
                session=session,
                product_id="vex",
                admin_mgmt=mgmt,
            )

    async def test_revoke_role_raises_validation_on_invalid_transition(self) -> None:
        from src.api.routes.admin_management import AdminManagementController
        from src.api.services.admin_management import InvalidRoleTransitionError

        mgmt = self._make_mgmt_service()
        mgmt.revoke_role = AsyncMock(side_effect=InvalidRoleTransitionError("bad"))
        session = AsyncMock()

        with pytest.raises(ValidationException):
            await AdminManagementController.revoke_role.fn(  # type: ignore[attr-defined]
                MagicMock(),
                superadmin=self._make_superadmin(),
                user_id=uuid4(),
                session=session,
                product_id="vex",
                admin_mgmt=mgmt,
            )

    async def test_revoke_role_raises_permission_denied_on_self_modification(self) -> None:
        from src.api.routes.admin_management import AdminManagementController
        from src.api.services.admin_management import SelfModificationError

        mgmt = self._make_mgmt_service()
        mgmt.revoke_role = AsyncMock(side_effect=SelfModificationError("self"))
        session = AsyncMock()

        with pytest.raises(PermissionDeniedException):
            await AdminManagementController.revoke_role.fn(  # type: ignore[attr-defined]
                MagicMock(),
                superadmin=self._make_superadmin(),
                user_id=uuid4(),
                session=session,
                product_id="vex",
                admin_mgmt=mgmt,
            )

    async def test_revoke_role_raises_not_found_on_mgmt_error(self) -> None:
        from src.api.routes.admin_management import AdminManagementController
        from src.api.services.admin_management import AdminManagementError

        mgmt = self._make_mgmt_service()
        mgmt.revoke_role = AsyncMock(side_effect=AdminManagementError("not found"))
        session = AsyncMock()

        with pytest.raises(NotFoundException):
            await AdminManagementController.revoke_role.fn(  # type: ignore[attr-defined]
                MagicMock(),
                superadmin=self._make_superadmin(),
                user_id=uuid4(),
                session=session,
                product_id="vex",
                admin_mgmt=mgmt,
            )

    async def test_grant_permission_success(self) -> None:
        from src.api.routes.admin_management import AdminManagementController

        mgmt = self._make_mgmt_service()
        session = AsyncMock()
        session.commit = AsyncMock()
        data = MagicMock()
        data.permission = MagicMock()
        data.permission.value = "manage_users"

        result = await AdminManagementController.grant_permission.fn(  # type: ignore[attr-defined]
            MagicMock(),
            superadmin=self._make_superadmin(),
            user_id=uuid4(),
            data=data,
            session=session,
            product_id="vex",
            admin_mgmt=mgmt,
        )
        assert "granted" in result["message"]

    async def test_grant_permission_raises_validation_on_invalid_transition(self) -> None:
        from src.api.routes.admin_management import AdminManagementController
        from src.api.services.admin_management import InvalidRoleTransitionError

        mgmt = self._make_mgmt_service()
        mgmt.grant_permission = AsyncMock(side_effect=InvalidRoleTransitionError("bad"))
        session = AsyncMock()
        data = MagicMock()
        data.permission = MagicMock()
        data.permission.value = "manage_users"

        with pytest.raises(ValidationException):
            await AdminManagementController.grant_permission.fn(  # type: ignore[attr-defined]
                MagicMock(),
                superadmin=self._make_superadmin(),
                user_id=uuid4(),
                data=data,
                session=session,
                product_id="vex",
                admin_mgmt=mgmt,
            )

    async def test_grant_permission_raises_not_found_on_mgmt_error(self) -> None:
        from src.api.routes.admin_management import AdminManagementController
        from src.api.services.admin_management import AdminManagementError

        mgmt = self._make_mgmt_service()
        mgmt.grant_permission = AsyncMock(side_effect=AdminManagementError("not found"))
        session = AsyncMock()
        data = MagicMock()
        data.permission = MagicMock()
        data.permission.value = "manage_users"

        with pytest.raises(NotFoundException):
            await AdminManagementController.grant_permission.fn(  # type: ignore[attr-defined]
                MagicMock(),
                superadmin=self._make_superadmin(),
                user_id=uuid4(),
                data=data,
                session=session,
                product_id="vex",
                admin_mgmt=mgmt,
            )

    async def test_revoke_permission_success(self) -> None:
        from src.api.routes.admin_management import AdminManagementController

        mgmt = self._make_mgmt_service()
        session = AsyncMock()
        session.commit = AsyncMock()
        data = MagicMock()
        data.permission = MagicMock()
        data.permission.value = "manage_users"

        result = await AdminManagementController.revoke_permission.fn(  # type: ignore[attr-defined]
            MagicMock(),
            superadmin=self._make_superadmin(),
            user_id=uuid4(),
            data=data,
            session=session,
            product_id="vex",
            admin_mgmt=mgmt,
        )
        assert "revoked" in result["message"]

    async def test_get_audit_log_returns_list(self) -> None:
        from src.api.routes.admin_management import AdminManagementController
        from src.api.schemas.pagination import CursorPage

        mgmt = self._make_mgmt_service()
        response = await AdminManagementController.get_audit_log.fn(  # type: ignore[attr-defined]
            MagicMock(),
            superadmin=self._make_superadmin(),
            session=AsyncMock(),
            product_id="vex",
            admin_mgmt=mgmt,
        )

        assert isinstance(response, CursorPage)
        assert response.items == []
        assert response.has_more is False
        assert response.next_cursor is None


# ---------------------------------------------------------------------------
# Health route helpers
# ---------------------------------------------------------------------------


class TestHealthRouteHelpers:
    def test_to_category_builds_response(self) -> None:
        from src.api.routes.health import _to_category

        cat_data = {
            "status": "healthy",
            "components": [{"name": "postgres", "status": "healthy", "latency_ms": 5}],
        }
        result = _to_category(cat_data)
        assert result.status == "healthy"
        assert len(result.components) == 1

    def test_to_gpu_sessions_builds_response(self) -> None:
        from src.api.routes.health import _to_gpu_sessions

        gpu_data = {
            "status": "healthy",
            "total": 3,
            "healthy": 2,
            "stale": 1,
            "message": "ok",
        }
        result = _to_gpu_sessions(gpu_data)
        assert result.total == 3
        assert result.stale == 1


# ---------------------------------------------------------------------------
# HealthController / AdminHealthController route handlers
# ---------------------------------------------------------------------------


class TestHealthRouteHandlers:
    async def test_readiness_returns_200_when_ready(self) -> None:
        from src.api.routes.health import HealthController

        health_service = AsyncMock()
        health_service.readiness = AsyncMock(return_value=(True, {"postgres": "ok"}))

        response = await HealthController.readiness.fn(  # type: ignore[attr-defined]
            MagicMock(),
            health_service=health_service,
        )
        assert response.status_code == 200

    async def test_readiness_returns_503_when_not_ready(self) -> None:
        from src.api.routes.health import HealthController

        health_service = AsyncMock()
        health_service.readiness = AsyncMock(return_value=(False, {"postgres": "down"}))

        response = await HealthController.readiness.fn(  # type: ignore[attr-defined]
            MagicMock(),
            health_service=health_service,
        )
        assert response.status_code == 503

    async def test_detailed_builds_response(self) -> None:
        from src.api.routes.health import AdminHealthController

        health_service = AsyncMock()
        health_service.detailed = AsyncMock(
            return_value={
                "status": "healthy",
                "checked_at": "2026-01-01T00:00:00Z",
                "infrastructure": {"status": "healthy", "components": []},
                "platform_apis": {"status": "healthy", "components": []},
                "cloud_providers": {},
                "gpu_sessions": {
                    "status": "healthy",
                    "total": 0,
                    "healthy": 0,
                    "stale": 0,
                    "message": "",
                },
            }
        )

        result = await AdminHealthController.detailed.fn(  # type: ignore[attr-defined]
            MagicMock(),
            health_service=health_service,
            admin_user=MagicMock(),
        )
        assert result.status == "healthy"

    async def test_stream_returns_sse(self) -> None:
        from src.api.routes.health import AdminHealthController

        health_service = AsyncMock()

        with (
            patch("src.api.routes.health.get_settings") as mock_settings,
            patch("src.api.routes.health.health_sse_generator"),
        ):
            mock_settings.return_value = MagicMock()
            response = await AdminHealthController.stream.fn(  # type: ignore[attr-defined]
                MagicMock(),
                health_service=health_service,
                admin_user=MagicMock(),
            )
        assert response is not None

    async def test_history_returns_snapshot_list(self) -> None:
        from src.api.routes.health import AdminHealthController

        session = AsyncMock()
        snapshot = MagicMock()
        snapshot.checked_at = datetime.now(UTC)
        snapshot.overall_status = "healthy"
        snapshot.snapshot_data = {}

        with patch("src.api.routes.health.HealthSnapshotRepository") as repo_cls:
            repo = MagicMock()
            repo.list_range = AsyncMock(return_value=[snapshot])
            repo_cls.return_value = repo

            with patch("src.api.routes.health.parse_history_params") as parse_mock:
                query = MagicMock()
                query.after = None
                query.before = None
                query.limit = 60
                parse_mock.return_value = query

                result = await AdminHealthController.history.fn(  # type: ignore[attr-defined]
                    MagicMock(),
                    session=session,
                    admin_user=MagicMock(),
                )

        assert len(result) == 1


# ---------------------------------------------------------------------------
# ProvidersController (providers.py)
# ---------------------------------------------------------------------------


class TestProvidersRouteHandlers:
    async def test_list_providers_no_auth_no_grok(self) -> None:
        from src.api.routes.providers import ProvidersController
        from src.core.enums import Provider

        session = AsyncMock()
        generation_service = MagicMock()
        generation_service.configured_providers = frozenset()

        with patch("src.api.routes.providers.GenerationModelRepository") as repo_cls:
            repo = MagicMock()
            repo.list_enabled_for_product = AsyncMock(return_value=[])
            repo_cls.return_value = repo

            product_config = MagicMock()
            product_config.is_model_allowed.return_value = True

            result = await ProvidersController.list_providers.fn(  # type: ignore[attr-defined]
                MagicMock(),
                session=session,
                generation_service=generation_service,
                current_user_id=None,
                product_config=product_config,
                product_id="vex",
            )

        assert result.user_context is None
        grok_provider = next(p for p in result.providers if p.provider == Provider.GROK.value)
        assert grok_provider.available is False

    async def test_list_providers_with_authenticated_user(self) -> None:
        from src.api.routes.providers import ProvidersController
        from src.core.enums import Provider

        session = AsyncMock()
        user = MagicMock()
        user.subscription_tier = MagicMock()
        user.subscription_tier.value = "free"

        generation_service = MagicMock()
        generation_service.configured_providers = frozenset({Provider.GROK})

        with (
            patch("src.api.routes.providers.GenerationModelRepository") as repo_cls,
            patch("src.api.routes.providers.GpuSessionRepository") as gpu_repo_cls,
            patch("src.api.routes.providers.UserRepository") as user_repo_cls,
        ):
            repo = MagicMock()
            repo.list_enabled_for_product = AsyncMock(return_value=[])
            repo_cls.return_value = repo

            gpu_repo = MagicMock()
            gpu_repo.list_by_user = AsyncMock(return_value=[])
            gpu_repo_cls.return_value = gpu_repo

            user_repo = MagicMock()
            user_repo.get_active_user = AsyncMock(return_value=user)
            user_repo_cls.return_value = user_repo

            product_config = MagicMock()
            product_config.is_model_allowed.return_value = True

            result = await ProvidersController.list_providers.fn(  # type: ignore[attr-defined]
                MagicMock(),
                session=session,
                generation_service=generation_service,
                current_user_id=uuid4(),
                product_config=product_config,
                product_id="vex",
            )

        assert result.user_context is not None
        grok_provider = next(p for p in result.providers if p.provider == Provider.GROK.value)
        assert grok_provider.available is True

    def test_configured_providers_controls_availability(self) -> None:
        from src.api.services.generation.service import GenerationService
        from src.core.enums import Provider

        svc = GenerationService(
            providers={Provider.GROK: MagicMock()},
            billing_service=MagicMock(),
            pricing_service=MagicMock(),
            rate_limiter=MagicMock(),
        )
        assert Provider.GROK in svc.configured_providers
        assert Provider.AISHA not in svc.configured_providers

    async def test_list_providers_with_valid_model_record(self) -> None:
        from src.api.routes.providers import ProvidersController

        session = AsyncMock()
        record = MagicMock()
        record.model_key = "aisha-image"
        record.name = "Aisha Image"
        record.description = "Aisha image model"
        record.is_enabled = True

        generation_service = MagicMock()
        generation_service.configured_providers = frozenset()

        with patch("src.api.routes.providers.GenerationModelRepository") as repo_cls:
            repo = MagicMock()
            repo.list_enabled_for_product = AsyncMock(return_value=[record])
            repo_cls.return_value = repo

            product_config = MagicMock()
            product_config.is_model_allowed.return_value = True

            result = await ProvidersController.list_providers.fn(  # type: ignore[attr-defined]
                MagicMock(),
                session=session,
                generation_service=generation_service,
                current_user_id=None,
                product_config=product_config,
                product_id="vex",
            )

        # Verify model was built and grouped by provider
        aisha_provider = next(p for p in result.providers if p.provider == "aisha")
        assert len(aisha_provider.models) == 1
        assert aisha_provider.models[0].model_key == "aisha-image"

    async def test_list_providers_skips_unknown_model_key(self) -> None:
        from src.api.routes.providers import ProvidersController

        session = AsyncMock()
        record = MagicMock()
        record.model_key = "unknown-model-xyz"

        generation_service = MagicMock()
        generation_service.configured_providers = frozenset()

        with patch("src.api.routes.providers.GenerationModelRepository") as repo_cls:
            repo = MagicMock()
            repo.list_enabled_for_product = AsyncMock(return_value=[record])
            repo_cls.return_value = repo

            product_config = MagicMock()
            product_config.is_model_allowed.return_value = True

            result = await ProvidersController.list_providers.fn(  # type: ignore[attr-defined]
                MagicMock(),
                session=session,
                generation_service=generation_service,
                current_user_id=None,
                product_config=product_config,
                product_id="vex",
            )

        # Unknown key skipped — all providers have 0 models
        for provider in result.providers:
            assert provider.models == []

    async def test_list_providers_user_not_found_returns_no_context(self) -> None:
        from src.api.routes.providers import ProvidersController

        session = AsyncMock()
        generation_service = MagicMock()
        generation_service.configured_providers = frozenset()

        with (
            patch("src.api.routes.providers.GenerationModelRepository") as repo_cls,
            patch("src.api.routes.providers.GpuSessionRepository") as gpu_repo_cls,
            patch("src.api.routes.providers.UserRepository") as user_repo_cls,
        ):
            repo = MagicMock()
            repo.list_enabled_for_product = AsyncMock(return_value=[])
            repo_cls.return_value = repo

            gpu_repo = MagicMock()
            gpu_repo.list_by_user = AsyncMock(return_value=[])
            gpu_repo_cls.return_value = gpu_repo

            user_repo = MagicMock()
            user_repo.get_active_user = AsyncMock(return_value=None)
            user_repo_cls.return_value = user_repo

            product_config = MagicMock()
            product_config.is_model_allowed.return_value = True

            result = await ProvidersController.list_providers.fn(  # type: ignore[attr-defined]
                MagicMock(),
                session=session,
                generation_service=generation_service,
                current_user_id=uuid4(),  # authenticated but user deleted
                product_config=product_config,
                product_id="vex",
            )

        assert result.user_context is None

    async def test_providers_endpoint_exposes_edit_aspect_ratios(self) -> None:
        from src.api.routes.providers import ProvidersController
        from src.core.enums import AspectRatio

        session = AsyncMock()
        grok_record = MagicMock()
        grok_record.model_key = "grok-imagine-image"
        grok_record.name = "Grok Imagine"
        grok_record.description = "Grok image model"
        grok_record.is_enabled = True

        aisha_record = MagicMock()
        aisha_record.model_key = "aisha-image"
        aisha_record.name = "Aisha Image"
        aisha_record.description = "Aisha image model"
        aisha_record.is_enabled = True

        generation_service = MagicMock()
        generation_service.configured_providers = frozenset()

        with patch("src.api.routes.providers.GenerationModelRepository") as repo_cls:
            repo = MagicMock()
            repo.list_enabled_for_product = AsyncMock(return_value=[grok_record, aisha_record])
            repo_cls.return_value = repo

            product_config = MagicMock()
            product_config.is_model_allowed.return_value = True

            result = await ProvidersController.list_providers.fn(  # type: ignore[attr-defined]
                MagicMock(),
                session=session,
                generation_service=generation_service,
                current_user_id=None,
                product_config=product_config,
                product_id="vex",
            )

        grok_provider = next(p for p in result.providers if p.provider == "grok")
        grok_model = next(m for m in grok_provider.models if m.model_key == "grok-imagine-image")
        assert grok_model.image is not None
        assert grok_model.image.edit_aspect_ratios == []

        aisha_provider = next(p for p in result.providers if p.provider == "aisha")
        aisha_model = next(m for m in aisha_provider.models if m.model_key == "aisha-image")
        assert aisha_model.image is not None
        assert set(aisha_model.image.edit_aspect_ratios) == {ar.value for ar in AspectRatio}


# ---------------------------------------------------------------------------
# AuthController (auth.py)
# ---------------------------------------------------------------------------


class TestAuthRouteHandlers:
    async def test_register_success(self) -> None:
        from src.api.routes.auth import AuthController

        tokens = MagicMock()
        tokens.access_token = "acc"
        tokens.refresh_token = "ref"
        tokens.expires_in = 3600
        tokens.expires_at = None
        auth_service = AsyncMock()
        auth_service.register = AsyncMock(return_value=(MagicMock(), tokens))
        data = MagicMock()
        data.email = "user@example.com"
        data.password = "pw"
        data.display_name = "Test"
        jwt_service = MagicMock()
        jwt_service.create_content_token.return_value = ("content_tok", None)
        product_config = MagicMock()
        product_config.cookie_domain = "example.com"
        settings = MagicMock()
        settings.content_cookie_ttl_hours = 1
        settings.content_cookie_secure = True

        response = await AuthController.register.fn(  # type: ignore[attr-defined]
            MagicMock(),
            data=data,
            auth_service=auth_service,
            jwt_service=jwt_service,
            product_id="vex",
            product_config=product_config,
            settings=settings,
        )
        assert response.status_code == 201

    async def test_register_email_exists(self) -> None:
        from src.api.routes.auth import AuthController
        from src.api.services.auth import EmailAlreadyExistsError

        auth_service = AsyncMock()
        auth_service.register = AsyncMock(side_effect=EmailAlreadyExistsError("exists"))
        data = MagicMock()
        data.email = "x@x.com"
        data.password = "pw"
        data.display_name = "T"
        jwt_service = MagicMock()
        product_config = MagicMock()
        settings = MagicMock()

        response = await AuthController.register.fn(  # type: ignore[attr-defined]
            MagicMock(),
            data=data,
            auth_service=auth_service,
            jwt_service=jwt_service,
            product_id="vex",
            product_config=product_config,
            settings=settings,
        )
        assert response.status_code == 400

    async def test_verify_email_success(self) -> None:
        from src.api.routes.auth import AuthController

        session = AsyncMock()
        svc = AsyncMock()
        svc.verify_email = AsyncMock(return_value=MagicMock())
        data = MagicMock()
        data.token = "tok"

        response = await AuthController.verify_email.fn(  # type: ignore[attr-defined]
            MagicMock(),
            data=data,
            session=session,
            email_verification_service=svc,
        )
        assert response.status_code == 200

    async def test_verify_email_invalid_token(self) -> None:
        from src.api.routes.auth import AuthController
        from src.api.services.email_verification import InvalidTokenError

        session = AsyncMock()
        svc = AsyncMock()
        svc.verify_email = AsyncMock(side_effect=InvalidTokenError("expired"))
        data = MagicMock()
        data.token = "bad"

        response = await AuthController.verify_email.fn(  # type: ignore[attr-defined]
            MagicMock(),
            data=data,
            session=session,
            email_verification_service=svc,
        )
        assert response.status_code == 400

    async def test_resend_verification_user_not_found(self) -> None:
        from src.api.routes.auth import AuthController

        session = AsyncMock()
        svc = AsyncMock()

        with patch("src.api.routes.auth.UserRepository") as repo_cls:
            repo_cls.return_value.get_active_user = AsyncMock(return_value=None)
            response = await AuthController.resend_verification.fn(  # type: ignore[attr-defined]
                MagicMock(),
                current_user_id=uuid4(),
                session=session,
                email_verification_service=svc,
            )

        assert response.status_code == 400

    async def test_resend_verification_already_verified(self) -> None:
        from src.api.routes.auth import AuthController

        session = AsyncMock()
        svc = AsyncMock()
        user = MagicMock()
        user.email_verified_at = datetime.now(UTC)

        with patch("src.api.routes.auth.UserRepository") as repo_cls:
            repo_cls.return_value.get_active_user = AsyncMock(return_value=user)
            response = await AuthController.resend_verification.fn(  # type: ignore[attr-defined]
                MagicMock(),
                current_user_id=uuid4(),
                session=session,
                email_verification_service=svc,
            )

        assert response.status_code == 200

    async def test_resend_verification_sends_email(self) -> None:
        from src.api.routes.auth import AuthController

        session = AsyncMock()
        svc = AsyncMock()
        svc.send_verification_email = AsyncMock()
        user = MagicMock()
        user.email_verified_at = None

        with patch("src.api.routes.auth.UserRepository") as repo_cls:
            repo_cls.return_value.get_active_user = AsyncMock(return_value=user)
            response = await AuthController.resend_verification.fn(  # type: ignore[attr-defined]
                MagicMock(),
                current_user_id=uuid4(),
                session=session,
                email_verification_service=svc,
            )

        assert response.status_code == 200

    async def test_resend_verification_user_not_found_error(self) -> None:
        from src.api.routes.auth import AuthController
        from src.api.services.email_verification import UserNotFoundError

        session = AsyncMock()
        svc = AsyncMock()
        svc.send_verification_email = AsyncMock(side_effect=UserNotFoundError("gone"))
        user = MagicMock()
        user.email_verified_at = None

        with patch("src.api.routes.auth.UserRepository") as repo_cls:
            repo_cls.return_value.get_active_user = AsyncMock(return_value=user)
            response = await AuthController.resend_verification.fn(  # type: ignore[attr-defined]
                MagicMock(),
                current_user_id=uuid4(),
                session=session,
                email_verification_service=svc,
            )

        assert response.status_code == 400

    async def test_login_success(self) -> None:
        from src.api.routes.auth import AuthController

        tokens = MagicMock()
        tokens.access_token = "acc"
        tokens.refresh_token = "ref"
        tokens.expires_in = 3600
        tokens.expires_at = None
        auth_service = AsyncMock()
        auth_service.login = AsyncMock(return_value=(MagicMock(), tokens))
        data = MagicMock()
        data.email = "u@e.com"
        data.password = "pw"
        request = MagicMock()
        request.headers.get.return_value = ""
        request.client = MagicMock()
        request.client.host = "127.0.0.1"
        jwt_service = MagicMock()
        jwt_service.create_content_token.return_value = ("content_tok", None)
        product_config = MagicMock()
        product_config.cookie_domain = "example.com"
        settings = MagicMock()
        settings.content_cookie_ttl_hours = 1
        settings.content_cookie_secure = True

        response = await AuthController.login.fn(  # type: ignore[attr-defined]
            MagicMock(),
            request=request,
            data=data,
            auth_service=auth_service,
            jwt_service=jwt_service,
            product_id="vex",
            product_config=product_config,
            settings=settings,
        )
        assert response.status_code == 200

    async def test_login_invalid_credentials(self) -> None:
        from src.api.routes.auth import AuthController
        from src.api.services.auth import InvalidCredentialsError

        auth_service = AsyncMock()
        auth_service.login = AsyncMock(side_effect=InvalidCredentialsError("bad"))
        data = MagicMock()
        request = MagicMock()
        request.headers.get.return_value = ""
        request.client = None
        jwt_service = MagicMock()
        product_config = MagicMock()
        settings = MagicMock()

        response = await AuthController.login.fn(  # type: ignore[attr-defined]
            MagicMock(),
            request=request,
            data=data,
            auth_service=auth_service,
            jwt_service=jwt_service,
            product_id="vex",
            product_config=product_config,
            settings=settings,
        )
        assert response.status_code == 401

    async def test_login_inactive_user(self) -> None:
        from src.api.routes.auth import AuthController
        from src.api.services.auth import UserInactiveError

        auth_service = AsyncMock()
        auth_service.login = AsyncMock(side_effect=UserInactiveError("inactive"))
        data = MagicMock()
        request = MagicMock()
        request.headers.get.return_value = ""
        request.client = None
        jwt_service = MagicMock()
        product_config = MagicMock()
        settings = MagicMock()

        response = await AuthController.login.fn(  # type: ignore[attr-defined]
            MagicMock(),
            request=request,
            data=data,
            auth_service=auth_service,
            jwt_service=jwt_service,
            product_id="vex",
            product_config=product_config,
            settings=settings,
        )
        assert response.status_code == 401

    async def test_refresh_success(self) -> None:
        from src.api.routes.auth import AuthController

        tokens = MagicMock()
        tokens.access_token = "acc"
        tokens.refresh_token = "ref"
        tokens.expires_in = 3600
        tokens.expires_at = None
        auth_service = AsyncMock()
        auth_service.refresh_tokens = AsyncMock(return_value=(tokens, "user-uuid-123"))
        data = MagicMock()
        data.refresh_token = "old_ref"
        request = MagicMock()
        request.headers.get.return_value = ""
        request.client = None
        jwt_service = MagicMock()
        jwt_service.decode_access_token.return_value = None  # skip cookie on MagicMock token
        jwt_service.create_content_token.return_value = ("content_token", None)
        product_config = MagicMock()
        product_config.cookie_domain = "example.com"
        settings = MagicMock()
        settings.content_cookie_ttl_hours = 1
        settings.content_cookie_secure = True

        response = await AuthController.refresh_tokens.fn(  # type: ignore[attr-defined]
            MagicMock(),
            request=request,
            data=data,
            auth_service=auth_service,
            jwt_service=jwt_service,
            product_id="vex",
            product_config=product_config,
            settings=settings,
        )
        assert response.status_code == 200

    async def test_refresh_invalid_token(self) -> None:
        from src.api.routes.auth import AuthController
        from src.api.services.auth import InvalidRefreshTokenError

        auth_service = AsyncMock()
        auth_service.refresh_tokens = AsyncMock(side_effect=InvalidRefreshTokenError("bad"))
        data = MagicMock()
        request = MagicMock()
        request.headers.get.return_value = ""
        request.client = None
        jwt_service = MagicMock()
        product_config = MagicMock()
        settings = MagicMock()

        response = await AuthController.refresh_tokens.fn(  # type: ignore[attr-defined]
            MagicMock(),
            request=request,
            data=data,
            auth_service=auth_service,
            jwt_service=jwt_service,
            product_id="vex",
            product_config=product_config,
            settings=settings,
        )
        assert response.status_code == 401

    async def test_refresh_token_reuse(self) -> None:
        from src.api.routes.auth import AuthController
        from src.api.services.auth import TokenReuseDetectedError

        auth_service = AsyncMock()
        auth_service.refresh_tokens = AsyncMock(side_effect=TokenReuseDetectedError("reuse"))
        data = MagicMock()
        request = MagicMock()
        request.headers.get.return_value = ""
        request.client = None
        jwt_service = MagicMock()
        product_config = MagicMock()
        settings = MagicMock()

        response = await AuthController.refresh_tokens.fn(  # type: ignore[attr-defined]
            MagicMock(),
            request=request,
            data=data,
            auth_service=auth_service,
            jwt_service=jwt_service,
            product_id="vex",
            product_config=product_config,
            settings=settings,
        )
        assert response.status_code == 401

    async def test_refresh_user_inactive(self) -> None:
        from src.api.routes.auth import AuthController
        from src.api.services.auth import UserInactiveError

        auth_service = AsyncMock()
        auth_service.refresh_tokens = AsyncMock(side_effect=UserInactiveError("inactive"))
        data = MagicMock()
        request = MagicMock()
        request.headers.get.return_value = ""
        request.client = None
        jwt_service = MagicMock()
        product_config = MagicMock()
        settings = MagicMock()

        response = await AuthController.refresh_tokens.fn(  # type: ignore[attr-defined]
            MagicMock(),
            request=request,
            data=data,
            auth_service=auth_service,
            jwt_service=jwt_service,
            product_id="vex",
            product_config=product_config,
            settings=settings,
        )
        assert response.status_code == 401

    async def test_logout_success(self) -> None:
        from src.api.routes.auth import AuthController

        auth_service = AsyncMock()
        auth_service.logout = AsyncMock()
        data = MagicMock()
        data.refresh_token = "ref"
        product_config = MagicMock()
        product_config.cookie_domain = "example.com"
        settings = MagicMock()
        settings.content_cookie_secure = True
        request = MagicMock()
        request.headers.get.return_value = None
        jwt_service = MagicMock()
        token_revocation_service = AsyncMock()

        response = await AuthController.logout.fn(  # type: ignore[attr-defined]
            MagicMock(),
            request=request,
            data=data,
            auth_service=auth_service,
            jwt_service=jwt_service,
            token_revocation_service=token_revocation_service,
            product_config=product_config,
            settings=settings,
        )
        assert response.status_code == 200
        assert response.headers["Clear-Site-Data"] == '"cache", "storage"'
        # No Authorization header presented — nothing to denylist.
        jwt_service.decode_access_token.assert_not_called()
        token_revocation_service.revoke_token.assert_not_called()

    async def test_logout_with_bearer_denylists_presenting_token_jti(self) -> None:
        """issue #142 — POST /v1/auth/logout denylists the presenting access
        token's own jti alongside the existing refresh-token revocation."""
        from src.api.routes.auth import AuthController
        from src.api.security.jwt import JWTConfig, JWTService

        real_jwt_service = JWTService(
            JWTConfig(secret_key="test_secret_key_for_testing_only_256bits_long")
        )
        token, expires_at = real_jwt_service.create_access_token(uuid4(), product_id="vex")
        payload = real_jwt_service.decode_access_token(token)
        assert payload is not None

        auth_service = AsyncMock()
        auth_service.logout = AsyncMock()
        data = MagicMock()
        data.refresh_token = "ref"
        product_config = MagicMock()
        product_config.cookie_domain = "example.com"
        settings = MagicMock()
        settings.content_cookie_secure = True
        request = MagicMock()
        request.headers.get.return_value = f"Bearer {token}"
        token_revocation_service = AsyncMock()

        response = await AuthController.logout.fn(  # type: ignore[attr-defined]
            MagicMock(),
            request=request,
            data=data,
            auth_service=auth_service,
            jwt_service=real_jwt_service,
            token_revocation_service=token_revocation_service,
            product_config=product_config,
            settings=settings,
        )

        assert response.status_code == 200
        assert response.headers["Clear-Site-Data"] == '"cache", "storage"'
        auth_service.logout.assert_awaited_once_with("ref")
        token_revocation_service.revoke_token.assert_awaited_once_with(
            payload.jti, int(expires_at.timestamp())
        )

    async def test_logout_with_expired_bearer_skips_jti_denylist(self) -> None:
        """An already-expired presenting token decodes to None — nothing to
        denylist, only the refresh token is revoked (pre-#142 behavior)."""
        from src.api.routes.auth import AuthController
        from src.api.security.jwt import JWTConfig, JWTService

        expired_jwt_service = JWTService(
            JWTConfig(
                secret_key="test_secret_key_for_testing_only_256bits_long",
                access_token_expire_minutes=-1,
            )
        )
        token, _ = expired_jwt_service.create_access_token(uuid4(), product_id="vex")

        auth_service = AsyncMock()
        auth_service.logout = AsyncMock()
        data = MagicMock()
        data.refresh_token = "ref"
        product_config = MagicMock()
        product_config.cookie_domain = "example.com"
        settings = MagicMock()
        settings.content_cookie_secure = True
        request = MagicMock()
        request.headers.get.return_value = f"Bearer {token}"
        token_revocation_service = AsyncMock()

        response = await AuthController.logout.fn(  # type: ignore[attr-defined]
            MagicMock(),
            request=request,
            data=data,
            auth_service=auth_service,
            jwt_service=expired_jwt_service,
            token_revocation_service=token_revocation_service,
            product_config=product_config,
            settings=settings,
        )

        assert response.status_code == 200
        assert response.headers["Clear-Site-Data"] == '"cache", "storage"'
        token_revocation_service.revoke_token.assert_not_called()

    async def test_logout_with_unknown_refresh_token_still_returns_200_with_header(self) -> None:
        """D4 — AuthService.logout returning False (unknown/already-revoked
        refresh token) must not change this endpoint's response: still 200
        with Clear-Site-Data, since the route deliberately ignores the
        return value (see the handler docstring)."""
        from src.api.routes.auth import AuthController

        auth_service = AsyncMock()
        auth_service.logout = AsyncMock(return_value=False)
        data = MagicMock()
        data.refresh_token = "unknown-or-already-revoked"
        product_config = MagicMock()
        product_config.cookie_domain = "example.com"
        settings = MagicMock()
        settings.content_cookie_secure = True
        request = MagicMock()
        request.headers.get.return_value = None
        jwt_service = MagicMock()
        token_revocation_service = AsyncMock()

        response = await AuthController.logout.fn(  # type: ignore[attr-defined]
            MagicMock(),
            request=request,
            data=data,
            auth_service=auth_service,
            jwt_service=jwt_service,
            token_revocation_service=token_revocation_service,
            product_config=product_config,
            settings=settings,
        )

        assert response.status_code == 200
        assert response.headers["Clear-Site-Data"] == '"cache", "storage"'

    async def test_forgot_password_always_200(self) -> None:
        from src.api.routes.auth import AuthController

        session = AsyncMock()
        svc = AsyncMock()
        svc.send_password_reset_email = AsyncMock()
        data = MagicMock()
        data.email = "u@e.com"
        request = MagicMock()
        request.headers.get.return_value = "1.2.3.4"
        request.client = None

        response = await AuthController.forgot_password.fn(  # type: ignore[attr-defined]
            MagicMock(),
            request=request,
            data=data,
            session=session,
            email_verification_service=svc,
        )
        assert response.status_code == 200

    async def test_forgot_password_exception_still_200(self) -> None:
        from src.api.routes.auth import AuthController

        session = AsyncMock()
        svc = AsyncMock()
        svc.send_password_reset_email = AsyncMock(side_effect=Exception("fail"))
        data = MagicMock()
        data.email = "u@e.com"
        request = MagicMock()
        request.headers.get.return_value = ""
        request.client = MagicMock()
        request.client.host = "127.0.0.1"

        response = await AuthController.forgot_password.fn(  # type: ignore[attr-defined]
            MagicMock(),
            request=request,
            data=data,
            session=session,
            email_verification_service=svc,
        )
        assert response.status_code == 200

    async def test_reset_password_success(self) -> None:
        from src.api.routes.auth import AuthController

        session = AsyncMock()
        svc = AsyncMock()
        svc.reset_password = AsyncMock(return_value=MagicMock())
        data = MagicMock()
        data.token = "tok"
        data.new_password = "newpw"

        response = await AuthController.reset_password.fn(  # type: ignore[attr-defined]
            MagicMock(),
            data=data,
            session=session,
            email_verification_service=svc,
        )
        assert response.status_code == 200
        assert response.headers["Clear-Site-Data"] == '"cache", "storage"'

    async def test_reset_password_invalid_token(self) -> None:
        from src.api.routes.auth import AuthController
        from src.api.services.email_verification import InvalidTokenError

        session = AsyncMock()
        svc = AsyncMock()
        svc.reset_password = AsyncMock(side_effect=InvalidTokenError("expired"))
        data = MagicMock()
        data.token = "bad"
        data.new_password = "pw"

        response = await AuthController.reset_password.fn(  # type: ignore[attr-defined]
            MagicMock(),
            data=data,
            session=session,
            email_verification_service=svc,
        )
        assert response.status_code == 400
        # No session was ended — the error response must not purge the cache.
        assert "Clear-Site-Data" not in response.headers

    async def test_product_info(self) -> None:
        from src.api.routes.auth import AuthController

        product_config = MagicMock()
        product_config.slug = "vex"
        product_config.display_name = "Vex"
        product_config.age_gate = MagicMock()
        product_config.age_gate.value = "none"
        product_config.allowed_auth_methods = [MagicMock(value="email")]
        product_config.content_policy = MagicMock()
        product_config.content_policy.rating = MagicMock(value="permissive")
        product_config.payment_providers = [MagicMock(value="stripe")]

        result = await AuthController.product_info.fn(  # type: ignore[attr-defined]
            MagicMock(),
            product_config=product_config,
        )
        assert result.product == "vex"
        assert result.display_name == "Vex"


# ---------------------------------------------------------------------------
# UnifiedGenerationController (unified_generation.py)
# ---------------------------------------------------------------------------


class TestUnifiedGenerationRouteHandlers:
    async def _make_request(self):
        from src.api.schemas.unified_generation import UnifiedGenerationRequest
        from src.core.enums import GenerationType, ModelType

        return UnifiedGenerationRequest(
            prompt="a cat",
            generation_type=GenerationType.T2I,
            model=ModelType.AISHA_IMAGE,
        )

    async def test_generate_idempotency_replay(self) -> None:
        from src.api.routes.unified_generation import UnifiedGenerationController
        from src.api.services.idempotency import IdempotencyReplayResult

        replay = IdempotencyReplayResult(status_code=201, body={"job_id": "abc"})
        idempotency_service = AsyncMock()
        idempotency_service.check = AsyncMock(return_value=replay)

        response = await UnifiedGenerationController.generate.fn(  # type: ignore[attr-defined]
            MagicMock(),
            current_user_id=uuid4(),
            data=await self._make_request(),
            session=AsyncMock(),
            generation_service=AsyncMock(),
            product_config=MagicMock(),
            product_id="vex",
            idempotency_service=idempotency_service,
            idempotency_key_header="key-123",
        )
        assert response.status_code == 201

    async def test_generate_success(self) -> None:
        from src.api.routes.unified_generation import UnifiedGenerationController
        from src.api.schemas.jobs import JobCreatedResponse
        from src.core.enums import GenerationType, JobStatus

        job_result = JobCreatedResponse(
            job_id=uuid4(),
            status=JobStatus.QUEUED,
            name="test-job",
            model="aisha-image",
            generation_type=GenerationType.T2I,
            created_at=datetime.now(UTC),
        )
        session = AsyncMock()
        generation_service = AsyncMock()
        generation_service.generate = AsyncMock(return_value=job_result)
        idempotency_service = AsyncMock()
        idempotency_service.check = AsyncMock(return_value=uuid4())
        idempotency_service.complete = AsyncMock()

        response = await UnifiedGenerationController.generate.fn(  # type: ignore[attr-defined]
            MagicMock(),
            current_user_id=uuid4(),
            data=await self._make_request(),
            session=session,
            generation_service=generation_service,
            product_config=MagicMock(),
            product_id="vex",
            idempotency_service=idempotency_service,
            idempotency_key_header="key-123",
        )
        assert response.status_code == 201

    async def test_generate_no_active_session(self) -> None:
        from src.api.routes.unified_generation import UnifiedGenerationController
        from src.api.services.gpu_session.exceptions import NoActiveSessionError

        generation_service = AsyncMock()
        generation_service.generate = AsyncMock(side_effect=NoActiveSessionError("no gpu"))
        idempotency_service = AsyncMock()
        idempotency_service.check = AsyncMock(return_value=uuid4())

        response = await UnifiedGenerationController.generate.fn(  # type: ignore[attr-defined]
            MagicMock(),
            current_user_id=uuid4(),
            data=await self._make_request(),
            session=AsyncMock(),
            generation_service=generation_service,
            product_config=MagicMock(),
            product_id="vex",
            idempotency_service=idempotency_service,
            idempotency_key_header="key-123",
        )
        assert response.status_code == 409

    async def test_generate_rate_limited(self) -> None:
        from src.api.routes.unified_generation import UnifiedGenerationController
        from src.api.services.generation.rate_limiter import RateLimitExceededError
        from src.core.enums import ModelType

        err = RateLimitExceededError(ModelType.AISHA_IMAGE, retry_after=60)
        generation_service = AsyncMock()
        generation_service.generate = AsyncMock(side_effect=err)
        idempotency_service = AsyncMock()
        idempotency_service.check = AsyncMock(return_value=uuid4())

        response = await UnifiedGenerationController.generate.fn(  # type: ignore[attr-defined]
            MagicMock(),
            current_user_id=uuid4(),
            data=await self._make_request(),
            session=AsyncMock(),
            generation_service=generation_service,
            product_config=MagicMock(),
            product_id="vex",
            idempotency_service=idempotency_service,
            idempotency_key_header="key-123",
        )
        assert response.status_code == 429

    async def test_generate_provider_unavailable(self) -> None:
        from src.api.routes.unified_generation import UnifiedGenerationController
        from src.api.services.generation.service import ProviderUnavailableError

        generation_service = AsyncMock()
        generation_service.generate = AsyncMock(side_effect=ProviderUnavailableError("unavailable"))
        idempotency_service = AsyncMock()
        idempotency_service.check = AsyncMock(return_value=uuid4())

        response = await UnifiedGenerationController.generate.fn(  # type: ignore[attr-defined]
            MagicMock(),
            current_user_id=uuid4(),
            data=await self._make_request(),
            session=AsyncMock(),
            generation_service=generation_service,
            product_config=MagicMock(),
            product_id="vex",
            idempotency_service=idempotency_service,
            idempotency_key_header="key-123",
        )
        assert response.status_code == 503

    async def test_generate_model_not_allowed(self) -> None:
        from src.api.routes.unified_generation import UnifiedGenerationController
        from src.api.services.generation.service import ModelNotAllowedError

        generation_service = AsyncMock()
        generation_service.generate = AsyncMock(side_effect=ModelNotAllowedError("not allowed"))
        idempotency_service = AsyncMock()
        idempotency_service.check = AsyncMock(return_value=uuid4())

        response = await UnifiedGenerationController.generate.fn(  # type: ignore[attr-defined]
            MagicMock(),
            current_user_id=uuid4(),
            data=await self._make_request(),
            session=AsyncMock(),
            generation_service=generation_service,
            product_config=MagicMock(),
            product_id="vex",
            idempotency_service=idempotency_service,
            idempotency_key_header="key-123",
        )
        assert response.status_code == 403

    async def test_generate_model_disabled(self) -> None:
        from src.api.routes.unified_generation import UnifiedGenerationController
        from src.api.services.generation.service import ModelDisabledError

        generation_service = AsyncMock()
        generation_service.generate = AsyncMock(side_effect=ModelDisabledError("disabled"))
        idempotency_service = AsyncMock()
        idempotency_service.check = AsyncMock(return_value=uuid4())

        response = await UnifiedGenerationController.generate.fn(  # type: ignore[attr-defined]
            MagicMock(),
            current_user_id=uuid4(),
            data=await self._make_request(),
            session=AsyncMock(),
            generation_service=generation_service,
            product_config=MagicMock(),
            product_id="vex",
            idempotency_service=idempotency_service,
            idempotency_key_header="key-123",
        )
        assert response.status_code == 400

    async def test_generate_value_error(self) -> None:
        from src.api.routes.unified_generation import UnifiedGenerationController

        generation_service = AsyncMock()
        generation_service.generate = AsyncMock(side_effect=ValueError("bad value"))
        idempotency_service = AsyncMock()
        idempotency_service.check = AsyncMock(return_value=uuid4())

        response = await UnifiedGenerationController.generate.fn(  # type: ignore[attr-defined]
            MagicMock(),
            current_user_id=uuid4(),
            data=await self._make_request(),
            session=AsyncMock(),
            generation_service=generation_service,
            product_config=MagicMock(),
            product_id="vex",
            idempotency_service=idempotency_service,
            idempotency_key_header="key-123",
        )
        assert response.status_code == 400

    async def test_generate_generation_error(self) -> None:
        from src.api.routes.unified_generation import UnifiedGenerationController
        from src.api.services.generation.service import GenerationError

        generation_service = AsyncMock()
        generation_service.generate = AsyncMock(side_effect=GenerationError("failed"))
        idempotency_service = AsyncMock()
        idempotency_service.check = AsyncMock(return_value=uuid4())

        response = await UnifiedGenerationController.generate.fn(  # type: ignore[attr-defined]
            MagicMock(),
            current_user_id=uuid4(),
            data=await self._make_request(),
            session=AsyncMock(),
            generation_service=generation_service,
            product_config=MagicMock(),
            product_id="vex",
            idempotency_service=idempotency_service,
            idempotency_key_header="key-123",
        )
        assert response.status_code == 400

    async def test_generate_bare_exception_reraises(self) -> None:
        from src.api.routes.unified_generation import UnifiedGenerationController

        generation_service = AsyncMock()
        generation_service.generate = AsyncMock(side_effect=RuntimeError("boom"))
        idempotency_service = AsyncMock()
        idempotency_service.check = AsyncMock(return_value=uuid4())

        with pytest.raises(RuntimeError, match="boom"):
            await UnifiedGenerationController.generate.fn(  # type: ignore[attr-defined]
                MagicMock(),
                current_user_id=uuid4(),
                data=await self._make_request(),
                session=AsyncMock(),
                generation_service=generation_service,
                product_config=MagicMock(),
                product_id="vex",
                idempotency_service=idempotency_service,
                idempotency_key_header="key-123",
            )


# ---------------------------------------------------------------------------
# OrganizationController (organization.py)
# ---------------------------------------------------------------------------


class TestOrganizationRouteHandlers:
    async def test_create_organization(self) -> None:
        from src.api.routes.organization import OrganizationController

        org = MagicMock()
        org.id = uuid4()
        account = MagicMock()
        account.id = uuid4()
        owner_member = MagicMock()
        owner_member.user_id = uuid4()
        user_id = owner_member.user_id

        org_service = AsyncMock()
        org_service.create_organization = AsyncMock(return_value=(org, account))
        org_service.list_members = AsyncMock(return_value=[owner_member])

        billing_service = AsyncMock()
        billing_service.get_balance = AsyncMock(return_value=100)

        data = MagicMock()
        data.name = "My Org"

        response = await OrganizationController.create_organization.fn(  # type: ignore[attr-defined]
            MagicMock(),
            current_user_id=user_id,
            data=data,
            session=AsyncMock(),
            organization_service=org_service,
            billing_service=billing_service,
            product_id="vex",
        )
        assert response.status_code == 201

    async def test_get_my_organization_not_found(self) -> None:
        from src.api.routes.organization import OrganizationController

        org_service = AsyncMock()
        org_service.get_user_organization = AsyncMock(return_value=None)
        billing_service = AsyncMock()

        response = await OrganizationController.get_my_organization.fn(  # type: ignore[attr-defined]
            MagicMock(),
            current_user_id=uuid4(),
            session=AsyncMock(),
            organization_service=org_service,
            billing_service=billing_service,
        )
        assert response.status_code == 404

    async def test_get_my_organization_found(self) -> None:
        from src.api.routes.organization import OrganizationController

        org = MagicMock()
        membership = MagicMock()
        membership.role = "owner"
        account = MagicMock()
        account.id = uuid4()

        org_service = AsyncMock()
        org_service.get_user_organization = AsyncMock(return_value=org)
        org_service.get_membership = AsyncMock(return_value=membership)

        billing_service = AsyncMock()
        billing_service.resolve_account_for_user = AsyncMock(return_value=account)
        billing_service.get_balance = AsyncMock(return_value=50)

        response = await OrganizationController.get_my_organization.fn(  # type: ignore[attr-defined]
            MagicMock(),
            current_user_id=uuid4(),
            session=AsyncMock(),
            organization_service=org_service,
            billing_service=billing_service,
        )
        assert response.status_code == 200

    async def test_get_organization_no_membership(self) -> None:
        from src.api.routes.organization import OrganizationController

        org_service = AsyncMock()
        org_service.get_membership = AsyncMock(return_value=None)

        response = await OrganizationController.get_organization.fn(  # type: ignore[attr-defined]
            MagicMock(),
            current_user_id=uuid4(),
            org_id=uuid4(),
            session=AsyncMock(),
            organization_service=org_service,
        )
        assert response.status_code == 404

    async def test_get_organization_found(self) -> None:
        from src.api.routes.organization import OrganizationController

        org = MagicMock()
        membership = MagicMock()

        org_service = AsyncMock()
        org_service.get_membership = AsyncMock(return_value=membership)
        org_service.get_organization = AsyncMock(return_value=org)

        response = await OrganizationController.get_organization.fn(  # type: ignore[attr-defined]
            MagicMock(),
            current_user_id=uuid4(),
            org_id=uuid4(),
            session=AsyncMock(),
            organization_service=org_service,
        )
        assert response.status_code == 200

    async def test_list_members_no_membership(self) -> None:
        from src.api.routes.organization import OrganizationController

        org_service = AsyncMock()
        org_service.get_membership = AsyncMock(return_value=None)

        response = await OrganizationController.list_members.fn(  # type: ignore[attr-defined]
            MagicMock(),
            current_user_id=uuid4(),
            org_id=uuid4(),
            session=AsyncMock(),
            organization_service=org_service,
        )
        assert response.status_code == 404

    async def test_list_members_success(self) -> None:
        from src.api.routes.organization import OrganizationController

        member = MagicMock()
        org_service = AsyncMock()
        org_service.get_membership = AsyncMock(return_value=MagicMock())
        org_service.list_members = AsyncMock(return_value=[member])

        response = await OrganizationController.list_members.fn(  # type: ignore[attr-defined]
            MagicMock(),
            current_user_id=uuid4(),
            org_id=uuid4(),
            session=AsyncMock(),
            organization_service=org_service,
        )
        assert response.status_code == 200

    async def test_add_member(self) -> None:
        from src.api.routes.organization import OrganizationController

        member = MagicMock()
        org_service = AsyncMock()
        org_service.add_member = AsyncMock(return_value=member)
        data = MagicMock()
        data.user_id = uuid4()
        data.role = "member"

        response = await OrganizationController.add_member.fn(  # type: ignore[attr-defined]
            MagicMock(),
            current_user_id=uuid4(),
            org_id=uuid4(),
            data=data,
            session=AsyncMock(),
            organization_service=org_service,
            product_id="vex",
        )
        assert response.status_code == 201

    async def test_delete_organization_not_found(self) -> None:
        from src.api.routes.organization import OrganizationController

        org_service = AsyncMock()
        org_service.get_organization = AsyncMock(return_value=None)

        response = await OrganizationController.delete_organization.fn(  # type: ignore[attr-defined]
            MagicMock(),
            current_user_id=uuid4(),
            org_id=uuid4(),
            session=AsyncMock(),
            organization_service=org_service,
        )
        assert response.status_code == 404

    async def test_delete_organization_success(self) -> None:
        from src.api.routes.organization import OrganizationController

        org_service = AsyncMock()
        org_service.get_organization = AsyncMock(return_value=MagicMock())
        org_service.delete_organization = AsyncMock()

        result = await OrganizationController.delete_organization.fn(  # type: ignore[attr-defined]
            MagicMock(),
            current_user_id=uuid4(),
            org_id=uuid4(),
            session=AsyncMock(),
            organization_service=org_service,
        )
        assert result == {"message": "Organization deleted"}

    async def test_remove_member(self) -> None:
        from src.api.routes.organization import OrganizationController

        org_service = AsyncMock()
        org_service.remove_member = AsyncMock()

        result = await OrganizationController.remove_member.fn(  # type: ignore[attr-defined]
            MagicMock(),
            current_user_id=uuid4(),
            org_id=uuid4(),
            user_id=uuid4(),
            session=AsyncMock(),
            organization_service=org_service,
        )
        assert result == {"message": "Member removed"}

    async def test_change_member_role(self) -> None:
        from src.api.routes.organization import OrganizationController

        member = MagicMock()
        org_service = AsyncMock()
        org_service.change_role = AsyncMock(return_value=member)
        data = MagicMock()
        data.role = "admin"

        response = await OrganizationController.change_member_role.fn(  # type: ignore[attr-defined]
            MagicMock(),
            current_user_id=uuid4(),
            org_id=uuid4(),
            user_id=uuid4(),
            data=data,
            session=AsyncMock(),
            organization_service=org_service,
        )
        assert response.status_code == 200


# ---------------------------------------------------------------------------
# GpuSessionController (gpu_session.py)
# ---------------------------------------------------------------------------


class TestGpuSessionRouteHandlers:
    async def test_start_session_success(self) -> None:
        from src.api.routes.gpu_session import GpuSessionController

        gpu_session = MagicMock()
        gpu_session_service = AsyncMock()
        gpu_session_service.start_session = AsyncMock(return_value=gpu_session)
        billing_service = AsyncMock()
        billing_service.resolve_account_for_user = AsyncMock(return_value=MagicMock())
        data = MagicMock()
        data.model = MagicMock()
        data.bundle_override = None

        with patch(
            "src.api.routes.gpu_session.GpuSessionResponse.from_model",
            return_value=MagicMock(),
        ):
            response = await GpuSessionController.start.fn(  # type: ignore[attr-defined]
                MagicMock(),
                current_user_id=uuid4(),
                data=data,
                session=AsyncMock(),
                gpu_session_service=gpu_session_service,
                billing_service=billing_service,
                product_id="vex",
            )
        assert response.status_code == 201

    async def test_start_session_already_exists(self) -> None:
        from src.api.routes.gpu_session import GpuSessionController
        from src.api.services.gpu_session.exceptions import SessionAlreadyExistsError

        gpu_session_service = AsyncMock()
        gpu_session_service.start_session = AsyncMock(
            side_effect=SessionAlreadyExistsError("exists")
        )
        billing_service = AsyncMock()
        billing_service.resolve_account_for_user = AsyncMock(return_value=MagicMock())
        data = MagicMock()
        data.bundle_override = None

        response = await GpuSessionController.start.fn(  # type: ignore[attr-defined]
            MagicMock(),
            current_user_id=uuid4(),
            data=data,
            session=AsyncMock(),
            gpu_session_service=gpu_session_service,
            billing_service=billing_service,
            product_id="vex",
        )
        assert response.status_code == 409

    async def test_start_session_insufficient_balance(self) -> None:
        from src.api.routes.gpu_session import GpuSessionController
        from src.api.services.billing_errors import InsufficientBalanceError

        gpu_session_service = AsyncMock()
        gpu_session_service.start_session = AsyncMock(side_effect=InsufficientBalanceError(0, 100))
        billing_service = AsyncMock()
        billing_service.resolve_account_for_user = AsyncMock(return_value=MagicMock())
        data = MagicMock()
        data.bundle_override = None

        response = await GpuSessionController.start.fn(  # type: ignore[attr-defined]
            MagicMock(),
            current_user_id=uuid4(),
            data=data,
            session=AsyncMock(),
            gpu_session_service=gpu_session_service,
            billing_service=billing_service,
            product_id="vex",
        )
        assert response.status_code == 402

    async def test_start_session_no_capacity(self) -> None:
        from src.api.routes.gpu_session import GpuSessionController
        from src.api.services.vastai.exceptions import NoCapacityError

        gpu_session_service = AsyncMock()
        gpu_session_service.start_session = AsyncMock(side_effect=NoCapacityError("no gpu"))
        billing_service = AsyncMock()
        billing_service.resolve_account_for_user = AsyncMock(return_value=MagicMock())
        data = MagicMock()
        data.bundle_override = None

        response = await GpuSessionController.start.fn(  # type: ignore[attr-defined]
            MagicMock(),
            current_user_id=uuid4(),
            data=data,
            session=AsyncMock(),
            gpu_session_service=gpu_session_service,
            billing_service=billing_service,
            product_id="vex",
        )
        assert response.status_code == 503

    async def test_start_session_vastai_error(self) -> None:
        from src.api.routes.gpu_session import GpuSessionController
        from src.api.services.vastai.exceptions import VastAIError

        gpu_session_service = AsyncMock()
        gpu_session_service.start_session = AsyncMock(side_effect=VastAIError("api fail"))
        billing_service = AsyncMock()
        billing_service.resolve_account_for_user = AsyncMock(return_value=MagicMock())
        data = MagicMock()
        data.bundle_override = None

        response = await GpuSessionController.start.fn(  # type: ignore[attr-defined]
            MagicMock(),
            current_user_id=uuid4(),
            data=data,
            session=AsyncMock(),
            gpu_session_service=gpu_session_service,
            billing_service=billing_service,
            product_id="vex",
        )
        assert response.status_code == 503

    async def test_list_sessions(self) -> None:
        from src.api.routes.gpu_session import GpuSessionController

        gpu_session_service = AsyncMock()
        gpu_session_service.list_user_sessions = AsyncMock(return_value=[MagicMock()])

        with patch(
            "src.api.routes.gpu_session.GpuSessionResponse.from_model",
            return_value=MagicMock(),
        ):
            result = await GpuSessionController.list_sessions.fn(  # type: ignore[attr-defined]
                MagicMock(),
                current_user_id=uuid4(),
                gpu_session_service=gpu_session_service,
                product_id="vex",
            )
        assert len(result.sessions) == 1

    async def test_get_session_not_found(self) -> None:
        from src.api.routes.gpu_session import GpuSessionController

        gpu_session_service = AsyncMock()
        gpu_session_service.get_session = AsyncMock(return_value=None)

        with pytest.raises(NotFoundException):
            await GpuSessionController.get_session.fn(  # type: ignore[attr-defined]
                MagicMock(),
                current_user_id=uuid4(),
                session_id=uuid4(),
                gpu_session_service=gpu_session_service,
                product_id="vex",
                session=AsyncMock(),
            )

    async def test_get_session_found(self) -> None:
        from src.api.routes.gpu_session import GpuSessionController

        gpu_session_service = AsyncMock()
        gpu_session_service.get_session = AsyncMock(return_value=MagicMock())

        with patch(
            "src.api.routes.gpu_session.GpuSessionResponse.from_model",
            return_value=MagicMock(),
        ):
            result = await GpuSessionController.get_session.fn(  # type: ignore[attr-defined]
                MagicMock(),
                current_user_id=uuid4(),
                session_id=uuid4(),
                gpu_session_service=gpu_session_service,
                product_id="vex",
                session=AsyncMock(),
            )
        assert result is not None

    async def test_pause_success(self) -> None:
        from src.api.routes.gpu_session import GpuSessionController

        gpu_session_service = AsyncMock()
        gpu_session_service.pause_session = AsyncMock(return_value=MagicMock())

        with patch(
            "src.api.routes.gpu_session.GpuSessionResponse.from_model",
            return_value=MagicMock(),
        ):
            response = await GpuSessionController.pause.fn(  # type: ignore[attr-defined]
                MagicMock(),
                current_user_id=uuid4(),
                session_id=uuid4(),
                gpu_session_service=gpu_session_service,
                product_id="vex",
            )
        assert response.status_code == 200

    async def test_pause_invalid_state(self) -> None:
        from src.api.routes.gpu_session import GpuSessionController
        from src.api.services.gpu_session.exceptions import InvalidSessionStateError

        gpu_session_service = AsyncMock()
        gpu_session_service.pause_session = AsyncMock(
            side_effect=InvalidSessionStateError(
                "invalid", current_status="active", operation="pause"
            )
        )

        response = await GpuSessionController.pause.fn(  # type: ignore[attr-defined]
            MagicMock(),
            current_user_id=uuid4(),
            session_id=uuid4(),
            gpu_session_service=gpu_session_service,
            product_id="vex",
        )
        assert response.status_code == 409

    async def test_pause_not_found(self) -> None:
        from src.api.routes.gpu_session import GpuSessionController
        from src.api.services.gpu_session.exceptions import GpuSessionError

        gpu_session_service = AsyncMock()
        gpu_session_service.pause_session = AsyncMock(side_effect=GpuSessionError("not found"))

        response = await GpuSessionController.pause.fn(  # type: ignore[attr-defined]
            MagicMock(),
            current_user_id=uuid4(),
            session_id=uuid4(),
            gpu_session_service=gpu_session_service,
            product_id="vex",
        )
        assert response.status_code == 404

    async def test_resume_success(self) -> None:
        from src.api.routes.gpu_session import GpuSessionController

        gpu_session_service = AsyncMock()
        gpu_session_service.resume_session = AsyncMock(return_value=MagicMock())

        with patch(
            "src.api.routes.gpu_session.GpuSessionResponse.from_model",
            return_value=MagicMock(),
        ):
            response = await GpuSessionController.resume.fn(  # type: ignore[attr-defined]
                MagicMock(),
                current_user_id=uuid4(),
                session_id=uuid4(),
                gpu_session_service=gpu_session_service,
                product_id="vex",
            )
        assert response.status_code == 200

    async def test_resume_not_found(self) -> None:
        from src.api.routes.gpu_session import GpuSessionController
        from src.api.services.gpu_session.exceptions import GpuSessionError

        gpu_session_service = AsyncMock()
        gpu_session_service.resume_session = AsyncMock(side_effect=GpuSessionError("not found"))

        response = await GpuSessionController.resume.fn(  # type: ignore[attr-defined]
            MagicMock(),
            current_user_id=uuid4(),
            session_id=uuid4(),
            gpu_session_service=gpu_session_service,
            product_id="vex",
        )
        assert response.status_code == 404

    async def test_stop_returns_confirmation(self) -> None:
        from src.api.routes.gpu_session import GpuSessionController
        from src.api.services.gpu_session.schemas import StopConfirmation

        stop_conf = MagicMock(spec=StopConfirmation)
        stop_conf.active_duration_seconds = 300
        stop_conf.session_id = uuid4()
        stop_conf.model_type = "aisha-image"
        stop_conf.vastai_gpu_name = "RTX 4090"
        stop_conf.vastai_cost_per_hour_micros = 100000
        stop_conf.paused_duration_seconds = 0
        stop_conf.message = "Confirm stop"

        gpu_session_service = AsyncMock()
        gpu_session_service.stop_session = AsyncMock(return_value=stop_conf)
        settings = MagicMock()
        settings.gpu_session_tokens_per_minute = 10
        data = MagicMock()
        data.confirmed = False

        response = await GpuSessionController.stop.fn(  # type: ignore[attr-defined]
            MagicMock(),
            current_user_id=uuid4(),
            session_id=uuid4(),
            data=data,
            gpu_session_service=gpu_session_service,
            product_id="vex",
            settings=settings,
        )
        assert response.status_code == 200

    async def test_stop_executes(self) -> None:
        from src.api.routes.gpu_session import GpuSessionController

        gpu_session_service = AsyncMock()
        gpu_session_service.stop_session = AsyncMock(return_value=MagicMock())
        data = MagicMock()
        data.confirmed = True
        settings = MagicMock()

        with patch(
            "src.api.routes.gpu_session.GpuSessionResponse.from_model",
            return_value=MagicMock(),
        ):
            response = await GpuSessionController.stop.fn(  # type: ignore[attr-defined]
                MagicMock(),
                current_user_id=uuid4(),
                session_id=uuid4(),
                data=data,
                gpu_session_service=gpu_session_service,
                product_id="vex",
                settings=settings,
            )
        assert response.status_code == 200

    async def test_stop_invalid_state(self) -> None:
        from src.api.routes.gpu_session import GpuSessionController
        from src.api.services.gpu_session.exceptions import InvalidSessionStateError

        gpu_session_service = AsyncMock()
        gpu_session_service.stop_session = AsyncMock(
            side_effect=InvalidSessionStateError(
                "can't stop", current_status="active", operation="stop"
            )
        )
        data = MagicMock()
        data.confirmed = True
        settings = MagicMock()

        response = await GpuSessionController.stop.fn(  # type: ignore[attr-defined]
            MagicMock(),
            current_user_id=uuid4(),
            session_id=uuid4(),
            data=data,
            gpu_session_service=gpu_session_service,
            product_id="vex",
            settings=settings,
        )
        assert response.status_code == 409

    async def test_resume_invalid_state(self) -> None:
        from src.api.routes.gpu_session import GpuSessionController
        from src.api.services.gpu_session.exceptions import InvalidSessionStateError

        gpu_session_service = AsyncMock()
        gpu_session_service.resume_session = AsyncMock(
            side_effect=InvalidSessionStateError(
                "can't resume", current_status="stopped", operation="resume"
            )
        )

        response = await GpuSessionController.resume.fn(  # type: ignore[attr-defined]
            MagicMock(),
            current_user_id=uuid4(),
            session_id=uuid4(),
            gpu_session_service=gpu_session_service,
            product_id="vex",
        )
        assert response.status_code == 409

    async def test_stop_not_found(self) -> None:
        from src.api.routes.gpu_session import GpuSessionController
        from src.api.services.gpu_session.exceptions import GpuSessionError

        gpu_session_service = AsyncMock()
        gpu_session_service.stop_session = AsyncMock(side_effect=GpuSessionError("not found"))
        data = MagicMock()
        data.confirmed = True
        settings = MagicMock()

        response = await GpuSessionController.stop.fn(  # type: ignore[attr-defined]
            MagicMock(),
            current_user_id=uuid4(),
            session_id=uuid4(),
            data=data,
            gpu_session_service=gpu_session_service,
            product_id="vex",
            settings=settings,
        )
        assert response.status_code == 404

    async def test_start_bundle_override_non_admin_forbidden(self) -> None:
        from src.api.routes.gpu_session import GpuSessionController

        user_mock = MagicMock()
        user_mock.role = MagicMock()
        user_mock.role.value = "user"

        data = MagicMock()
        data.bundle_override = "custom-bundle"

        with patch("src.db.repositories.UserRepository") as MockRepo:
            repo_instance = AsyncMock()
            repo_instance.get_active_user = AsyncMock(return_value=user_mock)
            MockRepo.return_value = repo_instance

            billing_service = AsyncMock()
            billing_service.resolve_account_for_user = AsyncMock(return_value=MagicMock())

            with pytest.raises(PermissionDeniedException):
                await GpuSessionController.start.fn(  # type: ignore[attr-defined]
                    MagicMock(),
                    current_user_id=uuid4(),
                    data=data,
                    session=AsyncMock(),
                    gpu_session_service=AsyncMock(),
                    billing_service=billing_service,
                    product_id="vex",
                )

    def test_stop_confirmation_response_excludes_bundle_name(self) -> None:
        from src.api.schemas.gpu_session import StopConfirmationResponse

        fields = {f.name for f in msgspec.structs.fields(StopConfirmationResponse)}
        assert "bundle_name" not in fields


# ---------------------------------------------------------------------------
# SSEController (sse.py)
# ---------------------------------------------------------------------------


class TestSSERouteHandlers:
    async def test_create_sse_ticket_no_redis(self) -> None:
        from src.api.routes.sse import SSEController

        settings = MagicMock()
        settings.redis_url = None

        with patch("src.api.routes.sse.get_settings", return_value=settings):
            response = await SSEController.create_sse_ticket.fn(  # type: ignore[attr-defined]
                MagicMock(),
                current_user_id=uuid4(),
                sse_ticket_service=AsyncMock(),
            )
        assert response.status_code == 503

    async def test_create_sse_ticket_with_redis(self) -> None:
        from src.api.routes.sse import SSEController

        settings = MagicMock()
        settings.redis_url = "redis://localhost:6379"
        sse_ticket_service = AsyncMock()
        sse_ticket_service.create_ticket = AsyncMock(return_value="ticket-xyz")

        with patch("src.api.routes.sse.get_settings", return_value=settings):
            result = await SSEController.create_sse_ticket.fn(  # type: ignore[attr-defined]
                MagicMock(),
                current_user_id=uuid4(),
                sse_ticket_service=sse_ticket_service,
            )
        assert result.ticket == "ticket-xyz"

    async def test_stream_no_redis(self) -> None:
        from src.api.routes.sse import SSEController

        settings = MagicMock()
        settings.redis_url = None

        with patch("src.api.routes.sse.get_settings", return_value=settings):
            response = await SSEController.stream.fn(  # type: ignore[attr-defined]
                MagicMock(),
                ticket="some-ticket",
                sse_ticket_service=AsyncMock(),
                event_bus=AsyncMock(),
            )
        assert response.status_code == 503

    async def test_stream_invalid_ticket(self) -> None:
        from src.api.routes.sse import SSEController

        settings = MagicMock()
        settings.redis_url = "redis://localhost:6379"
        sse_ticket_service = AsyncMock()
        sse_ticket_service.redeem_ticket = AsyncMock(return_value=None)

        with patch("src.api.routes.sse.get_settings", return_value=settings):
            response = await SSEController.stream.fn(  # type: ignore[attr-defined]
                MagicMock(),
                ticket="bad-ticket",
                sse_ticket_service=sse_ticket_service,
                event_bus=AsyncMock(),
            )
        assert response.status_code == 401

    async def test_stream_valid_ticket_returns_sse(self) -> None:
        from src.api.routes.sse import SSEController

        settings = MagicMock()
        settings.redis_url = "redis://localhost:6379"
        sse_ticket_service = AsyncMock()
        sse_ticket_service.redeem_ticket = AsyncMock(return_value=uuid4())

        with patch("src.api.routes.sse.get_settings", return_value=settings):
            result = await SSEController.stream.fn(  # type: ignore[attr-defined]
                MagicMock(),
                ticket="valid-ticket",
                sse_ticket_service=sse_ticket_service,
                event_bus=MagicMock(),
            )
        assert isinstance(result, ServerSentEvent)


class TestContentStreamFromR2:
    @staticmethod
    def _make_r2_mock(
        content_type: str = "image/jpeg",
        content_length: int = 1234,
        content_range: str | None = None,
    ) -> MagicMock:
        from src.api.services.storage.r2 import ObjectStream

        @asynccontextmanager
        async def _stream_object(
            _storage_key: str, *, range_header: str | None = None
        ) -> AsyncIterator[ObjectStream]:
            del range_header

            async def _chunks() -> AsyncIterator[bytes]:
                yield b"chunk"

            yield ObjectStream(
                chunks=_chunks(),
                content_type=content_type,
                content_length=content_length,
                content_range=content_range,
            )

        r2_mock = MagicMock()
        r2_mock.stream_object = _stream_object
        return r2_mock

    async def test_stream_from_r2_success(self) -> None:
        from src.api.routes.content import ContentProxyController

        r2_mock = self._make_r2_mock(content_type="image/jpeg", content_length=1234)

        result = await ContentProxyController._stream_from_r2(
            r2_mock,
            "users/abc/outputs/img.jpg",
            "etag123",
            1234,
            3600,
            range_header=None,
            if_none_match=None,
        )
        assert isinstance(result, Stream)
        assert result.headers["Accept-Ranges"] == "bytes"

    async def test_stream_from_r2_storage_error_returns_502(self) -> None:
        from src.api.routes.content import ContentProxyController
        from src.api.services.storage.exceptions import StorageError

        r2_mock = MagicMock()

        @asynccontextmanager
        async def _failing_stream_object(
            _storage_key: str, *, range_header: str | None = None
        ) -> AsyncIterator[None]:
            del range_header
            raise StorageError("r2 unreachable")
            yield  # pragma: no cover — unreachable, satisfies generator typing

        r2_mock.stream_object = _failing_stream_object

        result = await ContentProxyController._stream_from_r2(
            r2_mock, "some/key", "etag123", 1234, 3600, range_header=None, if_none_match=None
        )
        assert isinstance(result, Response)
        assert result.status_code == 502

    async def test_stream_from_r2_serves_206_for_valid_range(self) -> None:
        from src.api.routes.content import ContentProxyController

        r2_mock = self._make_r2_mock(
            content_type="video/mp4", content_length=500, content_range="bytes 0-499/1234"
        )

        result = await ContentProxyController._stream_from_r2(
            r2_mock,
            "users/abc/outputs/vid.mp4",
            "etag123",
            1234,
            3600,
            range_header="bytes=0-499",
            if_none_match=None,
        )
        assert isinstance(result, Stream)
        assert result.status_code == 206
        assert result.headers["Content-Range"] == "bytes 0-499/1234"
        assert result.headers["Content-Length"] == "500"

    async def test_stream_from_r2_returns_416_for_out_of_bounds_range(self) -> None:
        from src.api.routes.content import ContentProxyController

        r2_mock = MagicMock()  # never called — rejected before any R2 round-trip

        result = await ContentProxyController._stream_from_r2(
            r2_mock,
            "users/abc/outputs/vid.mp4",
            "etag123",
            1234,
            3600,
            range_header="bytes=9999-10999",
            if_none_match=None,
        )
        assert isinstance(result, Response)
        assert result.status_code == 416
        assert result.headers["Content-Range"] == "bytes */1234"

    async def test_stream_from_r2_returns_304_on_matching_etag(self) -> None:
        from src.api.routes.content import ContentProxyController

        r2_mock = MagicMock()  # never called — 304 short-circuits before R2

        result = await ContentProxyController._stream_from_r2(
            r2_mock,
            "users/abc/outputs/img.jpg",
            "etag123",
            1234,
            3600,
            range_header=None,
            if_none_match='"etag123"',
        )
        assert isinstance(result, Response)
        assert result.status_code == 304
        assert result.headers["ETag"] == '"etag123"'

    async def test_stream_from_r2_non_matching_etag_proceeds_to_200(self) -> None:
        from src.api.routes.content import ContentProxyController

        r2_mock = self._make_r2_mock()

        result = await ContentProxyController._stream_from_r2(
            r2_mock,
            "users/abc/outputs/img.jpg",
            "etag123",
            1234,
            3600,
            range_header=None,
            if_none_match='"some-other-etag"',
        )
        assert isinstance(result, Stream)
