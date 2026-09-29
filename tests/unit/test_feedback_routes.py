"""Unit tests for FeedbackController and AdminFeedbackController.

Contracts: C1 (commit strictly before the ops publish), C14 (POST is
legal-exempt, PATCH is not) plus the HTTP error mapping. Handler logic is
exercised by calling the underlying coroutine (``.fn``), same convention as
test_admin_notification_routes.py; C14 goes through the real ``auth_guard``.
"""

from __future__ import annotations

import os
from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

os.environ.setdefault("JWT_SECRET_KEY", "test-feedback-routes-key-32-bytes-long")

import msgspec
import pytest
from litestar import Litestar
from litestar.datastructures import State
from litestar.di import Provide
from litestar.testing import TestClient
from sqlalchemy.exc import DBAPIError
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.app import legal_acceptance_required_handler
from src.api.dependencies.common import get_product_config, get_product_id
from src.api.middleware.product import ProductMiddleware
from src.api.routes.admin_feedback import AdminFeedbackController
from src.api.routes.feedback import FeedbackController
from src.api.schemas.feedback import FeedbackAdminPatch, FeedbackCreate, FeedbackReportAdmin
from src.api.schemas.pagination import CursorPage
from src.api.security.jwt import JWTConfig, JWTService
from src.api.services.feedback import (
    FeedbackContextNotFoundError,
    FeedbackNotFoundError,
    FeedbackService,
    InvalidFeedbackContextError,
    InvalidFeedbackMessageError,
    InvalidFeedbackTransitionError,
    to_admin_view,
)
from src.api.services.legal.errors import LegalAcceptanceRequiredError
from src.api.services.token_revocation import TokenRevocationService
from src.core.enums import FeedbackCategory, FeedbackStatus
from src.db.models.feedback import FeedbackReport
from tests.legal_support import make_legal_registry

pytestmark = pytest.mark.unit

_submit = FeedbackController.submit.fn  # type: ignore[attr-defined]
_list = AdminFeedbackController.list_reports.fn  # type: ignore[attr-defined]
_get = AdminFeedbackController.get_report.fn  # type: ignore[attr-defined]
_update = AdminFeedbackController.update_report.fn  # type: ignore[attr-defined]

_NOW = datetime(2026, 9, 28, 12, 0, tzinfo=UTC)


def _report() -> FeedbackReport:
    return FeedbackReport(
        id=uuid4(),
        product_id="vex",
        user_id=uuid4(),
        category="bug",
        status="open",
        message="Something is definitely broken",
        created_at=_NOW,
        updated_at=_NOW,
    )


def _view(report: FeedbackReport | None = None) -> FeedbackReportAdmin:
    report = report or _report()
    return FeedbackReportAdmin(
        id=report.id,
        category=FeedbackCategory.BUG,
        status=FeedbackStatus.OPEN,
        message=report.message,
        user_id=report.user_id,
        user_email=None,
        job_id=None,
        asset_ref=None,
        asset_url=None,
        client_path=None,
        app_version=None,
        user_agent=None,
        admin_note=None,
        resolved_at=None,
        resolved_by=None,
        created_at=_NOW,
        updated_at=_NOW,
    )


def _data() -> FeedbackCreate:
    return FeedbackCreate(category=FeedbackCategory.BUG, message="Something is broken")


def _request(user_agent: str | None = "UA/1") -> MagicMock:
    request = MagicMock()
    request.headers = {} if user_agent is None else {"user-agent": user_agent}
    return request


def _body(response: Any) -> dict[str, Any]:
    body: dict[str, Any] = msgspec.to_builtins(response.content)
    return body


# ---------------------------------------------------------------------------
# POST /v1/feedback
# ---------------------------------------------------------------------------


class TestSubmitRoute:
    async def test_commits_before_publishing(self) -> None:
        """C1 — the ops event is published strictly after the row is durable."""
        report = _report()
        session = MagicMock()
        session.commit = AsyncMock()
        service = MagicMock()
        service.submit = AsyncMock(return_value=report)
        service.publish_submitted = AsyncMock()
        order = MagicMock()
        order.attach_mock(service.submit, "submit")
        order.attach_mock(session.commit, "commit")
        order.attach_mock(service.publish_submitted, "publish")
        user_id = uuid4()

        response = await _submit(
            MagicMock(),
            request=_request("Browser/1"),
            current_user_id=user_id,
            product_id="vex",
            data=_data(),
            session=session,
            feedback_service=service,
        )

        assert [name for name, _args, _kwargs in order.mock_calls] == [
            "submit",
            "commit",
            "publish",
        ]
        service.submit.assert_awaited_once_with(
            user_id=user_id, product_id="vex", data=_data(), user_agent="Browser/1"
        )
        service.publish_submitted.assert_awaited_once_with(report)
        assert response.status_code == 201
        assert _body(response) == {
            "id": str(report.id),
            "status": "open",
            "created_at": _NOW.isoformat().replace("+00:00", "Z"),
        }

    async def test_missing_user_agent_passes_none(self) -> None:
        service = MagicMock()
        service.submit = AsyncMock(return_value=_report())
        service.publish_submitted = AsyncMock()
        await _submit(
            MagicMock(),
            request=_request(None),
            current_user_id=uuid4(),
            product_id="vex",
            data=_data(),
            session=AsyncMock(),
            feedback_service=service,
        )
        assert service.submit.await_args.kwargs["user_agent"] is None

    @pytest.mark.parametrize(
        ("exc", "status", "error"),
        [
            (InvalidFeedbackMessageError("too short"), 400, "validation_error"),
            (InvalidFeedbackContextError("Invalid asset reference"), 400, "validation_error"),
            (FeedbackContextNotFoundError("job"), 404, "job_not_found"),
            (FeedbackContextNotFoundError("asset"), 404, "asset_not_found"),
        ],
    )
    async def test_errors_map_without_commit_or_publish(
        self, exc: Exception, status: int, error: str
    ) -> None:
        session = AsyncMock()
        service = MagicMock()
        service.submit = AsyncMock(side_effect=exc)
        service.publish_submitted = AsyncMock()

        response = await _submit(
            MagicMock(),
            request=_request(),
            current_user_id=uuid4(),
            product_id="vex",
            data=_data(),
            session=session,
            feedback_service=service,
        )

        assert response.status_code == status
        body = _body(response)
        assert body["error"] == error
        assert body["status_code"] == status
        session.commit.assert_not_awaited()
        service.publish_submitted.assert_not_awaited()


# ---------------------------------------------------------------------------
# Admin routes
# ---------------------------------------------------------------------------


class TestAdminRoutes:
    async def test_list_passes_filters_and_product(self) -> None:
        service = MagicMock()
        page = CursorPage(items=[_view()], limit=5, has_more=False)
        service.list_for_admin = AsyncMock(return_value=page)

        response = await _list(
            MagicMock(),
            admin=MagicMock(),
            product_id="synthara",
            feedback_service=service,
            status=FeedbackStatus.OPEN,
            category=FeedbackCategory.BILLING,
            limit=5,
            cursor="c",
        )

        service.list_for_admin.assert_awaited_once_with(
            product_id="synthara",
            status=FeedbackStatus.OPEN,
            category=FeedbackCategory.BILLING,
            limit=5,
            cursor="c",
        )
        assert response.content is page

    async def test_list_bad_cursor_is_400(self) -> None:
        service = MagicMock()
        service.list_for_admin = AsyncMock(side_effect=ValueError("Invalid pagination cursor"))
        response = await _list(
            MagicMock(), admin=MagicMock(), product_id="vex", feedback_service=service, cursor="x"
        )
        assert response.status_code == 400
        assert _body(response)["error"] == "invalid_cursor"

    async def test_get_not_found(self) -> None:
        service = MagicMock()
        service.get_for_admin = AsyncMock(side_effect=FeedbackNotFoundError)
        response = await _get(
            MagicMock(),
            admin=MagicMock(),
            product_id="vex",
            report_id=uuid4(),
            feedback_service=service,
        )
        assert response.status_code == 404
        assert _body(response)["error"] == "feedback_not_found"

    async def test_get_found(self) -> None:
        view = _view()
        service = MagicMock()
        service.get_for_admin = AsyncMock(return_value=view)
        response = await _get(
            MagicMock(),
            admin=MagicMock(),
            product_id="vex",
            report_id=view.id,
            feedback_service=service,
        )
        service.get_for_admin.assert_awaited_once_with(view.id, product_id="vex")
        assert response.content is view

    async def test_empty_patch_is_400_without_lock(self) -> None:
        service = MagicMock()
        service.update_by_admin = AsyncMock()
        session = AsyncMock()
        response = await _update(
            MagicMock(),
            admin=MagicMock(),
            product_id="vex",
            report_id=uuid4(),
            data=FeedbackAdminPatch(),
            session=session,
            feedback_service=service,
        )
        assert response.status_code == 400
        assert _body(response)["error"] == "validation_error"
        service.update_by_admin.assert_not_awaited()
        session.commit.assert_not_awaited()

    async def test_patch_commits_and_maps_locked_row(self) -> None:
        report = _report()
        email = "reporter@example.com"
        admin = MagicMock()
        admin.id = uuid4()
        session = MagicMock()
        session.commit = AsyncMock()
        service = MagicMock()
        service.update_by_admin = AsyncMock(return_value=(report, email))
        service.get_for_admin = AsyncMock()
        order = MagicMock()
        order.attach_mock(service.update_by_admin, "update")
        order.attach_mock(session.commit, "commit")
        patch_ = FeedbackAdminPatch(status=FeedbackStatus.RESOLVED)

        response = await _update(
            MagicMock(),
            admin=admin,
            product_id="vex",
            report_id=report.id,
            data=patch_,
            session=session,
            feedback_service=service,
        )

        assert [name for name, _a, _k in order.mock_calls] == ["update", "commit"]
        service.update_by_admin.assert_awaited_once_with(
            report.id, product_id="vex", admin_id=admin.id, patch=patch_
        )
        service.get_for_admin.assert_not_awaited()
        assert response.content == to_admin_view(report, email)

    async def test_patch_commit_failure_propagates(self) -> None:
        """Review r2 D3 — a failed commit is not swallowed or mapped to a 4xx."""
        session = MagicMock()
        session.commit = AsyncMock(side_effect=DBAPIError("COMMIT", None, Exception("boom")))
        service = MagicMock()
        service.update_by_admin = AsyncMock(return_value=(_report(), None))

        with pytest.raises(DBAPIError):
            await _update(
                MagicMock(),
                admin=MagicMock(),
                product_id="vex",
                report_id=uuid4(),
                data=FeedbackAdminPatch(admin_note="n"),
                session=session,
                feedback_service=service,
            )

    @pytest.mark.parametrize(
        ("exc", "status", "error"),
        [
            (FeedbackNotFoundError(), 404, "feedback_not_found"),
            (
                InvalidFeedbackTransitionError(FeedbackStatus.RESOLVED, FeedbackStatus.DISMISSED),
                409,
                "invalid_status_transition",
            ),
        ],
    )
    async def test_patch_errors_do_not_commit(
        self, exc: Exception, status: int, error: str
    ) -> None:
        session = AsyncMock()
        service = MagicMock()
        service.update_by_admin = AsyncMock(side_effect=exc)
        response = await _update(
            MagicMock(),
            admin=MagicMock(),
            product_id="vex",
            report_id=uuid4(),
            data=FeedbackAdminPatch(admin_note="n"),
            session=session,
            feedback_service=service,
        )
        assert response.status_code == status
        assert _body(response)["error"] == error
        session.commit.assert_not_awaited()

    async def test_transition_conflict_carries_detail(self) -> None:
        service = MagicMock()
        service.update_by_admin = AsyncMock(
            side_effect=InvalidFeedbackTransitionError(FeedbackStatus.OPEN, FeedbackStatus.OPEN)
        )
        response = await _update(
            MagicMock(),
            admin=MagicMock(),
            product_id="vex",
            report_id=uuid4(),
            data=FeedbackAdminPatch(status=FeedbackStatus.OPEN),
            session=AsyncMock(),
            feedback_service=service,
        )
        assert _body(response)["detail"] == {"current": "open", "target": "open"}


# ---------------------------------------------------------------------------
# C14 — legal enforcement through the real auth_guard
# ---------------------------------------------------------------------------

_SECRET = "test_secret_key_for_feedback_routes_256bits"


class TestLegalEnforcement:
    @pytest.fixture
    def jwt_service(self) -> JWTService:
        return JWTService(JWTConfig(secret_key=_SECRET))

    def _app(self, jwt_service: JWTService, service: MagicMock) -> Litestar:
        return Litestar(
            route_handlers=[FeedbackController, AdminFeedbackController],
            middleware=[ProductMiddleware],
            dependencies={
                "product_config": Provide(get_product_config, sync_to_thread=False),
                "product_id": Provide(get_product_id, sync_to_thread=False),
                "session": Provide(lambda: AsyncMock(spec=AsyncSession), sync_to_thread=False),
                "feedback_service": Provide(lambda: service, sync_to_thread=False),
            },
            exception_handlers={LegalAcceptanceRequiredError: legal_acceptance_required_handler},
            state=State(
                {
                    "jwt_service": jwt_service,
                    "token_revocation": TokenRevocationService(None, max_token_ttl_seconds=0),
                    "legal_registry": make_legal_registry(),
                }
            ),
        )

    def _headers(self, jwt_service: JWTService) -> dict[str, str]:
        token, _ = jwt_service.create_access_token(
            uuid4(), product_id="vex", legal_digest="0000000000000000"
        )
        return {"Authorization": f"Bearer {token}", "X-Product-Id": "vex"}

    @pytest.mark.parametrize("path", ["/v1/feedback", "/v1/feedback/"])
    def test_post_with_stale_digest_is_created(self, jwt_service: JWTService, path: str) -> None:
        # spec= so Litestar's signature validation accepts the injected double.
        service = MagicMock(spec=FeedbackService)
        service.submit = AsyncMock(return_value=_report())
        service.publish_submitted = AsyncMock()
        with TestClient(app=self._app(jwt_service, service)) as client:
            resp = client.post(
                path,
                json={"category": "bug", "message": "Something is broken"},
                headers=self._headers(jwt_service),
            )
        assert resp.status_code == 201, resp.text
        assert resp.json()["status"] == "open"

    def test_patch_with_stale_digest_is_428(self, jwt_service: JWTService) -> None:
        # spec= so Litestar's signature validation accepts the injected double.
        service = MagicMock(spec=FeedbackService)
        service.update_by_admin = AsyncMock()
        with TestClient(app=self._app(jwt_service, service)) as client:
            resp = client.patch(
                f"/v1/admin/feedback/{uuid4()}",
                json={"status": "resolved"},
                headers=self._headers(jwt_service),
            )
        assert resp.status_code == 428
        assert resp.json()["error"] == "legal_acceptance_required"
        service.update_by_admin.assert_not_awaited()

    def test_unauthenticated_post_is_401(self, jwt_service: JWTService) -> None:
        # spec= so Litestar's signature validation accepts the injected double.
        service = MagicMock(spec=FeedbackService)
        service.submit = AsyncMock()
        with TestClient(app=self._app(jwt_service, service)) as client:
            resp = client.post(
                "/v1/feedback",
                json={"category": "bug", "message": "Something is broken"},
                headers={"X-Product-Id": "vex"},
            )
        assert resp.status_code == 401
        service.submit.assert_not_awaited()
