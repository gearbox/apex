"""Admin view of the asset a feedback report references (A1-A6), unit level.

T9 (``asset_url`` mapping), T10 (``_is_view_start`` table), T11 (audit log event
carries IDs only), plus the service's target resolution. The HTTP behaviour
against real PostgreSQL lives in ``tests/integration/test_feedback_asset_view.py``.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from structlog.testing import capture_logs

from src.api.routes.content import _is_view_start
from src.api.services.feedback import (
    FEEDBACK_ASSET_VIEW_ACTION,
    FeedbackAssetNotFoundError,
    FeedbackAssetTarget,
    FeedbackNotFoundError,
    FeedbackService,
    to_admin_view,
)
from src.api.services.media import FEEDBACK_ASSET_PATH
from src.api.utils.http_range import FullBody, ServedRange, Unsatisfiable
from src.core.enums import FeedbackCategory, FeedbackStatus
from src.core.library_ref import LibraryAssetSource
from src.db.models.feedback import FeedbackReport

pytestmark = pytest.mark.unit

_MODULE = "src.api.services.feedback"

SECRET_MESSAGE = "SECRET-MESSAGE my card 4242 was charged twice"
SECRET_PATH = "/secret-client-path/library"
SECRET_UA = "SecretAgent/9.9 (leaky)"
SECRET_NOTE = "SECRET-ADMIN-NOTE refund approved by finance"


def _report(**overrides: Any) -> FeedbackReport:
    now = datetime.now(UTC)
    fields: dict[str, Any] = {
        "id": uuid4(),
        "product_id": "vex",
        "user_id": uuid4(),
        "category": FeedbackCategory.BUG.value,
        "status": FeedbackStatus.OPEN.value,
        "message": SECRET_MESSAGE,
        "client_path": SECRET_PATH,
        "user_agent": SECRET_UA,
        "admin_note": SECRET_NOTE,
        "created_at": now,
        "updated_at": now,
    }
    fields.update(overrides)
    return FeedbackReport(**fields)


def _service(report: FeedbackReport | None) -> tuple[FeedbackService, MagicMock]:
    session = MagicMock()
    service = FeedbackService(session=session, ops_event_bus=MagicMock())
    service._repo = MagicMock()
    service._repo.get = AsyncMock(return_value=report)
    return service, session


class TestIsViewStart:
    """T10."""

    @pytest.mark.parametrize(
        ("parsed", "expected"),
        [
            (FullBody(size=100), True),
            (ServedRange(start=0, end=99, size=100), True),
            (ServedRange(start=0, end=9, size=100), True),
            (ServedRange(start=1, end=99, size=100), False),
            (ServedRange(start=100, end=199, size=1000), False),
            (Unsatisfiable(size=100), False),
        ],
    )
    def test_table(self, parsed: Any, expected: bool) -> None:
        assert _is_view_start(parsed) is expected


class TestAssetUrl:
    """T9 (unit half)."""

    def test_set_iff_asset_ref_is_set(self) -> None:
        asset_id = uuid4()
        report = _report(asset_source=LibraryAssetSource.OUTPUT.value, asset_id=asset_id)
        view = to_admin_view(report, None)
        assert view.asset_ref == f"output:{asset_id}"
        assert view.asset_url == f"{FEEDBACK_ASSET_PATH}/{report.id}"

    def test_none_without_asset(self) -> None:
        view = to_admin_view(_report(), None)
        assert view.asset_ref is None
        assert view.asset_url is None

    def test_path_constant(self) -> None:
        assert FEEDBACK_ASSET_PATH == "/v1/content/feedback"


class TestGetAssetTarget:
    async def test_resolves_reporter_scoped_target(self) -> None:
        asset_id = uuid4()
        report = _report(asset_source=LibraryAssetSource.UPLOAD.value, asset_id=asset_id)
        service, _ = _service(report)

        target = await service.get_asset_target_for_admin(report.id, product_id="vex")

        assert target == FeedbackAssetTarget(
            report_id=report.id,
            source=LibraryAssetSource.UPLOAD,
            asset_id=asset_id,
            owner_id=report.user_id,  # type: ignore[arg-type]
        )
        cast("AsyncMock", service._repo.get).assert_awaited_once_with(report.id, product_id="vex")

    async def test_missing_report(self) -> None:
        service, _ = _service(None)
        with pytest.raises(FeedbackNotFoundError):
            await service.get_asset_target_for_admin(uuid4(), product_id="vex")

    @pytest.mark.parametrize(
        "overrides",
        [
            {},  # no asset reference
            {"asset_source": "output", "asset_id": uuid4(), "user_id": None},  # reporter purged
        ],
    )
    async def test_no_asset_or_reporter_purged(self, overrides: dict[str, Any]) -> None:
        service, _ = _service(_report(**overrides))
        with pytest.raises(FeedbackAssetNotFoundError):
            await service.get_asset_target_for_admin(uuid4(), product_id="vex")


class TestRecordAssetView:
    async def test_stages_audit_row_without_committing(self) -> None:
        target = FeedbackAssetTarget(
            report_id=uuid4(),
            source=LibraryAssetSource.OUTPUT,
            asset_id=uuid4(),
            owner_id=uuid4(),
        )
        admin_id = uuid4()
        service, session = _service(None)
        session.commit = AsyncMock()

        with patch(f"{_MODULE}.AdminRepository") as repo_cls:
            repo_cls.return_value.write_audit = AsyncMock()
            await service.record_asset_view(target, admin_id=admin_id, product_id="vex")

        call = repo_cls.return_value.write_audit.await_args
        assert call is not None
        (entry,) = call.args
        assert entry.actor_id == admin_id
        assert entry.target_user_id == target.owner_id
        assert entry.product_id == "vex"
        assert entry.action == FEEDBACK_ASSET_VIEW_ACTION == "feedback.asset.view"
        assert entry.source == "api"
        assert entry.detail == f"report {target.report_id} asset output:{target.asset_id}"
        session.commit.assert_not_awaited()

    async def test_log_event_carries_ids_only(self) -> None:
        """T11 — no message, note, path or UA in the captured events."""
        report = _report(asset_source="output", asset_id=uuid4())
        service, _ = _service(report)

        with capture_logs() as logs, patch(f"{_MODULE}.AdminRepository") as repo_cls:
            repo_cls.return_value.write_audit = AsyncMock()
            target = await service.get_asset_target_for_admin(report.id, product_id="vex")
            await service.record_asset_view(target, admin_id=uuid4(), product_id="vex")

        viewed = next(e for e in logs if e["event"] == "content.feedback_asset.viewed")
        assert set(viewed) - {"event", "log_level"} == {
            "report_id",
            "asset_ref",
            "admin_id",
            "owner_id",
            "product_id",
        }
        rendered = repr(logs) + repr(repo_cls.return_value.write_audit.await_args)
        for secret in (SECRET_MESSAGE, "4242", SECRET_PATH, SECRET_UA, SECRET_NOTE):
            assert secret not in rendered
