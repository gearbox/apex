"""Unit tests for in-product problem reports: schemas, enums, service, ops payload.

Contracts: C2 (ops payload field set), C3 (Telegram text never carries user
text), C5 (product-scoped class + catalog entry), C8 (client_path shape), C10
(transition table), C16 (no user text in logs). DB-backed contracts live in
tests/integration/test_feedback_reports.py.
"""

from __future__ import annotations

import itertools
from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID, uuid4

import msgspec
import pytest
from structlog.testing import capture_logs

from src.api.schemas.feedback import FeedbackAdminPatch, FeedbackCreate
from src.api.schemas.ops_events import (
    FeedbackSubmittedOpsPayload,
    OpsEventEnvelope,
    OpsEventType,
)
from src.api.services.admin_notifications import AdminNotificationService
from src.api.services.feedback import (
    FeedbackContextNotFoundError,
    FeedbackNotFoundError,
    FeedbackService,
    InvalidFeedbackContextError,
    InvalidFeedbackMessageError,
    InvalidFeedbackTransitionError,
    to_admin_view,
)
from src.api.services.telegram.mapping import map_ops_event
from src.core.enums import (
    PLATFORM_SCOPED_NOTIFICATION_CLASSES,
    FeedbackCategory,
    FeedbackStatus,
    NotificationClass,
)
from src.core.library_ref import LibraryAssetSource
from src.db.models.feedback import FeedbackReport

pytestmark = pytest.mark.unit

_MODULE = "src.api.services.feedback"
_encoder = msgspec.json.Encoder()

# Distinctive user-authored / client values that must never leave apex.
SECRET_MESSAGE = "SECRET-MESSAGE my card 4242 was charged twice"
SECRET_PATH = "/secret-client-path/library"
SECRET_UA = "SecretAgent/9.9 (leaky)"
SECRET_NOTE = "SECRET-ADMIN-NOTE refund approved by finance"


def _decode_create(**fields: Any) -> FeedbackCreate:
    body = {"category": "bug", "message": "Something is definitely broken", **fields}
    return msgspec.json.decode(_encoder.encode(body), type=FeedbackCreate)


def _service(
    *, session: MagicMock | None = None, bus: MagicMock | None = None
) -> tuple[FeedbackService, MagicMock, MagicMock]:
    session = session or MagicMock()
    bus = bus or MagicMock()
    bus.publish = AsyncMock()
    return FeedbackService(session=session, ops_event_bus=bus), session, bus


def _report(**overrides: Any) -> FeedbackReport:
    now = datetime.now(UTC)
    fields: dict[str, Any] = {
        "id": uuid4(),
        "product_id": "vex",
        "user_id": uuid4(),
        "category": FeedbackCategory.BUG.value,
        "status": FeedbackStatus.OPEN.value,
        "message": "Something is definitely broken",
        "created_at": now,
        "updated_at": now,
    }
    fields.update(overrides)
    return FeedbackReport(**fields)


def _owned(found: bool) -> MagicMock:
    """Patchable repository class whose ``get`` finds (or not) the row."""
    repo_cls = MagicMock()
    repo_cls.return_value.get = AsyncMock(return_value=MagicMock() if found else None)
    return repo_cls


# ---------------------------------------------------------------------------
# Schemas (C8 + boundary validation)
# ---------------------------------------------------------------------------


class TestFeedbackCreateSchema:
    @pytest.mark.parametrize(
        "path",
        ["/library?tab=1", "/library#frag", "library", "", "/a\x00b", "/lib\n", "/" + "a" * 512],
    )
    def test_invalid_client_path_rejected(self, path: str) -> None:
        """C8 — query, fragment, relative, NUL, or over-long paths never reach the DB."""
        with pytest.raises(msgspec.ValidationError):
            _decode_create(client_path=path)

    @pytest.mark.parametrize("path", ["/", "/library", "/library/assets/output:abc"])
    def test_pathname_accepted(self, path: str) -> None:
        assert _decode_create(client_path=path).client_path == path

    @pytest.mark.parametrize("message", ["short", "x" * 4001, "         ", "x" * 4000 + "\n"])
    def test_message_length_not_enforced_by_schema(self, message: str) -> None:
        """R1/R2 — length is a post-strip() service rule (400 validation_error),
        never a raw schema bound (framework 400 bad_request)."""
        assert _decode_create(message=message).message == message

    @pytest.mark.parametrize("version", ["", "1.0\x00", "1.0\n", "v" * 65])
    def test_invalid_app_version_rejected(self, version: str) -> None:
        with pytest.raises(msgspec.ValidationError):
            _decode_create(app_version=version)

    def test_unknown_category_rejected(self) -> None:
        with pytest.raises(msgspec.ValidationError):
            _decode_create(category="abuse")

    def test_unknown_field_rejected(self) -> None:
        with pytest.raises(msgspec.ValidationError):
            _decode_create(email="me@example.com")

    def test_all_categories_accepted(self) -> None:
        assert {c.value for c in FeedbackCategory} == {
            "bug",
            "generation",
            "billing",
            "account",
            "content",
            "other",
        }
        for category in FeedbackCategory:
            assert _decode_create(category=category.value).category is category


class TestFeedbackAdminPatchSchema:
    def test_omitted_fields_are_unset(self) -> None:
        patch_ = msgspec.json.decode(b"{}", type=FeedbackAdminPatch)
        assert patch_.status is msgspec.UNSET
        assert patch_.admin_note is msgspec.UNSET

    def test_null_note_clears(self) -> None:
        patch_ = msgspec.json.decode(b'{"admin_note": null}', type=FeedbackAdminPatch)
        assert patch_.admin_note is None

    def test_note_with_nul_rejected(self) -> None:
        with pytest.raises(msgspec.ValidationError):
            msgspec.json.decode(b'{"admin_note": "a\\u0000b"}', type=FeedbackAdminPatch)

    def test_multiline_note_accepted(self) -> None:
        patch_ = msgspec.json.decode(b'{"admin_note": "line 1\\nline 2"}', type=FeedbackAdminPatch)
        assert patch_.admin_note == "line 1\nline 2"


# ---------------------------------------------------------------------------
# Enums (C5, C10)
# ---------------------------------------------------------------------------

_ALLOWED_TRANSITIONS = {
    (FeedbackStatus.OPEN, FeedbackStatus.IN_PROGRESS),
    (FeedbackStatus.OPEN, FeedbackStatus.RESOLVED),
    (FeedbackStatus.OPEN, FeedbackStatus.DISMISSED),
    (FeedbackStatus.IN_PROGRESS, FeedbackStatus.RESOLVED),
    (FeedbackStatus.IN_PROGRESS, FeedbackStatus.DISMISSED),
}
_ALL_PAIRS = list(itertools.product(FeedbackStatus, repeat=2))


class TestFeedbackStatusEnum:
    @pytest.mark.parametrize(("current", "target"), _ALL_PAIRS)
    def test_transition_table(self, current: FeedbackStatus, target: FeedbackStatus) -> None:
        assert current.can_transition_to(target) is ((current, target) in _ALLOWED_TRANSITIONS)

    def test_terminal_statuses(self) -> None:
        assert {s for s in FeedbackStatus if s.is_terminal} == {
            FeedbackStatus.RESOLVED,
            FeedbackStatus.DISMISSED,
        }

    @pytest.mark.parametrize("status", list(FeedbackStatus))
    def test_terminal_iff_no_outgoing_transitions(self, status: FeedbackStatus) -> None:
        """R4 — ``is_terminal`` is derived from the transition table."""
        has_outgoing = any(status.can_transition_to(target) for target in FeedbackStatus)
        assert status.is_terminal is (not has_outgoing)


class TestNotificationWiring:
    def test_feedback_class_is_product_scoped(self) -> None:
        """C5 — delivered only to admins of the report's product."""
        assert NotificationClass.FEEDBACK_SUBMITTED not in PLATFORM_SCOPED_NOTIFICATION_CLASSES

    def test_catalog_has_product_scoped_entry(self) -> None:
        service = AdminNotificationService(sender=None, link_token_ttl_seconds=60)
        entry = next(
            info
            for info in service.get_class_catalog()
            if info.notification_class == NotificationClass.FEEDBACK_SUBMITTED.value
        )
        assert entry.scope == "product"
        assert entry.description


# ---------------------------------------------------------------------------
# Ops payload (C2, C3)
# ---------------------------------------------------------------------------


class TestOpsPayload:
    def test_field_set_is_ids_and_enums_only(self) -> None:
        """C2 — adding a field here is a D10 review, not a drive-by change."""
        assert set(FeedbackSubmittedOpsPayload.__struct_fields__) == {
            "report_id",
            "user_id",
            "category",
            "job_id",
        }

    async def test_publish_submitted_sends_ids_only(self) -> None:
        service, _session, bus = _service()
        report = _report(job_id=uuid4(), message=SECRET_MESSAGE)

        await service.publish_submitted(report)

        bus.publish.assert_awaited_once()
        kwargs = bus.publish.await_args.kwargs
        assert kwargs["event_type"] is OpsEventType.FEEDBACK_SUBMITTED
        assert kwargs["product_id"] == "vex"
        assert kwargs["payload"] == FeedbackSubmittedOpsPayload(
            report_id=report.id,
            user_id=report.user_id,  # type: ignore[arg-type]
            category="bug",
            job_id=report.job_id,
        )

    async def test_telegram_text_never_contains_user_text(self) -> None:
        """C3 — end to end: submit → captured publish → map_ops_event."""
        service, _session, bus = _service()
        job_id = uuid4()
        data = _decode_create(
            message=SECRET_MESSAGE, client_path=SECRET_PATH, app_version="9.9.9", job_id=str(job_id)
        )
        with patch(f"{_MODULE}.JobRepository", _owned(True)):
            report = await service.submit(
                user_id=uuid4(), product_id="vex", data=data, user_agent=SECRET_UA
            )
        await service.publish_submitted(report)

        kwargs = bus.publish.await_args.kwargs
        envelope = OpsEventEnvelope(
            event_type=kwargs["event_type"],
            product_id=kwargs["product_id"],
            payload=msgspec.Raw(_encoder.encode(kwargs["payload"])),
            timestamp=datetime.now(UTC),
            event_id="evt-1",
        )
        wire = _encoder.encode(envelope).decode()
        notification = map_ops_event(msgspec.json.decode(wire, type=OpsEventEnvelope))

        assert notification is not None
        assert notification.notification_class is NotificationClass.FEEDBACK_SUBMITTED
        for secret in (SECRET_MESSAGE, "4242", SECRET_PATH, SECRET_UA, "9.9.9"):
            assert secret not in notification.text
            assert secret not in wire
        assert str(report.id) in notification.text
        assert str(job_id) in notification.text


# ---------------------------------------------------------------------------
# submit
# ---------------------------------------------------------------------------


class TestSubmit:
    async def test_stages_stripped_row_without_commit(self) -> None:
        service, session, bus = _service()
        user_id = uuid4()
        data = _decode_create(message="   Button does nothing   ", app_version="1.2.3")

        report = await service.submit(
            user_id=user_id, product_id="synthara", data=data, user_agent="UA/1"
        )

        session.add.assert_called_once_with(report)
        session.commit.assert_not_called()
        session.flush.assert_not_called()
        bus.publish.assert_not_called()
        assert isinstance(report.id, UUID)
        assert report.product_id == "synthara"
        assert report.user_id == user_id
        assert report.message == "Button does nothing"
        assert report.status == FeedbackStatus.OPEN
        assert report.category == FeedbackCategory.BUG
        assert report.user_agent == "UA/1"
        assert report.app_version == "1.2.3"
        assert report.job_id is None
        assert report.asset_source is None
        assert report.asset_id is None

    @pytest.mark.parametrize(
        ("message", "stored_length"),
        [
            ("x" * 10, 10),
            ("\n  " + "x" * 10 + "  \t", 10),
            ("x" * 4000, 4000),
            ("x" * 4000 + "\n", 4000),
            ("  " + "x" * 4000 + "  ", 4000),
            ("🙂" * 4000, 4000),  # code points, not UTF-16 units (R5)
        ],
        ids=["10", "10-padded", "4000", "4000-newline", "4000-padded", "4000-emoji"],
    )
    async def test_message_length_boundaries_accepted(
        self, message: str, stored_length: int
    ) -> None:
        service, session, _bus = _service()
        report = await service.submit(
            user_id=uuid4(),
            product_id="vex",
            data=_decode_create(message=message),
            user_agent=None,
        )
        session.add.assert_called_once_with(report)
        assert len(report.message) == stored_length
        assert report.message == message.strip()

    @pytest.mark.parametrize(
        "message",
        [
            "   short    ",
            "\n\t  123456789 \n",  # 9 after strip
            "         ",
            "x" * 4001,
            "  " + "x" * 4001 + "\n",  # 4001 after strip
            "🙂" * 4001,
            "ten chars \x00 but has NUL",
        ],
        ids=["short", "nine", "blank", "4001", "4001-padded", "4001-emoji", "nul"],
    )
    async def test_invalid_message_rejected(self, message: str) -> None:
        service, session, _bus = _service()
        with pytest.raises(InvalidFeedbackMessageError):
            await service.submit(
                user_id=uuid4(),
                product_id="vex",
                data=_decode_create(message=message),
                user_agent=None,
            )
        session.add.assert_not_called()

    async def test_job_ownership_checked_with_user_scope(self) -> None:
        service, _session, _bus = _service()
        user_id, job_id = uuid4(), uuid4()
        repo_cls = _owned(True)
        with patch(f"{_MODULE}.JobRepository", repo_cls):
            report = await service.submit(
                user_id=user_id,
                product_id="vex",
                data=_decode_create(job_id=str(job_id)),
                user_agent=None,
            )
        repo_cls.return_value.get.assert_awaited_once_with(job_id, user_id=user_id)
        assert report.job_id == job_id

    async def test_missing_or_foreign_job_is_not_found(self) -> None:
        service, session, _bus = _service()
        with (
            patch(f"{_MODULE}.JobRepository", _owned(False)),
            pytest.raises(FeedbackContextNotFoundError) as exc_info,
        ):
            await service.submit(
                user_id=uuid4(),
                product_id="vex",
                data=_decode_create(job_id=str(uuid4())),
                user_agent=None,
            )
        assert exc_info.value.kind == "job"
        session.add.assert_not_called()

    @pytest.mark.parametrize(
        ("source", "repo_name"),
        [
            (LibraryAssetSource.OUTPUT, "OutputRepository"),
            (LibraryAssetSource.UPLOAD, "UserImageRepository"),
        ],
    )
    async def test_asset_ownership_checked_per_source(
        self, source: LibraryAssetSource, repo_name: str
    ) -> None:
        service, _session, _bus = _service()
        user_id, asset_id = uuid4(), uuid4()
        repo_cls = _owned(True)
        with patch(f"{_MODULE}.{repo_name}", repo_cls):
            report = await service.submit(
                user_id=user_id,
                product_id="vex",
                data=_decode_create(asset_ref=f"{source.value}:{asset_id}"),
                user_agent=None,
            )
        repo_cls.return_value.get.assert_awaited_once_with(asset_id, user_id=user_id)
        assert report.asset_source == source
        assert report.asset_id == asset_id

    @pytest.mark.parametrize("repo_name", ["OutputRepository", "UserImageRepository"])
    async def test_missing_or_foreign_asset_is_not_found(self, repo_name: str) -> None:
        service, _session, _bus = _service()
        source = "output" if repo_name == "OutputRepository" else "upload"
        with (
            patch(f"{_MODULE}.{repo_name}", _owned(False)),
            pytest.raises(FeedbackContextNotFoundError) as exc_info,
        ):
            await service.submit(
                user_id=uuid4(),
                product_id="vex",
                data=_decode_create(asset_ref=f"{source}:{uuid4()}"),
                user_agent=None,
            )
        assert exc_info.value.kind == "asset"

    @pytest.mark.parametrize(
        "ref", ["nope", "job:0190e4a2-7a1b-7c3d-8e4f-5a6b7c8d9e0f", "output:not-a-uuid", "output:"]
    )
    async def test_malformed_asset_ref_is_invalid(self, ref: str) -> None:
        service, _session, _bus = _service()
        with pytest.raises(InvalidFeedbackContextError):
            await service.submit(
                user_id=uuid4(),
                product_id="vex",
                data=_decode_create(asset_ref=ref),
                user_agent=None,
            )

    @pytest.mark.parametrize(
        ("raw", "stored"),
        [
            (None, None),
            ("", None),
            ("A" * 600, "A" * 512),
            ("UA\x00/1", "UA/1"),
            ("\x00", None),
        ],
    )
    async def test_user_agent_sanitized(self, raw: str | None, stored: str | None) -> None:
        service, _session, _bus = _service()
        report = await service.submit(
            user_id=uuid4(), product_id="vex", data=_decode_create(), user_agent=raw
        )
        assert report.user_agent == stored


# ---------------------------------------------------------------------------
# Admin
# ---------------------------------------------------------------------------


class TestUpdateByAdmin:
    @pytest.mark.parametrize(("current", "target"), _ALL_PAIRS)
    async def test_transition_table(self, current: FeedbackStatus, target: FeedbackStatus) -> None:
        """C10 — every allowed pair applies; every other pair (incl. self) raises."""
        service, _session, _bus = _service()
        resolved_at = datetime(2026, 1, 1, tzinfo=UTC) if current.is_terminal else None
        report = _report(status=current.value, resolved_at=resolved_at)
        service._repo = MagicMock()
        service._repo.get_for_update = AsyncMock(return_value=(report, None))
        admin_id = uuid4()

        call = service.update_by_admin(
            report.id,
            product_id="vex",
            admin_id=admin_id,
            patch=FeedbackAdminPatch(status=target),
        )
        if (current, target) not in _ALLOWED_TRANSITIONS:
            with pytest.raises(InvalidFeedbackTransitionError) as exc_info:
                await call
            assert (exc_info.value.current, exc_info.value.target) == (current, target)
            assert report.status == current.value
            assert report.resolved_at == resolved_at
            return

        result, email = await call
        assert result is report
        assert email is None
        assert report.status == target.value
        if target.is_terminal:
            assert report.resolved_at is not None
            assert report.resolved_by == admin_id
        else:
            assert report.resolved_at is None
            assert report.resolved_by is None

    async def test_note_only_on_terminal_report_keeps_resolution(self) -> None:
        service, _session, _bus = _service()
        resolved_at = datetime(2026, 1, 1, tzinfo=UTC)
        resolver = uuid4()
        report = _report(
            status=FeedbackStatus.RESOLVED.value, resolved_at=resolved_at, resolved_by=resolver
        )
        service._repo = MagicMock()
        service._repo.get_for_update = AsyncMock(return_value=(report, None))

        await service.update_by_admin(
            report.id,
            product_id="vex",
            admin_id=uuid4(),
            patch=FeedbackAdminPatch(admin_note="followed up"),
        )

        assert report.admin_note == "followed up"
        assert report.resolved_at == resolved_at
        assert report.resolved_by == resolver

    async def test_returns_reporter_email(self) -> None:
        service, _session, _bus = _service()
        report = _report()
        service._repo = MagicMock()
        service._repo.get_for_update = AsyncMock(return_value=(report, "a@example.com"))

        result, email = await service.update_by_admin(
            report.id, product_id="vex", admin_id=uuid4(), patch=FeedbackAdminPatch(admin_note="n")
        )
        assert result is report
        assert email == "a@example.com"

    async def test_null_note_clears(self) -> None:
        service, _session, _bus = _service()
        report = _report(admin_note="old")
        service._repo = MagicMock()
        service._repo.get_for_update = AsyncMock(return_value=(report, None))

        await service.update_by_admin(
            report.id, product_id="vex", admin_id=uuid4(), patch=FeedbackAdminPatch(admin_note=None)
        )
        assert report.admin_note is None

    async def test_missing_report_not_found(self) -> None:
        service, _session, _bus = _service()
        service._repo = MagicMock()
        service._repo.get_for_update = AsyncMock(return_value=None)
        with pytest.raises(FeedbackNotFoundError):
            await service.update_by_admin(
                uuid4(),
                product_id="vex",
                admin_id=uuid4(),
                patch=FeedbackAdminPatch(status=FeedbackStatus.RESOLVED),
            )

    async def test_lookup_is_product_scoped(self) -> None:
        service, _session, _bus = _service()
        service._repo = MagicMock()
        service._repo.get_for_update = AsyncMock(return_value=None)
        report_id = uuid4()
        with pytest.raises(FeedbackNotFoundError):
            await service.update_by_admin(
                report_id, product_id="synthara", admin_id=uuid4(), patch=FeedbackAdminPatch()
            )
        service._repo.get_for_update.assert_awaited_once_with(report_id, product_id="synthara")


class TestAdminReads:
    async def test_get_for_admin_maps_asset_ref_and_email(self) -> None:
        service, _session, _bus = _service()
        asset_id = uuid4()
        report = _report(asset_source="output", asset_id=asset_id)
        service._repo = MagicMock()
        service._repo.get_with_email = AsyncMock(return_value=(report, "a@example.com"))

        view = await service.get_for_admin(report.id, product_id="vex")

        service._repo.get_with_email.assert_awaited_once_with(report.id, product_id="vex")
        assert view.asset_ref == f"output:{asset_id}"
        assert view.user_email == "a@example.com"
        assert view.status is FeedbackStatus.OPEN
        assert view.category is FeedbackCategory.BUG

    async def test_get_for_admin_missing(self) -> None:
        service, _session, _bus = _service()
        service._repo = MagicMock()
        service._repo.get_with_email = AsyncMock(return_value=None)
        with pytest.raises(FeedbackNotFoundError):
            await service.get_for_admin(uuid4(), product_id="vex")

    def test_view_without_asset_or_user(self) -> None:
        view = to_admin_view(_report(user_id=None), None)
        assert view.asset_ref is None
        assert view.user_id is None
        assert view.user_email is None

    async def test_list_pages_with_cursor(self) -> None:
        service, _session, _bus = _service()
        rows = [(_report(), None) for _ in range(3)]
        service._repo = MagicMock()
        service._repo.list_page = AsyncMock(return_value=rows)

        page = await service.list_for_admin(
            product_id="vex",
            status=FeedbackStatus.OPEN,
            category=FeedbackCategory.BILLING,
            limit=2,
        )

        kwargs = service._repo.list_page.await_args.kwargs
        assert kwargs == {
            "product_id": "vex",
            "status": FeedbackStatus.OPEN,
            "category": FeedbackCategory.BILLING,
            "limit": 2,
            "cursor_ts": None,
            "cursor_id": None,
        }
        assert page.has_more is True
        assert [item.id for item in page.items] == [rows[0][0].id, rows[1][0].id]
        assert page.next_cursor is not None

        service._repo.list_page = AsyncMock(return_value=[])
        await service.list_for_admin(product_id="vex", limit=2, cursor=page.next_cursor)
        kwargs = service._repo.list_page.await_args.kwargs
        assert kwargs["cursor_id"] == rows[1][0].id
        assert kwargs["cursor_ts"] == rows[1][0].created_at

    async def test_list_last_page_has_no_cursor(self) -> None:
        service, _session, _bus = _service()
        service._repo = MagicMock()
        service._repo.list_page = AsyncMock(return_value=[(_report(), None)])
        page = await service.list_for_admin(product_id="vex", limit=2)
        assert page.has_more is False
        assert page.next_cursor is None

    async def test_bad_cursor_raises_value_error(self) -> None:
        service, _session, _bus = _service()
        with pytest.raises(ValueError, match="cursor"):
            await service.list_for_admin(product_id="vex", cursor="%%%not-a-cursor")


# ---------------------------------------------------------------------------
# Logging (C16)
# ---------------------------------------------------------------------------


class TestNoUserTextInLogs:
    async def test_submit_and_update_logs_carry_ids_only(self) -> None:
        """C16 — no captured event contains message, UA, path, or note values."""
        service, _session, _bus = _service()
        with capture_logs() as logs:
            report = await service.submit(
                user_id=uuid4(),
                product_id="vex",
                data=_decode_create(message=SECRET_MESSAGE, client_path=SECRET_PATH),
                user_agent=SECRET_UA,
            )
            service._repo = MagicMock()
            service._repo.get_for_update = AsyncMock(return_value=(report, None))
            await service.update_by_admin(
                report.id,
                product_id="vex",
                admin_id=uuid4(),
                patch=FeedbackAdminPatch(status=FeedbackStatus.RESOLVED, admin_note=SECRET_NOTE),
            )
            await service.publish_submitted(report)

        events = {entry["event"] for entry in logs}
        assert {"feedback.submitted", "feedback.updated"} <= events
        submitted = next(e for e in logs if e["event"] == "feedback.submitted")
        assert set(submitted) - {"event", "log_level"} == {"report_id", "category", "product_id"}
        rendered = repr(logs)
        for secret in (SECRET_MESSAGE, "4242", SECRET_PATH, SECRET_UA, SECRET_NOTE):
            assert secret not in rendered
