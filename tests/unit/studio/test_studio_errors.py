"""One handler turns every StudioError into the unified ErrorEnvelope."""

from __future__ import annotations

from typing import TYPE_CHECKING

import msgspec
import pytest
from litestar import Litestar, get
from litestar.testing import TestClient
from structlog.testing import capture_logs

from src.api.app import global_exception_handler, studio_error_handler
from src.studio import errors as e

if TYPE_CHECKING:
    from collections.abc import Callable

pytestmark = pytest.mark.unit

_ISSUES = (e.DocumentIssue("sections[0].label", "reserved_characters"),)

CASES: list[tuple[Callable[[], Exception], int, str]] = [
    (e.StudioNotFound, 404, "not_found"),
    (e.ArtistNotFound, 404, "artist_not_found"),
    (e.ParentVersionNotFound, 404, "parent_version_not_found"),
    (e.PlanVersionNotFound, 404, "plan_version_not_found"),
    (e.InvalidCursor, 400, "invalid_cursor"),
    (lambda: e.StudioDocumentInvalid(_ISSUES), 422, "document_invalid"),
    (e.TrackOrderMismatch, 422, "track_order_mismatch"),
    (e.ArtistNameConflict, 409, "artist_name_conflict"),
    (e.ArtistLimitReached, 409, "artist_limit_reached"),
    (lambda: e.ArtistInUse(3), 409, "artist_in_use"),
    (e.ProjectHasTracks, 409, "project_has_tracks"),
    (e.TrackLimitReached, 409, "track_limit_reached"),
    (e.VersionLimitReached, 409, "version_limit_reached"),
    (e.VersionUnchanged, 409, "version_unchanged"),
    (e.PlanRequired, 409, "plan_required"),
    (lambda: e.LyricsIncompatibleWithPlan(_ISSUES), 409, "lyrics_incompatible_with_plan"),
    (e.VersionIsCurrent, 409, "version_is_current"),
    (lambda: e.VersionHasChildren(2), 409, "version_has_children"),
    (lambda: e.VersionIsLyricsBase(1), 409, "version_is_lyrics_base"),
]


def _client(raiser: Callable[[], Exception]) -> TestClient[Litestar]:
    @get("/boom")
    async def boom() -> None:
        raise raiser()

    return TestClient(
        app=Litestar(
            route_handlers=[boom],
            exception_handlers={
                e.StudioError: studio_error_handler,
                Exception: global_exception_handler,
            },
        )
    )


@pytest.mark.parametrize(("factory", "status", "code"), CASES, ids=[c[2] for c in CASES])
def test_envelope_shape_and_status(
    factory: Callable[[], Exception], status: int, code: str
) -> None:
    with _client(factory) as client:
        resp = client.get("/boom")
    assert resp.status_code == status
    body = resp.json()
    assert body["error"] == code
    assert body["status_code"] == status
    assert isinstance(body["message"], str)
    assert body["message"]


def test_every_studio_error_subclass_is_covered() -> None:
    covered = {type(factory()) for factory, _, _ in CASES}
    declared = {
        cls
        for cls in vars(e).values()
        if isinstance(cls, type) and issubclass(cls, e.StudioError) and cls is not e.StudioError
    }
    assert declared == covered


def test_codes_are_unique() -> None:
    codes = [factory().code for factory, _, _ in CASES]  # type: ignore[attr-defined]
    assert len(codes) == len(set(codes))


def test_detail_carries_issues_and_counts() -> None:
    with _client(lambda: e.StudioDocumentInvalid(_ISSUES)) as client:
        detail = client.get("/boom").json()["detail"]
    assert detail == {"issues": [{"path": "sections[0].label", "code": "reserved_characters"}]}
    with _client(lambda: e.ArtistInUse(3)) as client:
        assert client.get("/boom").json()["detail"] == {"project_count": 3}
    with _client(lambda: e.VersionHasChildren(2)) as client:
        assert client.get("/boom").json()["detail"] == {"count": 2}
    with _client(e.StudioNotFound) as client:
        assert client.get("/boom").json()["detail"] is None


def test_schema_errors_are_not_studio_errors_and_surface_as_500() -> None:
    assert not issubclass(e.StudioDocumentSchemaError, e.StudioError)
    with _client(lambda: e.StudioDocumentSchemaError("schema_version 9")) as client:
        resp = client.get("/boom")
    assert resp.status_code == 500
    assert "schema_version" not in resp.text


def test_handler_logs_only_the_code_not_the_payload() -> None:
    issue = e.DocumentIssue("sections[0].label", "reserved_characters")
    with capture_logs() as logs, _client(lambda: e.StudioDocumentInvalid((issue,))) as client:
        client.get("/boom")
    events = [log for log in logs if log["event"] == "studio.error"]
    assert len(events) == 1
    assert events[0]["code"] == "document_invalid"
    assert events[0]["log_level"] == "warning"
    assert "issues" not in events[0]
    assert msgspec.json.encode(events[0]).find(b"reserved_characters") == -1
