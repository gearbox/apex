"""Tests for the shared pagination schema and cursor utilities.

Covers:
  - encode_cursor / decode_cursor round-trips
  - decode_cursor error handling
  - CursorPage field values and msgspec serialization
  - has_more / next_cursor logic
"""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

import msgspec
import pytest

from src.api.schemas.jobs import UnifiedJobResponse
from src.api.schemas.pagination import (
    CursorPage,
    decode_cursor,
    encode_cursor,
)
from src.core.enums import GenerationType, JobStatus

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _job_response() -> UnifiedJobResponse:
    return UnifiedJobResponse(
        id=uuid4(),
        name="test job",
        status=JobStatus.COMPLETED,
        provider="grok",
        generation_type=GenerationType.T2I,
        prompt="a cat",
        created_at=datetime.now(UTC),
    )


# ---------------------------------------------------------------------------
# Cursor encode / decode
# ---------------------------------------------------------------------------


class TestCursorEncoding:
    def test_round_trip_preserves_values(self) -> None:
        ts = datetime(2026, 3, 15, 12, 0, 0, tzinfo=UTC)
        uid = uuid4()

        token = encode_cursor(ts, uid)
        ts2, uid2 = decode_cursor(token)

        assert ts2 == ts
        assert uid2 == uid

    def test_output_is_url_safe_string(self) -> None:
        token = encode_cursor(datetime.now(UTC), uuid4())
        # URL-safe base64 must not contain +/=
        assert "+" not in token
        assert "/" not in token

    def test_with_timezone_aware_datetime(self) -> None:
        ts = datetime(2025, 1, 1, 0, 0, 0, tzinfo=UTC)
        uid = uuid4()
        token = encode_cursor(ts, uid)
        ts2, uid2 = decode_cursor(token)
        assert ts2 == ts
        assert uid2 == uid

    def test_different_ids_produce_different_cursors(self) -> None:
        ts = datetime.now(UTC)
        token1 = encode_cursor(ts, uuid4())
        token2 = encode_cursor(ts, uuid4())
        assert token1 != token2

    def test_different_timestamps_produce_different_cursors(self) -> None:
        uid = uuid4()
        ts1 = datetime(2026, 1, 1, tzinfo=UTC)
        ts2 = datetime(2026, 1, 2, tzinfo=UTC)
        assert encode_cursor(ts1, uid) != encode_cursor(ts2, uid)

    def test_decode_invalid_base64_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="Invalid pagination cursor"):
            decode_cursor("not-valid-base64!!!")

    def test_decode_valid_base64_missing_fields_raises_value_error(self) -> None:
        import base64
        import json

        bad = base64.urlsafe_b64encode(
            json.dumps({"only_created_at": "2026-01-01"}).encode()
        ).decode()
        with pytest.raises(ValueError, match="Invalid pagination cursor"):
            decode_cursor(bad)

    def test_decode_corrupted_uuid_raises_value_error(self) -> None:
        import base64
        import json

        bad = base64.urlsafe_b64encode(
            json.dumps({"created_at": "2026-01-01T00:00:00+00:00", "id": "not-a-uuid"}).encode()
        ).decode()
        with pytest.raises(ValueError, match="Invalid pagination cursor"):
            decode_cursor(bad)


# ---------------------------------------------------------------------------
# CursorPage struct
# ---------------------------------------------------------------------------


class TestCursorPage:
    def test_fields_stored_correctly(self) -> None:
        resp: CursorPage[str] = CursorPage(
            items=["a", "b"],
            limit=5,
            has_more=True,
            next_cursor="tok",
        )
        assert resp.items == ["a", "b"]
        assert resp.limit == 5
        assert resp.has_more is True
        assert resp.next_cursor == "tok"

    def test_next_cursor_defaults_to_none(self) -> None:
        resp: CursorPage[str] = CursorPage(
            items=[],
            limit=20,
            has_more=False,
        )
        assert resp.next_cursor is None

    def test_msgspec_round_trip_with_job_items(self) -> None:
        job = _job_response()
        resp: CursorPage[UnifiedJobResponse] = CursorPage(
            items=[job],
            limit=10,
            has_more=True,
            next_cursor="abc123",
        )

        encoded = msgspec.json.encode(resp)
        decoded = msgspec.json.decode(encoded, type=CursorPage[UnifiedJobResponse])

        assert decoded.limit == 10
        assert decoded.has_more is True
        assert decoded.next_cursor == "abc123"
        assert len(decoded.items) == 1
        assert decoded.items[0].id == job.id

    def test_msgspec_round_trip_empty_page(self) -> None:
        resp: CursorPage[UnifiedJobResponse] = CursorPage(
            items=[],
            limit=20,
            has_more=False,
        )

        encoded = msgspec.json.encode(resp)
        decoded = msgspec.json.decode(encoded, type=CursorPage[UnifiedJobResponse])

        assert decoded.items == []
        assert decoded.has_more is False
        assert decoded.next_cursor is None

    def test_has_more_true_when_more_items_exist(self) -> None:
        items = [_job_response() for _ in range(10)]
        resp: CursorPage[UnifiedJobResponse] = CursorPage(
            items=items,
            limit=10,
            has_more=True,
        )
        assert resp.has_more is True

    def test_has_more_false_on_last_page(self) -> None:
        items = [_job_response() for _ in range(5)]
        resp: CursorPage[UnifiedJobResponse] = CursorPage(
            items=items,
            limit=10,
            has_more=False,
        )
        assert resp.has_more is False


# ---------------------------------------------------------------------------
# Cursor encode → decode used inside CursorPage workflow
# ---------------------------------------------------------------------------


class TestCursorWorkflow:
    def test_next_cursor_decodes_to_last_item_position(self) -> None:
        ts = datetime(2026, 3, 10, 8, 30, 0, tzinfo=UTC)
        uid = uuid4()
        cursor = encode_cursor(ts, uid)

        ts2, uid2 = decode_cursor(cursor)

        assert ts2 == ts
        assert uid2 == uid

    def test_full_pagination_cursor_flow(self) -> None:
        """Simulate two pages using cursor."""
        items_page1 = [_job_response() for _ in range(3)]
        last = items_page1[-1]
        last_ts = last.created_at
        last_id = last.id

        cursor = encode_cursor(last_ts, last_id)

        resp1: CursorPage[UnifiedJobResponse] = CursorPage(
            items=items_page1,
            limit=3,
            has_more=True,
            next_cursor=cursor,
        )

        # Client decodes cursor and passes it to next request
        decoded_ts, decoded_id = decode_cursor(resp1.next_cursor)  # type: ignore[arg-type]
        assert decoded_ts == last_ts
        assert decoded_id == last_id
