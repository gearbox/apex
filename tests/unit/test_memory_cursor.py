"""Unit tests for the artist-memory cursor in ``src/api/schemas/pagination.py``.

The memory list is in diary order, so its cursor is ``(occurred_on | null, created_at, id)``. The
``occurred_on`` key is always present, which is what makes a cursor issued by another endpoint (it
has no such key) invalid here.
"""

from __future__ import annotations

import base64
import json
from datetime import UTC, date, datetime
from typing import Any
from uuid import uuid4

import pytest
from hypothesis import given
from hypothesis import strategies as st

from src.api.schemas.pagination import (
    decode_memory_cursor,
    encode_cursor,
    encode_library_cursor,
    encode_memory_cursor,
)

pytestmark = pytest.mark.unit

_TS = datetime(2026, 1, 1, 12, 30, tzinfo=UTC)


def _token(payload: Any) -> str:
    return base64.urlsafe_b64encode(json.dumps(payload).encode()).decode()


class TestRoundTrip:
    @pytest.mark.parametrize("occurred_on", [date(2019, 6, 4), None])
    def test_a_dated_and_an_undated_cursor_come_back_as_they_went(
        self, occurred_on: date | None
    ) -> None:
        id_ = uuid4()

        decoded = decode_memory_cursor(encode_memory_cursor(occurred_on, _TS, id_))

        assert decoded == (occurred_on, _TS, id_)

    @given(
        occurred_on=st.none() | st.dates(),
        created_at=st.datetimes(timezones=st.just(UTC)),
        id_=st.uuids(),
    )
    def test_any_position_round_trips(
        self, occurred_on: date | None, created_at: datetime, id_: Any
    ) -> None:
        assert decode_memory_cursor(encode_memory_cursor(occurred_on, created_at, id_)) == (
            occurred_on,
            created_at,
            id_,
        )

    def test_an_undated_cursor_says_null_not_a_missing_key(self) -> None:
        raw = json.loads(base64.urlsafe_b64decode(encode_memory_cursor(None, _TS, uuid4())))

        assert "occurred_on" in raw
        assert raw["occurred_on"] is None


class TestInvalid:
    def test_a_cursor_from_another_endpoint_has_no_occurred_on(self) -> None:
        with pytest.raises(ValueError, match="Invalid pagination cursor"):
            decode_memory_cursor(encode_cursor(_TS, uuid4()))
        with pytest.raises(ValueError, match="Invalid pagination cursor"):
            decode_memory_cursor(encode_library_cursor(_TS, "upload", uuid4()))

    @pytest.mark.parametrize(
        "payload",
        [
            {"occurred_on": None, "created_at": _TS.isoformat()},  # no id
            {"occurred_on": None, "id": str(uuid4())},  # no created_at
            {"created_at": _TS.isoformat(), "id": str(uuid4())},  # no occurred_on
            {"occurred_on": "2019-13-45", "created_at": _TS.isoformat(), "id": str(uuid4())},
            {"occurred_on": "yesterday", "created_at": _TS.isoformat(), "id": str(uuid4())},
            {"occurred_on": 20190604, "created_at": _TS.isoformat(), "id": str(uuid4())},
            {"occurred_on": None, "created_at": "not a time", "id": str(uuid4())},
            {"occurred_on": None, "created_at": _TS.isoformat(), "id": "not-a-uuid"},
            # a timestamp with no zone would be compared with a timestamptz: refuse it
            {"occurred_on": None, "created_at": "2026-01-01T12:30:00", "id": str(uuid4())},
            ["occurred_on", "created_at", "id"],
            "occurred_on",
            None,
        ],
    )
    def test_a_malformed_position_is_refused(self, payload: Any) -> None:
        with pytest.raises(ValueError, match="Invalid pagination cursor"):
            decode_memory_cursor(_token(payload))

    @pytest.mark.parametrize("cursor", ["", "nope", "not-valid-base64!!!"])
    def test_a_token_that_is_not_a_cursor_is_refused(self, cursor: str) -> None:
        with pytest.raises(ValueError, match="Invalid pagination cursor"):
            decode_memory_cursor(cursor)

    def test_valid_base64_that_is_not_json_is_refused(self) -> None:
        with pytest.raises(ValueError, match="Invalid pagination cursor"):
            decode_memory_cursor(base64.urlsafe_b64encode(b"not json").decode())
