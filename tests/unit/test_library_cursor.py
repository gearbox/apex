"""Unit tests for library cursor helpers in src/api/schemas/pagination.py."""

from __future__ import annotations

import base64
import json
from datetime import UTC, datetime
from uuid import uuid4

import pytest

from src.api.schemas.pagination import (
    decode_library_cursor,
    encode_cursor,
    encode_library_cursor,
)

pytestmark = pytest.mark.unit


class TestEncodeDecodeRoundTrip:
    @pytest.mark.parametrize("source", ["upload", "output"])
    def test_round_trip(self, source: str) -> None:
        ts = datetime(2026, 1, 1, 12, 30, tzinfo=UTC)
        id_ = uuid4()

        cursor = encode_library_cursor(ts, source, id_)
        decoded_ts, decoded_source, decoded_id = decode_library_cursor(cursor)

        assert decoded_ts == ts
        assert decoded_source == source
        assert decoded_id == id_


class TestInvalidSource:
    def test_unknown_source_raises(self) -> None:
        ts = datetime(2026, 1, 1, tzinfo=UTC)
        cursor = encode_library_cursor(ts, "generation", uuid4())
        with pytest.raises(ValueError, match="Invalid pagination cursor"):
            decode_library_cursor(cursor)


class TestMalformedToken:
    def test_not_base64(self) -> None:
        with pytest.raises(ValueError, match="Invalid pagination cursor"):
            decode_library_cursor("not-valid-base64!!!")

    def test_valid_base64_but_not_json(self) -> None:
        token = base64.urlsafe_b64encode(b"not json").decode()
        with pytest.raises(ValueError, match="Invalid pagination cursor"):
            decode_library_cursor(token)

    def test_missing_fields(self) -> None:
        payload = {"created_at": datetime(2026, 1, 1, tzinfo=UTC).isoformat()}
        token = base64.urlsafe_b64encode(json.dumps(payload).encode()).decode()
        with pytest.raises(ValueError, match="Invalid pagination cursor"):
            decode_library_cursor(token)

    def test_empty_string(self) -> None:
        with pytest.raises(ValueError, match="Invalid pagination cursor"):
            decode_library_cursor("")


class TestIsolationFromExistingCursor:
    """The library cursor is a distinct format/helper set from the 2-field cursor."""

    def test_plain_cursor_not_decodable_by_library_decode(self) -> None:
        ts = datetime(2026, 1, 1, tzinfo=UTC)
        token = encode_cursor(ts, uuid4())
        # decode_library_cursor requires a "source" field, absent from the
        # plain 2-field cursor payload.
        with pytest.raises(ValueError, match="Invalid pagination cursor"):
            decode_library_cursor(token)
