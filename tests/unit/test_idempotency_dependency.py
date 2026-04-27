"""Unit tests for get_idempotency_key dependency."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.api.dependencies.idempotency import get_idempotency_key

pytestmark = pytest.mark.unit


def _connection(header_value: str | None) -> MagicMock:
    conn = MagicMock()
    conn.headers.get.return_value = header_value
    return conn


class TestGetIdempotencyKey:
    def test_returns_key_when_valid(self) -> None:
        conn = _connection("my-idempotency-key")
        result = get_idempotency_key(conn)
        assert result == "my-idempotency-key"

    def test_raises_when_header_missing(self) -> None:
        conn = _connection(None)
        with pytest.raises(ValueError, match="Idempotency-Key header is required"):
            get_idempotency_key(conn)

    def test_raises_when_header_empty_string(self) -> None:
        conn = _connection("")
        with pytest.raises(ValueError, match="Idempotency-Key header is required"):
            get_idempotency_key(conn)

    def test_raises_when_header_too_long(self) -> None:
        conn = _connection("x" * 65)
        with pytest.raises(ValueError, match="64 characters or fewer"):
            get_idempotency_key(conn)

    def test_accepts_exactly_64_chars(self) -> None:
        conn = _connection("a" * 64)
        result = get_idempotency_key(conn)
        assert len(result) == 64
