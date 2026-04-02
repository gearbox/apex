"""Tests for history query parsing and validation."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from litestar.exceptions import ValidationException

from src.api.services.health.history import parse_history_params


class TestParseHistoryParams:
    def test_all_none(self) -> None:
        q = parse_history_params(after=None, before=None, limit=60)
        assert q.after is None
        assert q.before is None
        assert q.limit == 60

    def test_valid_iso_datetimes(self) -> None:
        q = parse_history_params(
            after="2026-03-01T00:00:00+00:00",
            before="2026-03-31T23:59:59+00:00",
            limit=100,
        )
        assert q.after == datetime(2026, 3, 1, tzinfo=UTC)
        assert q.before is not None

    def test_z_suffix_normalized(self) -> None:
        q = parse_history_params(after="2026-03-01T00:00:00Z", before=None, limit=10)
        assert q.after is not None
        assert q.after.tzinfo is not None

    def test_malformed_after_raises_validation(self) -> None:
        with pytest.raises(ValidationException, match="Invalid datetime for 'after'"):
            parse_history_params(after="not-a-date", before=None, limit=10)

    def test_malformed_before_raises_validation(self) -> None:
        with pytest.raises(ValidationException, match="Invalid datetime for 'before'"):
            parse_history_params(after=None, before="2026-13-45", limit=10)

    def test_limit_clamped_min(self) -> None:
        q = parse_history_params(after=None, before=None, limit=0)
        assert q.limit == 1

    def test_limit_clamped_max(self) -> None:
        q = parse_history_params(after=None, before=None, limit=9999)
        assert q.limit == 1440
