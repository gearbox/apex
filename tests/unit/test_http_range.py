"""Exhaustive unit tests for the pure Range header parser (D1)."""

from __future__ import annotations

import pytest

from src.api.utils.http_range import FullBody, ServedRange, Unsatisfiable, parse_range

pytestmark = pytest.mark.unit

SIZE = 1000


class TestNoHeader:
    def test_none_header_is_full_body(self) -> None:
        assert parse_range(None, SIZE) == FullBody(size=SIZE)

    def test_empty_header_is_full_body(self) -> None:
        assert parse_range("", SIZE) == FullBody(size=SIZE)


class TestMalformedHeaders:
    @pytest.mark.parametrize(
        "header",
        [
            "items=0-499",  # wrong unit
            "bytes=",  # empty spec
            "bytes=abc",  # no dash
            "bytes=abc-def",  # non-digit bounds
            "bytes=100-abc",  # digit start, non-digit end
            "bytes=1.5-10",  # non-integer
            "bytes=-",  # dash with nothing on either side
            "bytes=10-5",  # explicit end < start (syntactically invalid)
            "bytes=+5-10",  # sign not allowed
            "byteswrong=0-10",  # doesn't start with "bytes="
        ],
    )
    def test_malformed_resolves_to_full_body(self, header: str) -> None:
        assert parse_range(header, SIZE) == FullBody(size=SIZE)

    def test_multipart_range_is_full_body(self) -> None:
        """Multipart ranges are explicitly out of scope (DO-NOT boundary)."""
        assert parse_range("bytes=0-99,200-299", SIZE) == FullBody(size=SIZE)


class TestOpenEndedRange:
    def test_bytes_0_dash_serves_entire_body_as_range(self) -> None:
        result = parse_range("bytes=0-", SIZE)
        assert result == ServedRange(start=0, end=SIZE - 1, size=SIZE)

    def test_open_ended_mid_file(self) -> None:
        result = parse_range("bytes=500-", SIZE)
        assert result == ServedRange(start=500, end=SIZE - 1, size=SIZE)
        assert isinstance(result, ServedRange)
        assert result.length == 500

    def test_open_ended_start_at_size_is_unsatisfiable(self) -> None:
        assert parse_range(f"bytes={SIZE}-", SIZE) == Unsatisfiable(size=SIZE)

    def test_open_ended_start_beyond_size_is_unsatisfiable(self) -> None:
        assert parse_range("bytes=5000-", SIZE) == Unsatisfiable(size=SIZE)


class TestSuffixRange:
    def test_suffix_bytes_minus_500(self) -> None:
        result = parse_range("bytes=-500", SIZE)
        assert result == ServedRange(start=500, end=999, size=SIZE)
        assert isinstance(result, ServedRange)
        assert result.length == 500

    def test_suffix_larger_than_resource_serves_whole_body(self) -> None:
        result = parse_range("bytes=-5000", SIZE)
        assert result == ServedRange(start=0, end=SIZE - 1, size=SIZE)

    def test_suffix_exact_size(self) -> None:
        result = parse_range(f"bytes=-{SIZE}", SIZE)
        assert result == ServedRange(start=0, end=SIZE - 1, size=SIZE)

    def test_suffix_zero_is_unsatisfiable(self) -> None:
        assert parse_range("bytes=-0", SIZE) == Unsatisfiable(size=SIZE)

    def test_suffix_non_digit_is_full_body(self) -> None:
        assert parse_range("bytes=-abc", SIZE) == FullBody(size=SIZE)


class TestExplicitRange:
    def test_explicit_start_end(self) -> None:
        result = parse_range("bytes=100-199", SIZE)
        assert result == ServedRange(start=100, end=199, size=SIZE)
        assert isinstance(result, ServedRange)
        assert result.length == 100

    def test_explicit_start_end_clamped_to_size(self) -> None:
        """An end beyond size is clamped to the last available byte, not rejected."""
        result = parse_range("bytes=0-999999", SIZE)
        assert result == ServedRange(start=0, end=SIZE - 1, size=SIZE)

    def test_single_byte_range(self) -> None:
        result = parse_range("bytes=0-0", SIZE)
        assert result == ServedRange(start=0, end=0, size=SIZE)
        assert isinstance(result, ServedRange)
        assert result.length == 1

    def test_last_byte(self) -> None:
        result = parse_range(f"bytes={SIZE - 1}-{SIZE - 1}", SIZE)
        assert result == ServedRange(start=SIZE - 1, end=SIZE - 1, size=SIZE)


class TestOutOfBounds:
    def test_start_equal_to_size_is_unsatisfiable(self) -> None:
        assert parse_range(f"bytes={SIZE}-{SIZE + 10}", SIZE) == Unsatisfiable(size=SIZE)

    def test_start_beyond_size_is_unsatisfiable(self) -> None:
        assert parse_range("bytes=5000-6000", SIZE) == Unsatisfiable(size=SIZE)

    def test_zero_size_resource_is_unsatisfiable(self) -> None:
        assert parse_range("bytes=0-", 0) == Unsatisfiable(size=0)
        assert parse_range("bytes=0-0", 0) == Unsatisfiable(size=0)


class TestWhitespaceTolerance:
    def test_whitespace_around_bounds_is_tolerated(self) -> None:
        result = parse_range("bytes=  100 - 199 ", SIZE)
        assert result == ServedRange(start=100, end=199, size=SIZE)
