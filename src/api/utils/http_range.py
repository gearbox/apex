"""Pure HTTP Range header parsing (RFC 7233 §2.1, single range only)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ServedRange:
    """A single satisfiable byte range to serve as `206 Partial Content`."""

    start: int
    end: int  # inclusive
    size: int  # total resource size

    @property
    def length(self) -> int:
        """Number of bytes in this range (inclusive of both endpoints)."""
        return self.end - self.start + 1


@dataclass(frozen=True, slots=True)
class FullBody:
    """No usable Range — serve the full resource as `200 OK`."""

    size: int


@dataclass(frozen=True, slots=True)
class Unsatisfiable:
    """A syntactically valid single range that falls outside the resource."""

    size: int


ParsedRange = ServedRange | FullBody | Unsatisfiable


def parse_range(header: str | None, size: int) -> ParsedRange:
    """Parse a `Range: bytes=...` header against a known resource size.

    Only a single byte-range-spec is supported (RFC 7233 §2.1) — multipart
    (comma-separated) ranges are out of scope. A missing header, a
    non-`bytes` unit, a multi-range header, or a syntactically invalid
    byte-range-spec (non-digit bounds, or an explicit last-byte-pos less
    than first-byte-pos) all resolve to `FullBody` — per RFC 7233 §3.1 a
    server should ignore an unsatisfying-to-parse Range header rather than
    reject the request. A syntactically valid range whose first-byte-pos is
    at or beyond `size` resolves to `Unsatisfiable` (416); an end beyond
    `size` is clamped to the last available byte, not rejected.

    Args:
        header: Raw `Range` header value, or None if absent.
        size: Total size in bytes of the resource being requested.

    Returns:
        ServedRange, FullBody, or Unsatisfiable.
    """
    if not header:
        return FullBody(size=size)

    if "," in header:
        # Multipart ranges are out of scope — treat as a full-body request.
        return FullBody(size=size)

    if not header.startswith("bytes="):
        return FullBody(size=size)

    spec = header[len("bytes=") :].strip()
    if "-" not in spec:
        return FullBody(size=size)

    start_str, _, end_str = spec.partition("-")
    start_str = start_str.strip()
    end_str = end_str.strip()

    if start_str == "":
        # Suffix range: bytes=-N -> the last N bytes of the resource.
        if end_str == "" or not end_str.isdigit():
            return FullBody(size=size)
        suffix_length = int(end_str)
        if suffix_length == 0:
            return Unsatisfiable(size=size)
        start = max(0, size - suffix_length)
        end = size - 1
    else:
        if not start_str.isdigit():
            return FullBody(size=size)
        start = int(start_str)
        if end_str == "":
            end = size - 1
        else:
            if not end_str.isdigit():
                return FullBody(size=size)
            end = int(end_str)
            if end < start:
                # last-byte-pos < first-byte-pos: syntactically invalid per
                # RFC 7233 §2.1 — ignore rather than reject.
                return FullBody(size=size)

    if start >= size:
        return Unsatisfiable(size=size)

    return ServedRange(start=start, end=min(end, size - 1), size=size)
