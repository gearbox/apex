"""History query parsing and validation."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from litestar.exceptions import ValidationException

_MAX_LIMIT = 1440


@dataclass(frozen=True, kw_only=True)
class HistoryQuery:
    """Parsed and validated history query parameters."""

    after: datetime | None
    before: datetime | None
    limit: int


def parse_history_params(
    *,
    after: str | None,
    before: str | None,
    limit: int,
) -> HistoryQuery:
    """Parse and validate history endpoint query parameters.

    Args:
        after: ISO 8601 datetime string or None.
        before: ISO 8601 datetime string or None.
        limit: Requested limit (clamped to [1, 1440]).

    Returns:
        Validated HistoryQuery.

    Raises:
        ValidationException: If datetime strings are malformed.
    """
    parsed_after = _parse_datetime_param(after, "after")
    parsed_before = _parse_datetime_param(before, "before")
    clamped_limit = min(max(limit, 1), _MAX_LIMIT)
    return HistoryQuery(after=parsed_after, before=parsed_before, limit=clamped_limit)


def _parse_datetime_param(value: str | None, name: str) -> datetime | None:
    """Parse a single datetime query parameter.

    Handles the common Z suffix (fromisoformat doesn't accept it on <3.11)
    and provides a clear 400-level error on malformed input.
    """
    if value is None:
        return None
    try:
        # Normalize trailing Z to +00:00 for fromisoformat compatibility
        normalized = value.replace("Z", "+00:00") if value.endswith("Z") else value
        return datetime.fromisoformat(normalized)
    except (ValueError, TypeError) as exc:
        raise ValidationException(
            f"Invalid datetime for '{name}': {value!r}. "
            "Expected ISO 8601 format (e.g. 2026-03-31T14:00:00Z)."
        ) from exc
