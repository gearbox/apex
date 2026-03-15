"""Shared pagination schema and cursor utilities.

All paginated list endpoints return ``PaginatedResponse[T]``.  Clients can
switch between traditional offset-based pagination and efficient cursor-based
(keyset) pagination by supplying a ``cursor`` query parameter.

Cursor encoding
---------------
A cursor is an opaque, URL-safe base64 token that encodes the
``created_at`` timestamp and ``id`` of the **last item returned** on the
previous page.  The backend decodes it into a keyset WHERE clause:

    WHERE (created_at < cursor_ts)
       OR (created_at = cursor_ts AND id < cursor_id)
    ORDER BY created_at DESC, id DESC

This guarantees stable paging even when new rows are inserted between
fetches, which is a known problem with OFFSET-based paging.
"""

from __future__ import annotations

import base64
import json
from datetime import datetime
from typing import Generic, TypeVar
from uuid import UUID

import msgspec

T = TypeVar("T")


class PaginatedResponse(msgspec.Struct, Generic[T], kw_only=True):  # noqa: UP046 Because msgspec resolves generic type annotations at decode-time and needs T as a named TypeVar in the module scope — PEP 695 [T] syntax doesn't expose it as a name, causing the NameError. The Generic[T] form is required here.
    """Unified paginated collection response returned by all list endpoints.

    Attributes:
        items: Page of results.
        total: Total number of matching records (ignores cursor / offset).
        limit: Requested page size echoed back for client convenience.
        offset: Page offset used (0 when cursor-based pagination is active).
        has_more: ``True`` when there are additional pages after this one.
        next_cursor: Opaque cursor token to pass as ``cursor=`` on the next
            request.  ``None`` when ``has_more`` is ``False`` or when the
            endpoint does not support cursor-based paging.
    """

    items: list[T]
    total: int
    limit: int
    offset: int
    has_more: bool
    next_cursor: str | None = None


# ---------------------------------------------------------------------------
# Cursor encode / decode helpers
# ---------------------------------------------------------------------------


def encode_cursor(created_at: datetime, id: UUID) -> str:
    """Encode a (created_at, id) pair into an opaque cursor string.

    Args:
        created_at: Timestamp of the last item on the current page.
        id: Primary key of the last item on the current page.

    Returns:
        URL-safe base64-encoded JSON token.
    """
    payload = {"created_at": created_at.isoformat(), "id": str(id)}
    return base64.urlsafe_b64encode(json.dumps(payload).encode()).decode()


def decode_cursor(cursor: str) -> tuple[datetime, UUID]:
    """Decode an opaque cursor string into a (created_at, id) pair.

    Args:
        cursor: Token returned by a previous ``PaginatedResponse.next_cursor``.

    Returns:
        ``(created_at, id)`` suitable for building a keyset WHERE clause.

    Raises:
        ValueError: If the cursor is malformed or missing required fields.
    """
    try:
        data = json.loads(base64.urlsafe_b64decode(cursor.encode()))
        return datetime.fromisoformat(data["created_at"]), UUID(data["id"])
    except Exception as exc:
        raise ValueError(f"Invalid pagination cursor: {exc}") from exc
