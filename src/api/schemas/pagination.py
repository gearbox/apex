"""Shared pagination schema and cursor utilities.

All paginated list endpoints return ``CursorPage[T]``.  Clients pass an
opaque ``cursor`` query parameter to fetch the next page.

Cursor encoding
---------------
A cursor is an opaque, URL-safe base64 token that encodes the
``created_at`` timestamp and ``id`` of the **last item returned** on the
previous page.  The backend decodes it into a keyset WHERE clause:

    WHERE (created_at, id) < (cursor_ts, cursor_id)
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


# Ignore UP046 because msgspec resolves generic type annotations at decode-time
# and needs T as a named TypeVar in the module scope — PEP 695 [T] syntax
# doesn't expose it as a name, causing the NameError.
# The Generic[T] form is required here.
class CursorPage(msgspec.Struct, Generic[T], kw_only=True):  # noqa: UP046
    """Cursor-paginated response used by all list endpoints.

    No total count — uses limit+1 fetch pattern.
    No offset — cursor-only pagination.

    Attributes:
        items: Page of results.
        limit: Requested page size echoed back for client convenience.
        has_more: ``True`` when there are additional pages after this one.
        next_cursor: Opaque cursor token to pass as ``cursor=`` on the next
            request.  ``None`` when ``has_more`` is ``False``.
    """

    items: list[T]
    limit: int
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
        cursor: Token returned by a previous ``CursorPage.next_cursor``.

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


# ---------------------------------------------------------------------------
# Library cursor encode / decode helpers
# ---------------------------------------------------------------------------
#
# Distinct from encode_cursor/decode_cursor above: the library read model
# unions two tables (uploads, outputs) ranked against each other, so the
# keyset needs a third component — a fixed per-source rank — to keep the
# tuple comparison consistent with the union's ORDER BY. See
# src/db/repositories/library.py.

_LIBRARY_CURSOR_SOURCES: frozenset[str] = frozenset({"upload", "output"})


def encode_library_cursor(created_at: datetime, source: str, id: UUID, sort: str = "newest") -> str:
    """Encode a (created_at, source, id) triple into an opaque cursor string.

    Args:
        created_at: Timestamp of the last item on the current page. For the
            ``expiring_soon`` sort this is the item's ``expires_at`` value —
            the field name is generic; it always holds whatever timestamp
            the requested sort keys off of.
        source: ``"upload"`` or ``"output"`` — the last item's asset source.
        id: Primary key of the last item on the current page.
        sort: The ``LibrarySort`` value this cursor was produced under.
            Embedded so ``decode_library_cursor`` can reject a cursor reused
            after a sort switch (the keyset column differs per sort — reusing
            it blindly would silently misorder the page rather than error).

    Returns:
        URL-safe base64-encoded JSON token.
    """
    payload = {"created_at": created_at.isoformat(), "source": source, "id": str(id), "sort": sort}
    return base64.urlsafe_b64encode(json.dumps(payload).encode()).decode()


def decode_library_cursor(
    cursor: str,
    *,
    expected_sort: str | None = None,
) -> tuple[datetime, str, UUID]:
    """Decode an opaque library cursor string into a (created_at, source, id) triple.

    Args:
        cursor: Token returned by a previous library ``CursorPage.next_cursor``.
        expected_sort: When provided, the cursor's embedded ``sort`` must
            match this value or decoding fails — guards against a client
            reusing a page-2 cursor from one sort under a different sort.
            Cursors encoded without a ``sort`` field (there are none from
            this codebase, but defensively) are treated as ``"newest"``.

    Returns:
        ``(created_at, source, id)`` suitable for building a keyset WHERE clause.

    Raises:
        ValueError: If the cursor is malformed, missing required fields,
            ``source`` is not one of ``{"upload", "output"}``, or
            ``expected_sort`` is given and doesn't match the cursor's sort.
    """
    try:
        data = json.loads(base64.urlsafe_b64decode(cursor.encode()))
        created_at = datetime.fromisoformat(data["created_at"])
        source = data["source"]
        id_ = UUID(data["id"])
        sort = data.get("sort", "newest")
    except Exception as exc:
        raise ValueError(f"Invalid pagination cursor: {exc}") from exc

    if source not in _LIBRARY_CURSOR_SOURCES:
        raise ValueError(f"Invalid pagination cursor: unknown source {source!r}")

    if expected_sort is not None and sort != expected_sort:
        raise ValueError(
            f"Invalid pagination cursor: cursor was created for sort={sort!r}, "
            f"expected {expected_sort!r}"
        )

    return created_at, source, id_
