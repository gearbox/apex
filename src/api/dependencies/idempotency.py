"""Idempotency key extraction dependency."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from litestar.connection import ASGIConnection


def get_idempotency_key(connection: ASGIConnection) -> str:  # type: ignore[type-arg]
    """Extract Idempotency-Key header.

    Raises:
        ValueError: If header is missing or exceeds 64 characters.
    """
    key = connection.headers.get("idempotency-key")
    if not key:
        raise ValueError("Idempotency-Key header is required")
    if len(key) > 64:
        raise ValueError("Idempotency-Key must be 64 characters or fewer")
    return key
