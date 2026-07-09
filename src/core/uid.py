# src/core/uid.py
"""Centralised UUID generation for the Apex project.

All application-generated primary keys and identifiers must go through
``new_id()``. This ensures a consistent UUID version across the codebase
and makes a future version upgrade a one-line change.

UUIDv7 properties vs UUIDv4:
- Time-ordered (ms-precision Unix timestamp in high bits) → append-like
  B-tree inserts, lower index fragmentation on write-heavy tables.
- Implicit creation-time ordering when sorting by PK.
- Still globally unique (74 random bits).
- No ``gen_random_uuid()`` server default required.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from uuid6 import uuid7

if TYPE_CHECKING:
    from uuid import UUID


def new_id() -> UUID:
    """Return a new time-ordered UUIDv7.

    Use this everywhere a new primary key or opaque identifier is needed.
    Never call ``uuid.uuid4()`` directly in application code.
    """
    return uuid7()  # type: ignore[return-value, unused-ignore]
