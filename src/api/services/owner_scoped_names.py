"""Shared normalization for user-owned, case-insensitively-unique names.

Used by both LibraryProjectService (max_length=100) and LibraryTagService
(max_length=50) — same trim/collapse/length-recheck logic, different bound.
"""

from __future__ import annotations


def normalize_owner_scoped_name(raw: str, *, max_length: int) -> str:
    """Trim, collapse inner whitespace; raise ValueError if empty or > max_length.

    msgspec's ``min_length``/``max_length`` constraints validate the raw wire
    value — a string of all whitespace passes that check but normalizes to
    empty, so length is re-checked here after normalization.

    Args:
        raw: Raw name as received from the client.
        max_length: Maximum allowed length after normalization.

    Returns:
        The normalized name.

    Raises:
        ValueError: If the normalized name is empty or exceeds ``max_length``.
    """
    normalized = " ".join(raw.split())
    if not (1 <= len(normalized) <= max_length):
        raise ValueError(f"name must be 1-{max_length} characters after trimming whitespace")
    return normalized
