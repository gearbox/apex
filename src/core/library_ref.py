"""Library asset reference — typed, parseable identity for uploads and outputs.

Pure module: no DB, no Litestar imports. An ``AssetRef`` uniquely identifies
a single row in either ``user_images`` or ``generation_outputs`` without the
caller needing to know which table ahead of time.
"""

from __future__ import annotations

from enum import StrEnum
from uuid import UUID

import msgspec


class LibraryAssetSource(StrEnum):
    """Which table a library asset lives in."""

    UPLOAD = "upload"
    OUTPUT = "output"


class AssetRef(msgspec.Struct, frozen=True):
    """Typed identity for a single library asset: source table + row id."""

    source: LibraryAssetSource
    asset_id: UUID


def parse_asset_ref(raw: str) -> AssetRef:
    """Parse a wire-format asset reference (``"upload:<uuid>"`` / ``"output:<uuid>"``).

    Splits on the first ``:`` only, so the UUID segment must not itself
    contain a colon — a malformed or ambiguous reference is rejected rather
    than silently truncated.

    Args:
        raw: Wire-format asset reference string.

    Returns:
        Parsed AssetRef.

    Raises:
        ValueError: If the reference is malformed, the source is unknown,
            or the UUID segment is invalid.
    """
    parts = raw.split(":", 1)
    if len(parts) != 2:
        raise ValueError("Invalid asset reference")

    source_raw, asset_id_raw = parts
    if not source_raw or not asset_id_raw:
        raise ValueError("Invalid asset reference")
    if ":" in asset_id_raw:
        raise ValueError("Invalid asset reference")

    try:
        source = LibraryAssetSource(source_raw)
    except ValueError as exc:
        raise ValueError("Invalid asset reference") from exc

    try:
        asset_id = UUID(asset_id_raw)
    except ValueError as exc:
        raise ValueError("Invalid asset reference") from exc

    return AssetRef(source=source, asset_id=asset_id)


def format_asset_ref(source: LibraryAssetSource, asset_id: UUID) -> str:
    """Format a source + id pair into the wire-format asset reference string."""
    return f"{source}:{asset_id}"
