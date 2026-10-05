"""Frame lineage for client-extracted video frames uploaded via ``POST /v1/storage/upload``.

Pure module: no DB, no Litestar imports. The browser decodes a video it already has
access to, captures a frame, and uploads it with two optional multipart fields naming
the source asset and the capture timestamp. This module parses and bounds-checks those
raw form strings; ownership/product/media-kind validation of the *source* is the
service's job (``UserContentService``), via ``SourceMediaResolver``.

Every failure here is surfaced publicly as a single ``invalid_frame_lineage`` error —
the specific ``reason`` is for server-side logs only, so the response is never an
existence oracle.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum

from src.core.library_ref import AssetRef, parse_asset_ref

# ``user_images.source_timestamp_ms`` is a signed 32-bit ``Integer`` column.
MAX_SOURCE_TIMESTAMP_MS = 2_147_483_647

# ASCII digits only: ``\d`` would also accept other Unicode digits that ``int()`` parses.
# ``fullmatch`` (not ``$``) so a trailing newline is not tolerated.
_TIMESTAMP_RE = re.compile(r"[0-9]{1,10}")


class FrameLineageReason(StrEnum):
    """Why a frame lineage was rejected. Logged server-side; never sent to the client."""

    PARTIAL_FIELDS = "partial_fields"
    MALFORMED_REF = "malformed_ref"
    MALFORMED_TIMESTAMP = "malformed_timestamp"
    VIDEO_FILE = "video_file"
    SOURCE_UNAVAILABLE = "source_unavailable"
    SOURCE_NOT_VIDEO = "source_not_video"
    SOURCE_DURATION_UNKNOWN = "source_duration_unknown"
    TIMESTAMP_OUT_OF_RANGE = "timestamp_out_of_range"


class FrameLineageError(ValueError):
    """Malformed or partial frame lineage fields."""

    def __init__(self, reason: FrameLineageReason) -> None:
        super().__init__(reason.value)
        self.reason = reason


@dataclass(frozen=True, slots=True)
class FrameLineage:
    """Validated-in-shape lineage of an uploaded frame: its source asset and capture time."""

    source: AssetRef
    timestamp_ms: int


def parse_frame_lineage(asset_ref: str | None, timestamp_ms: str | None) -> FrameLineage | None:
    """Parse the two optional lineage form fields.

    Args:
        asset_ref: Raw ``source_asset_ref`` field (``upload:<uuid>`` / ``output:<uuid>``).
        timestamp_ms: Raw ``source_timestamp_ms`` field (decimal integer string).

    Returns:
        ``None`` when both fields are absent (an ordinary upload), else the parsed lineage.

    Raises:
        FrameLineageError: Exactly one field present, a malformed reference, or a timestamp
            that is not a plain non-negative decimal integer within the column range.
    """
    if asset_ref is None and timestamp_ms is None:
        return None
    if asset_ref is None or timestamp_ms is None:
        raise FrameLineageError(FrameLineageReason.PARTIAL_FIELDS)

    try:
        source = parse_asset_ref(asset_ref)
    except ValueError as exc:
        raise FrameLineageError(FrameLineageReason.MALFORMED_REF) from exc

    if _TIMESTAMP_RE.fullmatch(timestamp_ms) is None:
        raise FrameLineageError(FrameLineageReason.MALFORMED_TIMESTAMP)
    parsed = int(timestamp_ms)
    if parsed > MAX_SOURCE_TIMESTAMP_MS:
        raise FrameLineageError(FrameLineageReason.MALFORMED_TIMESTAMP)

    return FrameLineage(source=source, timestamp_ms=parsed)
