"""Pure parsing of the optional ``source_asset_ref`` / ``source_timestamp_ms`` upload fields."""

from __future__ import annotations

from uuid import UUID

import pytest

from src.api.services.frame_lineage import (
    MAX_SOURCE_TIMESTAMP_MS,
    FrameLineage,
    FrameLineageError,
    FrameLineageReason,
    parse_frame_lineage,
)
from src.core.library_ref import AssetRef, LibraryAssetSource

pytestmark = pytest.mark.unit

# Fixed (not uuid4): parametrize ids must be identical across xdist workers.
_UPLOAD_ID = UUID("0f6c1c3e-6a54-4d0e-9d57-3c2d9b6f7a11")
_OUTPUT_ID = UUID("7b2f9e44-1d3a-4c58-8a9e-5e0c4d7b2f22")


def test_both_absent_is_an_ordinary_upload() -> None:
    assert parse_frame_lineage(None, None) is None


@pytest.mark.parametrize(
    ("raw_ref", "source", "asset_id"),
    [
        (f"upload:{_UPLOAD_ID}", LibraryAssetSource.UPLOAD, _UPLOAD_ID),
        (f"output:{_OUTPUT_ID}", LibraryAssetSource.OUTPUT, _OUTPUT_ID),
    ],
)
def test_valid_lineage(raw_ref: str, source: LibraryAssetSource, asset_id: object) -> None:
    assert parse_frame_lineage(raw_ref, "1500") == FrameLineage(
        source=AssetRef(source=source, asset_id=asset_id),  # type: ignore[arg-type]
        timestamp_ms=1500,
    )


@pytest.mark.parametrize("timestamp", ["0", "00012", str(MAX_SOURCE_TIMESTAMP_MS)])
def test_timestamp_bounds_accepted(timestamp: str) -> None:
    lineage = parse_frame_lineage(f"upload:{_UPLOAD_ID}", timestamp)
    assert lineage is not None
    assert lineage.timestamp_ms == int(timestamp)


@pytest.mark.parametrize(
    ("asset_ref", "timestamp"),
    [(f"upload:{_UPLOAD_ID}", None), (None, "100")],
)
def test_exactly_one_field_is_partial(asset_ref: str | None, timestamp: str | None) -> None:
    with pytest.raises(FrameLineageError) as exc:
        parse_frame_lineage(asset_ref, timestamp)
    assert exc.value.reason is FrameLineageReason.PARTIAL_FIELDS


@pytest.mark.parametrize(
    "asset_ref",
    [
        "",
        str(_UPLOAD_ID),  # bare uuid: never a try-both-tables lookup
        f"image:{_UPLOAD_ID}",
        "upload:not-a-uuid",
        f"upload:{_UPLOAD_ID}:extra",
        "upload:",
        ":",
    ],
)
def test_malformed_asset_ref(asset_ref: str) -> None:
    with pytest.raises(FrameLineageError) as exc:
        parse_frame_lineage(asset_ref, "100")
    assert exc.value.reason is FrameLineageReason.MALFORMED_REF


@pytest.mark.parametrize(
    "timestamp",
    [
        "",
        " 12",
        "12 ",
        "12\n",
        "-0",
        "-1",
        "+1",
        "12.0",
        "1e3",
        "0x10",
        "1_000",
        "٣",  # Arabic-Indic digit: ``int()`` would parse it, a ``\d`` regex would let it through
        "12345678901",  # 11 digits
        str(MAX_SOURCE_TIMESTAMP_MS + 1),
        "9999999999",
    ],
)
def test_malformed_timestamp(timestamp: str) -> None:
    with pytest.raises(FrameLineageError) as exc:
        parse_frame_lineage(f"upload:{_UPLOAD_ID}", timestamp)
    assert exc.value.reason is FrameLineageReason.MALFORMED_TIMESTAMP
