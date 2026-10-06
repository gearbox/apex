"""Unit tests for the pure MediaObject builder."""

from __future__ import annotations

from unittest.mock import MagicMock
from uuid import uuid4

import pytest

from src.api.services.library import _build_media_object
from src.api.services.media import (
    OUTPUT_PREFIX,
    UPLOAD_PREFIX,
    build_output_media,
    build_upload_media,
)
from src.core.enums import OutputMediaType
from src.core.library_ref import LibraryAssetSource

pytestmark = pytest.mark.unit


def _make_upload_row(
    *,
    content_type: str = "image/png",
    width: int | None = 1024,
    height: int | None = 768,
    size_bytes: int = 500_000,
    duration_ms: int | None = None,
    thumbnail_max_edge: int | None = None,
    is_thumbnail: bool = False,
) -> MagicMock:
    row = MagicMock()
    row.id = uuid4()
    row.content_type = content_type
    row.width = width
    row.height = height
    row.size_bytes = size_bytes
    row.duration_ms = duration_ms
    row.thumbnail_max_edge = thumbnail_max_edge
    row.is_thumbnail = is_thumbnail
    return row


def _make_output_row(
    *,
    content_type: str = "image/png",
    width: int | None = 1024,
    height: int | None = 768,
    size_bytes: int = 500_000,
    duration_ms: int | None = None,
    thumbnail_max_edge: int | None = None,
) -> MagicMock:
    row = MagicMock()
    row.id = uuid4()
    row.content_type = content_type
    row.width = width
    row.height = height
    row.size_bytes = size_bytes
    row.duration_ms = duration_ms
    row.thumbnail_max_edge = thumbnail_max_edge
    return row


class TestBuildUploadMedia:
    def test_original_url_uses_upload_prefix(self) -> None:
        full = _make_upload_row()
        media = build_upload_media(full, [])
        assert media.original.url == f"{UPLOAD_PREFIX}/{full.id}"

    def test_variant_url_uses_upload_prefix(self) -> None:
        full = _make_upload_row()
        sm = _make_upload_row(thumbnail_max_edge=150)
        media = build_upload_media(full, [sm])
        assert len(media.variants) == 1
        assert media.variants[0].url == f"{UPLOAD_PREFIX}/{sm.id}"

    def test_image_content_type_yields_image_media_type(self) -> None:
        full = _make_upload_row(content_type="image/png")
        media = build_upload_media(full, [])
        assert media.media_type == OutputMediaType.IMAGE

    def test_video_content_type_yields_video_media_type(self) -> None:
        full = _make_upload_row(content_type="video/mp4")
        media = build_upload_media(full, [])
        assert media.media_type == OutputMediaType.VIDEO

    def test_empty_derivatives_yields_no_variants(self) -> None:
        full = _make_upload_row()
        media = build_upload_media(full, [])
        assert media.variants == []

    def test_unknown_thumbnail_max_edge_omitted(self) -> None:
        full = _make_upload_row()
        # thumbnail_max_edge=999 is not a known label → omitted
        unknown = _make_upload_row(thumbnail_max_edge=999)
        media = build_upload_media(full, [unknown])
        assert media.variants == []

    def test_none_thumbnail_max_edge_omitted(self) -> None:
        full = _make_upload_row()
        no_edge = _make_upload_row(thumbnail_max_edge=None)
        media = build_upload_media(full, [no_edge])
        assert media.variants == []

    def test_variants_sorted_ascending_by_width(self) -> None:
        full = _make_upload_row()
        md = _make_upload_row(thumbnail_max_edge=512, width=512, height=384)
        sm = _make_upload_row(thumbnail_max_edge=150, width=150, height=113)
        media = build_upload_media(full, [md, sm])
        assert len(media.variants) == 2
        assert media.variants[0].width == 150
        assert media.variants[1].width == 512

    def test_original_carries_width_height(self) -> None:
        full = _make_upload_row(width=1920, height=1080)
        media = build_upload_media(full, [])
        assert media.original.width == 1920
        assert media.original.height == 1080

    def test_original_none_dimensions_passed_through(self) -> None:
        full = _make_upload_row(width=None, height=None)
        media = build_upload_media(full, [])
        assert media.original.width is None
        assert media.original.height is None


class TestBuildOutputMedia:
    def test_original_url_uses_output_prefix(self) -> None:
        full = _make_output_row()
        media = build_output_media(full, [])
        assert media.original.url == f"{OUTPUT_PREFIX}/{full.id}"

    def test_variant_url_uses_output_prefix(self) -> None:
        full = _make_output_row()
        sm = _make_output_row(thumbnail_max_edge=150)
        media = build_output_media(full, [sm])
        assert media.variants[0].url == f"{OUTPUT_PREFIX}/{sm.id}"

    def test_image_content_type_yields_image_media_type(self) -> None:
        full = _make_output_row(content_type="image/webp")
        media = build_output_media(full, [])
        assert media.media_type == OutputMediaType.IMAGE

    def test_video_content_type_yields_video_media_type(self) -> None:
        full = _make_output_row(content_type="video/mp4")
        media = build_output_media(full, [])
        assert media.media_type == OutputMediaType.VIDEO

    def test_variants_sorted_ascending_by_width(self) -> None:
        full = _make_output_row()
        md = _make_output_row(thumbnail_max_edge=512, width=512, height=288)
        sm = _make_output_row(thumbnail_max_edge=150, width=150, height=84)
        media = build_output_media(full, [md, sm])
        assert media.variants[0].width == 150
        assert media.variants[1].width == 512


class TestVariantMissingDims:
    def test_missing_width_skips_variant_and_logs(self) -> None:
        import structlog.testing

        full = _make_output_row()
        missing_width = _make_output_row(thumbnail_max_edge=150, width=None)
        with structlog.testing.capture_logs() as cap:
            media = build_output_media(full, [missing_width])

        assert media.variants == []
        assert len(cap) == 1
        assert cap[0]["event"] == "media.variant.missing_dims"
        assert cap[0]["log_level"] == "warning"

    def test_missing_height_skips_variant_and_logs(self) -> None:
        import structlog.testing

        full = _make_output_row()
        missing_height = _make_output_row(thumbnail_max_edge=150, width=150, height=None)
        with structlog.testing.capture_logs() as cap:
            media = build_output_media(full, [missing_height])

        assert media.variants == []
        assert len(cap) == 1
        assert cap[0]["event"] == "media.variant.missing_dims"

    def test_valid_variant_dims_are_int(self) -> None:
        full = _make_output_row()
        sm = _make_output_row(thumbnail_max_edge=150, width=150, height=100)
        media = build_output_media(full, [sm])

        assert len(media.variants) == 1
        assert isinstance(media.variants[0].width, int)
        assert isinstance(media.variants[0].height, int)


class TestOriginalDurationMs:
    """I3 (unit) — ``MediaOriginal.duration_ms`` mirrors the row; images serialize ``null``."""

    def test_upload_video_carries_row_duration(self) -> None:
        full = _make_upload_row(content_type="video/mp4", duration_ms=8_000)
        assert build_upload_media(full, []).original.duration_ms == 8_000

    def test_output_video_carries_row_duration(self) -> None:
        full = _make_output_row(content_type="video/mp4", duration_ms=1_234)
        assert build_output_media(full, []).original.duration_ms == 1_234

    def test_images_have_no_duration(self) -> None:
        assert build_upload_media(_make_upload_row(), []).original.duration_ms is None
        assert build_output_media(_make_output_row(), []).original.duration_ms is None

    def test_legacy_video_row_without_duration_is_null(self) -> None:
        full = _make_upload_row(content_type="video/mp4", duration_ms=None)
        assert build_upload_media(full, []).original.duration_ms is None

    def test_image_serializes_explicit_null(self) -> None:
        import msgspec

        original = build_upload_media(_make_upload_row(), []).original
        assert msgspec.json.decode(msgspec.json.encode(original))["duration_ms"] is None

    @pytest.mark.parametrize(
        ("source", "duration_ms"),
        [
            (LibraryAssetSource.UPLOAD, 5_000),
            (LibraryAssetSource.OUTPUT, 5_000),
            (LibraryAssetSource.UPLOAD, None),
        ],
    )
    def test_library_builder_threads_duration(
        self, source: LibraryAssetSource, duration_ms: int | None
    ) -> None:
        media = _build_media_object(
            source=source,
            asset_id=uuid4(),
            width=1,
            height=1,
            content_type="video/mp4" if duration_ms else "image/png",
            size_bytes=1,
            duration_ms=duration_ms,
            derivatives=[],
        )
        assert media.original.duration_ms == duration_ms
