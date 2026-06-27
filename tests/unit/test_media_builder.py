"""Unit tests for the pure MediaObject builder."""

from __future__ import annotations

from unittest.mock import MagicMock
from uuid import uuid4

import pytest

from src.api.services.media import (
    OUTPUT_PREFIX,
    UPLOAD_PREFIX,
    build_output_media,
    build_upload_media,
)
from src.core.enums import OutputMediaType

pytestmark = pytest.mark.unit


def _make_upload_row(
    *,
    content_type: str = "image/png",
    width: int | None = 1024,
    height: int | None = 768,
    size_bytes: int = 500_000,
    thumbnail_max_edge: int | None = None,
    is_thumbnail: bool = False,
) -> MagicMock:
    row = MagicMock()
    row.id = uuid4()
    row.content_type = content_type
    row.width = width
    row.height = height
    row.size_bytes = size_bytes
    row.thumbnail_max_edge = thumbnail_max_edge
    row.is_thumbnail = is_thumbnail
    return row


def _make_output_row(
    *,
    content_type: str = "image/png",
    width: int | None = 1024,
    height: int | None = 768,
    size_bytes: int = 500_000,
    thumbnail_max_edge: int | None = None,
) -> MagicMock:
    row = MagicMock()
    row.id = uuid4()
    row.content_type = content_type
    row.width = width
    row.height = height
    row.size_bytes = size_bytes
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
