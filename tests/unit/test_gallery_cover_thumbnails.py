"""Tests for gallery cover resolution with parent_output_id thumbnail linking."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from src.api.services.gallery import GalleryService
from src.api.services.media import build_output_media
from src.db.repositories.gallery import GalleryRepository

pytestmark = pytest.mark.unit


def _make_service() -> GalleryService:
    return GalleryService(session=AsyncMock())


def _make_output(
    *,
    is_thumbnail: bool = False,
    parent_output_id: object = None,
    content_type: str = "image/png",
    output_index: int = 0,
    job_id: object = None,
    thumbnail_max_edge: int | None = None,
    width: int | None = 512,
    height: int | None = 512,
    size_bytes: int = 1024,
) -> MagicMock:
    out = MagicMock()
    out.id = uuid4()
    out.is_thumbnail = is_thumbnail
    out.parent_output_id = parent_output_id
    out.content_type = content_type
    out.output_index = output_index
    out.job_id = job_id or uuid4()
    out.thumbnail_max_edge = thumbnail_max_edge
    out.width = width
    out.height = height
    out.size_bytes = size_bytes
    out.created_at = __import__("datetime").datetime.now(__import__("datetime").UTC)
    out.expires_at = out.created_at + __import__("datetime").timedelta(days=7)
    return out


# ---------------------------------------------------------------------------
# GalleryRepository.batch_cover_data — new CoverData shape
# ---------------------------------------------------------------------------


class TestBatchCoverDataThumbnailLink:
    async def test_primary_output_is_first_non_thumbnail(self) -> None:
        """primary_output resolves to the first full (non-thumbnail) output."""
        job_id = uuid4()
        cover_id = uuid4()
        thumb_id = uuid4()

        cover = _make_output(content_type="image/png", output_index=0, job_id=job_id)
        cover.id = cover_id
        thumb = _make_output(
            is_thumbnail=True,
            parent_output_id=cover_id,
            content_type="image/webp",
            output_index=0,
            job_id=job_id,
        )
        thumb.id = thumb_id

        session = AsyncMock()

        async def fake_execute(_stmt: object) -> MagicMock:
            result = MagicMock()
            result.scalars.return_value.all.return_value = [cover, thumb]
            return result

        session.execute = fake_execute
        repo = GalleryRepository(session)
        cover_map = await repo.batch_cover_data([job_id])

        data = cover_map[job_id]
        assert data.primary_output is cover
        assert len(data.primary_derivatives) == 1
        assert data.primary_derivatives[0] is thumb
        assert data.output_count == 1

    async def test_thumbnail_without_parent_output_id_not_in_derivatives(self) -> None:
        """Legacy thumbnails (parent_output_id=None) do not appear in primary_derivatives."""
        job_id = uuid4()
        cover_id = uuid4()
        thumb_id = uuid4()

        cover = _make_output(content_type="image/png", output_index=0, job_id=job_id)
        cover.id = cover_id
        # Legacy thumbnail without a parent link
        thumb = _make_output(is_thumbnail=True, parent_output_id=None, job_id=job_id)
        thumb.id = thumb_id

        session = MagicMock()

        async def fake_execute(_stmt: object) -> MagicMock:
            result = MagicMock()
            result.scalars.return_value.all.return_value = [cover, thumb]
            return result

        session.execute = fake_execute
        repo = GalleryRepository(session)
        cover_map = await repo.batch_cover_data([job_id])

        data = cover_map[job_id]
        assert data.primary_output is cover
        assert data.primary_derivatives == []  # legacy thumb not linked to cover

    async def test_no_thumbnail_returns_empty_derivatives(self) -> None:
        job_id = uuid4()
        cover_id = uuid4()
        cover = _make_output(content_type="image/png", output_index=0, job_id=job_id)
        cover.id = cover_id

        session = MagicMock()

        async def fake_execute(_stmt: object) -> MagicMock:
            result = MagicMock()
            result.scalars.return_value.all.return_value = [cover]
            return result

        session.execute = fake_execute
        repo = GalleryRepository(session)
        cover_map = await repo.batch_cover_data([job_id])

        data = cover_map[job_id]
        assert data.primary_output is cover
        assert data.primary_derivatives == []
        assert data.output_count == 1


# ---------------------------------------------------------------------------
# build_output_media — cover MediaObject variant linking
# ---------------------------------------------------------------------------


class TestBuildMediaVariantsFromCoverData:
    def test_derivatives_appear_as_variants(self) -> None:
        cover = _make_output(thumbnail_max_edge=None)
        thumb = _make_output(
            is_thumbnail=True,
            parent_output_id=cover.id,
            content_type="image/webp",
            thumbnail_max_edge=512,
            width=400,
            height=400,
        )

        media = build_output_media(cover, [thumb])
        assert len(media.variants) == 1
        assert media.variants[0].label == "md"
        assert media.variants[0].url == f"/v1/content/outputs/{thumb.id}"
        assert media.variants[0].width == 400

    def test_no_derivatives_returns_empty_variants(self) -> None:
        cover = _make_output(thumbnail_max_edge=None)

        media = build_output_media(cover, [])
        assert media.variants == []
        assert media.original.url == f"/v1/content/outputs/{cover.id}"

    def test_multiple_sizes_sorted_by_width(self) -> None:
        cover = _make_output(thumbnail_max_edge=None)
        sm_thumb = _make_output(
            is_thumbnail=True,
            content_type="image/webp",
            thumbnail_max_edge=150,
            width=100,
        )
        md_thumb = _make_output(
            is_thumbnail=True,
            content_type="image/webp",
            thumbnail_max_edge=512,
            width=400,
        )

        media = build_output_media(cover, [md_thumb, sm_thumb])  # intentionally reversed
        assert len(media.variants) == 2
        assert media.variants[0].label == "sm"  # sorted by width asc
        assert media.variants[1].label == "md"


# ---------------------------------------------------------------------------
# GalleryService._build_output_item — new list[derivative] signature
# ---------------------------------------------------------------------------


class TestGalleryDetailThumbnailMap:
    def test_build_output_item_attaches_variants(self) -> None:
        svc = _make_service()
        full_out = _make_output(content_type="image/png", output_index=0, thumbnail_max_edge=None)
        thumb_out = _make_output(
            is_thumbnail=True,
            parent_output_id=full_out.id,
            content_type="image/webp",
            thumbnail_max_edge=512,
            width=400,
        )

        item = svc._build_output_item(full_out, [thumb_out])
        assert item.output_index == 0
        assert len(item.media.variants) == 1
        assert item.media.variants[0].url == f"/v1/content/outputs/{thumb_out.id}"
        assert item.media.original.url == f"/v1/content/outputs/{full_out.id}"

    def test_build_output_item_empty_derivatives_gives_no_variants(self) -> None:
        svc = _make_service()
        full_out = _make_output(content_type="image/png", output_index=2, thumbnail_max_edge=None)

        item = svc._build_output_item(full_out, [])
        assert item.output_index == 2
        assert item.media.variants == []
        assert item.media.original.url == f"/v1/content/outputs/{full_out.id}"
