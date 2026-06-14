"""Tests for gallery cover resolution with parent_output_id thumbnail linking."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from src.api.services.gallery import GalleryService
from src.db.repositories.gallery import CoverData, GalleryRepository

pytestmark = pytest.mark.unit


def _make_service() -> GalleryService:
    from unittest.mock import AsyncMock

    return GalleryService(session=AsyncMock())


def _make_job(*, generation_type: str = "t2i") -> MagicMock:
    job = MagicMock()
    job.id = uuid4()
    job.generation_type = generation_type
    job.source_output_id = None
    job.input_image_id = None
    job.prompt = "a cat"
    job.outputs = []
    return job


def _make_output(
    *,
    is_thumbnail: bool = False,
    parent_output_id: object = None,
    content_type: str = "image/png",
    output_index: int = 0,
    job_id: object = None,
) -> MagicMock:
    out = MagicMock()
    out.id = uuid4()
    out.is_thumbnail = is_thumbnail
    out.parent_output_id = parent_output_id
    out.content_type = content_type
    out.output_index = output_index
    out.job_id = job_id or uuid4()
    return out


# ---------------------------------------------------------------------------
# GalleryRepository.batch_cover_data — thumbnail index by parent_output_id
# ---------------------------------------------------------------------------


class TestBatchCoverDataThumbnailLink:
    async def test_thumbnail_linked_to_cover_via_parent_output_id(self) -> None:
        """thumbnail_output_id resolves to the thumbnail of the cover output."""
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
        assert data.cover_output_id == cover_id
        assert data.thumbnail_output_id == thumb_id
        assert data.output_count == 1

    async def test_legacy_thumbnail_fallback_when_no_parent_output_id(self) -> None:
        """Legacy thumbnail rows (parent_output_id=None) use fallback slot."""
        job_id = uuid4()
        cover_id = uuid4()
        thumb_id = uuid4()

        cover = _make_output(content_type="image/png", output_index=0, job_id=job_id)
        cover.id = cover_id
        # Legacy thumbnail without parent_output_id
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
        assert data.thumbnail_output_id == thumb_id  # fallback used

    async def test_no_thumbnail_returns_none(self) -> None:
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
        assert data.thumbnail_output_id is None
        assert data.cover_output_id == cover_id


# ---------------------------------------------------------------------------
# GalleryService._resolve_cover — t2i prefers thumbnail
# ---------------------------------------------------------------------------


class TestResolveCoverImagePrefersThumbnail:
    def test_t2i_prefers_thumbnail_url_over_full_cover(self) -> None:
        svc = _make_service()
        cover_id = uuid4()
        thumb_id = uuid4()

        job = _make_job(generation_type="t2i")
        cover_data = CoverData(
            cover_output_id=cover_id,
            thumbnail_output_id=thumb_id,
        )
        url, video_url = svc._resolve_cover(job, cover_data)
        assert f"/v1/content/outputs/{thumb_id}" == url
        assert video_url is None

    def test_t2i_falls_back_to_full_cover_when_no_thumbnail(self) -> None:
        svc = _make_service()
        cover_id = uuid4()

        job = _make_job(generation_type="t2i")
        cover_data = CoverData(cover_output_id=cover_id, thumbnail_output_id=None)
        url, video_url = svc._resolve_cover(job, cover_data)
        assert f"/v1/content/outputs/{cover_id}" == url

    def test_t2i_fallback_unknown_when_no_cover_and_no_thumbnail(self) -> None:
        svc = _make_service()
        job = _make_job(generation_type="t2i")
        cover_data = CoverData()
        url, video_url = svc._resolve_cover(job, cover_data)
        assert "unknown" in url


# ---------------------------------------------------------------------------
# GalleryService.get_gallery_detail — thumbnail_map by parent_output_id
# ---------------------------------------------------------------------------


class TestGalleryDetailThumbnailMap:
    def test_build_output_item_attaches_thumbnail(self) -> None:
        svc = _make_service()
        full_id = uuid4()
        thumb_id = uuid4()

        full_out = _make_output(content_type="image/png", output_index=0)
        full_out.id = full_id
        full_out.size_bytes = 1024
        full_out.format = "png"
        full_out.created_at = __import__("datetime").datetime.now(__import__("datetime").UTC)

        thumb_out = _make_output(
            is_thumbnail=True,
            parent_output_id=full_id,
            content_type="image/webp",
        )
        thumb_out.id = thumb_id

        item = svc._build_output_item(full_out, thumb_out)
        assert item.thumbnail_url == f"/v1/content/outputs/{thumb_id}"

    def test_build_output_item_no_thumbnail_when_none_passed(self) -> None:
        svc = _make_service()
        full_out = _make_output(content_type="image/png", output_index=0)
        full_out.size_bytes = 1024
        full_out.format = "png"
        full_out.created_at = __import__("datetime").datetime.now(__import__("datetime").UTC)

        item = svc._build_output_item(full_out, None)
        assert item.thumbnail_url is None
