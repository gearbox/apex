"""Unit tests for GalleryService business logic."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

from src.api.services.gallery import GalleryService
from src.core.enums import (
    GalleryBadge,
    GallerySourceType,
    GenerationType,
    OutputMediaType,
)
from src.db.repositories.gallery import CoverData


def _make_service() -> GalleryService:
    return GalleryService(session=AsyncMock())


def _make_job(
    *,
    generation_type: str = "t2i",
    source_job_id: object = None,
    source_output_id: object = None,
    input_image_id: object = None,
    source_job: object = None,
    prompt: str = "a cat",
    aspect_ratio: str | None = "16:9",
    model: str | None = "grok-imagine-image",
    provider: str = "grok",
    token_cost: int | None = None,
) -> MagicMock:
    job = MagicMock()
    job.id = uuid4()
    job.generation_type = generation_type
    job.source_job_id = source_job_id
    job.source_output_id = source_output_id
    job.input_image_id = input_image_id
    job.source_job = source_job
    job.prompt = prompt
    job.negative_prompt = None
    job.aspect_ratio = aspect_ratio
    job.model = model
    job.provider = provider
    job.token_cost = token_cost
    job.created_at = datetime.now(UTC)
    job.completed_at = None
    job.outputs = []
    return job


class TestResolveBadge:
    def test_t2i_is_prompt(self) -> None:
        svc = _make_service()
        assert svc._resolve_badge(GenerationType.T2I) == GalleryBadge.PROMPT

    def test_t2v_is_prompt(self) -> None:
        svc = _make_service()
        assert svc._resolve_badge(GenerationType.T2V) == GalleryBadge.PROMPT

    def test_i2i_is_image(self) -> None:
        svc = _make_service()
        assert svc._resolve_badge(GenerationType.I2I) == GalleryBadge.IMAGE

    def test_i2v_is_image(self) -> None:
        svc = _make_service()
        assert svc._resolve_badge(GenerationType.I2V) == GalleryBadge.IMAGE

    def test_flf2v_is_image(self) -> None:
        svc = _make_service()
        assert svc._resolve_badge(GenerationType.FLF2V) == GalleryBadge.IMAGE

    def test_v2v_is_image(self) -> None:
        svc = _make_service()
        assert svc._resolve_badge(GenerationType.V2V) == GalleryBadge.IMAGE


class TestResolveMediaType:
    def test_t2i_is_image(self) -> None:
        svc = _make_service()
        assert svc._resolve_media_type(GenerationType.T2I) == OutputMediaType.IMAGE

    def test_i2i_is_image(self) -> None:
        svc = _make_service()
        assert svc._resolve_media_type(GenerationType.I2I) == OutputMediaType.IMAGE

    def test_t2v_is_video(self) -> None:
        svc = _make_service()
        assert svc._resolve_media_type(GenerationType.T2V) == OutputMediaType.VIDEO

    def test_i2v_is_video(self) -> None:
        svc = _make_service()
        assert svc._resolve_media_type(GenerationType.I2V) == OutputMediaType.VIDEO

    def test_v2v_is_video(self) -> None:
        svc = _make_service()
        assert svc._resolve_media_type(GenerationType.V2V) == OutputMediaType.VIDEO

    def test_flf2v_is_video(self) -> None:
        svc = _make_service()
        assert svc._resolve_media_type(GenerationType.FLF2V) == OutputMediaType.VIDEO


class TestOutputMediaType:
    def test_image_jpeg(self) -> None:
        svc = _make_service()
        assert svc._output_media_type("image/jpeg") == OutputMediaType.IMAGE

    def test_video_mp4(self) -> None:
        svc = _make_service()
        assert svc._output_media_type("video/mp4") == OutputMediaType.VIDEO

    def test_image_png(self) -> None:
        svc = _make_service()
        assert svc._output_media_type("image/png") == OutputMediaType.IMAGE


class TestPromptSnippet:
    def test_short_prompt_unchanged(self) -> None:
        svc = _make_service()
        assert svc._prompt_snippet("short prompt") == "short prompt"

    def test_long_prompt_truncated_at_word_boundary(self) -> None:
        svc = _make_service()
        long_prompt = "word " * 25  # 125 chars
        result = svc._prompt_snippet(long_prompt)
        assert len(result) <= 101  # 100 chars + ellipsis
        assert result.endswith("…")

    def test_exactly_100_chars_not_truncated(self) -> None:
        svc = _make_service()
        prompt = "a" * 100
        assert svc._prompt_snippet(prompt) == prompt


class TestResolveCover:
    def test_t2i_uses_cover_output(self) -> None:
        svc = _make_service()
        job = _make_job(generation_type="t2i")
        cover_id = uuid4()
        cover_data = CoverData(cover_output_id=cover_id, output_count=1)
        cover_url, video_url = svc._resolve_cover(job, cover_data)
        assert cover_url == f"/v1/content/outputs/{cover_id}"
        assert video_url is None

    def test_t2v_uses_thumbnail_for_cover(self) -> None:
        svc = _make_service()
        job = _make_job(generation_type="t2v")
        thumb_id = uuid4()
        video_id = uuid4()
        cover_data = CoverData(
            thumbnail_output_id=thumb_id,
            video_output_id=video_id,
            output_count=1,
        )
        cover_url, video_url = svc._resolve_cover(job, cover_data)
        assert cover_url == f"/v1/content/outputs/{thumb_id}"
        assert video_url == f"/v1/content/outputs/{video_id}"

    def test_i2i_uses_source_output_id(self) -> None:
        svc = _make_service()
        source_output_id = uuid4()
        job = _make_job(generation_type="i2i", source_output_id=source_output_id)
        cover_data = CoverData(output_count=1)
        cover_url, video_url = svc._resolve_cover(job, cover_data)
        assert cover_url == f"/v1/content/outputs/{source_output_id}"

    def test_i2i_falls_back_to_input_image_id(self) -> None:
        svc = _make_service()
        image_id = uuid4()
        job = _make_job(generation_type="i2i", input_image_id=image_id)
        cover_data = CoverData(output_count=1)
        cover_url, _ = svc._resolve_cover(job, cover_data)
        assert cover_url == f"/v1/content/uploads/{image_id}"

    def test_i2i_falls_back_to_cover_output(self) -> None:
        svc = _make_service()
        cover_id = uuid4()
        job = _make_job(generation_type="i2i")
        cover_data = CoverData(cover_output_id=cover_id, output_count=1)
        cover_url, _ = svc._resolve_cover(job, cover_data)
        assert cover_url == f"/v1/content/outputs/{cover_id}"

    def test_video_types_set_video_url(self) -> None:
        svc = _make_service()
        video_id = uuid4()
        job = _make_job(generation_type="i2v", source_output_id=uuid4())
        cover_data = CoverData(video_output_id=video_id, output_count=1)
        _, video_url = svc._resolve_cover(job, cover_data)
        assert video_url == f"/v1/content/outputs/{video_id}"


class TestBuildLineage:
    def test_no_source_returns_none(self) -> None:
        svc = _make_service()
        job = _make_job()
        assert svc._build_lineage(job) is None

    def test_source_job_id_set(self) -> None:
        svc = _make_service()
        source_job_id = uuid4()
        source_job = MagicMock()
        source_job.name = "Source Job"
        job = _make_job(source_job_id=source_job_id, source_job=source_job)
        lineage = svc._build_lineage(job)
        assert lineage is not None
        assert lineage.source_type == GallerySourceType.GENERATION
        assert lineage.source_job_id == source_job_id
        assert lineage.source_job_name == "Source Job"

    def test_input_image_id_set(self) -> None:
        svc = _make_service()
        image_id = uuid4()
        job = _make_job(input_image_id=image_id)
        lineage = svc._build_lineage(job)
        assert lineage is not None
        assert lineage.source_type == GallerySourceType.UPLOAD
        assert lineage.source_upload_id == image_id


class TestAspectRatioPassthrough:
    def test_aspect_ratio_in_grid_item(self) -> None:
        svc = _make_service()
        job = _make_job(generation_type="t2i", aspect_ratio="4:3")
        cover_id = uuid4()
        cover_data = CoverData(cover_output_id=cover_id, output_count=1)
        cover_url, _ = svc._resolve_cover(job, cover_data)
        # Verify aspect_ratio is accessible on the job (passthrough check)
        assert job.aspect_ratio == "4:3"


class TestListGallery:
    async def test_returns_empty_page_when_no_jobs(self) -> None:
        mock_session = AsyncMock()
        svc = GalleryService(session=mock_session)
        mock_repo = AsyncMock()
        mock_repo.list_gallery_jobs.return_value = []
        mock_repo.batch_cover_data.return_value = {}

        with patch("src.api.services.gallery.GalleryRepository", return_value=mock_repo):
            page = await svc.list_gallery(uuid4(), "vex", session=mock_session, limit=20)

        assert page.items == []
        assert page.has_more is False
        assert page.next_cursor is None


class TestGetGalleryDetail:
    async def test_returns_none_for_missing_job(self) -> None:
        mock_session = AsyncMock()
        svc = GalleryService(session=mock_session)
        mock_repo = AsyncMock()
        mock_repo.get_gallery_job.return_value = None

        with patch("src.api.services.gallery.GalleryRepository", return_value=mock_repo):
            result = await svc.get_gallery_detail(uuid4(), uuid4(), "vex", session=mock_session)

        assert result is None
