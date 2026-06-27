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
    source_output: object = None,
    input_image: object = None,
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
    job.source_output = source_output
    job.input_image = input_image
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


def _make_output_row(
    *,
    content_type: str = "image/png",
    width: int | None = 512,
    height: int | None = 512,
    size_bytes: int = 1024,
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


class TestBuildOutputItem:
    def test_image_output_no_derivatives(self) -> None:
        svc = _make_service()
        output = _make_output_row(content_type="image/png")
        output.is_thumbnail = False
        output.output_index = 0
        output.created_at = datetime.now(UTC)

        item = svc._build_output_item(output, [])

        assert item.media.media_type == OutputMediaType.IMAGE
        assert item.media.original.url == f"/v1/content/outputs/{output.id}"
        assert item.media.variants == []

    def test_video_output_with_webp_derivative(self) -> None:
        svc = _make_service()
        output = _make_output_row(content_type="video/mp4")
        output.is_thumbnail = False
        output.output_index = 0
        output.created_at = datetime.now(UTC)

        derivative = _make_output_row(
            content_type="image/webp",
            thumbnail_max_edge=512,
            width=400,
            height=225,
        )

        item = svc._build_output_item(output, [derivative])

        assert item.media.media_type == OutputMediaType.VIDEO
        assert len(item.media.variants) == 1
        assert item.media.variants[0].label == "md"
        assert item.media.variants[0].url == f"/v1/content/outputs/{derivative.id}"

    def test_output_index_preserved(self) -> None:
        svc = _make_service()
        output = _make_output_row(content_type="image/png")
        output.is_thumbnail = False
        output.output_index = 3
        output.created_at = datetime.now(UTC)

        item = svc._build_output_item(output, [])

        assert item.output_index == 3


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

    async def test_returns_page_with_jobs_and_cursor(self) -> None:
        mock_session = AsyncMock()
        svc = GalleryService(session=mock_session)
        mock_repo = AsyncMock()
        job = _make_job(generation_type="t2i")

        primary_output = _make_output_row(content_type="image/png")
        cover_data = CoverData(primary_output=primary_output, output_count=1)

        # Return limit+1 jobs to trigger has_more=True
        mock_repo.list_gallery_jobs.return_value = [job, job]
        mock_repo.batch_cover_data.return_value = {job.id: cover_data}

        with patch("src.api.services.gallery.GalleryRepository", return_value=mock_repo):
            page = await svc.list_gallery(uuid4(), "vex", session=mock_session, limit=1)

        assert len(page.items) == 1
        assert page.has_more is True
        assert page.next_cursor is not None

    async def test_uses_cursor_to_decode(self) -> None:
        from src.api.schemas.pagination import encode_cursor

        mock_session = AsyncMock()
        svc = GalleryService(session=mock_session)
        mock_repo = AsyncMock()
        mock_repo.list_gallery_jobs.return_value = []
        mock_repo.batch_cover_data.return_value = {}
        cursor = encode_cursor(datetime.now(UTC), uuid4())

        with patch("src.api.services.gallery.GalleryRepository", return_value=mock_repo):
            page = await svc.list_gallery(
                uuid4(), "vex", session=mock_session, limit=20, cursor=cursor
            )

        assert page.items == []
        mock_repo.list_gallery_jobs.assert_awaited_once()
        call_kwargs = mock_repo.list_gallery_jobs.call_args
        # Verify cursor was decoded and passed through
        assert call_kwargs.kwargs.get("cursor_ts") is not None or call_kwargs.args

    async def test_grid_item_cover_is_media_object(self) -> None:
        """Gallery grid cover is a MediaObject with original URL and optional variants."""
        mock_session = AsyncMock()
        svc = GalleryService(session=mock_session)
        mock_repo = AsyncMock()
        job = _make_job(generation_type="t2i")

        primary_output = _make_output_row(content_type="image/png")
        thumb = _make_output_row(content_type="image/webp", thumbnail_max_edge=512)
        cover_data = CoverData(
            primary_output=primary_output,
            primary_derivatives=[thumb],
            output_count=1,
        )

        mock_repo.list_gallery_jobs.return_value = [job]
        mock_repo.batch_cover_data.return_value = {job.id: cover_data}

        with patch("src.api.services.gallery.GalleryRepository", return_value=mock_repo):
            page = await svc.list_gallery(uuid4(), "vex", session=mock_session, limit=20)

        assert len(page.items) == 1
        cover = page.items[0].cover
        assert cover.original.url == f"/v1/content/outputs/{primary_output.id}"
        assert len(cover.variants) == 1
        assert cover.variants[0].label == "md"


class TestGetGalleryDetail:
    async def test_returns_none_for_missing_job(self) -> None:
        mock_session = AsyncMock()
        svc = GalleryService(session=mock_session)
        mock_repo = AsyncMock()
        mock_repo.get_gallery_job.return_value = None

        with patch("src.api.services.gallery.GalleryRepository", return_value=mock_repo):
            result = await svc.get_gallery_detail(uuid4(), uuid4(), "vex", session=mock_session)

        assert result is None

    async def test_returns_detail_with_no_input(self) -> None:
        mock_session = AsyncMock()
        svc = GalleryService(session=mock_session)
        mock_repo = AsyncMock()
        job = _make_job(generation_type="t2i", source_output=None, input_image=None)
        job.outputs = []
        mock_repo.get_gallery_job.return_value = job

        with patch("src.api.services.gallery.GalleryRepository", return_value=mock_repo):
            result = await svc.get_gallery_detail(uuid4(), uuid4(), "vex", session=mock_session)

        assert result is not None
        assert result.input_media is None

    async def test_returns_detail_for_found_job_with_source_output(self) -> None:
        mock_session = AsyncMock()
        svc = GalleryService(session=mock_session)
        mock_repo = AsyncMock()

        source_output = _make_output_row(content_type="image/png")
        source_output.derivatives = []

        job = _make_job(
            generation_type="i2i",
            source_output_id=source_output.id,
            source_output=source_output,
            input_image=None,
        )
        job.outputs = []
        mock_repo.get_gallery_job.return_value = job

        with patch("src.api.services.gallery.GalleryRepository", return_value=mock_repo):
            result = await svc.get_gallery_detail(uuid4(), uuid4(), "vex", session=mock_session)

        assert result is not None
        assert result.input_media is not None
        assert result.input_media.original.url == f"/v1/content/outputs/{source_output.id}"

    async def test_returns_detail_for_found_job_with_input_image(self) -> None:
        mock_session = AsyncMock()
        svc = GalleryService(session=mock_session)
        mock_repo = AsyncMock()

        input_image = _make_output_row(content_type="image/png")
        input_image.derivatives = []

        job = _make_job(
            generation_type="i2v",
            input_image_id=input_image.id,
            source_output=None,
            input_image=input_image,
        )
        job.outputs = []
        mock_repo.get_gallery_job.return_value = job

        with patch("src.api.services.gallery.GalleryRepository", return_value=mock_repo):
            result = await svc.get_gallery_detail(uuid4(), uuid4(), "vex", session=mock_session)

        assert result is not None
        assert result.input_media is not None
        assert result.input_media.original.url == f"/v1/content/uploads/{input_image.id}"
