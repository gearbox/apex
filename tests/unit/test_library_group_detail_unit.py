"""Unit tests for LibraryService.get_group_detail and its private helpers.

Ported from the now-removed test_gallery_service.py (TestResolveBadge,
TestBuildLineage, TestBuildOutputItem, TestGetGalleryDetail) — LibraryService's
group-detail path was ported byte-for-byte from GalleryService.get_gallery_detail,
so this file preserves the same behavioral coverage against the Library types.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

from src.api.services.library import LibraryService
from src.core.enums import GenerationType, LibraryBadge, LibraryGroupSourceType


def _make_service() -> LibraryService:
    return LibraryService(session=AsyncMock())


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
    row.expires_at = datetime.now(UTC) + timedelta(days=7)
    return row


class TestResolveGroupBadge:
    def test_t2i_is_prompt(self) -> None:
        svc = _make_service()
        assert svc._resolve_group_badge(GenerationType.T2I) == LibraryBadge.PROMPT

    def test_t2v_is_prompt(self) -> None:
        svc = _make_service()
        assert svc._resolve_group_badge(GenerationType.T2V) == LibraryBadge.PROMPT

    def test_i2i_is_image(self) -> None:
        svc = _make_service()
        assert svc._resolve_group_badge(GenerationType.I2I) == LibraryBadge.IMAGE

    def test_i2v_is_image(self) -> None:
        svc = _make_service()
        assert svc._resolve_group_badge(GenerationType.I2V) == LibraryBadge.IMAGE

    def test_flf2v_is_image(self) -> None:
        svc = _make_service()
        assert svc._resolve_group_badge(GenerationType.FLF2V) == LibraryBadge.IMAGE

    def test_v2v_is_image(self) -> None:
        svc = _make_service()
        assert svc._resolve_group_badge(GenerationType.V2V) == LibraryBadge.IMAGE


class TestBuildGroupLineage:
    def test_no_source_returns_none(self) -> None:
        svc = _make_service()
        job = _make_job()
        assert svc._build_group_lineage(job) is None

    def test_source_job_id_set(self) -> None:
        svc = _make_service()
        source_job_id = uuid4()
        source_job = MagicMock()
        source_job.name = "Source Job"
        job = _make_job(source_job_id=source_job_id, source_job=source_job)
        lineage = svc._build_group_lineage(job)
        assert lineage is not None
        assert lineage.source_type == LibraryGroupSourceType.OUTPUT
        assert lineage.source_job_id == source_job_id
        assert lineage.source_job_name == "Source Job"

    def test_input_image_id_set(self) -> None:
        svc = _make_service()
        image_id = uuid4()
        job = _make_job(input_image_id=image_id)
        lineage = svc._build_group_lineage(job)
        assert lineage is not None
        assert lineage.source_type == LibraryGroupSourceType.UPLOAD
        assert lineage.source_upload_id == image_id


class TestBuildGroupOutputItem:
    def test_image_output_no_derivatives(self) -> None:
        svc = _make_service()
        output = _make_output_row(content_type="image/png")
        output.is_thumbnail = False
        output.output_index = 0
        output.created_at = datetime.now(UTC)

        item = svc._build_group_output_item(output, [])

        assert item.media.original.url == f"/v1/content/outputs/{output.id}"
        assert item.media.variants == []
        assert item.asset_ref == f"output:{output.id}"

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

        item = svc._build_group_output_item(output, [derivative])

        assert len(item.media.variants) == 1
        assert item.media.variants[0].label == "md"
        assert item.media.variants[0].url == f"/v1/content/outputs/{derivative.id}"

    def test_output_index_preserved(self) -> None:
        svc = _make_service()
        output = _make_output_row(content_type="image/png")
        output.is_thumbnail = False
        output.output_index = 3
        output.created_at = datetime.now(UTC)

        item = svc._build_group_output_item(output, [])

        assert item.output_index == 3

    def test_expires_at_propagated(self) -> None:
        svc = _make_service()
        output = _make_output_row(content_type="image/png")
        output.is_thumbnail = False
        output.output_index = 0
        output.created_at = datetime.now(UTC)

        item = svc._build_group_output_item(output, [])

        assert item.expires_at == output.expires_at


class TestGetGroupDetail:
    async def test_returns_none_for_missing_job(self) -> None:
        mock_session = AsyncMock()
        svc = LibraryService(session=mock_session)
        mock_repo = AsyncMock()
        mock_repo.get_group_job.return_value = None

        with patch("src.api.services.library.LibraryRepository", return_value=mock_repo):
            result = await svc.get_group_detail(uuid4(), uuid4(), "vex", session=mock_session)

        assert result is None

    async def test_returns_detail_with_no_input(self) -> None:
        mock_session = AsyncMock()
        svc = LibraryService(session=mock_session)
        mock_repo = AsyncMock()
        job = _make_job(generation_type="t2i", source_output=None, input_image=None)
        job.outputs = []
        mock_repo.get_group_job.return_value = job
        source_repo = MagicMock()
        source_repo.list_for_job = AsyncMock(return_value=[])

        with (
            patch("src.api.services.library.LibraryRepository", return_value=mock_repo),
            patch(
                "src.api.services.library.GenerationJobSourceRepository",
                return_value=source_repo,
            ),
        ):
            result = await svc.get_group_detail(uuid4(), uuid4(), "vex", session=mock_session)

        assert result is not None
        assert result.input_media is None

    async def test_returns_detail_for_found_job_with_source_output(self) -> None:
        mock_session = AsyncMock()
        svc = LibraryService(session=mock_session)
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
        mock_repo.get_group_job.return_value = job
        source_repo = MagicMock()
        source_repo.list_for_job = AsyncMock(return_value=[])

        with (
            patch("src.api.services.library.LibraryRepository", return_value=mock_repo),
            patch(
                "src.api.services.library.GenerationJobSourceRepository",
                return_value=source_repo,
            ),
        ):
            result = await svc.get_group_detail(uuid4(), uuid4(), "vex", session=mock_session)

        assert result is not None
        assert result.input_media is not None
        assert result.input_media.original.url == f"/v1/content/outputs/{source_output.id}"

    async def test_returns_detail_for_found_job_with_input_image(self) -> None:
        mock_session = AsyncMock()
        svc = LibraryService(session=mock_session)
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
        mock_repo.get_group_job.return_value = job
        source_repo = MagicMock()
        source_repo.list_for_job = AsyncMock(return_value=[])

        with (
            patch("src.api.services.library.LibraryRepository", return_value=mock_repo),
            patch(
                "src.api.services.library.GenerationJobSourceRepository",
                return_value=source_repo,
            ),
        ):
            result = await svc.get_group_detail(uuid4(), uuid4(), "vex", session=mock_session)

        assert result is not None
        assert result.input_media is not None
        assert result.input_media.original.url == f"/v1/content/uploads/{input_image.id}"

    async def test_preserves_unavailable_source_positions(self) -> None:
        mock_session = AsyncMock()
        svc = LibraryService(session=mock_session)
        mock_repo = AsyncMock()
        job = _make_job(generation_type="i2i", source_output=None, input_image=None)
        job.outputs = []
        mock_repo.get_group_job.return_value = job

        upload = _make_output_row(content_type="image/png")
        upload.product_id = "vex"
        output = _make_output_row(content_type="image/png")
        output.product_id = "vex"
        missing_id = uuid4()
        source_rows = [
            SimpleNamespace(
                position=0,
                asset_ref=f"upload:{upload.id}",
                source_upload_id=upload.id,
                source_output_id=None,
            ),
            SimpleNamespace(
                position=1,
                asset_ref=f"upload:{missing_id}",
                source_upload_id=None,
                source_output_id=None,
            ),
            SimpleNamespace(
                position=2,
                asset_ref=f"output:{output.id}",
                source_upload_id=None,
                source_output_id=output.id,
            ),
        ]
        source_repo = MagicMock(list_for_job=AsyncMock(return_value=source_rows))
        image_repo = MagicMock(
            get_many=AsyncMock(return_value={upload.id: upload}),
            batch_derivatives=AsyncMock(return_value={}),
        )
        output_repo = MagicMock(
            get_many=AsyncMock(return_value={output.id: output}),
            batch_derivatives=AsyncMock(return_value={}),
        )

        with (
            patch("src.api.services.library.LibraryRepository", return_value=mock_repo),
            patch(
                "src.api.services.library.GenerationJobSourceRepository",
                return_value=source_repo,
            ),
            patch("src.api.services.library.UserImageRepository", return_value=image_repo),
            patch("src.api.services.library.OutputRepository", return_value=output_repo),
        ):
            result = await svc.get_group_detail(uuid4(), uuid4(), "vex", session=mock_session)

        assert result is not None
        assert [source.position for source in result.source_media] == [0, 1, 2]
        assert [source.available for source in result.source_media] == [True, False, True]
        assert result.source_media[1].asset_ref == f"upload:{missing_id}"
        assert result.source_media[1].media is None
        assert result.input_media is result.source_media[0].media
