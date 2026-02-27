"""Tests for the gallery query helper (build_gallery_response)."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID, uuid4

from src.api.routes.gallery import GalleryOutputItem, GalleryResponse, build_gallery_response
from src.api.services.storage.exceptions import StorageDownloadError
from src.core.enums import GenerationType, JobStatus

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_job(
    *,
    job_id: UUID | None = None,
    user_id: UUID | None = None,
    provider: str = "grok",
    model: str | None = "grok-imagine-image",
    generation_type: str = GenerationType.T2I.value,
    status: str = JobStatus.COMPLETED.value,
    prompt: str = "a cat",
    name: str = "Test Job",
    token_cost: int | None = 50,
    error_message: str | None = None,
) -> MagicMock:
    job = MagicMock()
    job.id = job_id or uuid4()
    job.user_id = user_id or uuid4()
    job.provider = provider
    job.model = model
    job.generation_type = generation_type
    job.status = status
    job.prompt = prompt
    job.name = name
    job.token_cost = token_cost
    job.error_message = error_message
    return job


def _make_output(
    *,
    output_id: UUID | None = None,
    job: MagicMock | None = None,
    user_id: UUID | None = None,
    content_type: str = "image/png",
    format: str = "png",
    size_bytes: int = 1024,
    is_thumbnail: bool = False,
    storage_key: str | None = None,
    created_at: datetime | None = None,
) -> MagicMock:
    if job is None:
        job = _make_job()
    out = MagicMock()
    out.id = output_id or uuid4()
    out.job_id = job.id
    out.job = job
    out.user_id = user_id or uuid4()
    out.content_type = content_type
    out.format = format
    out.size_bytes = size_bytes
    out.is_thumbnail = is_thumbnail
    out.storage_key = storage_key or f"users/uid/outputs/{uuid4()}/file.{format}"
    out.created_at = created_at or datetime.now(UTC)
    return out


def _make_session(
    count: int,
    outputs: list,
    thumbs: list | None = None,
) -> AsyncMock:
    """Build a mock AsyncSession for gallery queries.

    Provides side_effects for the 2 (or 3, when video outputs exist) execute calls:
      1. count query
      2. data query
      3. thumbnail query (only when thumbs is not None)
    """
    count_result = MagicMock()
    count_result.scalar_one.return_value = count

    outputs_result = MagicMock()
    outputs_result.scalars.return_value.all.return_value = outputs

    side_effects: list = [count_result, outputs_result]

    if thumbs is not None:
        thumb_result = MagicMock()
        thumb_result.scalars.return_value.all.return_value = thumbs
        side_effects.append(thumb_result)

    session = AsyncMock()
    session.execute = AsyncMock(side_effect=side_effects)
    return session


def _make_storage(presigned_url: str = "https://r2.example.com/file") -> AsyncMock:
    url_result = MagicMock()
    url_result.presigned_url = presigned_url
    storage = AsyncMock()
    storage.get_presigned_url = AsyncMock(return_value=url_result)
    return storage


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestBuildGalleryResponseEmpty:
    async def test_empty_result(self) -> None:
        user_id = uuid4()
        result = await build_gallery_response(
            user_id, _make_storage(), _make_session(count=0, outputs=[])
        )
        assert isinstance(result, GalleryResponse)
        assert result.total == 0
        assert result.items == []

    async def test_default_pagination(self) -> None:
        user_id = uuid4()
        result = await build_gallery_response(
            user_id, _make_storage(), _make_session(count=0, outputs=[])
        )
        assert result.limit == 20
        assert result.offset == 0

    async def test_limit_capped_at_100(self) -> None:
        user_id = uuid4()
        result = await build_gallery_response(
            user_id,
            _make_storage(),
            _make_session(count=0, outputs=[]),
            limit=500,
        )
        assert result.limit == 100

    async def test_pagination_params_reflected(self) -> None:
        user_id = uuid4()
        result = await build_gallery_response(
            user_id,
            _make_storage(),
            _make_session(count=200, outputs=[]),
            limit=10,
            offset=50,
        )
        assert result.limit == 10
        assert result.offset == 50
        assert result.total == 200


class TestBuildGalleryResponseSingleOutput:
    async def test_item_fields_mapped_correctly(self) -> None:
        user_id = uuid4()
        job = _make_job(user_id=user_id)
        output = _make_output(user_id=user_id, job=job)
        storage = _make_storage("https://r2.example.com/img.png")

        result = await build_gallery_response(
            user_id, storage, _make_session(count=1, outputs=[output])
        )

        assert result.total == 1
        assert len(result.items) == 1
        item = result.items[0]
        assert isinstance(item, GalleryOutputItem)
        assert item.output_id == output.id
        assert item.job_id == job.id
        assert item.job_name == job.name
        assert item.provider == job.provider
        assert item.model == job.model
        assert item.prompt == job.prompt
        assert item.content_type == output.content_type
        assert item.format == output.format
        assert item.size_bytes == output.size_bytes
        assert item.url == "https://r2.example.com/img.png"
        assert item.token_cost == job.token_cost

    async def test_generation_type_converted_to_enum(self) -> None:
        user_id = uuid4()
        job = _make_job(user_id=user_id, generation_type=GenerationType.T2I.value)
        output = _make_output(user_id=user_id, job=job)

        result = await build_gallery_response(
            user_id, _make_storage(), _make_session(count=1, outputs=[output])
        )

        assert result.items[0].generation_type == GenerationType.T2I

    async def test_job_status_converted_to_enum(self) -> None:
        user_id = uuid4()
        job = _make_job(user_id=user_id, status=JobStatus.COMPLETED.value)
        output = _make_output(user_id=user_id, job=job)

        result = await build_gallery_response(
            user_id, _make_storage(), _make_session(count=1, outputs=[output])
        )

        assert result.items[0].status == JobStatus.COMPLETED

    async def test_no_thumbnail_url_for_image_output(self) -> None:
        user_id = uuid4()
        job = _make_job(user_id=user_id)
        output = _make_output(user_id=user_id, job=job, content_type="image/png")

        result = await build_gallery_response(
            user_id, _make_storage(), _make_session(count=1, outputs=[output])
        )

        assert result.items[0].thumbnail_url is None

    async def test_model_none_preserved(self) -> None:
        user_id = uuid4()
        job = _make_job(user_id=user_id, model=None)
        output = _make_output(user_id=user_id, job=job)

        result = await build_gallery_response(
            user_id, _make_storage(), _make_session(count=1, outputs=[output])
        )

        assert result.items[0].model is None

    async def test_token_cost_none_preserved(self) -> None:
        user_id = uuid4()
        job = _make_job(user_id=user_id, token_cost=None)
        output = _make_output(user_id=user_id, job=job)

        result = await build_gallery_response(
            user_id, _make_storage(), _make_session(count=1, outputs=[output])
        )

        assert result.items[0].token_cost is None


class TestBuildGalleryResponseVideoThumbnails:
    def _make_keyed_storage(self, url_map: dict[str, str]) -> AsyncMock:
        """Return a storage mock that maps storage_key → presigned URL."""

        def side_effect(storage_key: str, **_: object) -> MagicMock:
            if storage_key not in url_map:
                raise KeyError(storage_key)
            result = MagicMock()
            result.presigned_url = url_map[storage_key]
            return result

        storage = AsyncMock()
        storage.get_presigned_url = AsyncMock(side_effect=side_effect)
        return storage

    async def test_video_output_gets_thumbnail_url(self) -> None:
        user_id = uuid4()
        job = _make_job(user_id=user_id)
        thumb_key = "users/uid/outputs/job/thumb.jpg"
        video_key = "users/uid/outputs/job/video.mp4"

        output = _make_output(
            user_id=user_id,
            job=job,
            content_type="video/mp4",
            format="mp4",
            storage_key=video_key,
        )
        thumb = MagicMock()
        thumb.id = uuid4()
        thumb.job_id = job.id
        thumb.storage_key = thumb_key

        storage = self._make_keyed_storage(
            {
                thumb_key: "https://r2.example.com/thumb.jpg",
                video_key: "https://r2.example.com/video.mp4",
            }
        )

        result = await build_gallery_response(
            user_id, storage, _make_session(count=1, outputs=[output], thumbs=[thumb])
        )

        assert len(result.items) == 1
        assert result.items[0].url == "https://r2.example.com/video.mp4"
        assert result.items[0].thumbnail_url == "https://r2.example.com/thumb.jpg"

    async def test_video_output_no_thumbnail_record(self) -> None:
        """When no thumbnail row exists for a video job, thumbnail_url is None."""
        user_id = uuid4()
        job = _make_job(user_id=user_id)
        video_key = "users/uid/outputs/job/video.mp4"

        output = _make_output(
            user_id=user_id,
            job=job,
            content_type="video/mp4",
            format="mp4",
            storage_key=video_key,
        )
        storage = self._make_keyed_storage({video_key: "https://r2.example.com/video.mp4"})

        result = await build_gallery_response(
            user_id, storage, _make_session(count=1, outputs=[output], thumbs=[])
        )

        assert result.items[0].thumbnail_url is None
        assert result.items[0].url == "https://r2.example.com/video.mp4"

    async def test_failed_thumbnail_url_does_not_skip_output(self) -> None:
        """A broken thumbnail presign should not drop the parent output from results."""
        user_id = uuid4()
        job = _make_job(user_id=user_id)
        thumb_key = "users/uid/outputs/job/thumb.jpg"
        video_key = "users/uid/outputs/job/video.mp4"

        output = _make_output(
            user_id=user_id,
            job=job,
            content_type="video/mp4",
            format="mp4",
            storage_key=video_key,
        )
        thumb = MagicMock()
        thumb.id = uuid4()
        thumb.job_id = job.id
        thumb.storage_key = thumb_key

        def side_effect(storage_key: str, **_: object) -> MagicMock:
            if storage_key == thumb_key:
                raise StorageDownloadError("R2 thumbnail error")
            r = MagicMock()
            r.presigned_url = "https://r2.example.com/video.mp4"
            return r

        storage = AsyncMock()
        storage.get_presigned_url = AsyncMock(side_effect=side_effect)

        result = await build_gallery_response(
            user_id, storage, _make_session(count=1, outputs=[output], thumbs=[thumb])
        )

        assert len(result.items) == 1
        assert result.items[0].url == "https://r2.example.com/video.mp4"
        assert result.items[0].thumbnail_url is None


class TestBuildGalleryResponseErrors:
    async def test_failed_presigned_url_skips_output(self) -> None:
        """When the output's own presign fails the item is omitted from results."""
        user_id = uuid4()
        job = _make_job(user_id=user_id)
        output = _make_output(user_id=user_id, job=job)

        storage = AsyncMock()
        storage.get_presigned_url = AsyncMock(side_effect=StorageDownloadError("R2 error"))

        result = await build_gallery_response(
            user_id, storage, _make_session(count=1, outputs=[output])
        )

        # total comes from the DB count; items is empty because the URL failed
        assert result.total == 1
        assert result.items == []

    async def test_partial_failure_keeps_successful_outputs(self) -> None:
        """Only outputs whose presign fails are dropped; others appear normally."""
        user_id = uuid4()
        job = _make_job(user_id=user_id)
        good_key = "users/uid/outputs/good.png"
        bad_key = "users/uid/outputs/bad.png"

        good_output = _make_output(user_id=user_id, job=job, storage_key=good_key)
        bad_output = _make_output(user_id=user_id, job=job, storage_key=bad_key)

        def side_effect(storage_key: str, **_: object) -> MagicMock:
            if storage_key == bad_key:
                raise StorageDownloadError("R2 error")
            r = MagicMock()
            r.presigned_url = "https://r2.example.com/good.png"
            return r

        storage = AsyncMock()
        storage.get_presigned_url = AsyncMock(side_effect=side_effect)

        result = await build_gallery_response(
            user_id, storage, _make_session(count=2, outputs=[good_output, bad_output])
        )

        assert result.total == 2
        assert len(result.items) == 1
        assert result.items[0].url == "https://r2.example.com/good.png"


class TestBuildGalleryResponseMultipleOutputs:
    async def test_multiple_outputs_returned(self) -> None:
        user_id = uuid4()
        job = _make_job(user_id=user_id)
        outputs = [_make_output(user_id=user_id, job=job) for _ in range(3)]

        result = await build_gallery_response(
            user_id, _make_storage(), _make_session(count=3, outputs=outputs)
        )

        assert result.total == 3
        assert len(result.items) == 3

    async def test_storage_called_once_per_output(self) -> None:
        user_id = uuid4()
        job = _make_job(user_id=user_id)
        outputs = [_make_output(user_id=user_id, job=job) for _ in range(4)]
        storage = _make_storage()

        await build_gallery_response(user_id, storage, _make_session(count=4, outputs=outputs))

        assert storage.get_presigned_url.await_count == 4
