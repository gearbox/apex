"""Tests for the unified jobs service and schemas.

Covers:
  - UnifiedJobService.get_job (including Grok poll-on-read)
  - UnifiedJobService.list_jobs (pagination, limit capping)
  - _build_response output/thumbnail logic (presigned URL failures, thumbnail flag)
  - Schema round-trip serialization (UnifiedJobResponse, UnifiedJobListResponse,
    JobOutputItem)
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID, uuid4

import msgspec
import pytest

from src.api.schemas.jobs import JobOutputItem, UnifiedJobListResponse, UnifiedJobResponse
from src.api.services.unified_jobs import UnifiedJobService
from src.core.enums import GenerationType, JobStatus

# ---------------------------------------------------------------------------
# Shared helpers
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
    negative_prompt: str | None = None,
    token_cost: int | None = 50,
    error_message: str | None = None,
    created_at: datetime | None = None,
    started_at: datetime | None = None,
    completed_at: datetime | None = None,
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
    job.negative_prompt = negative_prompt
    job.token_cost = token_cost
    job.error_message = error_message
    job.created_at = created_at or datetime.now(UTC)
    job.started_at = started_at
    job.completed_at = completed_at
    return job


def _make_output(
    *,
    output_id: UUID | None = None,
    job_id: UUID | None = None,
    content_type: str = "image/jpeg",
    format: str = "jpeg",
    size_bytes: int = 2048,
    output_index: int = 0,
    is_thumbnail: bool = False,
    storage_key: str | None = None,
) -> MagicMock:
    out = MagicMock()
    out.id = output_id or uuid4()
    out.job_id = job_id or uuid4()
    out.content_type = content_type
    out.format = format
    out.size_bytes = size_bytes
    out.output_index = output_index
    out.is_thumbnail = is_thumbnail
    out.storage_key = storage_key or f"users/uid/outputs/{uuid4()}/file.{format}"
    return out


def _make_storage(presigned_url: str = "https://r2.example.com/file") -> AsyncMock:
    url_result = MagicMock()
    url_result.presigned_url = presigned_url
    storage = AsyncMock()
    storage.get_presigned_url = AsyncMock(return_value=url_result)
    return storage


def _url_mock(url: str) -> MagicMock:
    """Build a presigned-URL result mock pointing to *url*."""
    r = MagicMock()
    r.presigned_url = url
    return r


def _session_for_get(
    job: MagicMock | None,
    outputs: list | None = None,
) -> AsyncMock:
    """Build a mock AsyncSession for ``get_job`` calls.

    Provides side-effects for up to two ``execute`` calls:
    1. ``_get_by_id_with_optional_owner`` — returns *job* via scalar_one_or_none.
    2. ``list_job_outputs`` — returns *outputs* via scalars().all()
       (only called when *job* is not None).
    """
    get_result = MagicMock()
    get_result.scalar_one_or_none.return_value = job

    out_result = MagicMock()
    out_result.scalars.return_value.all.return_value = outputs or []

    session = AsyncMock()
    session.execute = AsyncMock(side_effect=[get_result, out_result])
    return session


def _session_for_list(
    count: int,
    jobs: list,
    outputs_per_job: list[list] | None = None,
) -> AsyncMock:
    """Build a mock AsyncSession for ``list_jobs`` calls.

    Side-effect order:
    1. count query
    2. data (jobs) query
    3+. one list_job_outputs call per job
    """
    count_result = MagicMock()
    count_result.scalar_one.return_value = count

    jobs_result = MagicMock()
    jobs_result.scalars.return_value.all.return_value = jobs

    side_effects: list = [count_result, jobs_result]

    for outputs in outputs_per_job or [[] for _ in jobs]:
        out_result = MagicMock()
        out_result.scalars.return_value.all.return_value = outputs
        side_effects.append(out_result)

    session = AsyncMock()
    session.execute = AsyncMock(side_effect=side_effects)
    return session


def _service(
    storage: AsyncMock | None = None,
    grok_job_service: AsyncMock | None = None,
) -> UnifiedJobService:
    return UnifiedJobService(
        storage=storage or _make_storage(),
        grok_job_service=grok_job_service,
    )


# ---------------------------------------------------------------------------
# Tests: UnifiedJobService.get_job
# ---------------------------------------------------------------------------


class TestGetJob:
    async def test_returns_none_when_job_not_found(self) -> None:
        session = _session_for_get(None)
        result = await _service().get_job(uuid4(), uuid4(), session=session)

        assert result is None

    async def test_returns_unified_job_response_for_found_job(self) -> None:
        user_id = uuid4()
        job = _make_job(user_id=user_id)
        session = _session_for_get(job)

        result = await _service().get_job(job.id, user_id, session=session)

        assert isinstance(result, UnifiedJobResponse)
        assert result.id == job.id

    async def test_fields_mapped_from_db_job(self) -> None:
        user_id = uuid4()
        created = datetime.now(UTC)
        job = _make_job(
            user_id=user_id,
            name="My Generation",
            prompt="a sunset",
            negative_prompt="blurry",
            provider="comfyui",
            model=None,
            generation_type=GenerationType.I2I.value,
            status=JobStatus.FAILED.value,
            token_cost=75,
            error_message="GPU OOM",
            created_at=created,
        )
        session = _session_for_get(job)

        result = await _service().get_job(job.id, user_id, session=session)

        assert result is not None
        assert result.name == "My Generation"
        assert result.prompt == "a sunset"
        assert result.negative_prompt == "blurry"
        assert result.provider == "comfyui"
        assert result.model is None
        assert result.generation_type == GenerationType.I2I
        assert result.status == JobStatus.FAILED
        assert result.token_cost == 75
        assert result.error == "GPU OOM"
        assert result.created_at == created

    async def test_no_grok_poll_when_grok_service_is_none(self) -> None:
        user_id = uuid4()
        job = _make_job(
            user_id=user_id,
            provider="grok",
            generation_type=GenerationType.T2V.value,
            status=JobStatus.RUNNING.value,
        )
        session = _session_for_get(job)

        # Should not raise even though the job is a video job
        result = await UnifiedJobService(storage=_make_storage(), grok_job_service=None).get_job(
            job.id, user_id, session=session
        )

        assert result is not None
        assert result.status == JobStatus.RUNNING

    async def test_grok_video_queued_job_is_polled(self) -> None:
        user_id = uuid4()
        job = _make_job(
            user_id=user_id,
            provider="grok",
            generation_type=GenerationType.T2V.value,
            status=JobStatus.QUEUED.value,
        )
        updated = _make_job(
            job_id=job.id,
            user_id=user_id,
            provider="grok",
            generation_type=GenerationType.T2V.value,
            status=JobStatus.COMPLETED.value,
        )
        session = _session_for_get(job)

        grok = AsyncMock()
        grok.poll_video_job = AsyncMock(return_value=updated)

        result = await _service(grok_job_service=grok).get_job(job.id, user_id, session=session)

        grok.poll_video_job.assert_awaited_once_with(session, job.id)
        assert result is not None
        assert result.status == JobStatus.COMPLETED

    async def test_grok_video_running_job_is_polled(self) -> None:
        user_id = uuid4()
        job = _make_job(
            user_id=user_id,
            provider="grok",
            generation_type=GenerationType.I2V.value,
            status=JobStatus.RUNNING.value,
        )
        session = _session_for_get(job)

        grok = AsyncMock()
        grok.poll_video_job = AsyncMock(return_value=None)  # None means no update

        await _service(grok_job_service=grok).get_job(job.id, user_id, session=session)

        grok.poll_video_job.assert_awaited_once()

    async def test_grok_poll_not_triggered_for_image_job(self) -> None:
        user_id = uuid4()
        job = _make_job(
            user_id=user_id,
            provider="grok",
            generation_type=GenerationType.T2I.value,  # image, not video
            status=JobStatus.RUNNING.value,
        )
        session = _session_for_get(job)

        grok = AsyncMock()
        grok.poll_video_job = AsyncMock()

        await _service(grok_job_service=grok).get_job(job.id, user_id, session=session)

        grok.poll_video_job.assert_not_awaited()

    async def test_grok_poll_not_triggered_for_completed_job(self) -> None:
        user_id = uuid4()
        job = _make_job(
            user_id=user_id,
            provider="grok",
            generation_type=GenerationType.T2V.value,
            status=JobStatus.COMPLETED.value,  # not queued/running
        )
        session = _session_for_get(job)

        grok = AsyncMock()
        grok.poll_video_job = AsyncMock()

        await _service(grok_job_service=grok).get_job(job.id, user_id, session=session)

        grok.poll_video_job.assert_not_awaited()

    async def test_grok_poll_not_triggered_for_non_grok_provider(self) -> None:
        user_id = uuid4()
        job = _make_job(
            user_id=user_id,
            provider="comfyui",
            generation_type=GenerationType.T2V.value,
            status=JobStatus.RUNNING.value,
        )
        session = _session_for_get(job)

        grok = AsyncMock()
        grok.poll_video_job = AsyncMock()

        await _service(grok_job_service=grok).get_job(job.id, user_id, session=session)

        grok.poll_video_job.assert_not_awaited()

    async def test_grok_poll_failure_is_swallowed_and_original_job_returned(self) -> None:
        user_id = uuid4()
        job = _make_job(
            user_id=user_id,
            provider="grok",
            generation_type=GenerationType.T2V.value,
            status=JobStatus.QUEUED.value,
        )
        session = _session_for_get(job)

        grok = AsyncMock()
        grok.poll_video_job = AsyncMock(side_effect=RuntimeError("xAI unreachable"))

        # Should not propagate the exception
        result = await _service(grok_job_service=grok).get_job(job.id, user_id, session=session)

        assert result is not None
        assert result.status == JobStatus.QUEUED  # original status unchanged

    async def test_grok_poll_returning_none_falls_back_to_original_job(self) -> None:
        user_id = uuid4()
        job = _make_job(
            user_id=user_id,
            provider="grok",
            generation_type=GenerationType.T2V.value,
            status=JobStatus.RUNNING.value,
        )
        session = _session_for_get(job)

        grok = AsyncMock()
        grok.poll_video_job = AsyncMock(return_value=None)

        result = await _service(grok_job_service=grok).get_job(job.id, user_id, session=session)

        assert result is not None
        assert result.status == JobStatus.RUNNING


# ---------------------------------------------------------------------------
# Tests: UnifiedJobService.list_jobs
# ---------------------------------------------------------------------------


class TestListJobs:
    async def test_empty_result(self) -> None:
        session = _session_for_list(count=0, jobs=[])

        result = await _service().list_jobs(uuid4(), session=session)

        assert isinstance(result, UnifiedJobListResponse)
        assert result.total == 0
        assert result.items == []

    async def test_default_pagination_params(self) -> None:
        session = _session_for_list(count=0, jobs=[])

        result = await _service().list_jobs(uuid4(), session=session)

        assert result.limit == 20
        assert result.offset == 0

    async def test_limit_capped_at_100(self) -> None:
        session = _session_for_list(count=0, jobs=[])

        result = await _service().list_jobs(uuid4(), session=session, limit=500)

        assert result.limit == 100

    async def test_custom_pagination_reflected_in_response(self) -> None:
        session = _session_for_list(count=200, jobs=[])

        result = await _service().list_jobs(uuid4(), session=session, limit=10, offset=50)

        assert result.limit == 10
        assert result.offset == 50
        assert result.total == 200

    async def test_returns_one_item_per_job(self) -> None:
        user_id = uuid4()
        jobs = [_make_job(user_id=user_id) for _ in range(3)]
        session = _session_for_list(count=3, jobs=jobs)

        result = await _service().list_jobs(user_id, session=session)

        assert len(result.items) == 3

    async def test_all_items_are_unified_job_responses(self) -> None:
        user_id = uuid4()
        jobs = [_make_job(user_id=user_id) for _ in range(2)]
        session = _session_for_list(count=2, jobs=jobs)

        result = await _service().list_jobs(user_id, session=session)

        for item in result.items:
            assert isinstance(item, UnifiedJobResponse)

    async def test_job_fields_mapped_correctly_in_list(self) -> None:
        user_id = uuid4()
        job = _make_job(user_id=user_id, name="Gallery Job", prompt="neon city")
        session = _session_for_list(count=1, jobs=[job])

        result = await _service().list_jobs(user_id, session=session)

        item = result.items[0]
        assert item.id == job.id
        assert item.name == "Gallery Job"
        assert item.prompt == "neon city"
        assert item.provider == job.provider


# ---------------------------------------------------------------------------
# Tests: _build_response — output and thumbnail logic
# ---------------------------------------------------------------------------


class TestBuildResponseOutputs:
    async def test_no_outputs_returns_empty_list_and_no_thumbnail(self) -> None:
        user_id = uuid4()
        job = _make_job(user_id=user_id)
        session = _session_for_get(job, outputs=[])

        result = await _service().get_job(job.id, user_id, session=session)

        assert result is not None
        assert result.outputs == []
        assert result.thumbnail_url is None

    async def test_single_image_output_sets_thumbnail_url(self) -> None:
        user_id = uuid4()
        job = _make_job(user_id=user_id)
        out = _make_output(job_id=job.id, content_type="image/jpeg", format="jpeg")
        session = _session_for_get(job, outputs=[out])

        storage = _make_storage("https://r2.example.com/img.jpg")
        result = await _service(storage=storage).get_job(job.id, user_id, session=session)

        assert result is not None
        assert len(result.outputs) == 1
        assert result.thumbnail_url == "https://r2.example.com/img.jpg"

    async def test_image_output_item_fields_mapped(self) -> None:
        user_id = uuid4()
        job = _make_job(user_id=user_id)
        out_id = uuid4()
        out = _make_output(
            output_id=out_id,
            job_id=job.id,
            content_type="image/png",
            format="png",
            size_bytes=4096,
            output_index=1,
        )
        session = _session_for_get(job, outputs=[out])

        result = await _service().get_job(job.id, user_id, session=session)

        assert result is not None
        item = result.outputs[0]
        assert isinstance(item, JobOutputItem)
        assert item.id == out_id
        assert item.content_type == "image/png"
        assert item.format == "png"
        assert item.size_bytes == 4096
        assert item.output_index == 1
        assert item.is_thumbnail is False

    async def test_thumbnail_flagged_output_used_as_thumbnail_url(self) -> None:
        user_id = uuid4()
        job = _make_job(user_id=user_id)
        thumb_key = "users/uid/outputs/job/thumb.jpg"
        video_key = "users/uid/outputs/job/video.mp4"

        thumb = _make_output(
            job_id=job.id,
            content_type="image/jpeg",
            format="jpeg",
            is_thumbnail=True,
            output_index=-1,
            storage_key=thumb_key,
        )
        video = _make_output(
            job_id=job.id,
            content_type="video/mp4",
            format="mp4",
            output_index=0,
            storage_key=video_key,
        )
        session = _session_for_get(job, outputs=[thumb, video])

        url_map = {
            thumb_key: "https://r2.example.com/thumb.jpg",
            video_key: "https://r2.example.com/video.mp4",
        }
        storage = AsyncMock()
        storage.get_presigned_url = AsyncMock(side_effect=lambda key, **_: _url_mock(url_map[key]))

        result = await _service(storage=storage).get_job(job.id, user_id, session=session)

        assert result is not None
        assert result.thumbnail_url == "https://r2.example.com/thumb.jpg"
        assert len(result.outputs) == 2

    async def test_video_output_without_thumbnail_flag_has_no_thumbnail_url(self) -> None:
        user_id = uuid4()
        job = _make_job(user_id=user_id)
        video = _make_output(
            job_id=job.id,
            content_type="video/mp4",
            format="mp4",
            is_thumbnail=False,
        )
        session = _session_for_get(job, outputs=[video])

        result = await _service().get_job(job.id, user_id, session=session)

        assert result is not None
        # video/mp4 content type does not trigger the "image" fallback
        assert result.thumbnail_url is None

    async def test_failed_presigned_url_skips_output(self) -> None:
        user_id = uuid4()
        job = _make_job(user_id=user_id)
        out = _make_output(job_id=job.id)
        session = _session_for_get(job, outputs=[out])

        storage = AsyncMock()
        storage.get_presigned_url = AsyncMock(side_effect=Exception("R2 error"))

        result = await _service(storage=storage).get_job(job.id, user_id, session=session)

        assert result is not None
        assert result.outputs == []
        assert result.thumbnail_url is None

    async def test_partial_presigned_failure_keeps_successful_outputs(self) -> None:
        user_id = uuid4()
        job = _make_job(user_id=user_id)
        good_key = "users/uid/outputs/good.jpg"
        bad_key = "users/uid/outputs/bad.jpg"

        good_out = _make_output(job_id=job.id, content_type="image/jpeg", storage_key=good_key)
        bad_out = _make_output(job_id=job.id, content_type="image/jpeg", storage_key=bad_key)
        session = _session_for_get(job, outputs=[good_out, bad_out])

        def _side_effect(storage_key: str, **_: object) -> MagicMock:
            if storage_key == bad_key:
                raise Exception("R2 failure")
            return _url_mock("https://r2.example.com/good.jpg")

        storage = AsyncMock()
        storage.get_presigned_url = AsyncMock(side_effect=_side_effect)

        result = await _service(storage=storage).get_job(job.id, user_id, session=session)

        assert result is not None
        assert len(result.outputs) == 1
        assert result.outputs[0].url == "https://r2.example.com/good.jpg"

    async def test_no_storage_configured_skips_all_outputs(self) -> None:
        user_id = uuid4()
        job = _make_job(user_id=user_id)
        out = _make_output(job_id=job.id)
        session = _session_for_get(job, outputs=[out])

        result = await UnifiedJobService(storage=None, grok_job_service=None).get_job(
            job.id, user_id, session=session
        )

        assert result is not None
        assert result.outputs == []
        assert result.thumbnail_url is None

    async def test_multiple_image_outputs_first_becomes_thumbnail(self) -> None:
        user_id = uuid4()
        job = _make_job(user_id=user_id)
        outputs = [
            _make_output(
                job_id=job.id,
                content_type="image/png",
                output_index=i,
                storage_key=f"users/uid/outputs/img_{i}.png",
            )
            for i in range(3)
        ]
        session = _session_for_get(job, outputs=outputs)

        first_url = "https://r2.example.com/img_0.png"
        storage = AsyncMock()
        storage.get_presigned_url = AsyncMock(
            side_effect=lambda key, **_: _url_mock(
                first_url if "img_0" in key else "https://r2.example.com/other.png"
            )
        )

        result = await _service(storage=storage).get_job(job.id, user_id, session=session)

        assert result is not None
        assert len(result.outputs) == 3
        assert result.thumbnail_url == first_url


# ---------------------------------------------------------------------------
# Tests: Schema serialization (msgspec round-trips)
# ---------------------------------------------------------------------------


class TestSchemas:
    def test_unified_job_response_round_trip(self) -> None:
        job_id = uuid4()
        now = datetime.now(UTC)
        response = UnifiedJobResponse(
            id=job_id,
            name="Round-trip test",
            status=JobStatus.COMPLETED,
            provider="grok",
            generation_type=GenerationType.T2I,
            prompt="a dog in space",
            created_at=now,
        )

        encoded = msgspec.json.encode(response)
        decoded = msgspec.json.decode(encoded, type=UnifiedJobResponse)

        assert decoded.id == job_id
        assert decoded.status == JobStatus.COMPLETED
        assert decoded.generation_type == GenerationType.T2I
        assert decoded.outputs == []
        assert decoded.thumbnail_url is None
        assert decoded.model is None
        assert decoded.error is None

    def test_unified_job_response_optional_fields_preserved(self) -> None:
        now = datetime.now(UTC)
        response = UnifiedJobResponse(
            id=uuid4(),
            name="Full response",
            status=JobStatus.FAILED,
            provider="comfyui",
            generation_type=GenerationType.I2V,
            prompt="ocean waves",
            model="aisha",
            negative_prompt="blurry",
            token_cost=120,
            created_at=now,
            started_at=now,
            completed_at=now,
            thumbnail_url="https://r2.example.com/thumb.jpg",
            error="Connection reset by peer",
        )

        encoded = msgspec.json.encode(response)
        decoded = msgspec.json.decode(encoded, type=UnifiedJobResponse)

        assert decoded.status == JobStatus.FAILED
        assert decoded.generation_type == GenerationType.I2V
        assert decoded.model == "aisha"
        assert decoded.negative_prompt == "blurry"
        assert decoded.token_cost == 120
        assert decoded.thumbnail_url == "https://r2.example.com/thumb.jpg"
        assert decoded.error == "Connection reset by peer"

    def test_unified_job_list_response_round_trip(self) -> None:
        resp = UnifiedJobListResponse(items=[], total=42, limit=10, offset=20)

        encoded = msgspec.json.encode(resp)
        decoded = msgspec.json.decode(encoded, type=UnifiedJobListResponse)

        assert decoded.total == 42
        assert decoded.limit == 10
        assert decoded.offset == 20
        assert decoded.items == []

    def test_job_output_item_defaults(self) -> None:
        item = JobOutputItem(
            id=uuid4(),
            url="https://r2.example.com/img.jpg",
            content_type="image/jpeg",
            format="jpeg",
            size_bytes=1024,
            output_index=0,
        )

        assert item.is_thumbnail is False

    def test_job_output_item_round_trip(self) -> None:
        item_id = uuid4()
        item = JobOutputItem(
            id=item_id,
            url="https://r2.example.com/video.mp4",
            content_type="video/mp4",
            format="mp4",
            size_bytes=1_048_576,
            output_index=0,
            is_thumbnail=True,
        )

        encoded = msgspec.json.encode(item)
        decoded = msgspec.json.decode(encoded, type=JobOutputItem)

        assert decoded.id == item_id
        assert decoded.content_type == "video/mp4"
        assert decoded.size_bytes == 1_048_576
        assert decoded.is_thumbnail is True

    def test_job_output_item_negative_index_for_thumbnail(self) -> None:
        item = JobOutputItem(
            id=uuid4(),
            url="https://r2.example.com/thumb.jpg",
            content_type="image/jpeg",
            format="jpeg",
            size_bytes=512,
            output_index=-1,
            is_thumbnail=True,
        )
        assert item.output_index == -1
