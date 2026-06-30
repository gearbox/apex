"""Tests for the unified jobs service and schemas.

Covers:
  - UnifiedJobService.get_job (including Grok poll-on-read)
  - UnifiedJobService.list_jobs (pagination, limit capping)
  - _build_response output/derivative logic (proxy URLs, MediaObject variants)
  - Schema round-trip serialization (UnifiedJobResponse, CursorPage, JobOutputItem)
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID, uuid4

import msgspec

from src.api.schemas.jobs import (
    JobCreatedResponse,
    JobOutputItem,
    UnifiedJobResponse,
)
from src.api.schemas.media import MediaObject, MediaOriginal
from src.api.schemas.pagination import CursorPage
from src.api.services.unified_jobs import UnifiedJobService
from src.core.enums import GenerationType, JobStatus, OutputMediaType

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
    is_deleted: bool = False,
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
    job.aspect_ratio = None
    job.token_cost = token_cost
    job.error_message = error_message
    job.is_deleted = is_deleted
    job.created_at = created_at or datetime.now(UTC)
    job.started_at = started_at
    job.completed_at = completed_at
    return job


def _make_output(
    *,
    output_id: UUID | None = None,
    job_id: UUID | None = None,
    content_type: str = "image/jpeg",
    size_bytes: int = 2048,
    output_index: int = 0,
    is_thumbnail: bool = False,
    parent_output_id: UUID | None = None,
    thumbnail_max_edge: int | None = None,
    width: int | None = 512,
    height: int | None = 512,
) -> MagicMock:
    out = MagicMock()
    out.id = output_id or uuid4()
    out.job_id = job_id or uuid4()
    out.content_type = content_type
    out.size_bytes = size_bytes
    out.output_index = output_index
    out.is_thumbnail = is_thumbnail
    out.parent_output_id = parent_output_id
    out.thumbnail_max_edge = thumbnail_max_edge
    out.width = width
    out.height = height
    return out


def _make_media() -> MediaObject:
    return MediaObject(
        media_type=OutputMediaType.IMAGE,
        original=MediaOriginal(
            url="/v1/content/outputs/some-id",
            width=512,
            height=512,
            content_type="image/png",
            size_bytes=1024,
        ),
        variants=[],
    )


def _session_for_get(
    job: MagicMock | None,
    full_outputs: list | None = None,
    derivative_outputs: list | None = None,
) -> AsyncMock:
    """Build a mock AsyncSession for ``get_job`` calls.

    Side-effect order:
    1. ``job_repo.get`` → scalar_one_or_none
    2. ``output_repo.list_by_job`` → scalars().all()  (only when job is found)
    3. ``output_repo.batch_derivatives`` → scalars().all()  (only when job is found)
    """
    get_result = MagicMock()
    get_result.scalar_one_or_none.return_value = job

    out_result = MagicMock()
    out_result.scalars.return_value.all.return_value = full_outputs or []

    deriv_result = MagicMock()
    deriv_result.scalars.return_value.all.return_value = derivative_outputs or []

    session = AsyncMock()
    side_effects = [get_result]
    if job is not None:
        side_effects += [out_result, deriv_result]
    session.execute = AsyncMock(side_effect=side_effects)
    return session


def _session_for_list(
    jobs: list,
    full_outputs_per_job: list[list] | None = None,
    derivatives_per_job: list[list] | None = None,
) -> AsyncMock:
    """Build a mock AsyncSession for ``list_jobs`` calls.

    Side-effect order:
    1. ``job_repo.list_by_user`` → scalars().all()
    For each job:
      2i. ``output_repo.list_by_job`` → scalars().all()
      2i+1. ``output_repo.batch_derivatives`` → scalars().all()
    """
    jobs_result = MagicMock()
    jobs_result.scalars.return_value.all.return_value = jobs

    side_effects: list = [jobs_result]
    for i in range(len(jobs)):
        out_result = MagicMock()
        out_result.scalars.return_value.all.return_value = (
            full_outputs_per_job[i] if full_outputs_per_job else []
        )
        side_effects.append(out_result)

        deriv_result = MagicMock()
        deriv_result.scalars.return_value.all.return_value = (
            derivatives_per_job[i] if derivatives_per_job else []
        )
        side_effects.append(deriv_result)

    session = AsyncMock()
    session.execute = AsyncMock(side_effect=side_effects)
    return session


def _service(grok_job_service: AsyncMock | None = None) -> UnifiedJobService:
    return UnifiedJobService(grok_job_service=grok_job_service)


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
            provider="aisha",
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
        assert result.provider == "aisha"
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

        result = await UnifiedJobService(grok_job_service=None).get_job(
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
        # get_job fetches the original job; then 2 more execute calls for _build_response
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
            generation_type=GenerationType.T2I.value,
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
            status=JobStatus.COMPLETED.value,
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
            provider="aisha",
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

        result = await _service(grok_job_service=grok).get_job(job.id, user_id, session=session)

        assert result is not None
        assert result.status == JobStatus.QUEUED

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
# Tests: _build_response — soft-delete regression
# ---------------------------------------------------------------------------


class TestBuildResponse:
    async def test_soft_deleted_job_error_field_not_leaked(self) -> None:
        """A soft-deleted job should not expose '__hidden__' as its error."""
        user_id = uuid4()
        job = _make_job(user_id=user_id, is_deleted=True, error_message=None)
        session = _session_for_get(job)

        result = await _service().get_job(job.id, user_id, session=session)

        assert result is not None
        assert result.error is None


# ---------------------------------------------------------------------------
# Tests: UnifiedJobService.list_jobs
# ---------------------------------------------------------------------------


class TestListJobs:
    async def test_empty_result(self) -> None:
        session = _session_for_list(jobs=[])

        result = await _service().list_jobs(uuid4(), session=session)

        assert isinstance(result, CursorPage)
        assert result.has_more is False
        assert result.items == []

    async def test_default_pagination_params(self) -> None:
        session = _session_for_list(jobs=[])

        result = await _service().list_jobs(uuid4(), session=session)

        assert result.limit == 20

    async def test_limit_capped_at_100(self) -> None:
        session = _session_for_list(jobs=[])

        result = await _service().list_jobs(uuid4(), session=session, limit=500)

        assert result.limit == 100

    async def test_custom_limit_reflected_in_response(self) -> None:
        session = _session_for_list(jobs=[])

        result = await _service().list_jobs(uuid4(), session=session, limit=10)

        assert result.limit == 10

    async def test_returns_one_item_per_job(self) -> None:
        user_id = uuid4()
        jobs = [_make_job(user_id=user_id) for _ in range(3)]
        session = _session_for_list(jobs=jobs)

        result = await _service().list_jobs(user_id, session=session)

        assert len(result.items) == 3

    async def test_all_items_are_unified_job_responses(self) -> None:
        user_id = uuid4()
        jobs = [_make_job(user_id=user_id) for _ in range(2)]
        session = _session_for_list(jobs=jobs)

        result = await _service().list_jobs(user_id, session=session)

        for item in result.items:
            assert isinstance(item, UnifiedJobResponse)

    async def test_job_fields_mapped_correctly_in_list(self) -> None:
        user_id = uuid4()
        job = _make_job(user_id=user_id, name="Gallery Job", prompt="neon city")
        session = _session_for_list(jobs=[job])

        result = await _service().list_jobs(user_id, session=session)

        item = result.items[0]
        assert item.id == job.id
        assert item.name == "Gallery Job"
        assert item.prompt == "neon city"
        assert item.provider == job.provider


# ---------------------------------------------------------------------------
# Tests: _build_response — output and media object logic
# ---------------------------------------------------------------------------


class TestBuildResponseOutputs:
    async def test_no_outputs_returns_empty_list(self) -> None:
        user_id = uuid4()
        job = _make_job(user_id=user_id)
        session = _session_for_get(job, full_outputs=[], derivative_outputs=[])

        result = await _service().get_job(job.id, user_id, session=session)

        assert result is not None
        assert result.outputs == []

    async def test_single_output_has_proxy_url(self) -> None:
        user_id = uuid4()
        job = _make_job(user_id=user_id)
        out_id = uuid4()
        out = _make_output(output_id=out_id, content_type="image/png")
        session = _session_for_get(job, full_outputs=[out], derivative_outputs=[])

        result = await _service().get_job(job.id, user_id, session=session)

        assert result is not None
        assert len(result.outputs) == 1
        item = result.outputs[0]
        assert isinstance(item, JobOutputItem)
        assert item.id == out_id
        assert item.media.original.url == f"/v1/content/outputs/{out_id}"

    async def test_output_item_fields_mapped(self) -> None:
        user_id = uuid4()
        job = _make_job(user_id=user_id)
        out_id = uuid4()
        out = _make_output(
            output_id=out_id,
            content_type="image/png",
            size_bytes=4096,
            output_index=2,
            width=1024,
            height=768,
        )
        session = _session_for_get(job, full_outputs=[out])

        result = await _service().get_job(job.id, user_id, session=session)

        assert result is not None
        item = result.outputs[0]
        assert item.output_index == 2
        assert item.media.original.content_type == "image/png"
        assert item.media.original.size_bytes == 4096
        assert item.media.original.width == 1024
        assert item.media.original.height == 768

    async def test_derivative_appears_as_variant(self) -> None:
        """Thumbnail row returned by batch_derivatives surfaces as media.variants entry."""
        user_id = uuid4()
        job = _make_job(user_id=user_id)
        out_id = uuid4()
        thumb_id = uuid4()

        out = _make_output(output_id=out_id, content_type="image/png")
        thumb = _make_output(
            output_id=thumb_id,
            content_type="image/webp",
            is_thumbnail=True,
            parent_output_id=out_id,
            thumbnail_max_edge=512,
            width=400,
            height=400,
        )
        session = _session_for_get(job, full_outputs=[out], derivative_outputs=[thumb])

        result = await _service().get_job(job.id, user_id, session=session)

        assert result is not None
        item = result.outputs[0]
        assert len(item.media.variants) == 1
        assert item.media.variants[0].url == f"/v1/content/outputs/{thumb_id}"
        assert item.media.variants[0].label == "md"
        assert item.media.variants[0].width == 400

    async def test_multiple_outputs_all_returned(self) -> None:
        # list_by_job in the DB already orders by output_index; the mock returns
        # them in whatever order we pass, which is already sorted here.
        user_id = uuid4()
        job = _make_job(user_id=user_id)
        out0 = _make_output(output_index=0, content_type="image/png")
        out1 = _make_output(output_index=1, content_type="image/png")
        out2 = _make_output(output_index=2, content_type="image/png")
        session = _session_for_get(job, full_outputs=[out0, out1, out2])

        result = await _service().get_job(job.id, user_id, session=session)

        assert result is not None
        assert len(result.outputs) == 3
        indices = [item.output_index for item in result.outputs]
        assert sorted(indices) == [0, 1, 2]

    async def test_no_variants_when_no_derivatives(self) -> None:
        user_id = uuid4()
        job = _make_job(user_id=user_id)
        out = _make_output(content_type="image/png")
        session = _session_for_get(job, full_outputs=[out], derivative_outputs=[])

        result = await _service().get_job(job.id, user_id, session=session)

        assert result is not None
        assert result.outputs[0].media.variants == []

    async def test_video_output_media_type_is_video(self) -> None:
        user_id = uuid4()
        job = _make_job(user_id=user_id)
        out = _make_output(content_type="video/mp4")
        session = _session_for_get(job, full_outputs=[out])

        result = await _service().get_job(job.id, user_id, session=session)

        assert result is not None
        assert result.outputs[0].media.media_type == OutputMediaType.VIDEO


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
        assert decoded.model is None
        assert decoded.error is None

    def test_unified_job_response_optional_fields_preserved(self) -> None:
        now = datetime.now(UTC)
        response = UnifiedJobResponse(
            id=uuid4(),
            name="Full response",
            status=JobStatus.FAILED,
            provider="aisha",
            generation_type=GenerationType.I2V,
            prompt="ocean waves",
            model="aisha",
            negative_prompt="blurry",
            token_cost=120,
            created_at=now,
            started_at=now,
            completed_at=now,
            error="Connection reset by peer",
        )

        encoded = msgspec.json.encode(response)
        decoded = msgspec.json.decode(encoded, type=UnifiedJobResponse)

        assert decoded.status == JobStatus.FAILED
        assert decoded.generation_type == GenerationType.I2V
        assert decoded.model == "aisha"
        assert decoded.negative_prompt == "blurry"
        assert decoded.token_cost == 120
        assert decoded.error == "Connection reset by peer"

    def test_cursor_page_round_trip(self) -> None:
        resp: CursorPage[UnifiedJobResponse] = CursorPage(
            items=[], limit=10, has_more=True, next_cursor="tok"
        )

        encoded = msgspec.json.encode(resp)
        decoded = msgspec.json.decode(encoded, type=CursorPage[UnifiedJobResponse])

        assert decoded.limit == 10
        assert decoded.has_more is True
        assert decoded.next_cursor == "tok"
        assert decoded.items == []

    def test_job_output_item_defaults(self) -> None:
        item = JobOutputItem(
            id=uuid4(),
            output_index=0,
            media=_make_media(),
        )

        assert item.output_index == 0
        assert isinstance(item.media, MediaObject)

    def test_job_output_item_round_trip(self) -> None:
        item_id = uuid4()
        item = JobOutputItem(
            id=item_id,
            output_index=0,
            media=_make_media(),
        )

        encoded = msgspec.json.encode(item)
        decoded = msgspec.json.decode(encoded, type=JobOutputItem)

        assert decoded.id == item_id
        assert decoded.output_index == 0
        assert decoded.media.original.content_type == "image/png"

    def test_job_output_item_with_variants_round_trip(self) -> None:
        from src.api.schemas.media import ImageVariant

        item = JobOutputItem(
            id=uuid4(),
            output_index=1,
            media=MediaObject(
                media_type=OutputMediaType.IMAGE,
                original=MediaOriginal(
                    url="/v1/content/outputs/abc",
                    width=1024,
                    height=1024,
                    content_type="image/png",
                    size_bytes=50000,
                ),
                variants=[
                    ImageVariant(
                        label="sm", width=100, height=100, url="/v1/content/outputs/sm-id"
                    ),
                    ImageVariant(
                        label="md", width=400, height=400, url="/v1/content/outputs/md-id"
                    ),
                ],
            ),
        )

        encoded = msgspec.json.encode(item)
        decoded = msgspec.json.decode(encoded, type=JobOutputItem)

        assert len(decoded.media.variants) == 2
        assert decoded.media.variants[0].label == "sm"
        assert decoded.media.variants[1].label == "md"

    def test_job_created_response_round_trip(self) -> None:
        now = datetime.now(UTC)
        response = JobCreatedResponse(
            job_id=uuid4(),
            status=JobStatus.QUEUED,
            name="Test job",
            model="grok-imagine-image",
            generation_type=GenerationType.T2I,
            created_at=now,
            tokens_charged=100,
            balance_remaining=900,
        )
        encoded = msgspec.json.encode(response)
        decoded = msgspec.json.decode(encoded, type=JobCreatedResponse)
        assert decoded.status == JobStatus.QUEUED
        assert decoded.model == "grok-imagine-image"
        assert decoded.tokens_charged == 100


# ---------------------------------------------------------------------------
# Tests: list_jobs — cursor and has_more paths
# ---------------------------------------------------------------------------


class TestListJobsPaginationCoverage:
    async def test_cursor_decoded_when_provided(self) -> None:
        """Covers line 130: decode_cursor called when cursor is not None."""
        from src.api.schemas.pagination import encode_cursor

        user_id = uuid4()
        now = datetime.now(UTC)
        cursor = encode_cursor(now, user_id)

        session = _session_for_list(jobs=[])
        result = await _service().list_jobs(user_id, session=session, cursor=cursor)

        assert isinstance(result, CursorPage)
        assert result.has_more is False

    async def test_has_more_and_next_cursor_when_extra_jobs_returned(self) -> None:
        """Covers lines 148, 156-157: jobs sliced and next_cursor encoded."""
        user_id = uuid4()
        jobs = [_make_job(user_id=user_id) for _ in range(3)]
        session = _session_for_list(jobs=jobs)

        result = await _service().list_jobs(user_id, session=session, limit=2)

        assert result.has_more is True
        assert len(result.items) == 2
        assert result.next_cursor is not None


# ---------------------------------------------------------------------------
# Tests: _build_response — eager-load and NoInspectionAvailable paths
# ---------------------------------------------------------------------------


class TestBuildResponseEagerLoadCoverage:
    async def test_no_inspection_available_falls_back_to_query_path(self) -> None:
        """Covers lines 187-188: except NoInspectionAvailable sets _outputs_loaded=False."""
        from unittest.mock import patch

        from sqlalchemy.exc import NoInspectionAvailable

        user_id = uuid4()
        job = _make_job(user_id=user_id)
        session = _session_for_get(job, full_outputs=[], derivative_outputs=[])

        with patch(
            "src.api.services.unified_jobs.inspect",
            side_effect=NoInspectionAvailable(),
        ):
            result = await _service().get_job(job.id, user_id, session=session)

        assert result is not None
        assert result.outputs == []

    async def test_eager_loaded_outputs_used_directly(self) -> None:
        """Covers lines 192-199: eagerly-loaded outputs separated in Python."""
        from unittest.mock import patch

        user_id = uuid4()
        job = _make_job(user_id=user_id)

        out_id = uuid4()
        full_out = _make_output(output_id=out_id, is_thumbnail=False, output_index=0)
        thumb_out = _make_output(
            is_thumbnail=True,
            parent_output_id=out_id,
            output_index=0,
            thumbnail_max_edge=512,
            width=400,
            height=400,
        )
        job.outputs = [full_out, thumb_out]

        inspect_result = MagicMock()
        inspect_result.dict = {"outputs": [full_out, thumb_out]}

        get_result = MagicMock()
        get_result.scalar_one_or_none.return_value = job
        session = AsyncMock()
        session.execute = AsyncMock(return_value=get_result)

        with patch("src.api.services.unified_jobs.inspect", return_value=inspect_result):
            result = await _service().get_job(job.id, user_id, session=session)

        assert result is not None
        assert len(result.outputs) == 1  # Only the full output
        assert result.outputs[0].id == out_id

    async def test_eager_loaded_thumbnail_appears_as_variant(self) -> None:
        """Covers lines 195-197: is_thumbnail loop builds derivatives_map."""
        from unittest.mock import patch

        user_id = uuid4()
        job = _make_job(user_id=user_id)

        out_id = uuid4()
        thumb_id = uuid4()
        full_out = _make_output(output_id=out_id, is_thumbnail=False, output_index=0)
        thumb_out = _make_output(
            output_id=thumb_id,
            is_thumbnail=True,
            parent_output_id=out_id,
            output_index=0,
            thumbnail_max_edge=512,
            width=400,
            height=400,
        )
        job.outputs = [full_out, thumb_out]

        inspect_result = MagicMock()
        inspect_result.dict = {"outputs": [full_out, thumb_out]}

        get_result = MagicMock()
        get_result.scalar_one_or_none.return_value = job
        session = AsyncMock()
        session.execute = AsyncMock(return_value=get_result)

        with patch("src.api.services.unified_jobs.inspect", return_value=inspect_result):
            result = await _service().get_job(job.id, user_id, session=session)

        assert result is not None
        assert len(result.outputs[0].media.variants) == 1
        assert result.outputs[0].media.variants[0].label == "md"
