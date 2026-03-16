"""Tests for AishaJobService — DB-backed ComfyUI image job lifecycle.

Covers:
  - poll_image_job precondition checks (missing job, terminal status, no prompt_id)
  - History entry handling: COMPLETED with outputs, FAILED on error status
  - Queue state: RUNNING transition
  - Error resilience: history/queue failures, partial output failures
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID, uuid4

from src.api.services.aisha_job_service import AishaJobService
from src.core.enums import JobStatus

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_job(
    *,
    job_id: UUID | None = None,
    user_id: UUID | None = None,
    status: str = JobStatus.QUEUED.value,
    external_request_id: str | None = "prompt-abc123",
) -> MagicMock:
    job = MagicMock()
    job.id = job_id or uuid4()
    job.user_id = user_id or uuid4()
    job.status = status
    job.external_request_id = external_request_id
    job.started_at = None
    job.completed_at = None
    job.error_message = None
    return job


def _make_session(job: MagicMock | None = None) -> AsyncMock:
    """Build a mock AsyncSession where session.get() returns *job*."""
    session = AsyncMock()
    session.get = AsyncMock(return_value=job)
    session.flush = AsyncMock()
    session.add = MagicMock()
    return session


def _make_comfyui(
    *,
    history: dict | None = None,
    queue: dict | None = None,
    image_bytes: bytes = b"fakeimage",
) -> AsyncMock:
    client = AsyncMock()
    client.get_history = AsyncMock(return_value=history or {})
    client.get_queue = AsyncMock(return_value=queue or {"queue_running": [], "queue_pending": []})
    client.get_image = AsyncMock(return_value=image_bytes)
    return client


def _make_storage() -> AsyncMock:
    result = MagicMock()
    result.id = uuid4()
    result.storage_key = "users/uid/outputs/job/img_0.png"
    result.expires_at = None
    storage = AsyncMock()
    storage.upload = AsyncMock(return_value=result)
    return storage


def _service(
    comfyui: AsyncMock | None = None,
    storage: AsyncMock | None = None,
) -> AishaJobService:
    return AishaJobService(
        comfyui_client=comfyui or _make_comfyui(),
        storage=storage or _make_storage(),
        retention_days=7,
    )


def _history_with_outputs(prompt_id: str, filenames: list[str]) -> dict:
    outputs: dict = {
        str(i): {"images": [{"filename": fname, "subfolder": "", "type": "output"}]}
        for i, fname in enumerate(filenames)
    }
    return {prompt_id: {"outputs": outputs}}


def _history_with_error(prompt_id: str) -> dict:
    return {prompt_id: {"status": {"status_str": "error", "messages": [["Error", "OOM"]]}}}


def _queue_with_running(prompt_id: str) -> dict:
    return {"queue_running": [["0", prompt_id]], "queue_pending": []}


def _queue_with_pending(prompt_id: str) -> dict:
    return {"queue_running": [], "queue_pending": [["0", prompt_id]]}


# ---------------------------------------------------------------------------
# Precondition / early-exit tests
# ---------------------------------------------------------------------------


class TestPollJobPreconditions:
    async def test_returns_none_when_job_not_found(self) -> None:
        session = _make_session(job=None)
        result = await _service().poll_image_job(session, uuid4())
        assert result is None

    async def test_returns_none_when_job_already_completed(self) -> None:
        job = _make_job(status=JobStatus.COMPLETED.value)
        session = _make_session(job)
        result = await _service().poll_image_job(session, job.id)
        assert result is None

    async def test_returns_none_when_job_already_failed(self) -> None:
        job = _make_job(status=JobStatus.FAILED.value)
        session = _make_session(job)
        result = await _service().poll_image_job(session, job.id)
        assert result is None

    async def test_returns_none_when_no_prompt_id(self) -> None:
        job = _make_job(external_request_id=None)
        session = _make_session(job)
        result = await _service().poll_image_job(session, job.id)
        assert result is None

    async def test_returns_none_when_history_call_raises(self) -> None:
        job = _make_job()
        session = _make_session(job)
        comfyui = _make_comfyui()
        comfyui.get_history = AsyncMock(side_effect=RuntimeError("ComfyUI down"))
        result = await _service(comfyui=comfyui).poll_image_job(session, job.id)
        assert result is None


# ---------------------------------------------------------------------------
# History entry — COMPLETED
# ---------------------------------------------------------------------------


class TestHistoryCompleted:
    async def test_transitions_job_to_completed(self) -> None:
        prompt_id = "prompt-xyz"
        job = _make_job(external_request_id=prompt_id)
        session = _make_session(job)
        history = _history_with_outputs(prompt_id, ["output_0.png"])
        comfyui = _make_comfyui(history=history)

        result = await _service(comfyui=comfyui).poll_image_job(session, job.id)

        assert result is not None
        assert result.status == JobStatus.COMPLETED.value
        assert result.completed_at is not None

    async def test_flushes_after_completing(self) -> None:
        prompt_id = "prompt-flush"
        job = _make_job(external_request_id=prompt_id)
        session = _make_session(job)
        history = _history_with_outputs(prompt_id, ["output_0.png"])
        comfyui = _make_comfyui(history=history)

        await _service(comfyui=comfyui).poll_image_job(session, job.id)

        session.flush.assert_awaited()

    async def test_downloads_image_from_comfyui(self) -> None:
        prompt_id = "prompt-dl"
        job = _make_job(external_request_id=prompt_id)
        session = _make_session(job)
        history = _history_with_outputs(prompt_id, ["gen_abc123.png"])
        comfyui = _make_comfyui(history=history)

        await _service(comfyui=comfyui).poll_image_job(session, job.id)

        comfyui.get_image.assert_awaited_once_with(
            filename="gen_abc123.png", subfolder="", folder_type="output"
        )

    async def test_uploads_image_to_r2(self) -> None:
        prompt_id = "prompt-r2"
        job = _make_job(external_request_id=prompt_id)
        session = _make_session(job)
        history = _history_with_outputs(prompt_id, ["out.png"])
        comfyui = _make_comfyui(history=history, image_bytes=b"pngdata")
        storage = _make_storage()

        await _service(comfyui=comfyui, storage=storage).poll_image_job(session, job.id)

        storage.upload.assert_awaited_once()
        call_kwargs = storage.upload.call_args.kwargs
        assert call_kwargs["user_id"] == job.user_id
        assert call_kwargs["job_id"] == job.id
        assert call_kwargs["content_type"] == "image/png"
        assert call_kwargs["data"] == b"pngdata"

    async def test_stores_multiple_outputs(self) -> None:
        prompt_id = "prompt-multi"
        job = _make_job(external_request_id=prompt_id)
        session = _make_session(job)
        history = _history_with_outputs(prompt_id, ["img_0.png", "img_1.png", "img_2.png"])
        comfyui = _make_comfyui(history=history)
        storage = _make_storage()

        await _service(comfyui=comfyui, storage=storage).poll_image_job(session, job.id)

        assert storage.upload.await_count == 3
        assert session.add.call_count == 3

    async def test_infers_jpeg_content_type_from_extension(self) -> None:
        prompt_id = "prompt-jpeg"
        job = _make_job(external_request_id=prompt_id)
        session = _make_session(job)
        history = _history_with_outputs(prompt_id, ["output.jpg"])
        comfyui = _make_comfyui(history=history)
        storage = _make_storage()

        await _service(comfyui=comfyui, storage=storage).poll_image_job(session, job.id)

        call_kwargs = storage.upload.call_args.kwargs
        assert call_kwargs["content_type"] == "image/jpeg"

    async def test_partial_store_failure_still_marks_completed(self) -> None:
        """If one output fails to store the job still transitions to COMPLETED."""
        prompt_id = "prompt-partial"
        job = _make_job(external_request_id=prompt_id)
        session = _make_session(job)
        history = _history_with_outputs(prompt_id, ["ok.png", "bad.png"])
        comfyui = _make_comfyui(history=history)
        storage = _make_storage()
        storage.upload = AsyncMock(
            side_effect=[MagicMock(id=uuid4(), storage_key="key"), RuntimeError("R2 error")]
        )

        result = await _service(comfyui=comfyui, storage=storage).poll_image_job(session, job.id)

        assert result is not None
        assert result.status == JobStatus.COMPLETED.value

    async def test_output_with_missing_filename_is_skipped(self) -> None:
        prompt_id = "prompt-nofname"
        job = _make_job(external_request_id=prompt_id)
        session = _make_session(job)
        # img_info with empty filename
        history = {
            prompt_id: {
                "outputs": {"0": {"images": [{"filename": "", "subfolder": "", "type": "output"}]}}
            }
        }
        comfyui = _make_comfyui(history=history)
        storage = _make_storage()

        result = await _service(comfyui=comfyui, storage=storage).poll_image_job(session, job.id)

        assert result is not None
        assert result.status == JobStatus.COMPLETED.value
        storage.upload.assert_not_awaited()


# ---------------------------------------------------------------------------
# History entry — FAILED
# ---------------------------------------------------------------------------


class TestHistoryFailed:
    async def test_transitions_job_to_failed_on_error_status(self) -> None:
        prompt_id = "prompt-err"
        job = _make_job(external_request_id=prompt_id)
        session = _make_session(job)
        history = _history_with_error(prompt_id)
        comfyui = _make_comfyui(history=history)

        result = await _service(comfyui=comfyui).poll_image_job(session, job.id)

        assert result is not None
        assert result.status == JobStatus.FAILED.value
        assert result.error_message is not None
        assert result.completed_at is not None

    async def test_returns_none_for_history_without_outputs_or_error(self) -> None:
        prompt_id = "prompt-empty"
        job = _make_job(external_request_id=prompt_id)
        session = _make_session(job)
        # History entry exists but has neither outputs nor error status
        history = {prompt_id: {"status": {"status_str": "success"}}}
        comfyui = _make_comfyui(history=history)

        result = await _service(comfyui=comfyui).poll_image_job(session, job.id)

        assert result is None


# ---------------------------------------------------------------------------
# Queue state
# ---------------------------------------------------------------------------


class TestQueueState:
    async def test_transitions_queued_job_to_running_when_in_running_queue(self) -> None:
        prompt_id = "prompt-run"
        job = _make_job(status=JobStatus.QUEUED.value, external_request_id=prompt_id)
        session = _make_session(job)
        queue = _queue_with_running(prompt_id)
        comfyui = _make_comfyui(history={}, queue=queue)

        result = await _service(comfyui=comfyui).poll_image_job(session, job.id)

        assert result is not None
        assert result.status == JobStatus.RUNNING.value

    async def test_sets_started_at_when_transitioning_to_running(self) -> None:
        prompt_id = "prompt-start"
        job = _make_job(status=JobStatus.QUEUED.value, external_request_id=prompt_id)
        job.started_at = None
        session = _make_session(job)
        queue = _queue_with_running(prompt_id)
        comfyui = _make_comfyui(history={}, queue=queue)

        result = await _service(comfyui=comfyui).poll_image_job(session, job.id)

        assert result is not None
        assert result.started_at is not None

    async def test_already_running_job_not_updated_again(self) -> None:
        prompt_id = "prompt-alreadyrun"
        job = _make_job(status=JobStatus.RUNNING.value, external_request_id=prompt_id)
        session = _make_session(job)
        queue = _queue_with_running(prompt_id)
        comfyui = _make_comfyui(history={}, queue=queue)

        result = await _service(comfyui=comfyui).poll_image_job(session, job.id)

        assert result is None

    async def test_returns_none_when_prompt_still_pending(self) -> None:
        prompt_id = "prompt-pending"
        job = _make_job(external_request_id=prompt_id)
        session = _make_session(job)
        queue = _queue_with_pending(prompt_id)
        comfyui = _make_comfyui(history={}, queue=queue)

        result = await _service(comfyui=comfyui).poll_image_job(session, job.id)

        assert result is None

    async def test_returns_none_when_not_in_history_or_queue(self) -> None:
        prompt_id = "prompt-lost"
        job = _make_job(external_request_id=prompt_id)
        session = _make_session(job)
        comfyui = _make_comfyui(history={}, queue={"queue_running": [], "queue_pending": []})

        result = await _service(comfyui=comfyui).poll_image_job(session, job.id)

        assert result is None

    async def test_returns_none_when_queue_check_fails(self) -> None:
        prompt_id = "prompt-qfail"
        job = _make_job(external_request_id=prompt_id)
        session = _make_session(job)
        comfyui = _make_comfyui(history={})
        comfyui.get_queue = AsyncMock(side_effect=RuntimeError("queue unreachable"))

        result = await _service(comfyui=comfyui).poll_image_job(session, job.id)

        assert result is None


# ---------------------------------------------------------------------------
# UnifiedJobService Aisha poll-on-read integration
# ---------------------------------------------------------------------------


class TestUnifiedJobServiceAishaPollOnRead:
    """Verify that UnifiedJobService triggers AishaJobService.poll_image_job
    on read for QUEUED/RUNNING Aisha image jobs."""

    def _make_unified_service(
        self,
        aisha_mock: AsyncMock | None = None,
    ) -> object:
        from src.api.services.unified_jobs import UnifiedJobService

        storage = AsyncMock()
        url_result = MagicMock()
        url_result.presigned_url = "https://r2.example.com/img.jpg"
        storage.get_presigned_url = AsyncMock(return_value=url_result)

        return UnifiedJobService(
            storage=storage,
            grok_job_service=None,
            aisha_job_service=aisha_mock,
        )

    def _make_aisha_db_job(
        self,
        *,
        status: str = JobStatus.QUEUED.value,
        generation_type: str = "t2i",
    ) -> MagicMock:
        job = MagicMock()
        job.id = uuid4()
        job.user_id = uuid4()
        job.provider = "aisha"
        job.model = "aisha-image"
        job.generation_type = generation_type
        job.status = status
        job.prompt = "a cat"
        job.name = "Test"
        job.negative_prompt = None
        job.aspect_ratio = None
        job.token_cost = 50
        job.error_message = None
        job.created_at = datetime.now(UTC)
        job.started_at = None
        job.completed_at = None
        return job

    def _make_session_for_job(self, job: MagicMock) -> AsyncMock:
        get_result = MagicMock()
        get_result.scalar_one_or_none.return_value = job
        out_result = MagicMock()
        out_result.scalars.return_value.all.return_value = []
        session = AsyncMock()
        session.execute = AsyncMock(side_effect=[get_result, out_result])
        return session

    async def test_aisha_queued_job_is_polled(self) -> None:
        job = self._make_aisha_db_job(status=JobStatus.QUEUED.value)
        session = self._make_session_for_job(job)

        aisha_mock = AsyncMock()
        aisha_mock.poll_image_job = AsyncMock(return_value=None)

        svc = self._make_unified_service(aisha_mock=aisha_mock)
        await svc.get_job(job.id, job.user_id, session=session)  # type: ignore[union-attr]

        aisha_mock.poll_image_job.assert_awaited_once_with(session, job.id)

    async def test_aisha_running_job_is_polled(self) -> None:
        job = self._make_aisha_db_job(status=JobStatus.RUNNING.value)
        session = self._make_session_for_job(job)

        aisha_mock = AsyncMock()
        aisha_mock.poll_image_job = AsyncMock(return_value=None)

        svc = self._make_unified_service(aisha_mock=aisha_mock)
        await svc.get_job(job.id, job.user_id, session=session)  # type: ignore[union-attr]

        aisha_mock.poll_image_job.assert_awaited_once()

    async def test_aisha_completed_job_not_polled(self) -> None:
        job = self._make_aisha_db_job(status=JobStatus.COMPLETED.value)
        session = self._make_session_for_job(job)

        aisha_mock = AsyncMock()
        aisha_mock.poll_image_job = AsyncMock()

        svc = self._make_unified_service(aisha_mock=aisha_mock)
        await svc.get_job(job.id, job.user_id, session=session)  # type: ignore[union-attr]

        aisha_mock.poll_image_job.assert_not_awaited()

    async def test_aisha_poll_failure_swallowed(self) -> None:
        job = self._make_aisha_db_job(status=JobStatus.QUEUED.value)
        session = self._make_session_for_job(job)

        aisha_mock = AsyncMock()
        aisha_mock.poll_image_job = AsyncMock(side_effect=RuntimeError("ComfyUI down"))

        svc = self._make_unified_service(aisha_mock=aisha_mock)
        result = await svc.get_job(job.id, job.user_id, session=session)  # type: ignore[union-attr]

        assert result is not None
        assert result.status == JobStatus.QUEUED

    async def test_no_aisha_poll_when_service_is_none(self) -> None:
        job = self._make_aisha_db_job(status=JobStatus.QUEUED.value)
        session = self._make_session_for_job(job)

        svc = self._make_unified_service(aisha_mock=None)
        result = await svc.get_job(job.id, job.user_id, session=session)  # type: ignore[union-attr]

        assert result is not None

    async def test_grok_job_not_polled_by_aisha_service(self) -> None:
        job = self._make_aisha_db_job(status=JobStatus.QUEUED.value)
        job.provider = "grok"
        session = self._make_session_for_job(job)

        aisha_mock = AsyncMock()
        aisha_mock.poll_image_job = AsyncMock()

        svc = self._make_unified_service(aisha_mock=aisha_mock)
        await svc.get_job(job.id, job.user_id, session=session)  # type: ignore[union-attr]

        aisha_mock.poll_image_job.assert_not_awaited()

    async def test_poll_returns_updated_job_used_for_response(self) -> None:
        job = self._make_aisha_db_job(status=JobStatus.QUEUED.value)
        updated = self._make_aisha_db_job(status=JobStatus.COMPLETED.value)
        updated.id = job.id
        session = self._make_session_for_job(job)

        aisha_mock = AsyncMock()
        aisha_mock.poll_image_job = AsyncMock(return_value=updated)

        svc = self._make_unified_service(aisha_mock=aisha_mock)
        result = await svc.get_job(job.id, job.user_id, session=session)  # type: ignore[union-attr]

        assert result is not None
        assert result.status == JobStatus.COMPLETED
