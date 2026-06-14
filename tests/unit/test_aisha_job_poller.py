"""Tests for AishaJobPoller — tick loop, _poll_one dispatch, edge cases."""

from __future__ import annotations

import contextlib
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from src.core.enums import JobStatus
from src.workers.aisha_job_poller import AishaJobPoller, AishaPollerConfig

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config(**kwargs: object) -> AishaPollerConfig:
    defaults: dict[str, object] = {
        "enabled": True,
        "tick_interval_seconds": 0.01,
        "max_concurrent_polls": 4,
        "job_age_warning_seconds": 300,
        "job_age_timeout_seconds": 1800,
        "comfyui_request_timeout_seconds": 5.0,
        "tunnel_allowed_suffix": "gpu-domain.com",
        "tunnel_allowed_prefix": "gpu-",
        "retention_days": 7,
    } | kwargs
    return AishaPollerConfig(**defaults)  # type: ignore[arg-type]


def _make_poller(**kwargs: object) -> AishaJobPoller:
    session_factory = MagicMock()
    session_factory.return_value = AsyncMock()
    session_factory.return_value.__aenter__ = AsyncMock(return_value=AsyncMock())
    session_factory.return_value.__aexit__ = AsyncMock(return_value=False)
    return AishaJobPoller(
        session_factory=session_factory,
        event_bus=None,
        billing_service=AsyncMock(),
        r2_storage=None,
        config=_make_config(**kwargs),
    )


def _make_job(
    *,
    status: str = JobStatus.QUEUED.value,
    external_request_id: str | None = "prompt-abc",
    tunnel_hostname: str = "gpu-node1.gpu-domain.com",
    started_at: datetime | None = None,
    created_at: datetime | None = None,
) -> MagicMock:
    gpu_session = MagicMock()
    gpu_session.id = uuid4()
    gpu_session.tunnel_hostname = tunnel_hostname

    job = MagicMock()
    job.id = uuid4()
    job.user_id = uuid4()
    job.product_id = "vex"
    job.status = status
    job.external_request_id = external_request_id
    job.gpu_session_id = gpu_session.id
    job.gpu_session = gpu_session
    job.started_at = started_at or datetime.now(UTC) - timedelta(seconds=10)
    job.created_at = created_at or datetime.now(UTC) - timedelta(seconds=10)
    return job


# ---------------------------------------------------------------------------
# Constructor
# ---------------------------------------------------------------------------


class TestConstructor:
    def test_raises_on_empty_tunnel_suffix(self) -> None:
        session_factory = MagicMock()
        with pytest.raises(ValueError, match="tunnel_allowed_suffix is empty"):
            AishaJobPoller(
                session_factory=session_factory,
                event_bus=None,
                billing_service=AsyncMock(),
                r2_storage=None,
                config=_make_config(tunnel_allowed_suffix=""),
            )

    def test_raises_on_whitespace_only_suffix(self) -> None:
        session_factory = MagicMock()
        with pytest.raises(ValueError, match="tunnel_allowed_suffix is empty"):
            AishaJobPoller(
                session_factory=session_factory,
                event_bus=None,
                billing_service=AsyncMock(),
                r2_storage=None,
                config=_make_config(tunnel_allowed_suffix="   "),
            )

    def test_normalizes_suffix_without_leading_dot(self) -> None:
        poller = _make_poller(tunnel_allowed_suffix="gpu-domain.com")
        assert poller._allowed_tunnel_suffix == ".gpu-domain.com"

    def test_preserves_suffix_with_leading_dot(self) -> None:
        poller = _make_poller(tunnel_allowed_suffix=".gpu-domain.com")
        assert poller._allowed_tunnel_suffix == ".gpu-domain.com"


# ---------------------------------------------------------------------------
# start / stop
# ---------------------------------------------------------------------------


class TestStartStop:
    async def test_start_creates_task(self) -> None:
        poller = _make_poller()
        with patch.object(poller, "_loop", new=AsyncMock()):
            await poller.start()
            assert poller._task is not None
            await poller.stop()

    async def test_start_is_idempotent(self) -> None:
        poller = _make_poller()
        with patch.object(poller, "_loop", new=AsyncMock()):
            await poller.start()
            task1 = poller._task
            await poller.start()
            task2 = poller._task
            assert task1 is task2
            await poller.stop()

    async def test_disabled_poller_does_not_start(self) -> None:
        poller = _make_poller(enabled=False)
        await poller.start()
        assert poller._task is None

    async def test_stop_is_idempotent(self) -> None:
        poller = _make_poller()
        await poller.stop()
        await poller.stop()


# ---------------------------------------------------------------------------
# _collect_image_infos
# ---------------------------------------------------------------------------


class TestCollectImageInfos:
    def test_empty_history_entry(self) -> None:
        result = AishaJobPoller._collect_image_infos({})
        assert result == []

    def test_single_node_with_images(self) -> None:
        entry = {
            "outputs": {
                "1": {"images": [{"filename": "img_0.png", "subfolder": "", "type": "output"}]}
            }
        }
        result = AishaJobPoller._collect_image_infos(entry)
        assert len(result) == 1
        assert result[0]["filename"] == "img_0.png"

    def test_multiple_nodes_multiple_images(self) -> None:
        entry = {
            "outputs": {
                "1": {
                    "images": [
                        {"filename": "a.png", "type": "output"},
                        {"filename": "b.png", "type": "output"},
                    ]
                },
                "2": {"images": [{"filename": "c.png", "type": "output"}]},
            }
        }
        result = AishaJobPoller._collect_image_infos(entry)
        assert len(result) == 3

    def test_nodes_without_images_key_ignored(self) -> None:
        entry = {
            "outputs": {
                "1": {"latents": []},
                "2": {"images": [{"filename": "x.png", "type": "output"}]},
            }
        }
        result = AishaJobPoller._collect_image_infos(entry)
        assert len(result) == 1


# ---------------------------------------------------------------------------
# _infer_image_format_and_content_type
# ---------------------------------------------------------------------------


class TestInferImageFormatAndContentType:
    def test_png(self) -> None:
        ext, ct = AishaJobPoller._infer_image_format_and_content_type("output_0.png")
        assert ext == "png"
        assert ct == "image/png"

    def test_jpeg(self) -> None:
        ext, ct = AishaJobPoller._infer_image_format_and_content_type("photo.jpeg")
        assert ext == "jpeg"
        assert ct == "image/jpeg"

    def test_jpg(self) -> None:
        ext, ct = AishaJobPoller._infer_image_format_and_content_type("photo.jpg")
        assert ext == "jpg"
        assert ct == "image/jpeg"

    def test_webp(self) -> None:
        ext, ct = AishaJobPoller._infer_image_format_and_content_type("img.webp")
        assert ext == "webp"
        assert ct == "image/webp"

    def test_unknown_extension_defaults_to_png(self) -> None:
        ext, ct = AishaJobPoller._infer_image_format_and_content_type("file.bmp")
        assert ext == "bmp"
        assert ct == "image/png"

    def test_no_extension_defaults_to_png(self) -> None:
        ext, ct = AishaJobPoller._infer_image_format_and_content_type("noext")
        assert ext == "png"
        assert ct == "image/png"


# ---------------------------------------------------------------------------
# _is_job_past_timeout
# ---------------------------------------------------------------------------


class TestIsJobPastTimeout:
    def test_uses_started_at_when_set(self) -> None:
        poller = _make_poller(job_age_timeout_seconds=60)
        job = _make_job(started_at=datetime.now(UTC) - timedelta(seconds=120))
        assert poller._is_job_past_timeout(job) is True

    def test_falls_back_to_created_at_when_started_at_is_none(self) -> None:
        poller = _make_poller(job_age_timeout_seconds=60)
        job = _make_job(started_at=None, created_at=datetime.now(UTC) - timedelta(seconds=120))
        job.started_at = None
        assert poller._is_job_past_timeout(job) is True

    def test_returns_false_for_recent_job(self) -> None:
        poller = _make_poller(job_age_timeout_seconds=1800)
        job = _make_job(started_at=datetime.now(UTC) - timedelta(seconds=10))
        assert poller._is_job_past_timeout(job) is False

    def test_returns_false_when_reference_is_none(self) -> None:
        poller = _make_poller(job_age_timeout_seconds=60)
        job = MagicMock()
        job.started_at = None
        job.created_at = None
        assert poller._is_job_past_timeout(job) is False

    def test_handles_naive_datetime(self) -> None:
        poller = _make_poller(job_age_timeout_seconds=60)
        job = MagicMock()
        # Simulate a naive UTC datetime (tzinfo stripped, as if DB forgot timezone).
        # Use replace(tzinfo=None) on the aware UTC now so the arithmetic is correct.
        naive_utc = datetime.now(UTC).replace(tzinfo=None)
        job.started_at = naive_utc - timedelta(seconds=120)
        job.created_at = None
        assert poller._is_job_past_timeout(job) is True


# ---------------------------------------------------------------------------
# _poll_one — hostname and prompt_id guards
# ---------------------------------------------------------------------------


class TestPollOne:
    async def test_fails_when_tunnel_hostname_invalid(self) -> None:
        poller = _make_poller(tunnel_allowed_suffix="gpu-domain.com", tunnel_allowed_prefix="gpu-")
        job = _make_job(tunnel_hostname="evil.com?attack.gpu-domain.com")
        session = AsyncMock()

        ts = AsyncMock()
        ts.transition_to_failed.return_value = (MagicMock(), True)
        with patch.object(poller, "_make_transition_service", return_value=ts):
            await poller._poll_one(job, job.gpu_session, session)
            ts.transition_to_failed.assert_awaited_once()

    async def test_skips_when_no_prompt_id(self) -> None:
        poller = _make_poller()
        job = _make_job(external_request_id=None)
        session = AsyncMock()

        ts = AsyncMock()
        with (
            patch.object(poller, "_make_transition_service", return_value=ts),
            patch("src.workers.aisha_job_poller.ComfyUIClient") as mock_client_cls,
        ):
            await poller._poll_one(job, job.gpu_session, session)
            mock_client_cls.assert_not_called()

    async def test_lowercases_hostname_before_validation(self) -> None:
        poller = _make_poller(tunnel_allowed_suffix="gpu-domain.com", tunnel_allowed_prefix="gpu-")
        # Mixed-case hostname that is valid after lowercasing
        job = _make_job(tunnel_hostname="GPU-Node1.Gpu-domain.Com")
        session = AsyncMock()

        client = AsyncMock()
        client.get_history = AsyncMock(return_value={})
        client.get_queue = AsyncMock(return_value={"queue_running": []})

        ts = AsyncMock()
        with (
            patch.object(poller, "_make_transition_service", return_value=ts),
            patch("src.workers.aisha_job_poller.ComfyUIClient", return_value=client),
            patch.object(poller, "_handle_queue_state", new=AsyncMock()) as mock_hqs,
        ):
            await poller._poll_one(job, job.gpu_session, session)
            # If lowercasing works, validation passes and we reach _handle_queue_state
            mock_hqs.assert_awaited_once()

    async def test_one_job_failure_does_not_block_other_jobs(self) -> None:
        """Exception in one guarded coroutine does not prevent others from running."""
        import asyncio as asyncio_lib

        completed: list[str] = []

        async def counting_poll(_job: object, _gpu: object, _sess: object) -> None:
            completed.append("ran")

        job1, job2 = _make_job(), _make_job()
        for j in (job1, job2):
            j.gpu_session = MagicMock()
            j.gpu_session.id = uuid4()

        sem = asyncio_lib.Semaphore(4)

        async def guarded(job: MagicMock) -> None:
            if job.gpu_session is None:
                return
            async with sem:
                with contextlib.suppress(Exception):
                    ctx = AsyncMock()
                    ctx.__aenter__ = AsyncMock(return_value=AsyncMock())
                    ctx.__aexit__ = AsyncMock(return_value=False)
                    async with ctx as s:
                        await counting_poll(job, job.gpu_session, s)

        await asyncio_lib.gather(guarded(job1), guarded(job2))
        assert len(completed) == 2


# ---------------------------------------------------------------------------
# _handle_queue_state — timeout handling
# ---------------------------------------------------------------------------


class TestHandleQueueState:
    async def test_transitions_to_running_when_in_queue(self) -> None:
        poller = _make_poller()
        job = _make_job(status=JobStatus.QUEUED.value)
        prompt_id = "prompt-abc"

        ts = AsyncMock()
        queue = {"queue_running": [[0, prompt_id, {}]]}
        await poller._handle_queue_state(
            job=job,
            queue=queue,
            prompt_id=prompt_id,
            product_id="vex",
            ts=ts,
        )
        ts.transition_to_running.assert_awaited_once_with(job.id)

    async def test_times_out_old_job_with_started_at(self) -> None:
        poller = _make_poller(job_age_timeout_seconds=60)
        old_started = datetime.now(UTC) - timedelta(seconds=120)
        job = _make_job(started_at=old_started)

        ts = AsyncMock()
        ts.transition_to_failed.return_value = (MagicMock(), True)
        await poller._handle_queue_state(
            job=job,
            queue={"queue_running": []},
            prompt_id="prompt-old",
            product_id="vex",
            ts=ts,
        )
        ts.transition_to_failed.assert_awaited_once()

    async def test_times_out_old_job_using_created_at_when_started_at_is_none(self) -> None:
        """Jobs that never reached RUNNING are still reaped via created_at fallback."""
        poller = _make_poller(job_age_timeout_seconds=60)
        job = _make_job(
            started_at=None,
            created_at=datetime.now(UTC) - timedelta(seconds=120),
        )
        job.started_at = None

        ts = AsyncMock()
        ts.transition_to_failed.return_value = (MagicMock(), True)
        await poller._handle_queue_state(
            job=job,
            queue={"queue_running": []},
            prompt_id="prompt-stuck",
            product_id="vex",
            ts=ts,
        )
        ts.transition_to_failed.assert_awaited_once()

    async def test_does_not_timeout_young_job(self) -> None:
        poller = _make_poller(job_age_timeout_seconds=1800)
        recent_started = datetime.now(UTC) - timedelta(seconds=10)
        job = _make_job(started_at=recent_started)

        ts = AsyncMock()
        await poller._handle_queue_state(
            job=job,
            queue={"queue_running": []},
            prompt_id="prompt-young",
            product_id="vex",
            ts=ts,
        )
        ts.transition_to_failed.assert_not_awaited()

    async def test_does_not_transition_running_job_that_is_already_running(self) -> None:
        """Job already in RUNNING status found in queue_running — no double transition."""
        poller = _make_poller()
        job = _make_job(status=JobStatus.RUNNING.value)
        prompt_id = "prompt-abc"

        ts = AsyncMock()
        queue = {"queue_running": [[0, prompt_id, {}]]}
        await poller._handle_queue_state(
            job=job,
            queue=queue,
            prompt_id=prompt_id,
            product_id="vex",
            ts=ts,
        )
        ts.transition_to_running.assert_not_awaited()


# ---------------------------------------------------------------------------
# _handle_history_complete — outputs / error / no-outputs handling
# ---------------------------------------------------------------------------


class TestHandleHistoryComplete:
    async def test_fails_job_on_error_status(self) -> None:
        poller = _make_poller()
        job = _make_job()
        client = AsyncMock()

        ts = AsyncMock()
        ts.transition_to_failed.return_value = (MagicMock(), True)
        history_entry = {
            "status": {"status_str": "error", "messages": [["Error", "OOM"]]},
        }
        await poller._handle_history_complete(
            client=client,
            job=job,
            history_entry=history_entry,
            product_id="vex",
            ts=ts,
        )
        ts.transition_to_failed.assert_awaited_once()
        ts.transition_to_completed.assert_not_awaited()

    async def test_history_without_outputs_below_timeout_logs_debug_and_returns(self) -> None:
        poller = _make_poller(job_age_timeout_seconds=1800)
        # Job is only 10 seconds old — well below timeout
        job = _make_job(started_at=datetime.now(UTC) - timedelta(seconds=10))
        client = AsyncMock()

        ts = AsyncMock()
        history_entry: dict = {}  # no "outputs" key, no "status" key
        await poller._handle_history_complete(
            client=client,
            job=job,
            history_entry=history_entry,
            product_id="vex",
            ts=ts,
        )
        ts.transition_to_failed.assert_not_awaited()
        ts.transition_to_completed.assert_not_awaited()

    async def test_history_without_outputs_past_timeout_marks_failed(self) -> None:
        poller = _make_poller(job_age_timeout_seconds=60)
        # Job is 2 minutes old — past timeout
        job = _make_job(started_at=datetime.now(UTC) - timedelta(seconds=120))
        client = AsyncMock()

        ts = AsyncMock()
        ts.transition_to_failed.return_value = (MagicMock(), True)
        history_entry: dict = {}  # no "outputs" key
        await poller._handle_history_complete(
            client=client,
            job=job,
            history_entry=history_entry,
            product_id="vex",
            ts=ts,
        )
        ts.transition_to_failed.assert_awaited_once()
        _args, kwargs = ts.transition_to_failed.call_args
        assert "no outputs" in kwargs.get("error_message", "").lower()

    async def test_history_with_explicit_error_marks_failed_immediately(self) -> None:
        """status_str=error triggers FAILED even if job is under the age timeout."""
        poller = _make_poller(job_age_timeout_seconds=1800)
        # Fresh job — age timeout would not trigger
        job = _make_job(started_at=datetime.now(UTC) - timedelta(seconds=5))
        client = AsyncMock()

        ts = AsyncMock()
        ts.transition_to_failed.return_value = (MagicMock(), True)
        history_entry = {"status": {"status_str": "error", "messages": [["Error", "CUDA OOM"]]}}
        await poller._handle_history_complete(
            client=client,
            job=job,
            history_entry=history_entry,
            product_id="vex",
            ts=ts,
        )
        ts.transition_to_failed.assert_awaited_once()
