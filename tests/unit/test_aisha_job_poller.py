"""Tests for AishaJobPoller — tick loop, _poll_one dispatch, edge cases."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

from src.core.enums import GpuSessionStatus, JobStatus
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
        "tunnel_allowed_suffix": "gpu.cloudin.space",
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
    tunnel_hostname: str = "node1.gpu.cloudin.space",
    session_status: str = GpuSessionStatus.active.value,
    started_at: datetime | None = None,
) -> MagicMock:
    gpu_session = MagicMock()
    gpu_session.id = uuid4()
    gpu_session.status = session_status
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
    return job


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
                "1": {"images": [{"filename": "a.png"}, {"filename": "b.png"}]},
                "2": {"images": [{"filename": "c.png"}]},
            }
        }
        result = AishaJobPoller._collect_image_infos(entry)
        assert len(result) == 3

    def test_nodes_without_images_key_ignored(self) -> None:
        entry = {"outputs": {"1": {"latents": []}, "2": {"images": [{"filename": "x.png"}]}}}
        result = AishaJobPoller._collect_image_infos(entry)
        assert len(result) == 1


# ---------------------------------------------------------------------------
# _poll_one — session-not-active guard
# ---------------------------------------------------------------------------


class TestPollOneSessionGuard:
    async def test_skips_when_session_not_active(self) -> None:
        poller = _make_poller()
        job = _make_job(session_status="stopped")
        session = AsyncMock()

        with patch.object(poller, "_make_transition_service") as mock_ts:
            await poller._poll_one(job, job.gpu_session, session)
            mock_ts.assert_not_called()

    async def test_fails_when_tunnel_hostname_invalid(self) -> None:
        poller = _make_poller(tunnel_allowed_suffix="gpu.cloudin.space")
        job = _make_job(tunnel_hostname="evil.com?attack.gpu.cloudin.space")
        session = AsyncMock()

        ts = AsyncMock()
        with patch(
            "src.workers.aisha_job_poller.JobStateTransitionService",
            return_value=ts,
        ):
            await poller._poll_one(job, job.gpu_session, session)
            ts.transition_to_failed.assert_awaited_once()

    async def test_skips_when_no_prompt_id(self) -> None:
        poller = _make_poller()
        job = _make_job(external_request_id=None)
        session = AsyncMock()

        with patch("src.workers.aisha_job_poller.ComfyUIClient") as mock_client_cls:
            await poller._poll_one(job, job.gpu_session, session)
            mock_client_cls.assert_not_called()


# ---------------------------------------------------------------------------
# _handle_queue_state — timeout handling
# ---------------------------------------------------------------------------


class TestHandleQueueState:
    async def test_transitions_to_running_when_in_queue(self) -> None:
        poller = _make_poller()
        job = _make_job(status=JobStatus.QUEUED.value)
        session = AsyncMock()
        prompt_id = "prompt-abc"

        ts = AsyncMock()
        with patch(
            "src.workers.aisha_job_poller.JobStateTransitionService",
            return_value=ts,
        ):
            queue = {"queue_running": [[0, prompt_id, {}]]}
            await poller._handle_queue_state(
                job=job,
                queue=queue,
                prompt_id=prompt_id,
                session=session,
                product_id="vex",
            )
            ts.transition_to_running.assert_awaited_once_with(job.id)

    async def test_times_out_old_job(self) -> None:
        poller = _make_poller(job_age_timeout_seconds=60)
        old_started = datetime.now(UTC) - timedelta(seconds=120)
        job = _make_job(started_at=old_started)
        session = AsyncMock()

        ts = AsyncMock()
        with patch(
            "src.workers.aisha_job_poller.JobStateTransitionService",
            return_value=ts,
        ):
            await poller._handle_queue_state(
                job=job,
                queue={"queue_running": []},
                prompt_id="prompt-old",
                session=session,
                product_id="vex",
            )
            ts.transition_to_failed.assert_awaited_once()

    async def test_does_not_timeout_young_job(self) -> None:
        poller = _make_poller(job_age_timeout_seconds=1800)
        recent_started = datetime.now(UTC) - timedelta(seconds=10)
        job = _make_job(started_at=recent_started)
        session = AsyncMock()

        ts = AsyncMock()
        with patch(
            "src.workers.aisha_job_poller.JobStateTransitionService",
            return_value=ts,
        ):
            await poller._handle_queue_state(
                job=job,
                queue={"queue_running": []},
                prompt_id="prompt-young",
                session=session,
                product_id="vex",
            )
            ts.transition_to_failed.assert_not_awaited()


# ---------------------------------------------------------------------------
# _handle_history_complete — error entry
# ---------------------------------------------------------------------------


class TestHandleHistoryComplete:
    async def test_fails_job_on_error_status(self) -> None:
        poller = _make_poller()
        job = _make_job()
        session = AsyncMock()
        client = AsyncMock()

        ts = AsyncMock()
        history_entry = {
            "status": {"status_str": "error", "messages": [["Error", "OOM"]]},
        }
        with patch(
            "src.workers.aisha_job_poller.JobStateTransitionService",
            return_value=ts,
        ):
            await poller._handle_history_complete(
                client=client,
                job=job,
                history_entry=history_entry,
                session=session,
                product_id="vex",
            )
            ts.transition_to_failed.assert_awaited_once()
            ts.transition_to_completed.assert_not_awaited()
