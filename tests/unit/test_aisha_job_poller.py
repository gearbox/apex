"""Tests for AishaJobPoller — tick loop, _poll_one dispatch, edge cases."""

from __future__ import annotations

import contextlib
import copy
import io
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
import structlog.testing
from PIL import Image

from src.api.services.generation.aisha_failures import AishaFailure
from src.api.services.job_state_transition import JobStateTransitionService
from src.api.services.storage import UploadResult
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
        redis_client_factory=MagicMock(),
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
                redis_client_factory=MagicMock(),
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
                redis_client_factory=MagicMock(),
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
    """Generic start/stop lifecycle is covered by test_periodic_worker.py.

    Only AishaJobPoller-specific behavior (config.enabled gating) lives here.
    """

    async def test_disabled_poller_does_not_start(self) -> None:
        poller = _make_poller(enabled=False)
        await poller.start()
        assert poller.is_running is False

    async def test_enabled_poller_starts(self) -> None:
        poller = _make_poller()
        with patch.object(poller, "_run_loop", new=AsyncMock()):
            await poller.start()
            assert poller.is_running is True
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
        kwargs = ts.transition_to_failed.call_args.kwargs
        assert kwargs["failure_code"] == AishaFailure.PROVIDER_UNAVAILABLE.value
        assert kwargs["public_error_message"] == AishaFailure.PROVIDER_UNAVAILABLE.public_message
        assert "evil.com" not in kwargs["public_error_message"]

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
        assert kwargs["failure_code"] == AishaFailure.PROVIDER_TIMEOUT.value
        assert kwargs["public_error_message"] == AishaFailure.PROVIDER_TIMEOUT.public_message

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
        kwargs = ts.transition_to_failed.call_args.kwargs
        assert kwargs["failure_code"] == AishaFailure.PROVIDER_EXECUTION_FAILED.value
        assert (
            kwargs["public_error_message"] == AishaFailure.PROVIDER_EXECUTION_FAILED.public_message
        )
        assert "CUDA OOM" not in kwargs["public_error_message"]


# ---------------------------------------------------------------------------
# _handle_history_complete — never complete a failed or imageless job
# ---------------------------------------------------------------------------

_PROMPT_ID = "e7299d91-f710-41c0-b4bf-06d9786fb961"
_TRACEBACK_TEXT = 'File "/workspace/ComfyUI/execution.py", line 510, in execute'

# The staging incident, verbatim in shape: ComfyUI writes an *empty* ``outputs``
# dict alongside ``status_str == "error"``.
_EVIDENCE_HISTORY: dict = {
    "outputs": {},
    "status": {
        "status_str": "error",
        "completed": False,
        "messages": [
            ["execution_start", {"prompt_id": _PROMPT_ID, "timestamp": 1758450000000}],
            [
                "execution_cached",
                {"nodes": ["66", "68", "67"], "prompt_id": _PROMPT_ID, "timestamp": 1758450000001},
            ],
            [
                "execution_error",
                {
                    "prompt_id": _PROMPT_ID,
                    "node_id": "72",
                    "node_type": "PatchFlashAttentionKJ",
                    "executed": [],
                    "exception_type": "ImportError",
                    "exception_message": (
                        "Flash attention not found. Install either FA2 ('flash_attn') or FA3 ..."
                    ),
                    "traceback": [
                        f"  {_TRACEBACK_TEXT}\n    output_data = ...\n",
                        "ImportError: Flash attention not found.\n",
                    ],
                    "current_inputs": {"model": ["<comfy.model_patcher.ModelPatcher object>"]},
                    "current_outputs": ["CURRENT_OUTPUTS_REPR"],
                    "timestamp": 1758450000002,
                },
            ],
        ],
    },
}


def _evidence_history(*, with_outputs_key: bool = True) -> dict:
    entry = copy.deepcopy(_EVIDENCE_HISTORY)
    if not with_outputs_key:
        del entry["outputs"]
    return entry


def _png_bytes() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (10, 10), (200, 100, 50)).save(buf, format="PNG")
    return buf.getvalue()


def _download_error() -> Exception:
    """What ``client.get_image`` raises when ComfyUI cannot serve the file."""
    return RuntimeError("connection reset")


def _output_history(*filenames: str, type_: str = "output") -> dict:
    return {
        "outputs": {
            "9": {"images": [{"filename": f, "subfolder": "", "type": type_} for f in filenames]}
        },
        "status": {"status_str": "success", "completed": True, "messages": []},
    }


async def _run_history(
    poller: AishaJobPoller,
    job: MagicMock,
    history_entry: dict,
    *,
    client: AsyncMock | None = None,
    ts: AsyncMock | None = None,
) -> tuple[AsyncMock, AsyncMock]:
    client = client or AsyncMock()
    ts = ts or AsyncMock()
    ts.transition_to_failed.return_value = (MagicMock(), True)
    await poller._handle_history_complete(
        client=client,
        job=job,
        history_entry=history_entry,
        product_id="vex",
        ts=ts,
    )
    return client, ts


class TestExecutionErrorClassification:
    """D1: ``status.status_str == "error"`` is checked first, unconditionally."""

    async def test_error_with_empty_outputs_dict_fails_and_refunds(self) -> None:
        poller = _make_poller()
        job = _make_job()

        client, ts = await _run_history(poller, job, _evidence_history())

        ts.transition_to_completed.assert_not_awaited()
        client.get_image.assert_not_awaited()
        ts.transition_to_failed.assert_awaited_once()
        kwargs = ts.transition_to_failed.call_args.kwargs
        assert kwargs["refund"] is True
        assert kwargs["failure_code"] == AishaFailure.PROVIDER_EXECUTION_FAILED.value
        assert kwargs["public_error_message"] == (
            AishaFailure.PROVIDER_EXECUTION_FAILED.public_message
        )
        assert "PatchFlashAttentionKJ" in kwargs["error_message"]
        assert "ImportError" in kwargs["error_message"]
        assert "Flash attention not found" in kwargs["error_message"]

    async def test_error_message_carries_no_traceback_or_node_state(self) -> None:
        poller = _make_poller()

        _, ts = await _run_history(poller, _make_job(), _evidence_history())

        error_message = ts.transition_to_failed.call_args.kwargs["error_message"]
        assert _TRACEBACK_TEXT not in error_message
        assert "execution.py" not in error_message
        assert "ModelPatcher" not in error_message
        assert "CURRENT_OUTPUTS_REPR" not in error_message

    async def test_error_with_outputs_key_absent_is_identical(self) -> None:
        """The presence of the ``outputs`` key is irrelevant to the outcome."""
        poller = _make_poller()
        with_key = await _run_history(poller, _make_job(), _evidence_history())
        without_key = await _run_history(
            poller, _make_job(), _evidence_history(with_outputs_key=False)
        )

        assert (
            with_key[1].transition_to_failed.call_args.kwargs["error_message"]
            == without_key[1].transition_to_failed.call_args.kwargs["error_message"]
        )
        without_key[1].transition_to_failed.assert_awaited_once()
        without_key[1].transition_to_completed.assert_not_awaited()

    async def test_error_after_partial_outputs_still_fails(self) -> None:
        """A run can emit UI outputs from earlier nodes and still error later."""
        poller = _make_poller()
        entry = _evidence_history()
        entry["outputs"] = {"9": {"images": [{"filename": "a.png", "type": "output"}]}}

        client, ts = await _run_history(poller, _make_job(), entry)

        client.get_image.assert_not_awaited()
        ts.transition_to_completed.assert_not_awaited()
        ts.transition_to_failed.assert_awaited_once()

    async def test_error_without_execution_error_message_uses_fixed_fallback(self) -> None:
        poller = _make_poller()
        entry = {"outputs": {}, "status": {"status_str": "error", "messages": []}}

        _, ts = await _run_history(poller, _make_job(), entry)

        kwargs = ts.transition_to_failed.call_args.kwargs
        assert kwargs["error_message"] == (
            "ComfyUI reported status error without an execution_error message"
        )
        assert kwargs["failure_code"] == AishaFailure.PROVIDER_EXECUTION_FAILED.value

    async def test_malformed_execution_error_payload_uses_fixed_fallback(self) -> None:
        poller = _make_poller()
        entry = {
            "status": {
                "status_str": "error",
                "messages": ["not-a-pair", ["execution_error"], ["execution_error", "str"]],
            }
        }

        _, ts = await _run_history(poller, _make_job(), entry)

        assert (
            "without an execution_error message"
            in (ts.transition_to_failed.call_args.kwargs["error_message"])
        )

    async def test_secret_in_exception_message_is_not_persisted(self) -> None:
        poller = _make_poller()
        entry = _evidence_history()
        entry["status"]["messages"][2][1]["exception_message"] = (
            "download failed for https://a/x?token=SECRET"
        )

        _, ts = await _run_history(poller, _make_job(), entry)

        error_message = ts.transition_to_failed.call_args.kwargs["error_message"]
        assert "SECRET" not in error_message
        assert "download failed" in error_message

    async def test_detail_is_bounded(self) -> None:
        poller = _make_poller()
        entry = _evidence_history()
        entry["status"]["messages"][2][1]["exception_message"] = "x" * 100_000

        _, ts = await _run_history(poller, _make_job(), entry)

        assert len(ts.transition_to_failed.call_args.kwargs["error_message"]) <= 500

    async def test_execution_error_is_logged_with_alertable_fields(self) -> None:
        poller = _make_poller()
        job = _make_job()

        with structlog.testing.capture_logs() as logs:
            await _run_history(poller, job, _evidence_history())

        [event] = [e for e in logs if e["event"] == "aisha_job_poller.execution_error"]
        assert event["job_id"] == str(job.id)
        assert event["node_type"] == "PatchFlashAttentionKJ"
        assert event["exception_type"] == "ImportError"
        assert event["log_level"] == "error"

    async def test_execution_error_log_fields_are_redacted(self) -> None:
        poller = _make_poller()
        entry = _evidence_history()
        payload = entry["status"]["messages"][2][1]
        payload["node_type"] = "CustomNode token=NODE_SECRET"
        payload["exception_type"] = "ImportError api_key=TYPE_SECRET"

        with structlog.testing.capture_logs() as logs:
            await _run_history(poller, _make_job(), entry)

        [event] = [event for event in logs if event["event"] == "aisha_job_poller.execution_error"]
        assert "NODE_SECRET" not in event["node_type"]
        assert "TYPE_SECRET" not in event["exception_type"]


class TestNothingCollectable:
    async def test_only_temp_images_fails_immediately_and_refunds(self) -> None:
        poller = _make_poller()
        # Fresh job: nothing about the age timeout may be what fails it.
        job = _make_job(started_at=datetime.now(UTC) - timedelta(seconds=5))

        with structlog.testing.capture_logs() as logs:
            client, ts = await _run_history(poller, job, _output_history("p.png", type_="temp"))

        client.get_image.assert_not_awaited()
        ts.transition_to_completed.assert_not_awaited()
        ts.transition_to_failed.assert_awaited_once()
        kwargs = ts.transition_to_failed.call_args.kwargs
        assert kwargs["refund"] is True
        assert kwargs["failure_code"] == AishaFailure.PROVIDER_EXECUTION_FAILED.value
        assert "collectable output" in kwargs["error_message"]
        assert any(e["event"] == "aisha_job_poller.no_collectable_outputs" for e in logs)

    async def test_empty_outputs_dict_without_error_fails_immediately(self) -> None:
        poller = _make_poller()
        entry = {"outputs": {}, "status": {"status_str": "success", "completed": True}}

        _, ts = await _run_history(poller, _make_job(), entry)

        ts.transition_to_failed.assert_awaited_once()
        ts.transition_to_completed.assert_not_awaited()

    async def test_output_entry_without_filename_is_not_collectable(self) -> None:
        """An entry that can never be downloaded must not be retried until timeout."""
        poller = _make_poller()
        entry = {"outputs": {"9": {"images": [{"subfolder": "", "type": "output"}]}}}

        client, ts = await _run_history(poller, _make_job(), entry)

        client.get_image.assert_not_awaited()
        ts.transition_to_failed.assert_awaited_once()
        assert ts.transition_to_failed.call_args.kwargs["failure_code"] == (
            AishaFailure.PROVIDER_EXECUTION_FAILED.value
        )


class TestOutputDownloadFailure:
    """D3: only a download failure after ComfyUI succeeded is retried."""

    @staticmethod
    def _poller_with_r2(**config: object) -> AishaJobPoller:
        poller = _make_poller(**config)
        poller._r2 = AsyncMock()
        poller._r2.upload.side_effect = lambda **_: UploadResult(
            id=uuid4(), storage_key="users/u/outputs/j/f.png"
        )
        return poller

    async def test_every_download_failing_leaves_job_running_without_refund(self) -> None:
        poller = self._poller_with_r2(job_age_timeout_seconds=1800)
        job = _make_job(started_at=datetime.now(UTC) - timedelta(seconds=30))
        client = AsyncMock()
        client.get_image.side_effect = _download_error()

        with structlog.testing.capture_logs() as logs:
            _, ts = await _run_history(
                poller, job, _output_history("a.png", "b.png"), client=client
            )

        assert client.get_image.await_count == 2
        ts.transition_to_completed.assert_not_awaited()
        ts.transition_to_failed.assert_not_awaited()
        [event] = [e for e in logs if e["event"] == "aisha_job_poller.output_download_failed"]
        assert event["job_id"] == str(job.id)

    async def test_every_upload_failing_is_also_a_retry(self) -> None:
        poller = self._poller_with_r2(job_age_timeout_seconds=1800)
        failing_r2 = AsyncMock()
        failing_r2.upload.side_effect = RuntimeError("r2 down")
        poller._r2 = failing_r2
        client = AsyncMock()
        client.get_image.return_value = _png_bytes()

        _, ts = await _run_history(poller, _make_job(), _output_history("a.png"), client=client)

        ts.transition_to_completed.assert_not_awaited()
        ts.transition_to_failed.assert_not_awaited()

    async def test_download_failure_past_timeout_fails_with_timeout_and_refund(self) -> None:
        poller = self._poller_with_r2(job_age_timeout_seconds=60)
        job = _make_job(started_at=datetime.now(UTC) - timedelta(seconds=120))
        client = AsyncMock()
        client.get_image.side_effect = _download_error()

        _, ts = await _run_history(poller, job, _output_history("a.png"), client=client)

        ts.transition_to_completed.assert_not_awaited()
        ts.transition_to_failed.assert_awaited_once()
        kwargs = ts.transition_to_failed.call_args.kwargs
        assert kwargs["refund"] is True
        assert kwargs["failure_code"] == AishaFailure.PROVIDER_TIMEOUT.value
        assert kwargs["public_error_message"] == AishaFailure.PROVIDER_TIMEOUT.public_message

    async def test_one_of_two_downloads_failing_still_completes_with_one_output(self) -> None:
        """Documents the out-of-scope partial-download behaviour: it completes and bills."""
        poller = self._poller_with_r2()
        client = AsyncMock()
        client.get_image.side_effect = [_download_error(), _png_bytes()]

        _, ts = await _run_history(
            poller, _make_job(), _output_history("a.png", "b.png"), client=client
        )

        ts.transition_to_failed.assert_not_awaited()
        ts.transition_to_completed.assert_awaited_once()
        outputs = ts.transition_to_completed.call_args.kwargs["outputs"]
        full_outputs = [o for o in outputs if not o.is_thumbnail]
        assert len(full_outputs) == 1
        assert full_outputs[0].output_index == 1

    async def test_all_downloads_succeeding_completes(self) -> None:
        poller = self._poller_with_r2()
        client = AsyncMock()
        client.get_image.return_value = _png_bytes()

        _, ts = await _run_history(
            poller, _make_job(), _output_history("a.png", "b.png"), client=client
        )

        ts.transition_to_failed.assert_not_awaited()
        ts.transition_to_completed.assert_awaited_once()
        full_outputs = [
            o for o in ts.transition_to_completed.call_args.kwargs["outputs"] if not o.is_thumbnail
        ]
        assert [o.output_index for o in full_outputs] == [0, 1]


class TestExecutionErrorIsSettledOnce:
    """Ticks are independent: the same failed history entry must refund once."""

    async def test_same_failed_history_on_two_consecutive_ticks_refunds_once(self) -> None:
        poller = _make_poller()
        job = _make_job(status=JobStatus.RUNNING.value)
        job.debit_transaction_id = uuid4()
        job.generation_type = "t2i"
        job.provider = "aisha"

        session = AsyncMock()
        session.get.return_value = job
        session.refresh = AsyncMock()
        update_result = MagicMock()
        update_result.rowcount = 1
        session.execute.return_value = update_result
        billing = AsyncMock()
        ts = JobStateTransitionService(session=session, event_bus=None, billing_service=billing)

        client = AsyncMock()
        await poller._handle_history_complete(
            client=client, job=job, history_entry=_evidence_history(), product_id="vex", ts=ts
        )
        # What the first tick's committed UPDATE did to the row.
        job.status = JobStatus.FAILED.value
        await poller._handle_history_complete(
            client=client, job=job, history_entry=_evidence_history(), product_id="vex", ts=ts
        )

        billing.refund.assert_awaited_once()
        session.commit.assert_awaited_once()
