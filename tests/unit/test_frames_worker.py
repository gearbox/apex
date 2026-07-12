"""Unit tests for FrameExtractionWorker.

Fully mocked: DB sessions, R2 storage, and ffmpeg subprocess calls are all
AsyncMock/patched (matches this project's established convention — see
tests/unit/test_thumbnail.py and tests/unit/test_frames_ffmpeg.py — real
ffmpeg is never invoked from unit tests).
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from src.api.services.frames.ffmpeg import FfprobeError, VideoProbe
from src.api.services.frames.worker import FrameExtractionWorker
from src.core.enums import FrameExtractionKind

pytestmark = pytest.mark.unit


def _cm(value: object) -> MagicMock:
    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=value)
    cm.__aexit__ = AsyncMock(return_value=False)
    return cm


def _make_db_manager(sessions: list[AsyncMock]) -> MagicMock:
    """DatabaseManager stub whose .session() hands out sessions in order."""
    iterator = iter(sessions)

    def _next_session() -> MagicMock:
        return _cm(next(iterator))

    manager = MagicMock()
    manager.session = MagicMock(side_effect=_next_session)
    return manager


def _make_session() -> AsyncMock:
    session = AsyncMock()
    session.commit = AsyncMock()
    return session


def _make_settings(**overrides: object) -> MagicMock:
    settings = MagicMock()
    settings.frame_extract_poll_interval_seconds = 2.0
    settings.frame_extract_ffmpeg_timeout_seconds = 30
    settings.frame_preview_max_edge = 512
    settings.retention_days = 7
    settings.frame_extract_stale_running_seconds = 300
    for k, v in overrides.items():
        setattr(settings, k, v)
    return settings


def _make_job(**overrides: object) -> MagicMock:
    job = MagicMock()
    job.id = uuid4()
    job.user_id = uuid4()
    job.product_id = "vex"
    job.kind = FrameExtractionKind.PREVIEW.value
    job.source_output_id = uuid4()
    job.source_upload_id = None
    job.params = {"frame_count": 3}
    for k, v in overrides.items():
        setattr(job, k, v)
    return job


def _make_worker(
    *,
    sessions: list[AsyncMock] | None = None,
    settings: MagicMock | None = None,
) -> tuple[FrameExtractionWorker, MagicMock, AsyncMock]:
    sessions = sessions or [_make_session() for _ in range(4)]
    db_manager = _make_db_manager(sessions)
    r2_storage = AsyncMock()
    worker = FrameExtractionWorker(
        db_manager=db_manager,
        r2_storage=r2_storage,
        settings=settings or _make_settings(),
    )
    return worker, db_manager, r2_storage


class TestWorkerClaimsAndCompletesPreview:
    async def test_worker_claims_queued_job_and_completes_preview(self) -> None:
        job = _make_job(kind=FrameExtractionKind.PREVIEW.value, params={"frame_count": 2})
        output = MagicMock(storage_key="users/u/uploads/vid.mp4", content_type="video/mp4")

        sessions = [_make_session(), _make_session(), _make_session()]
        worker, _db_manager, r2_storage = _make_worker(sessions=sessions)
        r2_storage.download = AsyncMock(return_value=b"fake video bytes")

        job_repo = AsyncMock()
        job_repo.fail_stale_running = AsyncMock(return_value=0)
        job_repo.claim_next = AsyncMock(return_value=job)
        job_repo.mark_completed = AsyncMock()
        output_repo = AsyncMock()
        output_repo.get = AsyncMock(return_value=output)

        probe_result = VideoProbe(duration_ms=10_000, width=640, height=480, codec="h264")

        with (
            patch.multiple(
                "src.api.services.frames.worker",
                FrameExtractionJobRepository=MagicMock(return_value=job_repo),
                OutputRepository=MagicMock(return_value=output_repo),
            ),
            patch(
                "src.api.services.frames.worker.frame_ffmpeg.probe",
                AsyncMock(return_value=probe_result),
            ),
            patch(
                "src.api.services.frames.worker.frame_ffmpeg.compute_uniform_timestamps",
                MagicMock(return_value=[0, 5000]),
            ),
            patch(
                "src.api.services.frames.worker.frame_ffmpeg.extract_frame",
                AsyncMock(return_value=b"webpbytes"),
            ),
        ):
            r2_storage._get_client = MagicMock(return_value=_cm(AsyncMock()))
            r2_storage._settings = MagicMock(bucket_name="test-bucket")
            await worker.run_once()

        job_repo.claim_next.assert_awaited_once()
        job_repo.mark_completed.assert_awaited_once()
        result = job_repo.mark_completed.call_args.kwargs["result"]
        assert len(result["frames"]) == 2
        assert result["frames"][0]["timestamp_ms"] == 0
        assert result["frames"][1]["timestamp_ms"] == 5000
        assert all("key" in f for f in result["frames"])

    async def test_worker_noop_when_nothing_queued(self) -> None:
        sessions = [_make_session(), _make_session()]
        worker, _db_manager, _r2 = _make_worker(sessions=sessions)

        job_repo = AsyncMock()
        job_repo.fail_stale_running = AsyncMock(return_value=0)
        job_repo.claim_next = AsyncMock(return_value=None)

        with patch.multiple(
            "src.api.services.frames.worker",
            FrameExtractionJobRepository=MagicMock(return_value=job_repo),
        ):
            await worker.run_once()

        job_repo.claim_next.assert_awaited_once()


class TestWorkerExtract:
    async def test_worker_extract_creates_user_images_with_lineage(self) -> None:
        job = _make_job(
            kind=FrameExtractionKind.EXTRACT.value,
            params={"timestamps_ms": [1000, 2000]},
            source_output_id=uuid4(),
            source_upload_id=None,
        )
        output = MagicMock(storage_key="users/u/outputs/j/vid.mp4", content_type="video/mp4")

        sessions = [_make_session(), _make_session(), _make_session(), _make_session()]
        worker, _db_manager, r2_storage = _make_worker(sessions=sessions)
        r2_storage.download = AsyncMock(return_value=b"fake video bytes")
        upload_result = MagicMock(id=uuid4(), storage_key="users/u/uploads/frame.png")
        r2_storage.upload = AsyncMock(return_value=upload_result)

        job_repo = AsyncMock()
        job_repo.fail_stale_running = AsyncMock(return_value=0)
        job_repo.claim_next = AsyncMock(return_value=job)
        job_repo.mark_completed = AsyncMock()
        output_repo = AsyncMock()
        output_repo.get = AsyncMock(return_value=output)
        image_repo = AsyncMock()
        db_image = MagicMock(id=upload_result.id)
        image_repo.create = AsyncMock(return_value=db_image)

        probe_result = VideoProbe(duration_ms=10_000, width=640, height=480, codec="h264")

        with (
            patch.multiple(
                "src.api.services.frames.worker",
                FrameExtractionJobRepository=MagicMock(return_value=job_repo),
                OutputRepository=MagicMock(return_value=output_repo),
                UserImageRepository=MagicMock(return_value=image_repo),
            ),
            patch(
                "src.api.services.frames.worker.frame_ffmpeg.probe",
                AsyncMock(return_value=probe_result),
            ),
            patch(
                "src.api.services.frames.worker.frame_ffmpeg.extract_frame",
                AsyncMock(return_value=b"\x89PNGfakepng"),
            ),
            patch(
                "src.api.services.frames.worker.read_dimensions",
                AsyncMock(return_value=None),
            ),
            patch(
                "src.api.services.frames.worker.make_image_thumbnails",
                AsyncMock(return_value=[]),
            ),
        ):
            await worker.run_once()

        assert image_repo.create.await_count == 2
        for call in image_repo.create.await_args_list:
            kwargs = call.kwargs
            assert kwargs["source_output_id"] == job.source_output_id
            assert kwargs["source_upload_id"] is None
            assert kwargs["source_timestamp_ms"] in (1000, 2000)

        job_repo.mark_completed.assert_awaited_once()
        result = job_repo.mark_completed.call_args.kwargs["result"]
        assert len(result["frames"]) == 2
        assert result["frames"][0]["upload_id"] == str(db_image.id)

    async def test_worker_marks_failed_on_out_of_range_timestamp(self) -> None:
        job = _make_job(
            kind=FrameExtractionKind.EXTRACT.value,
            params={"timestamps_ms": [999_999]},
            source_output_id=uuid4(),
            source_upload_id=None,
        )
        output = MagicMock(storage_key="users/u/outputs/j/vid.mp4", content_type="video/mp4")

        sessions = [_make_session(), _make_session(), _make_session()]
        worker, _db_manager, r2_storage = _make_worker(sessions=sessions)
        r2_storage.download = AsyncMock(return_value=b"fake video bytes")

        job_repo = AsyncMock()
        job_repo.fail_stale_running = AsyncMock(return_value=0)
        job_repo.claim_next = AsyncMock(return_value=job)
        job_repo.mark_failed = AsyncMock()
        output_repo = AsyncMock()
        output_repo.get = AsyncMock(return_value=output)

        probe_result = VideoProbe(duration_ms=10_000, width=640, height=480, codec="h264")

        with (
            patch.multiple(
                "src.api.services.frames.worker",
                FrameExtractionJobRepository=MagicMock(return_value=job_repo),
                OutputRepository=MagicMock(return_value=output_repo),
            ),
            patch(
                "src.api.services.frames.worker.frame_ffmpeg.probe",
                AsyncMock(return_value=probe_result),
            ),
        ):
            await worker.run_once()

        job_repo.mark_failed.assert_awaited_once()
        error_message = job_repo.mark_failed.call_args.kwargs["error"]
        assert "out of range" in error_message.lower()


class TestWorkerFailureHandling:
    async def test_worker_marks_failed_on_corrupt_video_with_error_message(self) -> None:
        job = _make_job()
        output = MagicMock(storage_key="users/u/outputs/j/vid.mp4", content_type="video/mp4")

        sessions = [_make_session(), _make_session(), _make_session()]
        worker, _db_manager, r2_storage = _make_worker(sessions=sessions)
        r2_storage.download = AsyncMock(return_value=b"not a real video")

        job_repo = AsyncMock()
        job_repo.fail_stale_running = AsyncMock(return_value=0)
        job_repo.claim_next = AsyncMock(return_value=job)
        job_repo.mark_failed = AsyncMock()
        output_repo = AsyncMock()
        output_repo.get = AsyncMock(return_value=output)

        with (
            patch.multiple(
                "src.api.services.frames.worker",
                FrameExtractionJobRepository=MagicMock(return_value=job_repo),
                OutputRepository=MagicMock(return_value=output_repo),
            ),
            patch(
                "src.api.services.frames.worker.frame_ffmpeg.probe",
                AsyncMock(side_effect=FfprobeError("moov atom not found")),
            ),
        ):
            await worker.run_once()

        job_repo.mark_failed.assert_awaited_once()
        assert "moov atom not found" in job_repo.mark_failed.call_args.kwargs["error"]

    async def test_worker_survives_job_failure_and_continues_loop(self) -> None:
        """run_once() must never raise — a failed job is recorded, not propagated."""
        job = _make_job()
        sessions = [_make_session(), _make_session(), _make_session()]
        worker, _db_manager, r2_storage = _make_worker(sessions=sessions)
        r2_storage.download = AsyncMock(side_effect=RuntimeError("R2 unreachable"))

        job_repo = AsyncMock()
        job_repo.fail_stale_running = AsyncMock(return_value=0)
        job_repo.claim_next = AsyncMock(return_value=job)
        job_repo.mark_failed = AsyncMock()
        output_repo = AsyncMock()
        output_repo.get = AsyncMock(
            return_value=MagicMock(storage_key="k", content_type="video/mp4")
        )

        with patch.multiple(
            "src.api.services.frames.worker",
            FrameExtractionJobRepository=MagicMock(return_value=job_repo),
            OutputRepository=MagicMock(return_value=output_repo),
        ):
            await worker.run_once()  # must not raise

        job_repo.mark_failed.assert_awaited_once()
        assert "R2 unreachable" in job_repo.mark_failed.call_args.kwargs["error"]

    async def test_worker_rejects_non_video_source_defensively(self) -> None:
        job = _make_job()
        sessions = [_make_session(), _make_session(), _make_session()]
        worker, _db_manager, r2_storage = _make_worker(sessions=sessions)

        job_repo = AsyncMock()
        job_repo.fail_stale_running = AsyncMock(return_value=0)
        job_repo.claim_next = AsyncMock(return_value=job)
        job_repo.mark_failed = AsyncMock()
        output_repo = AsyncMock()
        output_repo.get = AsyncMock(
            return_value=MagicMock(storage_key="k", content_type="image/png")
        )

        with patch.multiple(
            "src.api.services.frames.worker",
            FrameExtractionJobRepository=MagicMock(return_value=job_repo),
            OutputRepository=MagicMock(return_value=output_repo),
        ):
            await worker.run_once()

        job_repo.mark_failed.assert_awaited_once()
        assert "not a video" in job_repo.mark_failed.call_args.kwargs["error"]
        r2_storage.download.assert_not_awaited()


class TestWorkerStaleRunningSweep:
    async def test_sweep_runs_before_claim_in_its_own_transaction(self) -> None:
        """The sweep must commit in a separate, prior transaction from the claim."""
        sessions = [_make_session(), _make_session()]
        worker, _db_manager, _r2 = _make_worker(sessions=sessions)

        job_repo = AsyncMock()
        job_repo.fail_stale_running = AsyncMock(return_value=2)
        job_repo.claim_next = AsyncMock(return_value=None)

        with patch.multiple(
            "src.api.services.frames.worker",
            FrameExtractionJobRepository=MagicMock(return_value=job_repo),
        ):
            await worker.run_once()

        job_repo.fail_stale_running.assert_awaited_once()
        cutoff = job_repo.fail_stale_running.call_args.kwargs["cutoff"]
        assert cutoff < datetime.now(UTC)
        # Sweep session committed before the claim session was ever touched.
        sessions[0].commit.assert_awaited_once()
        job_repo.claim_next.assert_awaited_once()

    async def test_no_stale_jobs_does_not_short_circuit_claim(self) -> None:
        sessions = [_make_session(), _make_session()]
        worker, _db_manager, _r2 = _make_worker(sessions=sessions)

        job_repo = AsyncMock()
        job_repo.fail_stale_running = AsyncMock(return_value=0)
        job_repo.claim_next = AsyncMock(return_value=None)

        with patch.multiple(
            "src.api.services.frames.worker",
            FrameExtractionJobRepository=MagicMock(return_value=job_repo),
        ):
            await worker.run_once()

        job_repo.claim_next.assert_awaited_once()
