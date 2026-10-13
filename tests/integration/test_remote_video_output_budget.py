"""R1 / P0-1: provider-generated videos use their own byte budget (real PostgreSQL + ffmpeg).

On 0ed11cd both providers' materializers shared the 20 MB *upload* ingest
service, so a ~31 MB 1080p/30 s Wan output was rejected with ``InvalidMediaError``,
settled as ``OUTPUT_NOT_DELIVERED`` and charged — the user lost the finished video.

The fixture is generated at test time (never committed): a CBR 8 Mbps H.264/AAC
1920x1080, 30 s file of about 31 MB.
"""

from __future__ import annotations

import shutil
import subprocess
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession

from src.api.services.generation.provider_failures import ProviderFailureKind
from src.api.services.media_ingest import InvalidMediaError
from src.api.services.media_ingest.factory import (
    build_media_ingest_service,
    build_provider_video_ingest_service,
)
from src.api.services.video_jobs.outcome import RemoteVideoResult
from src.core.config import Settings
from src.core.enums import JobStatus, Provider
from src.db.models.storage import GenerationJob, GenerationMaterializationAttempt, GenerationOutput
from tests.integration.test_remote_video_materialization import (
    _PROVIDERS,
    _cleanup,
    _http,
    _materializer,
    _no_settle,
    _seed,
    _storage,
)

if TYPE_CHECKING:
    from src.api.services.media_ingest.profile_log import VideoProfileOrigin

pytestmark = pytest.mark.asyncio

_MIN_LARGE_BYTES = 21 * 1024 * 1024  # strictly above the default 20 MB upload cap


@pytest.fixture(scope="module")
def large_1080p_video(tmp_path_factory: pytest.TempPathFactory) -> bytes:
    """A ~31 MB, 30 s, 1920x1080 CBR H.264/AAC MP4 produced by ffmpeg."""
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg is not installed")
    target = tmp_path_factory.mktemp("wan_output") / "wan_1080p_30s.mp4"
    subprocess.run(  # noqa: S603
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=1920x1080:rate=30",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:sample_rate=48000",
            "-t",
            "30",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-b:v",
            "8M",
            "-minrate",
            "8M",
            "-maxrate",
            "8M",
            "-bufsize",
            "16M",
            "-x264-params",
            "nal-hrd=cbr",
            "-c:a",
            "aac",
            "-b:a",
            "128k",
            "-movflags",
            "+faststart",
            str(target),
        ],
        check=True,
    )
    data = target.read_bytes()
    assert len(data) > _MIN_LARGE_BYTES, "fixture must exceed the user-upload cap"
    return data


@pytest.mark.parametrize(("provider", "origin"), _PROVIDERS)
async def test_large_provider_output_completes_under_the_provider_budget(
    db_engine: AsyncEngine,
    large_1080p_video: bytes,
    provider: Provider,
    origin: VideoProfileOrigin,
) -> None:
    """Inverted 0ed11cd probe: the same ~31 MB file now completes for both providers."""
    settings = Settings.model_construct()
    seed = await _seed(db_engine, provider)
    storage = _storage()
    materializer = _materializer(
        provider,
        origin,
        storage=storage,
        ingestor=build_provider_video_ingest_service(settings),
        http=_http(content=large_1080p_video),
        max_output_bytes=settings.media_ingest_provider_video_max_bytes,
    )
    try:
        with patch(
            "src.api.services.video_jobs.materializer.extract_video_thumbnail",
            new=AsyncMock(return_value=None),
        ):
            async with AsyncSession(bind=db_engine, expire_on_commit=False) as session:
                job = await materializer.finalize_completed_video(
                    session,
                    job_id=seed.job_id,
                    result=RemoteVideoResult(url="https://oss.invalid/v.mp4"),
                    product_id="vex",
                    settle_failure=_no_settle,
                )

        assert job.status == JobStatus.COMPLETED.value
        assert job.failure_code is None
        async with AsyncSession(bind=db_engine, expire_on_commit=False) as verify:
            outputs = (
                await verify.scalars(
                    select(GenerationOutput).where(GenerationOutput.job_id == seed.job_id)
                )
            ).all()
            attempts = (
                await verify.scalars(
                    select(GenerationMaterializationAttempt).where(
                        GenerationMaterializationAttempt.job_id == seed.job_id
                    )
                )
            ).all()
        (original,) = outputs
        assert (original.width, original.height) == (1920, 1080)
        assert original.duration_ms is not None
        assert abs(original.duration_ms - 30_000) < 1_000
        assert original.size_bytes > _MIN_LARGE_BYTES
        (attempt,) = attempts
        assert attempt.state == "committed"
    finally:
        await _cleanup(db_engine, seed)


async def test_the_upload_budget_would_still_reject_it(large_1080p_video: bytes) -> None:
    """Negative control: the cap — not the codec or ffmpeg — was the only blocker."""
    upload_service = build_media_ingest_service(Settings.model_construct())

    with pytest.raises(InvalidMediaError, match="byte limit"):
        await upload_service.prepare_video(large_1080p_video)


@pytest.mark.parametrize(("provider", "origin"), _PROVIDERS)
async def test_over_budget_output_is_settled_before_any_upload_or_attempt(
    db_engine: AsyncEngine, provider: Provider, origin: VideoProfileOrigin
) -> None:
    """An abnormal (over-budget) body aborts the stream: no R2 write, no attempt row."""
    seed = await _seed(db_engine, provider)
    storage = _storage()
    ingestor = MagicMock()
    ingestor.prepare_video = AsyncMock()
    materializer = _materializer(
        provider,
        origin,
        storage=storage,
        ingestor=ingestor,
        http=_http(content=b"x" * 4096),
        max_output_bytes=1024,
    )
    settle_failure = AsyncMock(return_value=MagicMock(spec=GenerationJob))
    try:
        async with AsyncSession(bind=db_engine, expire_on_commit=False) as session:
            await materializer.finalize_completed_video(
                session,
                job_id=seed.job_id,
                result=RemoteVideoResult(url="https://oss.invalid/v.mp4"),
                product_id="vex",
                settle_failure=settle_failure,
            )

        settle_failure.assert_awaited_once()
        assert settle_failure.await_args is not None
        outcome: Any = settle_failure.await_args.args[2]
        assert outcome.failure.kind is ProviderFailureKind.OUTPUT_NOT_DELIVERED
        ingestor.prepare_video.assert_not_awaited()
        storage.put_raw.assert_not_awaited()
        async with AsyncSession(bind=db_engine, expire_on_commit=False) as verify:
            attempts = (
                await verify.scalars(
                    select(GenerationMaterializationAttempt).where(
                        GenerationMaterializationAttempt.job_id == seed.job_id
                    )
                )
            ).all()
        assert attempts == []
    finally:
        await _cleanup(db_engine, seed)
