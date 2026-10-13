"""Shared remote-video completion across providers (real PostgreSQL).

C15: a Wan moderation rejection refunds; an undelivered output is charged with
the billed notice. C16: Wan completes through the shared materializer exactly
like Grok (output row, thumbnails, provider profile origin, committed attempt),
and a crash after the planned-keys commit leaves a reconcilable attempt that
``ContentRetentionService`` cleans.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID, uuid4

import httpx
import pytest
from pydantic import SecretStr
from sqlalchemy import delete, func, select, text, update
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from src.api.services.billing import BillingService
from src.api.services.content_retention import ContentRetentionService
from src.api.services.generation.provider_billing_policy import ProviderBillingPolicyRegistry
from src.api.services.image_thumbnail import GeneratedThumbnail, ThumbnailResult
from src.api.services.media_ingest import InvalidMediaError, PreparedVideo, VideoStreamProfile
from src.api.services.media_ingest.profile_log import VideoProfileOrigin
from src.api.services.model_studio import ModelStudioClient
from src.api.services.model_studio.wan_job_service import WanJobService
from src.api.services.video_jobs.download import VideoDownloadClient
from src.api.services.video_jobs.materializer import RemoteVideoMaterializer
from src.api.services.video_jobs.outcome import RemoteVideoResult, VideoPollOutcome
from src.api.services.video_jobs.transition import make_transition_service_factory
from src.core.config import Settings
from src.core.enums import (
    GenerationType,
    JobStatus,
    MediaFormat,
    Provider,
    TransactionType,
    VideoPollStatus,
)
from src.core.media_hash import HashSample, HashSet, PdqHash
from src.core.thumbnails import ThumbnailSpec
from src.core.uid import new_id
from src.db.models.billing import TokenAccount, TokenTransaction
from src.db.models.storage import GenerationJob, GenerationMaterializationAttempt, GenerationOutput
from src.db.models.user import User

pytestmark = pytest.mark.asyncio

_PROVIDERS = [
    pytest.param(Provider.GROK, VideoProfileOrigin.GROK_OUTPUT, id="grok"),
    pytest.param(Provider.ALIBABA, VideoProfileOrigin.ALIBABA_OUTPUT, id="alibaba"),
]
_WEBP = b"RIFF\x00\x00\x00\x00WEBP" + b"\x00" * 20


@dataclass(frozen=True)
class _Seed:
    user_id: UUID
    account_id: UUID | None
    job_id: UUID


async def _seed(
    engine: AsyncEngine,
    provider: Provider,
    *,
    with_debit: bool = False,
    external_id: str = "task-1",
) -> _Seed:
    user = User(
        id=new_id(),
        email=f"remote-video-{uuid4().hex[:8]}@example.com",
        password_hash="x" * 64,
        product_id="vex",
        is_active=True,
    )
    job = GenerationJob(
        id=new_id(),
        user_id=user.id,
        name="video",
        prompt="a test video",
        generation_type=GenerationType.T2V.value,
        status=JobStatus.RUNNING.value,
        provider=provider.value,
        model="wan3.0-video" if provider is Provider.ALIBABA else "grok-imagine-video",
        product_id="vex",
        external_request_id=external_id,
        started_at=datetime.now(UTC),
    )
    async with AsyncSession(bind=engine, expire_on_commit=False) as session:
        session.add(user)
        await session.flush()
        session.add(job)
        account_id: UUID | None = None
        if with_debit:
            account = TokenAccount(
                id=new_id(), user_id=user.id, account_type="personal", product_id="vex"
            )
            credit = TokenTransaction(
                id=new_id(),
                account_id=account.id,
                transaction_type=TransactionType.CREDIT.value,
                amount=100,
                balance_after=100,
                product_id="vex",
            )
            session.add_all([account, credit])
            await session.flush()
            reserve = await BillingService().check_and_reserve(
                account.id,
                25,
                job.id,
                metadata={"type": "generation", "provider": provider.value},
                session=session,
                product_id="vex",
            )
            job.token_cost = 25
            job.debit_transaction_id = reserve.txn.id
            account_id = account.id
        await session.commit()
    return _Seed(user_id=user.id, account_id=account_id, job_id=job.id)


async def _cleanup(engine: AsyncEngine, seed: _Seed) -> None:
    async with AsyncSession(bind=engine, expire_on_commit=False) as session:
        await session.execute(text("SET LOCAL session_replication_role = 'replica'"))
        if seed.account_id is not None:
            await session.execute(
                delete(TokenTransaction).where(TokenTransaction.account_id == seed.account_id)
            )
        await session.execute(
            delete(GenerationMaterializationAttempt).where(
                GenerationMaterializationAttempt.job_id == seed.job_id
            )
        )
        await session.execute(
            delete(GenerationOutput).where(GenerationOutput.job_id == seed.job_id)
        )
        await session.execute(delete(GenerationJob).where(GenerationJob.id == seed.job_id))
        if seed.account_id is not None:
            await session.execute(delete(TokenAccount).where(TokenAccount.id == seed.account_id))
        await session.execute(delete(User).where(User.id == seed.user_id))
        await session.commit()


def _prepared_video() -> PreparedVideo:
    return PreparedVideo(
        data=b"prepared-video",
        format=MediaFormat.MP4,
        width=1280,
        height=720,
        duration_ms=8000,
        hash_set=HashSet(
            profile_id="pdq-video-rgb-white-v2-edge-512",
            sampling_profile="uniform-pts-v1",
            samples=(
                HashSample(
                    pdq=PdqHash(bits=b"\x00" * 32, quality=100),
                    sample_index=0,
                    frame_timestamp_ms=0,
                ),
            ),
        ),
        stream_profile=VideoStreamProfile(
            container=MediaFormat.MP4,
            codec="h264",
            codec_profile="High",
            pix_fmt="yuv420p",
            color_transfer="bt709",
            color_primaries="bt709",
            rotation_degrees=0,
            sample_aspect_ratio="1:1",
            has_audio=False,
        ),
    )


def _thumbnails() -> list[GeneratedThumbnail]:
    return [
        GeneratedThumbnail(
            spec=ThumbnailSpec("sm", 150), result=ThumbnailResult(data=_WEBP, width=100, height=56)
        ),
        GeneratedThumbnail(
            spec=ThumbnailSpec("md", 512), result=ThumbnailResult(data=_WEBP, width=400, height=225)
        ),
    ]


def _storage() -> MagicMock:
    storage = MagicMock()
    storage.build_storage_key.side_effect = lambda **kw: (
        f"users/{kw['user_id']}/outputs/{kw['job_id']}/{kw['file_id']}.{kw['format'].value}"
    )
    storage.put_raw = AsyncMock()
    storage.delete_many = AsyncMock()
    return storage


def _http(content: bytes = b"video-bytes", status: int = 200) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _: httpx.Response(status, content=content))
    )


def _materializer(
    provider: Provider,
    origin: VideoProfileOrigin,
    *,
    storage: MagicMock,
    ingestor: Any,
    http: httpx.AsyncClient,
    policy: ProviderBillingPolicyRegistry | None = None,
    max_output_bytes: int = 200 * 1024 * 1024,
) -> RemoteVideoMaterializer:
    policy = policy or ProviderBillingPolicyRegistry.from_settings(Settings.model_construct())
    return RemoteVideoMaterializer(
        provider=provider,
        profile_origin=origin,
        storage=storage,
        media_ingestor=ingestor,
        http_client_provider=lambda: http,
        retention_days=7,
        finalization_lease_seconds=120,
        max_output_bytes=max_output_bytes,
        billing_policy=policy,
        transition_service_factory=make_transition_service_factory(
            billing_service=BillingService(), event_bus=None, ops_event_bus=None
        ),
    )


async def _no_settle(*_: object) -> Any:
    raise AssertionError("settle_failure must not be called on the happy path")


@pytest.mark.parametrize(("provider", "origin"), _PROVIDERS)
async def test_c16_completion_commits_output_thumbnails_and_attempt(
    db_engine: AsyncEngine, provider: Provider, origin: VideoProfileOrigin
) -> None:
    seed = await _seed(db_engine, provider)
    ingestor = MagicMock()
    ingestor.prepare_video = AsyncMock(return_value=_prepared_video())
    storage = _storage()
    materializer = _materializer(provider, origin, storage=storage, ingestor=ingestor, http=_http())
    events: list[dict[str, Any]] = []
    try:
        with (
            patch(
                "src.api.services.video_jobs.materializer.extract_video_thumbnail",
                new=AsyncMock(return_value=b"\xff\xd8\xff\xe0jpeg"),
            ),
            patch(
                "src.api.services.video_jobs.materializer.make_image_thumbnails",
                new=AsyncMock(return_value=_thumbnails()),
            ),
            patch(
                "src.api.services.media_ingest.profile_log.logger.info",
                side_effect=lambda event, **kw: events.append({"event": event, **kw}),
            ),
        ):
            async with AsyncSession(bind=db_engine, expire_on_commit=False) as session:
                job = await materializer.finalize_completed_video(
                    session,
                    job_id=seed.job_id,
                    result=RemoteVideoResult(url="https://provider.invalid/v.mp4"),
                    product_id="vex",
                    settle_failure=_no_settle,
                )

        assert job.status == JobStatus.COMPLETED.value
        async with AsyncSession(bind=db_engine, expire_on_commit=False) as verify:
            outputs = (
                await verify.scalars(
                    select(GenerationOutput)
                    .where(GenerationOutput.job_id == seed.job_id)
                    .order_by(GenerationOutput.is_thumbnail)
                )
            ).all()
            attempts = (
                await verify.scalars(
                    select(GenerationMaterializationAttempt).where(
                        GenerationMaterializationAttempt.job_id == seed.job_id
                    )
                )
            ).all()
        original, *thumbs = outputs
        assert original.is_thumbnail is False
        assert [t.parent_output_id for t in thumbs] == [original.id, original.id]
        assert {t.thumbnail_max_edge for t in thumbs} == {150, 512}
        (attempt,) = attempts
        assert attempt.state == "committed"
        assert set(attempt.planned_storage_keys) == {o.storage_key for o in outputs}
        (profile,) = [e for e in events if e["event"] == "media.video_profile"]
        assert profile["origin"] == origin.value
        assert profile["asset_ref"] == f"output:{original.id}"
        assert profile["job_id"] == str(seed.job_id)
    finally:
        await _cleanup(db_engine, seed)


@pytest.mark.parametrize(("provider", "origin"), _PROVIDERS)
async def test_c16_a_crash_after_the_planned_commit_is_cleaned_by_retention(
    db_engine: AsyncEngine, provider: Provider, origin: VideoProfileOrigin
) -> None:
    """The durable attempt, not process memory, lets a later sweep delete orphaned R2 keys."""
    seed = await _seed(db_engine, provider)
    storage = _storage()
    materializer = _materializer(
        provider, origin, storage=storage, ingestor=MagicMock(), http=_http()
    )
    planned = [f"users/{seed.user_id}/outputs/{seed.job_id}/{uuid4()}.mp4"]
    try:
        async with AsyncSession(bind=db_engine, expire_on_commit=False) as session:
            token = await materializer._claim_video_finalization(session, seed.job_id)
            assert token is not None
            await materializer._create_materialization_attempt(
                session,
                job_id=seed.job_id,
                product_id="vex",
                claim_token=token,
                planned_storage_keys=planned,
            )
        # The process dies here: the claim is later cleared by a takeover/expiry sweep.
        async with AsyncSession(bind=db_engine, expire_on_commit=False) as session:
            await session.execute(
                update(GenerationJob)
                .where(GenerationJob.id == seed.job_id)
                .values(finalization_claim_token=None, finalization_lease_expires_at=None)
            )
            await session.commit()

        retention_storage = MagicMock()
        retention_storage.delete_many = AsyncMock()
        service = ContentRetentionService(
            session_factory=async_sessionmaker(db_engine, expire_on_commit=False),
            storage=retention_storage,
            batch_size=50,
            max_batches_per_run=2,
        )
        reconciled, _ = await service.reconcile_storage_artifacts(product_id="vex")

        assert reconciled >= 1
        retention_storage.delete_many.assert_awaited_with(planned)
        async with AsyncSession(bind=db_engine, expire_on_commit=False) as verify:
            state = await verify.scalar(
                select(GenerationMaterializationAttempt.state).where(
                    GenerationMaterializationAttempt.job_id == seed.job_id
                )
            )
        assert state == "cleaned"
    finally:
        await _cleanup(db_engine, seed)


# ---- C15: Wan settlement through the service --------------------------------------------


def _wan_service(
    materializer: RemoteVideoMaterializer,
    handler: Any,
    *,
    policy: ProviderBillingPolicyRegistry | None = None,
) -> WanJobService:
    client = ModelStudioClient(
        base_url="https://ws.maas.aliyuncs.com",
        api_key=SecretStr("sk-test"),
        timeout_seconds=30,
        transport=httpx.MockTransport(handler),
    )
    return WanJobService(
        client,
        materializer,
        VideoDownloadClient(),
        billing_service=BillingService(),
        billing_policy=policy
        or ProviderBillingPolicyRegistry.from_settings(Settings.model_construct()),
    )


async def _refunds(engine: AsyncEngine, job_id: UUID) -> int:
    async with AsyncSession(bind=engine, expire_on_commit=False) as session:
        return int(
            await session.scalar(
                select(func.count())
                .select_from(TokenTransaction)
                .where(
                    TokenTransaction.job_id == job_id,
                    TokenTransaction.transaction_type == TransactionType.REFUND.value,
                )
            )
            or 0
        )


async def test_c15_a_wan_moderation_rejection_fails_the_job_and_refunds(
    db_engine: AsyncEngine,
) -> None:
    seed = await _seed(db_engine, Provider.ALIBABA, with_debit=True)
    materializer = _materializer(
        Provider.ALIBABA,
        VideoProfileOrigin.ALIBABA_OUTPUT,
        storage=_storage(),
        ingestor=MagicMock(),
        http=_http(),
    )

    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "request_id": "r",
                "output": {
                    "task_id": "task-1",
                    "task_status": "FAILED",
                    "code": "DataInspectionFailed",
                    "message": "Input data may contain inappropriate content.",
                },
            },
        )

    service = _wan_service(materializer, handler)
    await service._client.connect()
    try:
        async with AsyncSession(bind=db_engine, expire_on_commit=False) as session:
            job = await service.poll_video_job(session, seed.job_id)

        assert job is not None
        assert job.status == JobStatus.FAILED.value
        assert job.failure_code == "provider_moderation_rejected"
        assert "charged" not in (job.public_error_message or "")
        assert await _refunds(db_engine, seed.job_id) == 1
    finally:
        await service._client.close()
        await _cleanup(db_engine, seed)


async def test_c15_an_undelivered_wan_output_is_charged_with_the_billed_notice(
    db_engine: AsyncEngine,
) -> None:
    seed = await _seed(db_engine, Provider.ALIBABA, with_debit=True)
    ingestor = MagicMock()
    ingestor.prepare_video = AsyncMock(side_effect=InvalidMediaError("not a video"))
    materializer = _materializer(
        Provider.ALIBABA,
        VideoProfileOrigin.ALIBABA_OUTPUT,
        storage=_storage(),
        ingestor=ingestor,
        http=_http(b"garbage"),
    )

    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "request_id": "r",
                "output": {
                    "task_id": "task-1",
                    "task_status": "SUCCEEDED",
                    "video_url": "https://oss.example.com/v.mp4",
                },
                "usage": {"duration": 5.0},
            },
        )

    service = _wan_service(materializer, handler)
    await service._client.connect()
    try:
        async with AsyncSession(bind=db_engine, expire_on_commit=False) as session:
            job = await service.poll_video_job(session, seed.job_id)

        assert job is not None
        assert job.status == JobStatus.FAILED.value
        assert job.failure_code == "provider_output_not_delivered"
        assert "This generation was charged" in (job.public_error_message or "")
        assert await _refunds(db_engine, seed.job_id) == 0
    finally:
        await service._client.close()
        await _cleanup(db_engine, seed)


async def test_wan_overdue_job_times_out_and_refunds(db_engine: AsyncEngine) -> None:
    seed = await _seed(db_engine, Provider.ALIBABA, with_debit=True)
    async with AsyncSession(bind=db_engine, expire_on_commit=False) as session:
        await session.execute(
            update(GenerationJob)
            .where(GenerationJob.id == seed.job_id)
            .values(started_at=datetime.now(UTC) - timedelta(hours=2))
        )
        await session.commit()
    materializer = _materializer(
        Provider.ALIBABA,
        VideoProfileOrigin.ALIBABA_OUTPUT,
        storage=_storage(),
        ingestor=MagicMock(),
        http=_http(),
    )
    called = MagicMock()
    service = _wan_service(materializer, called)
    await service._client.connect()
    try:
        async with AsyncSession(bind=db_engine, expire_on_commit=False) as session:
            job = await service.poll_video_job(session, seed.job_id)

        assert job is not None
        assert job.failure_code == "provider_timeout"
        assert await _refunds(db_engine, seed.job_id) == 1
        called.assert_not_called()
    finally:
        await service._client.close()
        await _cleanup(db_engine, seed)


async def test_wan_worker_settlement_of_still_running_is_a_no_op_for_a_running_job(
    db_engine: AsyncEngine,
) -> None:
    seed = await _seed(db_engine, Provider.ALIBABA)
    materializer = _materializer(
        Provider.ALIBABA,
        VideoProfileOrigin.ALIBABA_OUTPUT,
        storage=_storage(),
        ingestor=MagicMock(),
        http=_http(),
    )
    service = _wan_service(materializer, MagicMock())
    try:
        async with AsyncSession(bind=db_engine, expire_on_commit=False) as session:
            job = await service.settle_video_poll_outcome(
                session,
                job_id=seed.job_id,
                outcome=VideoPollOutcome(status=VideoPollStatus.STILL_RUNNING),
                product_id="vex",
            )
        assert job.status == JobStatus.RUNNING.value
    finally:
        await _cleanup(db_engine, seed)
