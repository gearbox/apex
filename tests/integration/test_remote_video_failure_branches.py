"""Shared materializer failure branches, per provider (real PostgreSQL).

``RemoteVideoMaterializer`` is now the completion path of two providers, so its
uncertain-outcome handling is pinned for both: an ``IntegrityError`` after
materialization, cancellation mid-upload and mid-transition, a lost transition,
and a commit outcome that cannot be proven. In every case R2 keys are deleted
only after a database read proves them unreferenced.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from sqlalchemy import select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession

from src.api.services.billing import BillingService
from src.api.services.job_state_transition import GenerationOutputData
from src.api.services.video_jobs.materializer import RemoteVideoMaterializer, _MaterializedVideo
from src.api.services.video_jobs.outcome import RemoteVideoResult
from src.api.services.video_jobs.transition import make_transition_service_factory
from src.core.enums import JobStatus, MediaFormat, Provider
from src.db.models.storage import GenerationJob, GenerationMaterializationAttempt, GenerationOutput
from tests.integration.test_remote_video_materialization import (
    _PROVIDERS,
    _cleanup,
    _http,
    _materializer,
    _no_settle,
    _prepared_video,
    _seed,
    _storage,
    _thumbnails,
)

if TYPE_CHECKING:
    from collections.abc import Callable
    from contextlib import AbstractContextManager
    from uuid import UUID

    from src.api.services.media_ingest.profile_log import VideoProfileOrigin

pytestmark = pytest.mark.asyncio

_URL = RemoteVideoResult(url="https://provider.invalid/v.mp4")


def _ingestor() -> MagicMock:
    ingestor = MagicMock()
    ingestor.prepare_video = AsyncMock(return_value=_prepared_video())
    return ingestor


def _derivatives() -> AbstractContextManager[Any]:
    """Original + two thumbnails, so three keys are planned."""
    import contextlib

    stack = contextlib.ExitStack()
    stack.enter_context(
        patch(
            "src.api.services.video_jobs.materializer.extract_video_thumbnail",
            new=AsyncMock(return_value=b"\xff\xd8\xff\xe0jpeg"),
        )
    )
    stack.enter_context(
        patch(
            "src.api.services.video_jobs.materializer.make_image_thumbnails",
            new=AsyncMock(return_value=_thumbnails()),
        )
    )
    return stack


def _with_transition(
    materializer: RemoteVideoMaterializer, factory: Callable[[AsyncSession], Any]
) -> RemoteVideoMaterializer:
    materializer._transition_service_factory = factory
    return materializer


def _real_factory() -> Callable[[AsyncSession], Any]:
    return make_transition_service_factory(
        billing_service=BillingService(), event_bus=None, ops_event_bus=None
    )


async def _attempt(engine: AsyncEngine, job_id: UUID) -> GenerationMaterializationAttempt:
    async with AsyncSession(bind=engine, expire_on_commit=False) as session:
        return (
            await session.scalars(
                select(GenerationMaterializationAttempt).where(
                    GenerationMaterializationAttempt.job_id == job_id
                )
            )
        ).one()


async def _job(engine: AsyncEngine, job_id: UUID) -> GenerationJob:
    async with AsyncSession(bind=engine, expire_on_commit=False) as session:
        return (
            await session.scalars(select(GenerationJob).where(GenerationJob.id == job_id))
        ).one()


def _deleted(storage: MagicMock) -> list[str]:
    return [key for call in storage.delete_many.await_args_list for key in call.args[0]]


@pytest.mark.parametrize(("provider", "origin"), _PROVIDERS)
async def test_integrity_error_before_commit_reconciles_and_deletes_only_proven_orphans(
    db_engine: AsyncEngine, provider: Provider, origin: VideoProfileOrigin
) -> None:
    seed = await _seed(db_engine, provider)
    storage = _storage()
    materializer = _materializer(
        provider, origin, storage=storage, ingestor=_ingestor(), http=_http()
    )
    failing = MagicMock()
    failing.transition_claimed_video_to_completed = AsyncMock(
        side_effect=IntegrityError("INSERT", {}, Exception("duplicate"))
    )
    _with_transition(materializer, lambda _session: failing)
    try:
        with _derivatives():
            async with AsyncSession(bind=db_engine, expire_on_commit=False) as session:
                job = await materializer.finalize_completed_video(
                    session,
                    job_id=seed.job_id,
                    result=_URL,
                    product_id="vex",
                    settle_failure=_no_settle,
                )

        attempt = await _attempt(db_engine, seed.job_id)
        assert job.status == JobStatus.RUNNING.value  # the uncommitted transition did not stick
        assert len(attempt.planned_storage_keys) == 3
        assert sorted(_deleted(storage)) == sorted(attempt.planned_storage_keys)
        assert attempt.state == "cleaned"
    finally:
        await _cleanup(db_engine, seed)


@pytest.mark.parametrize(("provider", "origin"), _PROVIDERS)
async def test_integrity_error_after_the_commit_landed_deletes_nothing(
    db_engine: AsyncEngine, provider: Provider, origin: VideoProfileOrigin
) -> None:
    """The acknowledgement was lost but PostgreSQL committed: referenced keys must survive."""
    seed = await _seed(db_engine, provider)
    storage = _storage()
    materializer = _materializer(
        provider, origin, storage=storage, ingestor=_ingestor(), http=_http()
    )
    real = _real_factory()

    def factory(session: AsyncSession) -> Any:
        service = real(session)
        original = service.transition_claimed_video_to_completed

        async def commit_then_lose_the_ack(*args: Any, **kwargs: Any) -> Any:
            await original(*args, **kwargs)
            await session.commit()
            raise IntegrityError("COMMIT", {}, Exception("ack lost"))

        wrapped = MagicMock()
        wrapped.transition_claimed_video_to_completed = commit_then_lose_the_ack
        return wrapped

    _with_transition(materializer, factory)
    try:
        with _derivatives():
            async with AsyncSession(bind=db_engine, expire_on_commit=False) as session:
                job = await materializer.finalize_completed_video(
                    session,
                    job_id=seed.job_id,
                    result=_URL,
                    product_id="vex",
                    settle_failure=_no_settle,
                )

        assert job.status == JobStatus.COMPLETED.value
        assert (await _attempt(db_engine, seed.job_id)).state == "committed"
        storage.delete_many.assert_not_awaited()
    finally:
        await _cleanup(db_engine, seed)


@pytest.mark.parametrize(("provider", "origin"), _PROVIDERS)
async def test_cancellation_mid_upload_releases_the_claim_and_leaves_durable_cleanup(
    db_engine: AsyncEngine, provider: Provider, origin: VideoProfileOrigin
) -> None:
    seed = await _seed(db_engine, provider)
    storage = _storage()
    storage.put_raw = AsyncMock(side_effect=asyncio.CancelledError)
    materializer = _materializer(
        provider, origin, storage=storage, ingestor=_ingestor(), http=_http()
    )
    try:
        with _derivatives(), pytest.raises(asyncio.CancelledError):
            async with AsyncSession(bind=db_engine, expire_on_commit=False) as session:
                await materializer.finalize_completed_video(
                    session,
                    job_id=seed.job_id,
                    result=_URL,
                    product_id="vex",
                    settle_failure=_no_settle,
                )

        job = await _job(db_engine, seed.job_id)
        assert job.finalization_claim_token is None  # released, not left to expire
        assert job.status == JobStatus.RUNNING.value
        assert (await _attempt(db_engine, seed.job_id)).state == "cleanup_pending"
    finally:
        await _cleanup(db_engine, seed)


@pytest.mark.parametrize(("provider", "origin"), _PROVIDERS)
async def test_cancellation_after_materialization_is_reconciled_under_shield(
    db_engine: AsyncEngine, provider: Provider, origin: VideoProfileOrigin
) -> None:
    seed = await _seed(db_engine, provider)
    storage = _storage()
    materializer = _materializer(
        provider, origin, storage=storage, ingestor=_ingestor(), http=_http()
    )
    cancelled = MagicMock()
    cancelled.transition_claimed_video_to_completed = AsyncMock(side_effect=asyncio.CancelledError)
    _with_transition(materializer, lambda _session: cancelled)
    try:
        with _derivatives(), pytest.raises(asyncio.CancelledError):
            async with AsyncSession(bind=db_engine, expire_on_commit=False) as session:
                await materializer.finalize_completed_video(
                    session,
                    job_id=seed.job_id,
                    result=_URL,
                    product_id="vex",
                    settle_failure=_no_settle,
                )

        attempt = await _attempt(db_engine, seed.job_id)
        assert sorted(_deleted(storage)) == sorted(attempt.planned_storage_keys)
        assert attempt.state == "cleaned"
    finally:
        await _cleanup(db_engine, seed)


@pytest.mark.parametrize(("provider", "origin"), _PROVIDERS)
async def test_a_lost_transition_cleans_up_instead_of_attaching_outputs(
    db_engine: AsyncEngine, provider: Provider, origin: VideoProfileOrigin
) -> None:
    """The lease was taken over mid-upload: the real CAS refuses, so every key is an orphan."""
    seed = await _seed(db_engine, provider)
    storage = _storage()

    async def takeover(*_: Any, **__: Any) -> None:
        async with AsyncSession(bind=db_engine, expire_on_commit=False) as other:
            await other.execute(
                update(GenerationJob)
                .where(GenerationJob.id == seed.job_id)
                .values(finalization_claim_token="someone-else")
            )
            await other.commit()

    storage.put_raw = AsyncMock(side_effect=takeover)
    materializer = _materializer(
        provider, origin, storage=storage, ingestor=_ingestor(), http=_http()
    )
    try:
        with _derivatives():
            async with AsyncSession(bind=db_engine, expire_on_commit=False) as session:
                await materializer.finalize_completed_video(
                    session,
                    job_id=seed.job_id,
                    result=_URL,
                    product_id="vex",
                    settle_failure=_no_settle,
                )

        attempt = await _attempt(db_engine, seed.job_id)
        assert (await _job(db_engine, seed.job_id)).status == JobStatus.RUNNING.value
        assert attempt.state == "cleaned"
        assert sorted(_deleted(storage)) == sorted(attempt.planned_storage_keys)
        async with AsyncSession(bind=db_engine, expire_on_commit=False) as verify:
            outputs = (
                await verify.scalars(
                    select(GenerationOutput).where(GenerationOutput.job_id == seed.job_id)
                )
            ).all()
        assert outputs == []
    finally:
        await _cleanup(db_engine, seed)


@pytest.mark.parametrize(("provider", "origin"), _PROVIDERS)
async def test_reconcile_deletes_only_the_keys_no_output_row_references(
    db_engine: AsyncEngine, provider: Provider, origin: VideoProfileOrigin
) -> None:
    """Another terminal transition won with the original; only the derivatives are orphans."""
    seed = await _seed(db_engine, provider)
    storage = _storage()
    materializer = _materializer(
        provider, origin, storage=storage, ingestor=_ingestor(), http=_http()
    )
    prefix = f"users/{seed.user_id}/outputs/{seed.job_id}"
    original_key = f"{prefix}/{uuid4()}.mp4"
    thumb_keys = [f"{prefix}/{uuid4()}.webp", f"{prefix}/{uuid4()}.webp"]
    original_id = uuid4()
    try:
        async with AsyncSession(bind=db_engine, expire_on_commit=False) as session:
            token = await materializer._claim_video_finalization(session, seed.job_id)
            assert token is not None
            attempt_id = await materializer._create_materialization_attempt(
                session,
                job_id=seed.job_id,
                product_id="vex",
                claim_token=token,
                planned_storage_keys=[original_key, *thumb_keys],
            )
            # The competing transition: job completed, original attached.
            job = await session.get(GenerationJob, seed.job_id)
            assert job is not None
            job.status = JobStatus.COMPLETED.value
            session.add(
                GenerationOutput(
                    id=original_id,
                    user_id=seed.user_id,
                    job_id=seed.job_id,
                    storage_key=original_key,
                    content_type="video/mp4",
                    size_bytes=1,
                    format=MediaFormat.MP4.value,
                    output_index=0,
                    product_id="vex",
                    expires_at=datetime.now(UTC) + timedelta(days=7),
                )
            )
            await session.commit()

            await materializer._reconcile_materialization_attempt(
                session,
                job_id=seed.job_id,
                product_id="vex",
                materialized=_MaterializedVideo(
                    outputs=[
                        GenerationOutputData(
                            id=original_id,
                            storage_key=original_key,
                            content_type="video/mp4",
                            size_bytes=1,
                            format=MediaFormat.MP4.value,
                            output_index=0,
                            expires_at=datetime.now(UTC) + timedelta(days=7),
                        )
                    ],
                    storage_keys=[original_key, *thumb_keys],
                    stream_profile=_prepared_video().stream_profile,
                    attempt_id=attempt_id,
                    product_id="vex",
                ),
            )

        assert sorted(_deleted(storage)) == sorted(thumb_keys)
        assert original_key not in _deleted(storage)
        assert (await _attempt(db_engine, seed.job_id)).state == "cleaned"
    finally:
        await _cleanup(db_engine, seed)


@pytest.mark.parametrize(("provider", "origin"), _PROVIDERS)
async def test_an_unprovable_commit_outcome_is_ambiguous_and_deletes_nothing(
    db_engine: AsyncEngine, provider: Provider, origin: VideoProfileOrigin
) -> None:
    seed = await _seed(db_engine, provider)
    storage = _storage()
    materializer = _materializer(
        provider, origin, storage=storage, ingestor=_ingestor(), http=_http()
    )
    broken = MagicMock()
    broken.transition_claimed_video_to_completed = AsyncMock(side_effect=RuntimeError("commit?"))
    _with_transition(materializer, lambda _session: broken)

    real_refresh = materializer._refresh_authoritative_job
    calls = {"count": 0}

    async def refresh(session: AsyncSession, job_id: UUID) -> GenerationJob:
        calls["count"] += 1
        if calls["count"] > 1:  # the first call reads user_id; reconciliation cannot re-read
            raise ConnectionError("database went away")
        return await real_refresh(session, job_id)

    materializer._refresh_authoritative_job = refresh  # type: ignore[method-assign]
    try:
        with _derivatives(), pytest.raises(RuntimeError, match="commit"):
            async with AsyncSession(bind=db_engine, expire_on_commit=False) as session:
                await materializer.finalize_completed_video(
                    session,
                    job_id=seed.job_id,
                    result=_URL,
                    product_id="vex",
                    settle_failure=_no_settle,
                )

        attempt = await _attempt(db_engine, seed.job_id)
        assert attempt.state == "ambiguous"
        storage.delete_many.assert_not_awaited()
    finally:
        await _cleanup(db_engine, seed)
