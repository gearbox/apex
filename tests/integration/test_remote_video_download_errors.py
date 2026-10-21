"""R2 / P1-1: how a result-URL HTTP error is settled, per provider (real PostgreSQL).

A *permanent* 4xx on the provider's presigned result URL (403/404/410, ...) will never
heal inside the poll window. It used to loop on the failing download until
``max_poll_time`` and then be refunded as a timeout, bypassing the configured
undelivered-output billing policy although the provider had completed and billed the
generation. It now settles on the first poll as ``OUTPUT_NOT_DELIVERED``.
A *transient* error (5xx, 408, 425, 429, transport failure) keeps retrying.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.services.billing import BillingService
from src.api.services.generation.provider_billing_policy import ProviderBillingPolicyRegistry
from src.api.services.grok import GrokVideoResult
from src.api.services.media_ingest.profile_log import VideoProfileOrigin
from src.core.config import Settings
from src.core.enums import JobStatus, Provider
from src.db.models.storage import GenerationJob, GenerationMaterializationAttempt
from tests.grok_support import make_grok_job_service
from tests.integration.test_remote_video_materialization import (
    _cleanup,
    _materializer,
    _refunds,
    _seed,
    _storage,
    _wan_service,
)

if TYPE_CHECKING:
    from collections.abc import Callable
    from uuid import UUID

    from sqlalchemy.ext.asyncio import AsyncEngine

pytestmark = pytest.mark.asyncio

_PROVIDERS = [pytest.param(Provider.GROK, id="grok"), pytest.param(Provider.ALIBABA, id="alibaba")]
Policy = Literal["charge", "refund"]


def _policy(undelivered: Policy) -> ProviderBillingPolicyRegistry:
    return ProviderBillingPolicyRegistry.from_settings(
        Settings.model_construct(
            grok_undelivered_output_billing_policy=undelivered,
            wan_undelivered_output_billing_policy=undelivered,
        )
    )


def _wan_succeeded(_: httpx.Request) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "request_id": "r",
            "output": {
                "task_id": "task-1",
                "task_status": "SUCCEEDED",
                "video_url": "https://oss.invalid/expired.mp4",
            },
            "usage": {"duration": 5.0},
        },
    )


@dataclass
class _Rig:
    poll: Callable[[AsyncSession, UUID], Any]
    close: Callable[[], Any]
    ingestor: MagicMock
    storage: MagicMock
    downloads: list[str]


async def _rig(
    provider: Provider,
    *,
    answer: Callable[[httpx.Request], httpx.Response],
    undelivered: Policy = "charge",
) -> _Rig:
    """One service per provider whose result URL answers through ``answer``."""
    downloads: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        downloads.append(str(request.url))
        return answer(request)

    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    storage = _storage()
    ingestor = MagicMock()
    ingestor.prepare_video = AsyncMock()
    policy = _policy(undelivered)

    if provider is Provider.ALIBABA:
        materializer = _materializer(
            Provider.ALIBABA,
            VideoProfileOrigin.ALIBABA_OUTPUT,
            storage=storage,
            ingestor=ingestor,
            http=http,
            policy=policy,
        )
        wan = _wan_service(materializer, _wan_succeeded, policy=policy)
        await wan._client.connect()
        return _Rig(wan.poll_video_job, wan._client.close, ingestor, storage, downloads)

    grok_client = MagicMock()
    grok_client.get_video_result = AsyncMock(
        return_value=GrokVideoResult(url="https://vidgen.x.ai/expired.mp4")
    )
    grok = make_grok_job_service(
        grok_client,
        storage,
        billing_service=BillingService(),
        media_ingestor=ingestor,
        provider_video_ingestor=ingestor,
        billing_policy=policy,
    )
    grok._http_client = http
    return _Rig(grok.poll_video_job, grok.close, ingestor, storage, downloads)


async def _job(engine: AsyncEngine, job_id: UUID) -> GenerationJob:
    async with AsyncSession(bind=engine, expire_on_commit=False) as session:
        return (
            await session.scalars(select(GenerationJob).where(GenerationJob.id == job_id))
        ).one()


async def _attempts(engine: AsyncEngine, job_id: UUID) -> int:
    async with AsyncSession(bind=engine, expire_on_commit=False) as session:
        return int(
            await session.scalar(
                select(func.count())
                .select_from(GenerationMaterializationAttempt)
                .where(GenerationMaterializationAttempt.job_id == job_id)
            )
            or 0
        )


@pytest.mark.parametrize("provider", _PROVIDERS)
@pytest.mark.parametrize("status", [403, 404, 410])
async def test_a_permanent_result_url_error_settles_as_undelivered_on_the_first_poll(
    db_engine: AsyncEngine, provider: Provider, status: int
) -> None:
    seed = await _seed(db_engine, provider, with_debit=True)
    rig = await _rig(provider, answer=lambda _: httpx.Response(status, content=b"gone"))
    try:
        async with AsyncSession(bind=db_engine, expire_on_commit=False) as session:
            polled = await rig.poll(session, seed.job_id)

        assert polled is not None
        assert polled.status == JobStatus.FAILED.value
        assert polled.failure_code == "provider_output_not_delivered"
        assert "This generation was charged" in (polled.public_error_message or "")
        assert await _refunds(db_engine, seed.job_id) == 0  # default policy: charge

        job = await _job(db_engine, seed.job_id)
        assert job.status == JobStatus.FAILED.value
        assert job.finalization_claim_token is None
        assert job.finalization_lease_expires_at is None
        assert await _attempts(db_engine, seed.job_id) == 0
        assert len(rig.downloads) == 1
        rig.ingestor.prepare_video.assert_not_awaited()
        rig.storage.put_raw.assert_not_awaited()
    finally:
        await rig.close()
        await _cleanup(db_engine, seed)


@pytest.mark.parametrize("provider", _PROVIDERS)
async def test_a_permanent_result_url_error_honours_the_refund_policy(
    db_engine: AsyncEngine, provider: Provider
) -> None:
    seed = await _seed(db_engine, provider, with_debit=True)
    rig = await _rig(
        provider, answer=lambda _: httpx.Response(404, content=b"gone"), undelivered="refund"
    )
    try:
        async with AsyncSession(bind=db_engine, expire_on_commit=False) as session:
            polled = await rig.poll(session, seed.job_id)

        assert polled is not None
        assert polled.status == JobStatus.FAILED.value
        assert polled.failure_code == "provider_output_not_delivered"
        assert "This generation was charged" not in (polled.public_error_message or "")
        assert await _refunds(db_engine, seed.job_id) == 1
    finally:
        await rig.close()
        await _cleanup(db_engine, seed)


def _connect_error(request: httpx.Request) -> httpx.Response:
    raise httpx.ConnectError("boom", request=request)


@pytest.mark.parametrize("provider", _PROVIDERS)
@pytest.mark.parametrize(
    ("answer", "expected"),
    [
        pytest.param(lambda _: httpx.Response(500), httpx.HTTPStatusError, id="500"),
        pytest.param(lambda _: httpx.Response(503), httpx.HTTPStatusError, id="503"),
        pytest.param(lambda _: httpx.Response(408), httpx.HTTPStatusError, id="408"),
        pytest.param(lambda _: httpx.Response(429), httpx.HTTPStatusError, id="429"),
        pytest.param(_connect_error, httpx.ConnectError, id="connect-error"),
    ],
)
async def test_a_transient_result_url_error_keeps_the_job_running_and_retries(
    db_engine: AsyncEngine,
    provider: Provider,
    answer: Callable[[httpx.Request], httpx.Response],
    expected: type[Exception],
) -> None:
    seed = await _seed(db_engine, provider, with_debit=True)
    rig = await _rig(provider, answer=answer)
    try:
        for attempt in (1, 2):
            async with AsyncSession(bind=db_engine, expire_on_commit=False) as session:
                with pytest.raises(expected):
                    await rig.poll(session, seed.job_id)

            job = await _job(db_engine, seed.job_id)
            assert job.status == JobStatus.RUNNING.value
            assert job.failure_code is None
            assert job.finalization_claim_token is None  # released, so the next poll retries
            assert await _refunds(db_engine, seed.job_id) == 0
            assert await _attempts(db_engine, seed.job_id) == 0
            assert len(rig.downloads) == attempt
        rig.ingestor.prepare_video.assert_not_awaited()
        rig.storage.put_raw.assert_not_awaited()
    finally:
        await rig.close()
        await _cleanup(db_engine, seed)
