"""R1 / P0-1: the provider-output byte budget, the streaming cap and the ingest factories."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import httpx
import pytest
from pydantic import ValidationError
from structlog.testing import capture_logs

from src.api.services.generation.provider_billing_policy import DEFAULT_PROVIDER_BILLING_POLICIES
from src.api.services.media_ingest import InvalidMediaError
from src.api.services.media_ingest.factory import (
    build_media_ingest_service,
    build_provider_video_ingest_service,
)
from src.api.services.media_ingest.profile_log import VideoProfileOrigin
from src.api.services.video_jobs.materializer import RemoteVideoMaterializer
from src.api.services.video_jobs.outcome import RemoteVideoResult
from src.core.config import Settings
from src.core.enums import Provider

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

_CAP = 1024
_URL = "https://oss.invalid/private/presigned?sig=SECRET"
_PROVIDERS = [
    pytest.param(Provider.GROK, VideoProfileOrigin.GROK_OUTPUT, id="grok"),
    pytest.param(Provider.ALIBABA, VideoProfileOrigin.ALIBABA_OUTPUT, id="alibaba"),
]


def _materializer(
    provider: Provider,
    origin: VideoProfileOrigin,
    handler: httpx.MockTransport,
    *,
    storage: MagicMock | None = None,
    ingestor: MagicMock | None = None,
) -> RemoteVideoMaterializer:
    http = httpx.AsyncClient(transport=handler)
    return RemoteVideoMaterializer(
        provider=provider,
        profile_origin=origin,
        storage=storage or MagicMock(),
        media_ingestor=ingestor or MagicMock(),
        http_client_provider=lambda: http,
        retention_days=7,
        finalization_lease_seconds=120,
        max_output_bytes=_CAP,
        billing_policy=DEFAULT_PROVIDER_BILLING_POLICIES,
        transition_service_factory=MagicMock(),
    )


def _declared(size: int) -> httpx.MockTransport:
    return httpx.MockTransport(
        lambda _: httpx.Response(200, content=b"x" * size, headers={"content-length": str(size)})
    )


def _chunked(chunk: int, count: int) -> httpx.MockTransport:
    async def body() -> AsyncIterator[bytes]:
        for _ in range(count):
            yield b"x" * chunk

    return httpx.MockTransport(lambda _: httpx.Response(200, content=body()))


@pytest.mark.parametrize(("provider", "origin"), _PROVIDERS)
async def test_download_over_the_cap_via_content_length_is_refused_before_any_work(
    provider: Provider, origin: VideoProfileOrigin
) -> None:
    storage = MagicMock(put_raw=AsyncMock())
    ingestor = MagicMock(prepare_video=AsyncMock())
    materializer = _materializer(
        provider, origin, _declared(_CAP + 1), storage=storage, ingestor=ingestor
    )
    materializer._create_materialization_attempt = AsyncMock()  # type: ignore[method-assign]

    with capture_logs() as logs, pytest.raises(InvalidMediaError, match="output byte limit"):
        await materializer._materialize_video_result(
            user_id=uuid4(),
            job_id=uuid4(),
            result=RemoteVideoResult(url=_URL),
            product_id="vex",
            session=AsyncMock(),
            claim_token="claim",
        )

    ingestor.prepare_video.assert_not_awaited()
    storage.put_raw.assert_not_awaited()
    materializer._create_materialization_attempt.assert_not_awaited()  # type: ignore[attr-defined]
    (event,) = [e for e in logs if e["event"] == "video_job.output_too_large"]
    assert event["content_length"] == _CAP + 1
    assert event["max_output_bytes"] == _CAP
    assert event["provider"] == provider.value
    assert "SECRET" not in repr(logs)


@pytest.mark.parametrize(("provider", "origin"), _PROVIDERS)
async def test_download_over_the_cap_without_content_length_aborts_while_streaming(
    provider: Provider, origin: VideoProfileOrigin
) -> None:
    storage = MagicMock(put_raw=AsyncMock())
    ingestor = MagicMock(prepare_video=AsyncMock())
    # 100 chunks of 100 B = 10 kB on the wire; the cap is 1 kB.
    transport = _chunked(chunk=100, count=100)
    materializer = _materializer(provider, origin, transport, storage=storage, ingestor=ingestor)
    materializer._create_materialization_attempt = AsyncMock()  # type: ignore[method-assign]

    with capture_logs() as logs, pytest.raises(InvalidMediaError, match="output byte limit"):
        await materializer._materialize_video_result(
            user_id=uuid4(),
            job_id=uuid4(),
            result=RemoteVideoResult(url=_URL),
            product_id="vex",
            session=AsyncMock(),
            claim_token="claim",
        )

    ingestor.prepare_video.assert_not_awaited()
    storage.put_raw.assert_not_awaited()
    materializer._create_materialization_attempt.assert_not_awaited()  # type: ignore[attr-defined]
    (event,) = [e for e in logs if e["event"] == "video_job.output_too_large"]
    assert event["content_length"] is None


async def test_a_body_exactly_at_the_cap_is_accepted() -> None:
    materializer = _materializer(
        Provider.ALIBABA, VideoProfileOrigin.ALIBABA_OUTPUT, _declared(_CAP)
    )

    data = await materializer._download_video(_URL, uuid4())

    assert len(data) == _CAP


async def test_an_http_error_status_still_raises_before_buffering() -> None:
    transport = httpx.MockTransport(lambda _: httpx.Response(403, content=b"denied"))
    materializer = _materializer(Provider.ALIBABA, VideoProfileOrigin.ALIBABA_OUTPUT, transport)

    with pytest.raises(httpx.HTTPStatusError):
        await materializer._download_video(_URL, uuid4())


def test_provider_and_upload_services_have_independent_byte_budgets() -> None:
    settings = Settings.model_construct(
        max_upload_size_mb=20, media_ingest_provider_video_max_mb=200
    )

    uploads = build_media_ingest_service(settings)
    provider = build_provider_video_ingest_service(settings)

    assert uploads._max_input_bytes == 20 * 1024 * 1024
    assert provider._max_input_bytes == 200 * 1024 * 1024


def test_both_factories_share_one_video_and_image_semaphore_when_supplied() -> None:
    settings = Settings.model_construct()
    video_slots = asyncio.Semaphore(2)
    image_slots = asyncio.Semaphore(4)

    uploads = build_media_ingest_service(settings, video_slots=video_slots, image_slots=image_slots)
    provider = build_provider_video_ingest_service(
        settings, video_slots=video_slots, image_slots=image_slots
    )

    assert uploads._video_slots is video_slots
    assert provider._video_slots is video_slots
    assert uploads._image_slots is image_slots
    assert provider._image_slots is image_slots


def test_without_supplied_semaphores_each_service_owns_its_slots() -> None:
    settings = Settings.model_construct()

    assert (
        build_media_ingest_service(settings)._video_slots
        is not build_provider_video_ingest_service(settings)._video_slots
    )


def test_provider_budget_below_the_upload_budget_fails_startup() -> None:
    with pytest.raises(ValidationError, match="media_ingest_provider_video_max_mb"):
        Settings(max_upload_size_mb=50, media_ingest_provider_video_max_mb=40)  # type: ignore[call-arg]


def test_provider_budget_defaults_to_200_mb_and_is_bounded() -> None:
    assert Settings.model_construct().media_ingest_provider_video_max_mb == 200
    for bad in (19, 1001):
        with pytest.raises(ValidationError):
            Settings(media_ingest_provider_video_max_mb=bad)  # type: ignore[call-arg]
