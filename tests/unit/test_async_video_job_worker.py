"""Provider independence of ``AsyncVideoJobWorker`` (one instance per provider)."""

from __future__ import annotations

import asyncio
from typing import Any, ClassVar
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from src.api.services.generation.provider_failures import ProviderFailure, ProviderFailureKind
from src.api.services.video_jobs.outcome import VideoPollOutcome
from src.api.services.video_jobs.worker import AsyncVideoJobWorker
from src.core.enums import JobStatus, Provider, VideoPollStatus
from src.workers.base import lease_key

pytestmark = pytest.mark.unit


class _TransientError(Exception):
    def __init__(self) -> None:
        super().__init__("transient")
        self.failure = ProviderFailure(
            kind=ProviderFailureKind.RATE_LIMITED,
            provider=Provider.ALIBABA,
            sanitized_message="rate limited",
        )


class _FakeService:
    """Minimal ``AsyncVideoJobService`` with an optional gate on polling."""

    provider: ClassVar[Provider] = Provider.GROK
    transient_poll_errors: ClassVar[tuple[type[Exception], ...]] = (_TransientError,)
    running_progress_pct: ClassVar[int | None] = None

    def __init__(self, gate: asyncio.Event | None = None) -> None:
        self._gate = gate
        self.polled: list[Any] = []
        self.settled: list[Any] = []

    async def poll_video_job_for_worker(self, session: object, job_id: object) -> VideoPollOutcome:
        del session
        if self._gate is not None:
            await self._gate.wait()
        self.polled.append(job_id)
        return VideoPollOutcome(status=VideoPollStatus.COMPLETED)

    async def settle_video_poll_outcome(self, session: object, **kwargs: Any) -> None:
        del session
        self.settled.append(kwargs["job_id"])


class _GrokService(_FakeService):
    provider = Provider.GROK


class _GrokProgressService(_GrokService):
    running_progress_pct = 50


class _WanService(_FakeService):
    provider = Provider.ALIBABA


class _SessionCtx:
    async def __aenter__(self) -> object:
        return AsyncMock()

    async def __aexit__(self, *exc: object) -> bool:
        return False


def _job() -> MagicMock:
    job = MagicMock()
    job.id = uuid4()
    job.user_id = uuid4()
    job.product_id = "vex"
    job.status = JobStatus.QUEUED.value
    job.generation_type = "t2v"
    return job


def _worker(name: str, service: _FakeService) -> AsyncVideoJobWorker:
    db_manager = MagicMock()
    db_manager.session.side_effect = _SessionCtx
    return AsyncVideoJobWorker(
        name=name,
        job_service=service,  # type: ignore[arg-type]
        interval_seconds=5,
        max_concurrent_polls=2,
        db_manager=db_manager,
        redis_client_factory=MagicMock(),
    )


async def test_c2_a_blocked_provider_does_not_delay_the_other() -> None:
    gate = asyncio.Event()
    grok_job, wan_job = _job(), _job()
    grok_service, wan_service = _GrokService(), _WanService(gate=gate)
    grok_worker = _worker("grok_video_poller", grok_service)
    wan_worker = _worker("alibaba_video_poller", wan_service)

    async def list_pending(*, provider: Provider) -> list[MagicMock]:
        return [grok_job] if provider is Provider.GROK else [wan_job]

    repo = MagicMock()
    repo.list_pending_video_jobs = AsyncMock(side_effect=list_pending)

    with patch("src.api.services.video_jobs.worker.JobRepository", return_value=repo):
        wan_tick = asyncio.create_task(wan_worker.run_once())
        await asyncio.sleep(0)  # let the Wan poll start and block on the gate
        await asyncio.wait_for(grok_worker.run_once(), timeout=2)

        assert grok_service.settled == [grok_job.id]
        assert not wan_tick.done()
        assert wan_service.settled == []

        gate.set()
        await asyncio.wait_for(wan_tick, timeout=2)

    assert wan_service.settled == [wan_job.id]


def test_c2_each_provider_worker_has_its_own_lease_key() -> None:
    grok = _worker("grok_video_poller", _GrokService())
    wan = _worker("alibaba_video_poller", _WanService())

    assert grok.name != wan.name
    assert lease_key(grok.name) != lease_key(wan.name)
    assert lease_key("grok_video_poller").endswith(":worker:grok_video_poller:lease")


async def test_running_job_without_progress_value_is_neither_settled_nor_announced() -> None:
    service = _GrokService()
    event_bus = AsyncMock()
    worker = _worker("alibaba_video_poller", service)
    worker._event_bus = event_bus
    job = _job()
    job.status = JobStatus.RUNNING.value
    service.poll_video_job_for_worker = AsyncMock(  # type: ignore[method-assign]
        return_value=VideoPollOutcome(status=VideoPollStatus.STILL_RUNNING)
    )

    await worker._poll_one(job, AsyncMock())

    event_bus.publish.assert_not_awaited()
    assert service.settled == []


async def test_running_job_with_legacy_progress_value_emits_it() -> None:
    service = _GrokProgressService()
    event_bus = AsyncMock()
    worker = _worker("grok_video_poller", service)
    worker._event_bus = event_bus
    job = _job()
    job.status = JobStatus.RUNNING.value
    service.poll_video_job_for_worker = AsyncMock(  # type: ignore[method-assign]
        return_value=VideoPollOutcome(status=VideoPollStatus.STILL_RUNNING)
    )

    await worker._poll_one(job, AsyncMock())

    event_bus.publish.assert_awaited_once()
    assert event_bus.publish.await_args.kwargs["payload"].progress_pct == 50


async def test_transient_error_declared_by_the_service_skips_the_tick() -> None:
    service = _WanService()
    service.poll_video_job_for_worker = AsyncMock(  # type: ignore[method-assign]
        side_effect=_TransientError()
    )
    worker = _worker("alibaba_video_poller", service)

    await worker._poll_one(_job(), AsyncMock())

    assert service.settled == []
