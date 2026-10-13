"""WanJobService: submission saga, poll mapping and failure policy; WanGenerationProvider."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from structlog.testing import capture_logs

from src.api.schemas.unified_generation import SourceMediaReference, UnifiedGenerationRequest
from src.api.services.generation.provider_billing_policy import ProviderBillingPolicyRegistry
from src.api.services.generation.provider_failures import (
    ProviderFailureKind,
    ProviderModerationRejectedError,
    ProviderSubmissionFailedError,
)
from src.api.services.generation.source_media import ResolvedSourceMedia
from src.api.services.generation.wan_provider import WanGenerationProvider
from src.api.services.model_studio import (
    ModelStudioAPIError,
    ModelStudioFailureClassifier,
    ModelStudioTask,
    ModelStudioTaskRef,
    ModelStudioTaskStatus,
    ModelStudioTransientError,
    ModelStudioVideoUsage,
)
from src.api.services.model_studio.wan_job_service import WanJobError, WanJobService
from src.core.config import Settings
from src.core.enums import (
    AspectRatio,
    GenerationType,
    JobStatus,
    MediaKind,
    ModelType,
    Provider,
    VideoPollStatus,
    VideoResolution,
)
from src.core.library_ref import AssetRef, LibraryAssetSource, format_asset_ref

pytestmark = pytest.mark.unit

_classifier = ModelStudioFailureClassifier()


class _FakeJobRepository:
    def __init__(self, job: Any = None) -> None:
        self.job = job

    async def create(self, **kwargs: object) -> SimpleNamespace:
        self.job = SimpleNamespace(**kwargs)
        return self.job

    async def get(self, job_id: object) -> Any:
        del job_id
        return self.job

    async def update_status(self, job_id: object, status: JobStatus, **kwargs: object) -> Any:
        if self.job is None:
            self.job = SimpleNamespace(id=job_id)
        self.job.status = status
        for key, value in kwargs.items():
            setattr(self.job, key, value)
        return self.job


def _billing() -> SimpleNamespace:
    return SimpleNamespace(
        check_and_reserve=AsyncMock(
            return_value=SimpleNamespace(
                txn=SimpleNamespace(id=uuid4(), balance_after=75), event=None
            )
        )
    )


def _service(client: Any = None, **kwargs: Any) -> WanJobService:
    return WanJobService(
        client or MagicMock(),
        MagicMock(),
        MagicMock(),
        billing_policy=ProviderBillingPolicyRegistry.from_settings(Settings.model_construct()),
        **kwargs,
    )


async def _start(service: WanJobService, repo: _FakeJobRepository, billing: Any) -> Any:
    session = SimpleNamespace(flush=AsyncMock(), rollback=AsyncMock())
    with patch("src.api.services.model_studio.wan_job_service.JobRepository", return_value=repo):
        return await service.start_video_job(
            session,  # type: ignore[arg-type]
            user_id=uuid4(),
            prompt="a fox",
            model=ModelType.WAN_3_VIDEO,
            generation_type=GenerationType.T2V,
            payload={"model": "wan3.0-video"},
            aspect_ratio=None,
            name=None,
            billing_service=billing,
            account_id=uuid4(),
            token_cost=40,
            billed_seconds=5,
            video_resolution=VideoResolution.RES_720P,
            product_id="vex",
        )


def _running_job(**overrides: Any) -> SimpleNamespace:
    fields: dict[str, Any] = {
        "id": uuid4(),
        "user_id": uuid4(),
        "product_id": "vex",
        "status": JobStatus.RUNNING.value,
        "external_request_id": "task-1",
        "started_at": datetime.now(UTC) - timedelta(seconds=30),
        "created_at": datetime.now(UTC) - timedelta(seconds=40),
        "debit_transaction_id": None,
    }
    return SimpleNamespace(**(fields | overrides))


def _task(status: ModelStudioTaskStatus, **kwargs: Any) -> ModelStudioTask:
    return ModelStudioTask(task_id="task-1", task_status=status, **kwargs)


async def _poll(service: WanJobService, job: Any) -> Any:
    return await service._poll_video_result(AsyncMock(), job)


# ---- start_video_job ---------------------------------------------------------------


async def test_start_stores_the_task_id_and_leaves_adaptive_ratio_unset() -> None:
    client = MagicMock()
    client.create_video_task = AsyncMock(return_value=ModelStudioTaskRef(task_id="task-9"))
    repo = _FakeJobRepository()
    billing = _billing()

    result = await _start(_service(client), repo, billing)

    assert result.job.status == JobStatus.QUEUED
    assert result.job.external_request_id == "task-9"
    assert repo.job is not None
    assert repo.job.provider is Provider.ALIBABA
    assert repo.job.aspect_ratio is None
    assert repo.job.token_cost == 40
    metadata = billing.check_and_reserve.await_args.kwargs["metadata"]
    assert metadata["provider"] == "alibaba"
    assert metadata["billed_seconds"] == 5
    assert metadata["video_resolution"] == "720p"


@pytest.mark.parametrize(
    ("code", "kind", "billable"),
    [
        ("DataInspectionFailed", ProviderFailureKind.MODERATION_REJECTED, False),
        ("InvalidParameter", ProviderFailureKind.INVALID_REQUEST, False),
    ],
)
async def test_start_provider_rejection_is_raised_for_the_orchestrator_to_settle(
    code: str, kind: ProviderFailureKind, billable: bool
) -> None:
    client = MagicMock()
    client.create_video_task = AsyncMock(
        side_effect=ModelStudioAPIError(
            _classifier.classify(code=code, http_status=400, provider_request_accepted=False)
        )
    )

    with pytest.raises(ProviderSubmissionFailedError) as raised:
        await _start(_service(client), _FakeJobRepository(), _billing())

    assert raised.value.failure.kind is kind
    assert raised.value.failure.billable is billable
    assert isinstance(raised.value, ProviderModerationRejectedError) is (
        kind is ProviderFailureKind.MODERATION_REJECTED
    )


async def test_start_ambiguous_timeout_is_not_billable_and_is_never_retried() -> None:
    client = MagicMock()
    client.create_video_task = AsyncMock(
        side_effect=ModelStudioAPIError(_classifier.timeout(provider_request_accepted=None))
    )

    with pytest.raises(ProviderSubmissionFailedError) as raised:
        await _start(_service(client), _FakeJobRepository(), _billing())

    assert raised.value.failure.billable is False
    client.create_video_task.assert_awaited_once()


async def test_start_unexpected_fault_after_reserve_becomes_unknown_failure() -> None:
    client = MagicMock()
    client.create_video_task = AsyncMock(side_effect=RuntimeError("boom"))

    with pytest.raises(ProviderSubmissionFailedError) as raised:
        await _start(_service(client), _FakeJobRepository(), _billing())

    assert raised.value.failure.kind is ProviderFailureKind.UNKNOWN


async def test_start_billing_failure_rolls_back_and_is_not_a_provider_failure() -> None:
    client = MagicMock()
    billing = SimpleNamespace(check_and_reserve=AsyncMock(side_effect=ValueError("insufficient")))
    session = SimpleNamespace(flush=AsyncMock(), rollback=AsyncMock())
    service = _service(client)

    with (
        patch(
            "src.api.services.model_studio.wan_job_service.JobRepository",
            return_value=_FakeJobRepository(),
        ),
        pytest.raises(ValueError, match="insufficient"),
    ):
        await service.start_video_job(
            session,  # type: ignore[arg-type]
            user_id=uuid4(),
            prompt="p",
            model=ModelType.WAN_3_VIDEO,
            generation_type=GenerationType.T2V,
            payload={},
            aspect_ratio=AspectRatio.RATIO_16_9,
            name="n",
            billing_service=billing,  # type: ignore[arg-type]
            account_id=uuid4(),
            token_cost=1,
            billed_seconds=5,
            video_resolution=VideoResolution.RES_720P,
            product_id="vex",
        )

    session.rollback.assert_awaited_once()
    client.create_video_task.assert_not_called()


# ---- polling -----------------------------------------------------------------------


@pytest.mark.parametrize("status", [ModelStudioTaskStatus.PENDING, ModelStudioTaskStatus.RUNNING])
async def test_poll_in_flight_task_is_still_running(status: ModelStudioTaskStatus) -> None:
    client = MagicMock()
    client.get_task = AsyncMock(return_value=_task(status))

    outcome = await _poll(_service(client), _running_job())

    assert outcome.status is VideoPollStatus.STILL_RUNNING


async def test_poll_succeeded_returns_the_result_url_and_task_id() -> None:
    client = MagicMock()
    client.get_task = AsyncMock(
        return_value=_task(ModelStudioTaskStatus.SUCCEEDED, video_url="https://oss/x.mp4")
    )

    outcome = await _poll(_service(client), _running_job())

    assert outcome.status is VideoPollStatus.COMPLETED
    assert outcome.result is not None
    assert (outcome.result.url, outcome.result.provider_request_id) == (
        "https://oss/x.mp4",
        "task-1",
    )


async def test_poll_succeeded_without_url_is_malformed_but_accepted() -> None:
    client = MagicMock()
    client.get_task = AsyncMock(return_value=_task(ModelStudioTaskStatus.SUCCEEDED))

    outcome = await _poll(_service(client), _running_job())

    assert outcome.status is VideoPollStatus.FAILED
    assert outcome.failure is not None
    assert outcome.failure.kind is ProviderFailureKind.MALFORMED_RESPONSE
    assert outcome.failure.provider_request_accepted is True


async def test_poll_failed_moderation_is_refunded_under_the_default_wan_policy() -> None:
    client = MagicMock()
    client.get_task = AsyncMock(
        return_value=_task(ModelStudioTaskStatus.FAILED, code="DataInspectionFailed")
    )

    outcome = await _poll(_service(client), _running_job())

    assert outcome.failure is not None
    assert outcome.failure.kind is ProviderFailureKind.MODERATION_REJECTED
    assert outcome.failure.billable is False


async def test_poll_failed_moderation_is_charged_when_the_policy_says_so() -> None:
    client = MagicMock()
    client.get_task = AsyncMock(
        return_value=_task(ModelStudioTaskStatus.FAILED, code="DataInspectionFailed")
    )
    service = WanJobService(
        client,
        MagicMock(),
        MagicMock(),
        billing_policy=ProviderBillingPolicyRegistry.from_settings(
            Settings.model_construct(wan_moderation_billing_policy="charge")
        ),
    )

    outcome = await _poll(service, _running_job())

    assert outcome.failure is not None
    assert outcome.failure.billable is True


@pytest.mark.parametrize("status", [ModelStudioTaskStatus.CANCELED, ModelStudioTaskStatus.UNKNOWN])
async def test_poll_canceled_or_expired_task_is_an_accepted_unknown_failure(
    status: ModelStudioTaskStatus,
) -> None:
    client = MagicMock()
    client.get_task = AsyncMock(return_value=_task(status))

    outcome = await _poll(_service(client), _running_job())

    assert outcome.failure is not None
    assert outcome.failure.kind is ProviderFailureKind.UNKNOWN
    assert outcome.failure.provider_request_accepted is True


async def test_poll_transient_errors_propagate_for_the_worker_to_skip_the_tick() -> None:
    client = MagicMock()
    client.get_task = AsyncMock(
        side_effect=ModelStudioTransientError(_classifier.classify(http_status=429))
    )

    with pytest.raises(ModelStudioTransientError):
        await _poll(_service(client), _running_job())


async def test_poll_non_transient_api_error_fails_the_job() -> None:
    client = MagicMock()
    client.get_task = AsyncMock(
        side_effect=ModelStudioAPIError(_classifier.malformed(provider_request_accepted=True))
    )

    outcome = await _poll(_service(client), _running_job())

    assert outcome.status is VideoPollStatus.FAILED
    assert outcome.failure is not None
    assert outcome.failure.kind is ProviderFailureKind.MALFORMED_RESPONSE


def _auth_failing_service(**kwargs: Any) -> WanJobService:
    """A real ``ModelStudioClient`` whose every task query is answered with a 401."""
    import httpx
    from pydantic import SecretStr

    from src.api.services.model_studio import ModelStudioClient

    client = ModelStudioClient(
        base_url="https://ws.maas.aliyuncs.com",
        api_key=SecretStr("sk-test"),
        timeout_seconds=5,
        transport=httpx.MockTransport(
            lambda _: httpx.Response(401, json={"code": "InvalidApiKey", "message": "bad key"})
        ),
    )
    return _service(client, **kwargs)


async def test_worker_poll_of_a_401_is_transient_no_transition_no_refund() -> None:
    """R4: a rotated/revoked key must not fail and refund every in-flight job."""
    from src.api.services.video_jobs.worker import AsyncVideoJobWorker

    service = _auth_failing_service()
    await service._client.connect()
    service.settle_video_poll_outcome = AsyncMock()  # type: ignore[method-assign]
    job = _running_job()
    worker = AsyncVideoJobWorker(
        name="alibaba_video_poller",
        job_service=service,
        interval_seconds=15,
        max_concurrent_polls=1,
        db_manager=MagicMock(),
        redis_client_factory=MagicMock(),
    )

    with (
        patch(
            "src.api.services.model_studio.wan_job_service.JobRepository",
            return_value=_FakeJobRepository(job),
        ),
        capture_logs() as logs,
    ):
        await worker._poll_one(job, MagicMock())  # type: ignore[arg-type]

    service.settle_video_poll_outcome.assert_not_awaited()
    assert job.status == JobStatus.RUNNING.value
    assert [e["log_level"] for e in logs if e["event"] == "alibaba.poll_auth_failed"] == ["error"]
    assert [e for e in logs if e["event"] == "video_job.poll_transient"]
    await service._client.close()


async def test_overdue_job_with_a_persisting_401_still_times_out_before_polling() -> None:
    """R4 relies on the overdue check running *before* the poll: retries are bounded."""
    service = _auth_failing_service(max_poll_time=60)
    await service._client.connect()
    job = _running_job(started_at=datetime.now(UTC) - timedelta(seconds=120))

    with (
        patch(
            "src.api.services.model_studio.wan_job_service.JobRepository",
            return_value=_FakeJobRepository(job),
        ),
        capture_logs() as logs,
    ):
        outcome = await service.poll_video_job_for_worker(MagicMock(), job.id)

    assert outcome.status is VideoPollStatus.FAILED
    assert outcome.failure is not None
    assert outcome.failure.kind is ProviderFailureKind.TIMEOUT
    assert not [e for e in logs if e["event"] == "alibaba.poll_auth_failed"]  # never queried
    await service._client.close()


async def test_poll_without_a_task_id_is_a_programming_error() -> None:
    with pytest.raises(WanJobError):
        await _poll(_service(), _running_job(external_request_id=None))


async def test_worker_poll_of_an_overdue_job_times_out_without_calling_upstream() -> None:
    client = MagicMock()
    client.get_task = AsyncMock()
    service = _service(client, max_poll_time=60)
    job = _running_job(started_at=datetime.now(UTC) - timedelta(seconds=120))

    with patch(
        "src.api.services.model_studio.wan_job_service.JobRepository",
        return_value=_FakeJobRepository(job),
    ):
        outcome = await service.poll_video_job_for_worker(MagicMock(), job.id)

    assert outcome.failure is not None
    assert outcome.failure.kind is ProviderFailureKind.TIMEOUT
    client.get_task.assert_not_called()


async def test_worker_poll_of_a_terminal_job_is_a_no_op() -> None:
    job = _running_job(status=JobStatus.COMPLETED.value)
    client = MagicMock()
    client.get_task = AsyncMock()

    with patch(
        "src.api.services.model_studio.wan_job_service.JobRepository",
        return_value=_FakeJobRepository(job),
    ):
        outcome = await _service(client).poll_video_job_for_worker(MagicMock(), job.id)

    assert outcome.status is VideoPollStatus.STILL_RUNNING
    client.get_task.assert_not_called()


# ---- billing-basis observation -----------------------------------------------------


async def test_billing_basis_mismatch_is_logged_and_nothing_else() -> None:
    client = MagicMock()
    client.get_task = AsyncMock(
        return_value=_task(
            ModelStudioTaskStatus.SUCCEEDED,
            video_url="https://oss/x.mp4",
            usage=ModelStudioVideoUsage(duration=12.0),
        )
    )
    session = AsyncMock()
    session.get = AsyncMock(return_value=SimpleNamespace(metadata_={"billed_seconds": 15}))
    job = _running_job(debit_transaction_id=uuid4())

    with capture_logs() as logs:
        outcome = await _service(client)._poll_video_result(session, job)  # type: ignore[arg-type]

    assert outcome.status is VideoPollStatus.COMPLETED
    (event,) = [e for e in logs if e["event"] == "alibaba.billing_basis_mismatch"]
    assert (event["reserved_seconds"], event["upstream_seconds"]) == (15, 12)
    assert event["upstream_seconds_exact"] == 12.0
    assert event["job_id"] == str(job.id)


async def test_matching_billing_basis_logs_nothing() -> None:
    client = MagicMock()
    client.get_task = AsyncMock(
        return_value=_task(
            ModelStudioTaskStatus.SUCCEEDED,
            video_url="https://oss/x.mp4",
            usage=ModelStudioVideoUsage(duration=15.0),
        )
    )
    session = AsyncMock()
    session.get = AsyncMock(return_value=SimpleNamespace(metadata_={"billed_seconds": 15}))

    job: Any = _running_job(debit_transaction_id=uuid4())
    with capture_logs() as logs:
        await _service(client)._poll_video_result(session, job)

    assert not [e for e in logs if e["event"] == "alibaba.billing_basis_mismatch"]


async def test_fractional_v2v_input_does_not_log_a_billing_basis_mismatch() -> None:
    """P3-1: 7.3 s input + 5 s = 12.3 s upstream vs 5 + ceil(7.3) = 13 reserved is not a mismatch."""
    from src.api.services.generation.video_validation import input_video_seconds
    from src.core.model_registry import get_model_meta

    video = get_model_meta(ModelType.WAN_3_VIDEO).video
    assert video is not None
    source = SimpleNamespace(media_kind=MediaKind.VIDEO, duration_ms=7300, position=0)
    reserved = 5 + input_video_seconds([source], video)  # type: ignore[list-item]
    assert reserved == 13
    client = MagicMock()
    client.get_task = AsyncMock(
        return_value=_task(
            ModelStudioTaskStatus.SUCCEEDED,
            video_url="https://oss/x.mp4",
            usage=ModelStudioVideoUsage(
                duration=12.3, input_video_duration=7.3, output_video_duration=5.0
            ),
        )
    )
    session = AsyncMock()
    session.get = AsyncMock(return_value=SimpleNamespace(metadata_={"billed_seconds": reserved}))

    job: Any = _running_job(debit_transaction_id=uuid4())
    with capture_logs() as logs:
        outcome = await _service(client)._poll_video_result(session, job)  # type: ignore[arg-type]

    assert outcome.status is VideoPollStatus.COMPLETED
    assert not [e for e in logs if e["event"].startswith("alibaba.billing_basis")]


# ---- provider adapter --------------------------------------------------------------


def _resolved(kind: MediaKind, key: str, position: int = 0) -> ResolvedSourceMedia:
    asset_id = uuid4()
    return ResolvedSourceMedia(
        position=position,
        ref=AssetRef(source=LibraryAssetSource.UPLOAD, asset_id=asset_id),
        asset_ref=format_asset_ref(LibraryAssetSource.UPLOAD, asset_id),
        media_kind=kind,
        content_type="video/mp4" if kind is MediaKind.VIDEO else "image/png",
        storage_key=key,
        size_bytes=1,
        duration_ms=7_200 if kind is MediaKind.VIDEO else None,
        job_id=None,
    )


async def _submit(request: UnifiedGenerationRequest, sources: list[ResolvedSourceMedia]) -> Any:
    wan = MagicMock()
    wan.start_video_job = AsyncMock(return_value="submitted")
    r2 = MagicMock()
    r2.get_presigned_url = AsyncMock(
        side_effect=lambda key, expires_in: SimpleNamespace(
            presigned_url=f"https://r2/{key}?ttl={expires_in}"
        )
    )
    provider = WanGenerationProvider(wan, r2, input_url_ttl_seconds=900)
    result = await provider.submit(
        request,
        user_id=uuid4(),
        session=MagicMock(),
        billing_service=MagicMock(),
        account_id=uuid4(),
        token_cost=10,
        product_id="vex",
        source_media=sources,
    )
    assert result == "submitted"
    return wan.start_video_job.await_args.kwargs


async def test_provider_presigns_inputs_with_the_configured_ttl_and_builds_the_payload() -> None:
    request = UnifiedGenerationRequest(
        prompt="Image 1 meets Image 2",
        generation_type=GenerationType.R2V,
        model=ModelType.WAN_3_VIDEO,
        source_media=[SourceMediaReference(asset_ref=f"upload:{uuid4()}") for _ in range(2)],
        duration=6,
        audio=False,
        seed=3,
    )
    sources = [_resolved(MediaKind.IMAGE, "k/a.png", 0), _resolved(MediaKind.IMAGE, "k/b.png", 1)]

    kwargs = await _submit(request, sources)

    assert kwargs["payload"]["input"]["media"] == [
        {"type": "reference_image", "url": "https://r2/k/a.png?ttl=900"},
        {"type": "reference_image", "url": "https://r2/k/b.png?ttl=900"},
    ]
    assert kwargs["payload"]["parameters"]["audio"] is False
    assert kwargs["payload"]["parameters"]["seed"] == 3
    assert kwargs["billed_seconds"] == 6
    assert kwargs["aspect_ratio"] is None


@pytest.mark.parametrize(
    ("requested", "expected"),
    [(None, AspectRatio.RATIO_16_9), (AspectRatio.RATIO_9_16, AspectRatio.RATIO_9_16)],
)
async def test_provider_persists_the_ratio_it_actually_sent_for_t2v(
    requested: AspectRatio | None, expected: AspectRatio
) -> None:
    request = UnifiedGenerationRequest(
        prompt="a fox", generation_type=GenerationType.T2V, model=ModelType.WAN_3_VIDEO,
        aspect_ratio=requested,
    )  # fmt: skip

    kwargs = await _submit(request, [])

    assert kwargs["payload"]["parameters"]["ratio"] == expected.value
    assert kwargs["aspect_ratio"] is expected


async def test_provider_leaves_adaptive_ratio_unset_when_media_decides() -> None:
    request = UnifiedGenerationRequest(
        prompt="animate",
        generation_type=GenerationType.I2V,
        model=ModelType.WAN_3_VIDEO,
        source_media=[SourceMediaReference(asset_ref=f"upload:{uuid4()}")],
    )

    kwargs = await _submit(request, [_resolved(MediaKind.IMAGE, "k/a.png")])

    assert kwargs["payload"]["parameters"]["ratio"] == "adaptive"
    assert kwargs["aspect_ratio"] is None


async def test_provider_bills_input_video_seconds_for_v2v() -> None:
    request = UnifiedGenerationRequest(
        prompt="extend",
        generation_type=GenerationType.V2V,
        model=ModelType.WAN_3_VIDEO,
        source_media=[SourceMediaReference(asset_ref=f"upload:{uuid4()}")],
        duration=10,
        resolution=VideoResolution.RES_1080P,
    )

    kwargs = await _submit(request, [_resolved(MediaKind.VIDEO, "k/v.mp4")])

    assert kwargs["billed_seconds"] == 10 + 8
    assert kwargs["video_resolution"] is VideoResolution.RES_1080P
    assert kwargs["payload"]["input"]["media"][0]["type"] == "reference_video"


async def test_provider_refresh_polls_only_in_flight_video_jobs() -> None:
    wan = MagicMock()
    wan.poll_video_job = AsyncMock(return_value="polled")
    provider = WanGenerationProvider(wan, MagicMock(), input_url_ttl_seconds=900)
    running = SimpleNamespace(id=uuid4(), generation_type="t2v", status=JobStatus.RUNNING.value)
    done = SimpleNamespace(id=uuid4(), generation_type="t2v", status=JobStatus.COMPLETED.value)

    assert await provider.refresh_job(MagicMock(), running) == "polled"  # type: ignore[arg-type,comparison-overlap]
    assert await provider.refresh_job(MagicMock(), done) is None  # type: ignore[arg-type]
    wan.poll_video_job.assert_awaited_once()


# ---- read-through polling and lifecycle ----------------------------------------------


def _patched_repo(job: Any) -> Any:
    return patch(
        "src.api.services.model_studio.wan_job_service.JobRepository",
        return_value=_FakeJobRepository(job),
    )


async def test_read_through_poll_returns_terminal_jobs_untouched() -> None:
    job = _running_job(status=JobStatus.COMPLETED.value)
    client = MagicMock()
    client.get_task = AsyncMock()

    with _patched_repo(job):
        assert await _service(client).poll_video_job(MagicMock(), job.id) is job

    client.get_task.assert_not_called()


async def test_read_through_poll_of_a_missing_job_raises() -> None:
    from src.api.services.model_studio.wan_job_service import WanJobNotFoundError

    with _patched_repo(None), pytest.raises(WanJobNotFoundError):
        await _service().poll_video_job(MagicMock(), uuid4())


async def test_read_through_poll_reports_a_transient_error_as_unchanged() -> None:
    job = _running_job()
    client = MagicMock()
    client.get_task = AsyncMock(
        side_effect=ModelStudioTransientError(_classifier.classify(http_status=503))
    )

    with _patched_repo(job):
        assert await _service(client).poll_video_job(MagicMock(), job.id) is job


async def test_read_through_poll_settles_the_observed_outcome() -> None:
    job = _running_job()
    client = MagicMock()
    client.get_task = AsyncMock(return_value=_task(ModelStudioTaskStatus.RUNNING))
    service = _service(client)
    settled = SimpleNamespace(status=JobStatus.RUNNING.value)
    service.settle_video_poll_outcome = AsyncMock(return_value=settled)  # type: ignore[method-assign]

    with _patched_repo(job):
        result = await service.poll_video_job(MagicMock(), job.id)

    assert result is settled
    outcome = service.settle_video_poll_outcome.await_args_list[-1].kwargs["outcome"]
    assert outcome.status is VideoPollStatus.STILL_RUNNING


async def test_read_through_poll_times_out_an_overdue_job_without_calling_upstream() -> None:
    job = _running_job(started_at=datetime.now(UTC) - timedelta(hours=3))
    client = MagicMock()
    client.get_task = AsyncMock()
    service = _service(client, max_poll_time=60)
    service.settle_video_poll_outcome = AsyncMock(return_value="settled")  # type: ignore[method-assign]

    with _patched_repo(job):
        assert await service.poll_video_job(MagicMock(), job.id) == "settled"

    client.get_task.assert_not_called()
    outcome = service.settle_video_poll_outcome.await_args_list[-1].kwargs["outcome"]
    assert outcome.failure.kind is ProviderFailureKind.TIMEOUT


async def test_settle_delegates_to_the_shared_settlement_with_its_own_collaborators() -> None:
    service = _service()
    outcome: Any = SimpleNamespace(status=VideoPollStatus.STILL_RUNNING)
    with patch(
        "src.api.services.model_studio.wan_job_service.settle_video_poll_outcome",
        new=AsyncMock(return_value="settled"),
    ) as shared:
        result = await service.settle_video_poll_outcome(
            MagicMock(),
            job_id=uuid4(),
            outcome=outcome,
            product_id="vex",  # type: ignore[arg-type]
        )

    assert result == "settled"
    kwargs = shared.await_args_list[-1].kwargs
    assert kwargs["materializer"] is service._materializer
    assert kwargs["settle_failure"] == service._settle_failed_outcome
    assert kwargs["outcome"] is outcome


async def test_failed_outcome_adapter_routes_back_through_settle() -> None:
    service = _service()
    service.settle_video_poll_outcome = AsyncMock(return_value="x")  # type: ignore[method-assign]
    job_id = uuid4()
    outcome: Any = SimpleNamespace()

    assert await service._settle_failed_outcome(MagicMock(), job_id, outcome, "vex") == "x"  # type: ignore[arg-type]
    assert service.settle_video_poll_outcome.await_args_list[-1].kwargs["job_id"] == job_id


async def test_connect_and_close_manage_the_download_client_only() -> None:
    download = MagicMock(connect=AsyncMock(), close=AsyncMock())
    client = MagicMock()
    service = WanJobService(client, MagicMock(), download)

    await service.connect()
    await service.close()

    download.connect.assert_awaited_once()
    download.close.assert_awaited_once()


async def test_long_prompts_get_a_truncated_default_name() -> None:
    client = MagicMock()
    client.create_video_task = AsyncMock(return_value=ModelStudioTaskRef(task_id="t"))
    repo = _FakeJobRepository()
    session = SimpleNamespace(flush=AsyncMock(), rollback=AsyncMock())
    service = _service(client)

    with patch("src.api.services.model_studio.wan_job_service.JobRepository", return_value=repo):
        await service.start_video_job(
            session,  # type: ignore[arg-type]
            user_id=uuid4(),
            prompt="x" * 80,
            model=ModelType.WAN_3_VIDEO,
            generation_type=GenerationType.T2V,
            payload={},
            aspect_ratio=None,
            name=None,
            billing_service=_billing(),  # type: ignore[arg-type]
            account_id=uuid4(),
            token_cost=1,
            billed_seconds=5,
            video_resolution=VideoResolution.RES_720P,
            product_id="vex",
        )

    assert repo.job is not None
    assert repo.job.name == "x" * 50 + "..."
