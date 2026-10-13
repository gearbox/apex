"""C18-C22: policy registry, provider discovery, completeness, settings, standalone CLI."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any
from unittest.mock import ANY, AsyncMock, MagicMock, patch

import msgspec
import pytest
from pydantic import ValidationError

from src.api.routes.providers import PROVIDER_DISPLAY_NAMES, ProvidersController
from src.api.services.generation.provider_billing_policy import ProviderBillingPolicyRegistry
from src.api.services.generation.provider_failures import ProviderFailure, ProviderFailureKind
from src.core.config import Settings
from src.core.enums import (
    GenerationType,
    ModelType,
    Product,
    Provider,
    ProvisioningMode,
)
from src.core.model_registry import MODEL_METADATA, VideoMeta
from src.core.product_registry import PRODUCT_REGISTRY
from src.workers.wan_video import WanVideoPollerCLI
from tests.unit.helpers import hermetic_settings as _hermetic_settings

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _jwt_secret(monkeypatch: pytest.MonkeyPatch) -> None:
    """``hermetic_settings`` ignores ``.env``, so supply the one required secret."""
    monkeypatch.setenv("JWT_SECRET_KEY", "test_secret_key_for_testing_only_256bits_long")


def hermetic_settings(**overrides: Any) -> Settings:
    return _hermetic_settings(**overrides)


# ---- C18 ---------------------------------------------------------------------------


def _failure(provider: Provider, kind: ProviderFailureKind) -> ProviderFailure:
    return ProviderFailure(
        kind=kind,
        provider=provider,
        sanitized_message=ProviderFailure.safe_message_for_kind(kind),
        provider_request_accepted=True,
    )


def test_c18_defaults_charge_grok_and_refund_wan_moderation() -> None:
    registry = ProviderBillingPolicyRegistry.from_settings(Settings.model_construct())

    grok = registry.policies[Provider.GROK]
    wan = registry.policies[Provider.ALIBABA]
    assert (grok.charge_on_moderation_rejection, grok.charge_on_undelivered_output) == (True, True)
    assert (wan.charge_on_moderation_rejection, wan.charge_on_undelivered_output) == (False, True)


def test_c18_each_provider_reads_only_its_own_settings() -> None:
    registry = ProviderBillingPolicyRegistry.from_settings(
        Settings.model_construct(
            grok_moderation_billing_policy="refund",
            wan_moderation_billing_policy="charge",
            wan_undelivered_output_billing_policy="refund",
        )
    )

    moderation_grok = _failure(Provider.GROK, ProviderFailureKind.MODERATION_REJECTED)
    moderation_wan = _failure(Provider.ALIBABA, ProviderFailureKind.MODERATION_REJECTED)
    undelivered_wan = _failure(Provider.ALIBABA, ProviderFailureKind.OUTPUT_NOT_DELIVERED)
    assert registry.apply(moderation_grok).billable is False
    assert registry.apply(moderation_wan).billable is True
    assert registry.apply(undelivered_wan).billable is False


def test_c18_a_provider_missing_from_the_registry_still_charges_nothing() -> None:
    registry = ProviderBillingPolicyRegistry(policies={})

    for provider in (Provider.GROK, Provider.ALIBABA):
        assert (
            registry.apply(_failure(provider, ProviderFailureKind.MODERATION_REJECTED)).billable
            is False
        )


# ---- C19 ---------------------------------------------------------------------------


async def _list_providers(product_slug: str) -> Any:
    product_config = PRODUCT_REGISTRY[Product(product_slug)]
    record = SimpleNamespace(
        model_key="wan3.0-video", name="Wan 3.0", description="Wan", is_enabled=True
    )
    generation_service = MagicMock()
    generation_service.configured_providers = frozenset({Provider.ALIBABA})
    generation_service.get_aisha_capabilities = MagicMock(return_value=None)

    async def list_enabled_for_product(config: Any) -> list[Any]:
        # The real repository applies the product allowlist; mirror that contract.
        return [record] if config.is_model_allowed(ModelType.WAN_3_VIDEO) else []

    with patch("src.api.routes.providers.GenerationModelRepository") as repo_cls:
        repo_cls.return_value.list_enabled_for_product = AsyncMock(
            side_effect=list_enabled_for_product
        )
        return await ProvidersController.list_providers.fn(
            MagicMock(),
            session=AsyncMock(),
            generation_service=generation_service,
            current_user_id=None,
            product_config=product_config,
            product_id=product_slug,
        )


@pytest.mark.parametrize("product", ["vex", "synthara"])
async def test_c19_discovery_lists_wan_with_five_modes_and_video_constraints(
    product: str,
) -> None:
    response = await _list_providers(product)

    provider = next(p for p in response.providers if p.provider == "alibaba")
    assert provider.name == "Alibaba Wan"
    assert provider.available is True
    assert provider.provisioning_mode == "always_on"
    (model,) = provider.models
    assert model.model_key == "wan3.0-video"
    assert list(model.generation_modes) == ["t2v", "i2v", "v2v", "flf2v", "r2v"]
    assert model.video is not None
    assert model.video.supported_parameters == ["audio", "prompt_extend"]
    assert (model.video.min_duration, model.video.max_duration) == (2, 30)
    assert model.video.resolutions == ["480p", "720p", "1080p"]
    assert model.video.max_combined_duration == 30
    assert msgspec.to_builtins(model.video.input_video) == {
        "formats": ["mov", "mp4"],
        "min_duration_ms": 1000,
        "max_duration_ms": 15000,
        "max_bytes": 100000000,
    }
    assert model.video.input_anchored_modes == ["flf2v", "i2v", "v2v"]
    r2v = model.generation_modes["r2v"].source_media
    assert r2v is not None
    assert (r2v.min, r2v.max) == (1, 8)


async def test_c19_a_model_the_admin_has_not_enabled_is_not_listed() -> None:
    generation_service = MagicMock()
    generation_service.configured_providers = frozenset()
    with patch("src.api.routes.providers.GenerationModelRepository") as repo_cls:
        repo_cls.return_value.list_enabled_for_product = AsyncMock(return_value=[])
        response = await ProvidersController.list_providers.fn(
            MagicMock(),
            session=AsyncMock(),
            generation_service=generation_service,
            current_user_id=None,
            product_config=PRODUCT_REGISTRY[Product.VEX],
            product_id="vex",
        )

    provider = next(p for p in response.providers if p.provider == "alibaba")
    assert provider.models == []
    assert provider.available is False


def test_c19_both_products_allow_wan() -> None:
    assert PRODUCT_REGISTRY[Product.VEX].is_model_allowed(ModelType.WAN_3_VIDEO)
    assert PRODUCT_REGISTRY[Product.SYNTHARA].is_model_allowed(ModelType.WAN_3_VIDEO)


# ---- C20 ---------------------------------------------------------------------------


def test_c20_every_provider_has_a_mode_and_a_display_name() -> None:
    for provider in Provider:
        assert isinstance(provider.provisioning_mode, ProvisioningMode)
        assert PROVIDER_DISPLAY_NAMES[provider]
    assert Provider.ALIBABA.provisioning_mode is ProvisioningMode.ALWAYS_ON


def test_c20_every_model_has_a_registry_entry_for_its_provider() -> None:
    for model in ModelType:
        assert model in MODEL_METADATA
    assert ModelType.WAN_3_VIDEO.provider is Provider.ALIBABA


def test_c20_video_meta_has_no_defaults_so_every_model_decides_each_field() -> None:
    import dataclasses

    for field in dataclasses.fields(VideoMeta):
        assert field.default is dataclasses.MISSING
        assert field.default_factory is dataclasses.MISSING

    for model, meta in MODEL_METADATA.items():
        if meta.video is not None:
            assert meta.video.min_duration >= 1, model
            assert meta.video.min_duration <= meta.video.max_duration, model


def test_c20_r2v_is_a_video_output_taking_images() -> None:
    from src.core.enums import MediaKind

    assert GenerationType.R2V.output_kind is MediaKind.VIDEO
    assert GenerationType.R2V.input_kinds == {MediaKind.IMAGE}


# ---- C21 ---------------------------------------------------------------------------


def test_c21_wan_lease_alone_can_violate_the_media_ingest_deadline_budget() -> None:
    # deadline * 4 <= lease * 3: 90 * 4 = 360 > 100 * 3 = 300 only for the Wan lease.
    with pytest.raises(ValidationError, match="Wan finalization lease"):
        hermetic_settings(
            media_ingest_video_deadline_seconds=90,
            grok_video_finalization_lease_seconds=120,
            wan_video_finalization_lease_seconds=100,
        )


def test_c21_grok_lease_alone_can_violate_the_budget() -> None:
    with pytest.raises(ValidationError, match="Grok finalization lease"):
        hermetic_settings(
            media_ingest_video_deadline_seconds=90,
            grok_video_finalization_lease_seconds=100,
            wan_video_finalization_lease_seconds=180,
        )


@pytest.mark.parametrize(
    "url",
    [
        "http://ws.ap-southeast-1.maas.aliyuncs.com",
        "https://ws.ap-southeast-1.maas.aliyuncs.com/",
        "https://ws.ap-southeast-1.maas.aliyuncs.com/api/v1",
        "https://ws.ap-southeast-1.maas.aliyuncs.com?x=1",
        "ws.ap-southeast-1.maas.aliyuncs.com",
    ],
)
def test_c21_a_base_url_with_a_path_or_without_https_fails_startup(url: str) -> None:
    with pytest.raises(ValidationError, match="model_studio_base_url"):
        hermetic_settings(model_studio_base_url=url)


def test_c21_configured_needs_both_url_and_key_and_defaults_are_unconfigured() -> None:
    assert hermetic_settings().model_studio_configured is False
    assert (
        hermetic_settings(
            model_studio_base_url="https://ws.maas.aliyuncs.com"
        ).model_studio_configured
        is False
    )
    assert (
        hermetic_settings(
            model_studio_base_url="https://ws.maas.aliyuncs.com", model_studio_api_key="sk-x"
        ).model_studio_configured
        is True
    )


def test_c21_the_api_key_is_a_secret() -> None:
    settings = hermetic_settings(model_studio_api_key="sk-leaky")

    assert "sk-leaky" not in repr(settings)
    assert "sk-leaky" not in settings.model_dump_json()


# ---- C22 ---------------------------------------------------------------------------


@pytest.fixture
def cli_settings() -> MagicMock:
    s = MagicMock()
    s.model_studio_configured = True
    s.model_studio_base_url = "https://ws.maas.aliyuncs.com"
    s.model_studio_api_key = MagicMock()
    s.model_studio_timeout_seconds = 60
    s.r2_configured = True
    s.database_url = "postgresql+asyncpg://apex:apex@localhost:5432/apex"
    s.db_pool_size = 5
    s.db_max_overflow = 10
    s.db_echo = False
    s.r2_account_id = "a"
    s.r2_access_key_id = "b"
    s.r2_secret_access_key = "c"
    s.r2_bucket_name = "d"
    s.retention_days = 7
    s.wan_video_poll_interval = 15
    s.wan_video_max_concurrent_polls = 8
    s.redis_url = None
    s.media_ingest_video_concurrency = 2
    s.media_ingest_image_concurrency = 4
    s.media_ingest_provider_video_max_bytes = 200 * 1024 * 1024
    return s


async def test_c22_exits_1_when_model_studio_is_not_configured(cli_settings: MagicMock) -> None:
    cli_settings.model_studio_configured = False

    with pytest.raises(SystemExit) as raised:
        await WanVideoPollerCLI(cli_settings).run()

    assert raised.value.code == 1


async def test_c22_exits_1_when_r2_is_not_configured(cli_settings: MagicMock) -> None:
    cli_settings.r2_configured = False

    with pytest.raises(SystemExit) as raised:
        await WanVideoPollerCLI(cli_settings).run()

    assert raised.value.code == 1


async def test_c22_starts_the_alibaba_poller_and_cleans_up_on_shutdown(
    cli_settings: MagicMock,
) -> None:
    db_manager = AsyncMock()
    r2 = AsyncMock()
    worker = AsyncMock()
    built = SimpleNamespace(service=MagicMock(), close=AsyncMock())

    runner = WanVideoPollerCLI(cli_settings)
    with (
        patch("src.api.services.video_jobs.cli.init_db", return_value=db_manager),
        patch("src.api.services.video_jobs.cli.R2StorageService", return_value=r2),
        patch("src.api.services.video_jobs.cli.R2StorageSettings"),
        patch("src.api.services.video_jobs.cli.BillingService"),
        patch("src.api.services.video_jobs.cli.build_media_ingest_service"),
        patch("src.api.services.video_jobs.cli.build_provider_video_ingest_service"),
        patch(
            "src.api.services.video_jobs.cli.AsyncVideoJobWorker", return_value=worker
        ) as worker_cls,
        patch.object(WanVideoPollerCLI, "_build_job_service", AsyncMock(return_value=built)),
    ):

        async def trigger_shutdown() -> None:
            await asyncio.sleep(0)
            runner.handle_signal()

        await asyncio.gather(runner.run(), trigger_shutdown())

    worker_cls.assert_called_once_with(
        name="alibaba_video_poller",
        job_service=built.service,
        interval_seconds=15,
        max_concurrent_polls=8,
        db_manager=db_manager,
        event_bus=ANY,
        redis_enabled=False,
        redis_client_factory=ANY,
    )
    worker.start.assert_awaited_once()
    worker.stop.assert_awaited_once()
    built.close.assert_awaited_once()
    r2.close.assert_awaited_once()
    db_manager.close.assert_awaited_once()


def test_the_two_standalone_workers_use_different_lease_names() -> None:
    from src.workers import grok_video, wan_video

    assert grok_video.WORKER_NAME == "grok_video_poller"
    assert wan_video.WORKER_NAME == "alibaba_video_poller"


async def test_wan_cli_builds_a_polling_service_from_settings(cli_settings: MagicMock) -> None:
    from src.api.services.video_jobs.cli import VideoWorkerDeps

    deps = VideoWorkerDeps(
        settings=cli_settings,
        storage=MagicMock(),
        billing_service=MagicMock(),
        event_bus=MagicMock(),
        ops_event_bus=MagicMock(),
        media_ingestor=MagicMock(),
        provider_video_ingestor=MagicMock(),
    )
    cli_settings.wan_video_max_poll_time = 1800
    cli_settings.wan_video_finalization_lease_seconds = 180
    cli_settings.wan_moderation_billing_policy = "refund"
    cli_settings.wan_undelivered_output_billing_policy = "charge"
    cli_settings.grok_moderation_billing_policy = "charge"
    cli_settings.grok_undelivered_output_billing_policy = "charge"

    with patch("src.workers.wan_video.ModelStudioClient") as client_cls:
        client_cls.return_value.connect = AsyncMock()
        client_cls.return_value.close = AsyncMock()
        built = await WanVideoPollerCLI._build_job_service(deps)
        await built.close()

    client_cls.return_value.connect.assert_awaited_once()
    client_cls.return_value.close.assert_awaited_once()
    assert type(built.service).__name__ == "WanJobService"
    assert built.service.provider is Provider.ALIBABA


async def test_wan_cli_refuses_to_build_without_an_api_key(cli_settings: MagicMock) -> None:
    from src.api.services.video_jobs.cli import VideoWorkerDeps

    cli_settings.model_studio_api_key = None
    deps = VideoWorkerDeps(
        settings=cli_settings,
        storage=MagicMock(),
        billing_service=MagicMock(),
        event_bus=MagicMock(),
        ops_event_bus=MagicMock(),
        media_ingestor=MagicMock(),
        provider_video_ingestor=MagicMock(),
    )

    with pytest.raises(RuntimeError, match="MODEL_STUDIO_API_KEY"):
        await WanVideoPollerCLI._build_job_service(deps)


async def test_wan_materializer_uses_the_provider_video_ingestor_and_cap(
    cli_settings: MagicMock,
) -> None:
    """R1: a Wan output must never be prepared under the user-upload byte budget."""
    from src.api.services.video_jobs.cli import VideoWorkerDeps

    upload_ingestor = MagicMock(name="upload")
    provider_ingestor = MagicMock(name="provider")
    deps = VideoWorkerDeps(
        settings=cli_settings,
        storage=MagicMock(),
        billing_service=MagicMock(),
        event_bus=MagicMock(),
        ops_event_bus=MagicMock(),
        media_ingestor=upload_ingestor,
        provider_video_ingestor=provider_ingestor,
    )
    cli_settings.wan_video_max_poll_time = 1800
    cli_settings.wan_video_finalization_lease_seconds = 180
    cli_settings.wan_moderation_billing_policy = "refund"
    cli_settings.wan_undelivered_output_billing_policy = "charge"
    cli_settings.grok_moderation_billing_policy = "charge"
    cli_settings.grok_undelivered_output_billing_policy = "charge"

    with patch("src.workers.wan_video.ModelStudioClient") as client_cls:
        client_cls.return_value.connect = AsyncMock()
        client_cls.return_value.close = AsyncMock()
        built = await WanVideoPollerCLI._build_job_service(deps)
        await built.close()

    materializer = built.service._materializer  # type: ignore[attr-defined]
    assert materializer._media_ingestor is provider_ingestor
    assert materializer._max_output_bytes == cli_settings.media_ingest_provider_video_max_bytes


async def test_grok_materializer_uses_the_provider_video_ingestor_and_cap() -> None:
    from src.api.services.video_jobs.cli import VideoWorkerDeps
    from src.workers.grok_video import GrokVideoPollerCLI

    settings = MagicMock()
    settings.media_ingest_provider_video_max_bytes = 200 * 1024 * 1024
    settings.retention_days = 7
    settings.grok_video_max_poll_time = 600
    settings.grok_video_finalization_lease_seconds = 120
    settings.wan_moderation_billing_policy = "refund"
    settings.wan_undelivered_output_billing_policy = "charge"
    settings.grok_moderation_billing_policy = "charge"
    settings.grok_undelivered_output_billing_policy = "charge"
    upload_ingestor = MagicMock(name="upload")
    provider_ingestor = MagicMock(name="provider")
    deps = VideoWorkerDeps(
        settings=settings,
        storage=MagicMock(),
        billing_service=MagicMock(),
        event_bus=MagicMock(),
        ops_event_bus=MagicMock(),
        media_ingestor=upload_ingestor,
        provider_video_ingestor=provider_ingestor,
    )

    with patch("src.workers.grok_video.GrokClient") as client_cls:
        client_cls.return_value.connect = AsyncMock()
        client_cls.return_value.close = AsyncMock()
        built = await GrokVideoPollerCLI._build_job_service(deps)
        await built.close()

    materializer = built.service._materializer  # type: ignore[attr-defined]
    assert materializer._media_ingestor is provider_ingestor
    assert materializer._max_output_bytes == 200 * 1024 * 1024
    # Images (Grok image outputs, i2v inputs) keep the user-facing ingestor.
    assert built.service._media_ingestor is upload_ingestor  # type: ignore[attr-defined]


@pytest.mark.parametrize(
    ("service_path", "expected"),
    [
        ("src.api.services.grok.job_service.GrokJobService", "grok_video_poller"),
        ("src.api.services.model_studio.wan_job_service.WanJobService", "alibaba_video_poller"),
    ],
)
def test_worker_name_comes_from_the_single_job_service_constant(
    service_path: str, expected: str
) -> None:
    """In-process and standalone workers share one lease only if they share one literal."""
    import importlib

    from src.api.services.video_jobs.worker import AsyncVideoJobWorker

    module_name, class_name = service_path.rsplit(".", 1)
    service_cls = getattr(importlib.import_module(module_name), class_name)
    service = MagicMock()
    service.VIDEO_POLLER_NAME = service_cls.VIDEO_POLLER_NAME

    worker = AsyncVideoJobWorker(
        job_service=service,
        interval_seconds=5,
        max_concurrent_polls=1,
        db_manager=MagicMock(),
        redis_client_factory=MagicMock(),
    )

    assert expected == service_cls.VIDEO_POLLER_NAME
    assert worker.name == expected
