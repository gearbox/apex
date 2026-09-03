"""Unit tests for GpuSessionService.

External clients (VastAIClient, CloudflareTunnelClient, BundleIndexService) are
mocked with AsyncMock/MagicMock. The session factory is mocked so no real DB is
required; GpuSessionRepository is patched at the service module level.
"""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import ANY, AsyncMock, MagicMock, patch
from uuid import uuid4

import msgspec
import pytest
from sqlalchemy.exc import IntegrityError

from src.api.schemas.events import EventType, GpuSessionStatusPayload
from src.api.services.billing import BillingResult, SettleUsageResult
from src.api.services.bundle_index import BundleIndexService
from src.api.services.cloudflare.client import CloudflareTunnelClient
from src.api.services.gpu_session import (
    GpuSessionService,
    InvalidSessionStateError,
    NullNodeCooldownStore,
    SessionAlreadyExistsError,
    SessionDurations,
    StopConfirmation,
    billable_minutes_for_active_seconds,
    compute_session_durations,
)
from src.api.services.vastai.client import VastAIClient
from src.api.services.vastai.exceptions import NoCapacityError, OfferTakenError, VastAIError
from src.api.services.vastai.schemas import VastAIOffer
from src.core.bundle_config import BundleMapping, HardwareRequirements, ReadinessMarker
from src.core.enums import GpuSessionStatus, ModelType
from src.db.models.gpu_session import GpuSession

_DEFAULT_COMFYUI_PORT: int = HardwareRequirements.__dataclass_fields__["comfyui_port"].default

_REPO_PATH = "src.api.services.gpu_session.service.GpuSessionRepository"
_OPERATION_REPO_PATH = "src.api.services.gpu_session.service.GpuSessionOperationRepository"
_JOB_REPO_PATH = "src.api.services.gpu_session.service.JobRepository"


# ---------------------------------------------------------------------------
# Helper builders
# ---------------------------------------------------------------------------


def _datetime_utc(*args: Any, **kwargs: Any) -> datetime:
    return datetime(*args, tzinfo=kwargs.pop("tzinfo", UTC), **kwargs)


def _make_billing_mock() -> MagicMock:
    """Return a MagicMock whose async billing methods are explicit AsyncMocks.

    Plain AsyncMock() refuses attributes whose names start with "assert" (Python
    mock treats them as spy-assertion calls). By building the mock manually we
    avoid the AttributeError for assert_sufficient_balance.
    """
    m = MagicMock()
    m.assert_sufficient_balance = AsyncMock(return_value=None)
    m.check_and_reserve = AsyncMock(return_value=BillingResult(txn=MagicMock(), event=None))
    m.refund = AsyncMock(return_value=BillingResult(txn=MagicMock(), event=None))
    m.partial_refund = AsyncMock(return_value=BillingResult(txn=MagicMock(), event=None))
    # SettleUsageResult(settled_tokens, new_balance, event) — overridden per-test as needed
    m.settle_session_usage = AsyncMock(
        return_value=SettleUsageResult(settled_tokens=0, new_balance=0, event=None)
    )
    return m


def _make_hardware() -> HardwareRequirements:
    return HardwareRequirements(
        gpu_whitelist=("RTX_4090",),
        min_disk_gb=100,
        min_network_upload_mbps=100,
        min_network_download_mbps=500,
        cuda_min_version="12.1",
        num_gpus=1,
        comfyui_port=_DEFAULT_COMFYUI_PORT,
    )


def _make_bundle_mapping(
    *,
    bundle_name: str = "wan_2.2_i2v",
    bundle_version: str | None = "260105-01",
    readiness_marker: ReadinessMarker | None = None,
    template_hash_id: str | None = None,
) -> BundleMapping:
    hardware = _make_hardware()
    if template_hash_id is not None:
        from dataclasses import replace

        hardware = replace(hardware, template_hash_id=template_hash_id)
    return BundleMapping(
        bundle_name=bundle_name,
        bundle_version=bundle_version,
        hardware=hardware,
        readiness_marker=readiness_marker,
    )


def _make_offer(*, offer_id: int = 1001, dph_total: float = 0.5) -> VastAIOffer:
    return VastAIOffer(id=offer_id, gpu_name="RTX_4090", num_gpus=1, dph_total=dph_total)


def _make_gpu_session(**kwargs: Any) -> GpuSession:
    now = datetime.now(UTC)
    session = GpuSession()
    session.id = uuid4()
    session.user_id = uuid4()
    session.product_id = "vex"
    session.status = GpuSessionStatus.active
    session.bundle_name = "wan_2.2_i2v"
    session.model_type = "aisha-image"
    session.bundle_version = "260105-01"
    session.vastai_instance_id = 12345
    session.vastai_offer_id = 67890
    session.vastai_cost_per_hour_micros = 500_000
    session.vastai_gpu_name = "RTX_4090"
    session.cf_tunnel_id = "tunnel-abc123"
    session.cf_dns_record_id = "dns-abc123"
    session.tunnel_hostname = "gpu-01234567.gpu-domain.com"
    session.callback_token_hash = hashlib.sha256(b"test-callback-token").hexdigest()
    session.stale_detected_at = None
    session.stale_notified = False
    session.paused_at = None
    session.resumed_at = None
    session.stopped_at = None
    session.started_at = now - timedelta(hours=2)
    session.created_at = now - timedelta(hours=3)
    session.error_message = None
    session.provision_attempt = 1
    session.total_paused_seconds = 0
    session.account_id = uuid4()
    for k, v in kwargs.items():
        setattr(session, k, v)
    return session


def _make_settings(
    *,
    offer_walk_depth: int = 10,
    recreation_attempts: int = 1,
    destroy_retry_attempts: int = 3,
) -> MagicMock:
    settings = MagicMock()
    settings.provisioning_offer_walk_depth = offer_walk_depth
    settings.provisioning_recreation_attempts = recreation_attempts
    settings.vastai_destroy_retry_attempts = destroy_retry_attempts
    settings.vastai_offer_search_limit = 20
    settings.apex_callback_url = "https://apex.example.com/callback"
    settings.hf_token = "test-hf-token"
    settings.civitai_api_token = "test-civitai-token"
    settings.gpu_session_tokens_per_minute = 100
    settings.ai_bundles_github_token = "ghp_test_token"
    settings.ai_bundles_repo_url = "https://github.com/gearbox/ai-bundles.git"
    settings.ai_bundles_branch = "master"
    settings.aisha_repo_url = "https://github.com/gearbox/aisha.git"
    settings.aisha_branch = "master"
    settings.aisha_comfyui_host = "0.0.0.0"  # noqa: S104
    settings.aisha_comfyui_extra_args = ""
    return settings


def _make_mock_session_factory() -> tuple[MagicMock, MagicMock]:
    """Return (factory, session) mocks that support async with patterns."""
    mock_session = MagicMock()
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=None)
    # GpuSessionOperationRepository creates the bootstrap operation during
    # start_session and flushes it within the transaction.
    mock_session.flush = AsyncMock(return_value=None)

    mock_begin = MagicMock()
    mock_begin.__aenter__ = AsyncMock(return_value=None)
    mock_begin.__aexit__ = AsyncMock(return_value=None)
    mock_session.begin = MagicMock(return_value=mock_begin)

    mock_factory = MagicMock(return_value=mock_session)
    return mock_factory, mock_session


def _make_service(
    **overrides: Any,
) -> tuple[GpuSessionService, dict[str, Any]]:
    mock_factory, mock_session = _make_mock_session_factory()
    mocks: dict[str, Any] = {
        "vastai_client": AsyncMock(spec=VastAIClient),
        "cf_client": AsyncMock(spec=CloudflareTunnelClient),
        "bundle_index": MagicMock(spec=BundleIndexService),
        "session_factory": mock_factory,
        "mock_session": mock_session,
        "settings": _make_settings(),
        "billing_service": _make_billing_mock(),
        "cooldown_store": NullNodeCooldownStore(),
        "account_id": uuid4(),
        "event_bus": None,
        "job_sweep_service": None,
    } | overrides
    service = GpuSessionService(
        vastai_client=mocks["vastai_client"],
        cf_client=mocks["cf_client"],
        bundle_index=mocks["bundle_index"],
        session_factory=mocks["session_factory"],
        settings=mocks["settings"],
        billing_service=mocks["billing_service"],
        cooldown_store=mocks["cooldown_store"],
        event_bus=mocks["event_bus"],
        job_sweep_service=mocks["job_sweep_service"],
    )
    return service, mocks


_TUNNEL_RESULT = ("tunnel-id-xyz", "tunnel-token-secret", "dns-id-xyz", "01abcdef.gpu.test")


# ---------------------------------------------------------------------------
# start_session tests
# ---------------------------------------------------------------------------


class TestStartSession:
    async def test_happy_path(self) -> None:
        service, mocks = _make_service()
        bundle = _make_bundle_mapping()
        mocks["bundle_index"].resolve_bundle.return_value = bundle
        offer = _make_offer()
        mocks["cf_client"].create_session_tunnel.return_value = _TUNNEL_RESULT
        mocks["vastai_client"].search_offers.return_value = [offer]
        mocks["vastai_client"].create_instance.return_value = 99999

        user_id = uuid4()
        expected_session = _make_gpu_session(
            user_id=user_id,
            status=GpuSessionStatus.pending,
        )

        with patch(_REPO_PATH) as MockRepo, patch(_OPERATION_REPO_PATH) as MockOperationRepo:
            mock_repo = AsyncMock()
            MockRepo.return_value = mock_repo
            mock_repo.get_non_terminal_for_model.return_value = None
            operation_repo = AsyncMock()
            MockOperationRepo.return_value = operation_repo

            async def create_session(**kwargs: Any) -> GpuSession:
                expected_session.bootstrap_operation_id = kwargs["bootstrap_operation_id"]
                return expected_session

            mock_repo.create.side_effect = create_session

            result = await service.start_session(
                user_id=user_id,
                product_id="vex",
                model_type=ModelType.AISHA_IMAGE,
                account_id=mocks["account_id"],
            )

        assert result is expected_session
        mocks["bundle_index"].resolve_bundle.assert_called_once_with("aisha-image")
        mock_repo.get_non_terminal_for_model.assert_called_once_with(user_id, "vex", "aisha-image")
        mocks["cf_client"].create_session_tunnel.assert_called_once()
        mocks["vastai_client"].search_offers.assert_called_once_with(
            bundle.hardware, limit=mocks["settings"].vastai_offer_search_limit
        )
        mocks["vastai_client"].create_instance.assert_called_once_with(
            offer.id,
            image="vastai/comfy:v0.15.1-cuda-12.9-py312",
            disk_gb=bundle.hardware.min_disk_gb,
            env=ANY,
            onstart_cmd=ANY,
        )
        mock_repo.create.assert_called_once()
        create_kwargs = mock_repo.create.await_args.kwargs
        operation_kwargs = operation_repo.create.await_args.kwargs
        env = mocks["vastai_client"].create_instance.await_args.kwargs["env"]
        assert expected_session.bootstrap_operation_id == operation_kwargs["id"]
        assert env["ACS_APEX_OPERATION_ID"] == str(operation_kwargs["id"])
        assert create_kwargs["bootstrap_operation_id"] == operation_kwargs["id"]
        mock_repo.update_bootstrap_operation_id.assert_not_awaited()

    async def test_start_session_persists_vastai_machine_id(self) -> None:
        service, mocks = _make_service()
        bundle = _make_bundle_mapping()
        mocks["bundle_index"].resolve_bundle.return_value = bundle
        offer = _make_offer(offer_id=1001)
        offer = VastAIOffer(
            id=offer.id,
            gpu_name=offer.gpu_name,
            dph_total=offer.dph_total,
            num_gpus=offer.num_gpus,
            machine_id=555111,
        )
        mocks["cf_client"].create_session_tunnel.return_value = _TUNNEL_RESULT
        mocks["vastai_client"].search_offers.return_value = [offer]
        mocks["vastai_client"].create_instance.return_value = 99999

        with patch(_REPO_PATH) as MockRepo:
            mock_repo = AsyncMock()
            MockRepo.return_value = mock_repo
            mock_repo.get_non_terminal_for_model.return_value = None
            mock_repo.create.return_value = _make_gpu_session(status=GpuSessionStatus.pending)

            await service.start_session(
                user_id=uuid4(),
                product_id="vex",
                model_type=ModelType.AISHA_IMAGE,
                account_id=mocks["account_id"],
            )

        create_kwargs = mock_repo.create.call_args.kwargs
        assert create_kwargs["vastai_machine_id"] == 555111

    async def test_start_session_passes_cooldown_store_to_provisioning(self) -> None:
        """A cooldown store that cools the cheapest offer's machine must cause the
        second-cheapest offer to be selected instead."""
        cheap_offer = VastAIOffer(id=1, gpu_name="RTX_4090", dph_total=0.3, machine_id=111)
        pricier_offer = VastAIOffer(id=2, gpu_name="RTX_4090", dph_total=0.5, machine_id=222)
        cooldown_store = AsyncMock()
        cooldown_store.cooled_machine_ids.return_value = {111}

        service, mocks = _make_service(cooldown_store=cooldown_store)
        bundle = _make_bundle_mapping()
        mocks["bundle_index"].resolve_bundle.return_value = bundle
        mocks["cf_client"].create_session_tunnel.return_value = _TUNNEL_RESULT
        mocks["vastai_client"].search_offers.return_value = [cheap_offer, pricier_offer]
        mocks["vastai_client"].create_instance.return_value = 99999

        with patch(_REPO_PATH) as MockRepo:
            mock_repo = AsyncMock()
            MockRepo.return_value = mock_repo
            mock_repo.get_non_terminal_for_model.return_value = None
            mock_repo.create.return_value = _make_gpu_session(status=GpuSessionStatus.pending)

            await service.start_session(
                user_id=uuid4(),
                product_id="vex",
                model_type=ModelType.AISHA_IMAGE,
                account_id=mocks["account_id"],
            )

        mocks["vastai_client"].create_instance.assert_called_once_with(
            pricier_offer.id,
            image="vastai/comfy:v0.15.1-cuda-12.9-py312",
            disk_gb=bundle.hardware.min_disk_gb,
            env=ANY,
            onstart_cmd=ANY,
        )

    async def test_bundle_override_takes_precedence(self) -> None:
        service, mocks = _make_service()
        bundle = _make_bundle_mapping(bundle_name="custom_bundle")
        mocks["bundle_index"].resolve_bundle_override.return_value = bundle
        offer = _make_offer()
        mocks["cf_client"].create_session_tunnel.return_value = _TUNNEL_RESULT
        mocks["vastai_client"].search_offers.return_value = [offer]
        mocks["vastai_client"].create_instance.return_value = 99999

        with patch(_REPO_PATH) as MockRepo:
            mock_repo = AsyncMock()
            MockRepo.return_value = mock_repo
            mock_repo.get_non_terminal_for_model.return_value = None
            mock_repo.create.return_value = _make_gpu_session(status=GpuSessionStatus.pending)

            await service.start_session(
                user_id=uuid4(),
                product_id="vex",
                model_type=ModelType.AISHA_IMAGE,
                bundle_override="custom_bundle:260201-01",
                account_id=mocks["account_id"],
            )

        mocks["bundle_index"].resolve_bundle_override.assert_called_once_with(
            "custom_bundle:260201-01"
        )
        mocks["bundle_index"].resolve_bundle.assert_not_called()

    async def test_unknown_model_type_raises_bundle_not_found(self) -> None:
        from src.api.services.bundle_index import BundleNotFoundError

        service, mocks = _make_service()
        mocks["bundle_index"].resolve_bundle.side_effect = BundleNotFoundError("not found")

        with pytest.raises(BundleNotFoundError):
            await service.start_session(
                user_id=uuid4(),
                product_id="vex",
                model_type=ModelType.AISHA_IMAGE,
                account_id=mocks["account_id"],
            )

    async def test_already_exists_precheck_raises_before_external_calls(self) -> None:
        service, mocks = _make_service()
        mocks["bundle_index"].resolve_bundle.return_value = _make_bundle_mapping()

        with patch(_REPO_PATH) as MockRepo:
            mock_repo = AsyncMock()
            MockRepo.return_value = mock_repo
            mock_repo.get_non_terminal_for_model.return_value = _make_gpu_session()

            with pytest.raises(SessionAlreadyExistsError):
                await service.start_session(
                    user_id=uuid4(),
                    product_id="vex",
                    model_type=ModelType.AISHA_IMAGE,
                    account_id=mocks["account_id"],
                )

        mocks["cf_client"].create_session_tunnel.assert_not_called()
        mocks["vastai_client"].search_offers.assert_not_called()
        mocks["vastai_client"].create_instance.assert_not_called()

    async def test_no_capacity_cleans_up_tunnel(self) -> None:
        service, mocks = _make_service()
        mocks["bundle_index"].resolve_bundle.return_value = _make_bundle_mapping()
        mocks["cf_client"].create_session_tunnel.return_value = _TUNNEL_RESULT
        mocks["vastai_client"].search_offers.side_effect = NoCapacityError("no capacity")

        with patch(_REPO_PATH) as MockRepo:
            mock_repo = AsyncMock()
            MockRepo.return_value = mock_repo
            mock_repo.get_non_terminal_for_model.return_value = None

            with pytest.raises(NoCapacityError):
                await service.start_session(
                    user_id=uuid4(),
                    product_id="vex",
                    model_type=ModelType.AISHA_IMAGE,
                    account_id=mocks["account_id"],
                )

        mocks["cf_client"].delete_session_tunnel.assert_called_once_with(
            _TUNNEL_RESULT[0], _TUNNEL_RESULT[2]
        )

    async def test_empty_offers_list_raises_no_capacity_and_cleans_up_tunnel(self) -> None:
        """Defensive: if search_offers returns [] instead of raising, we must not IndexError.

        The Vast.ai client should raise NoCapacityError on empty results, but the
        service guards the contract and raises explicitly rather than crashing
        when it indexes offers[0].
        """
        service, mocks = _make_service()
        mocks["bundle_index"].resolve_bundle.return_value = _make_bundle_mapping()
        mocks["cf_client"].create_session_tunnel.return_value = _TUNNEL_RESULT
        # Note: intentionally returning [] instead of raising, simulating a
        # (hypothetical) client contract drift.
        mocks["vastai_client"].search_offers.return_value = []

        with patch(_REPO_PATH) as MockRepo:
            mock_repo = AsyncMock()
            MockRepo.return_value = mock_repo
            mock_repo.get_non_terminal_for_model.return_value = None

            with pytest.raises(NoCapacityError):
                await service.start_session(
                    user_id=uuid4(),
                    product_id="vex",
                    model_type=ModelType.AISHA_IMAGE,
                    account_id=mocks["account_id"],
                )

        mocks["vastai_client"].create_instance.assert_not_called()
        mocks["cf_client"].delete_session_tunnel.assert_called_once_with(
            _TUNNEL_RESULT[0], _TUNNEL_RESULT[2]
        )
        mock_repo.create.assert_not_called()

    async def test_cf_tunnel_creation_failure_does_not_proceed(self) -> None:
        """If Cloudflare tunnel creation fails, no Vast.ai calls happen and no DB row is created.

        The CF client is expected to have already cleaned up its own partial
        resources (see CloudflareTunnelClient.create_session_tunnel rollback
        behavior), so the service just propagates the exception.
        """
        service, mocks = _make_service()
        mocks["bundle_index"].resolve_bundle.return_value = _make_bundle_mapping()
        mocks["cf_client"].create_session_tunnel.side_effect = RuntimeError(
            "cf tunnel creation failed"
        )

        with patch(_REPO_PATH) as MockRepo:
            mock_repo = AsyncMock()
            MockRepo.return_value = mock_repo
            mock_repo.get_non_terminal_for_model.return_value = None

            with pytest.raises(RuntimeError, match="cf tunnel creation failed"):
                await service.start_session(
                    user_id=uuid4(),
                    product_id="vex",
                    model_type=ModelType.AISHA_IMAGE,
                    account_id=mocks["account_id"],
                )

        # Must not have called any Vast.ai APIs
        mocks["vastai_client"].search_offers.assert_not_called()
        mocks["vastai_client"].create_instance.assert_not_called()
        # Must not have deleted a tunnel (the client is responsible for its own rollback)
        mocks["cf_client"].delete_session_tunnel.assert_not_called()
        # Must not have persisted any session row
        mock_repo.create.assert_not_called()

    async def test_port_mapping_uses_hardware_comfyui_port(self) -> None:
        """The env dict's port mapping must be derived from hardware.comfyui_port, not hardcoded."""
        service, mocks = _make_service()
        # Custom port — different from the default _DEFAULT_COMFYUI_PORT
        custom_hardware = HardwareRequirements(
            gpu_whitelist=("RTX_4090",),
            min_disk_gb=100,
            min_network_upload_mbps=100,
            min_network_download_mbps=500,
            cuda_min_version="12.1",
            num_gpus=1,
            comfyui_port=9999,
        )
        bundle = BundleMapping(
            bundle_name="custom_bundle",
            bundle_version=None,
            hardware=custom_hardware,
        )
        mocks["bundle_index"].resolve_bundle.return_value = bundle
        mocks["cf_client"].create_session_tunnel.return_value = _TUNNEL_RESULT
        mocks["vastai_client"].search_offers.return_value = [_make_offer()]
        mocks["vastai_client"].create_instance.return_value = 77777

        with patch(_REPO_PATH) as MockRepo:
            mock_repo = AsyncMock()
            MockRepo.return_value = mock_repo
            mock_repo.get_non_terminal_for_model.return_value = None
            mock_repo.create.return_value = MagicMock(spec=GpuSession)

            await service.start_session(
                user_id=uuid4(),
                product_id="vex",
                model_type=ModelType.AISHA_IMAGE,
                account_id=mocks["account_id"],
            )

        # Inspect the env dict passed to create_instance
        env_arg = mocks["vastai_client"].create_instance.call_args.kwargs["env"]
        assert "-p 9999:9999" in env_arg
        assert f"-p {_DEFAULT_COMFYUI_PORT}:{_DEFAULT_COMFYUI_PORT}" not in env_arg
        # Also verify CF tunnel was configured for the same port
        mocks["cf_client"].create_session_tunnel.assert_called_once_with(ANY, 9999)

    async def test_offer_taken_retries_with_next_offer(self) -> None:
        service, mocks = _make_service(settings=_make_settings(offer_walk_depth=3))
        mocks["bundle_index"].resolve_bundle.return_value = _make_bundle_mapping()
        mocks["cf_client"].create_session_tunnel.return_value = _TUNNEL_RESULT
        offer1 = _make_offer(offer_id=1001)
        offer2 = _make_offer(offer_id=1002, dph_total=0.6)
        mocks["vastai_client"].search_offers.return_value = [offer1, offer2]
        mocks["vastai_client"].create_instance.side_effect = [
            OfferTakenError("taken", status_code=409),
            88888,
        ]

        with patch(_REPO_PATH) as MockRepo:
            mock_repo = AsyncMock()
            MockRepo.return_value = mock_repo
            mock_repo.get_non_terminal_for_model.return_value = None
            mock_repo.create.return_value = _make_gpu_session(status=GpuSessionStatus.pending)

            result = await service.start_session(
                user_id=uuid4(),
                product_id="vex",
                model_type=ModelType.AISHA_IMAGE,
                account_id=mocks["account_id"],
            )

        assert result.status == GpuSessionStatus.pending
        assert mocks["vastai_client"].create_instance.call_count == 2
        # Second offer was used
        second_call_offer_id = mocks["vastai_client"].create_instance.call_args_list[1][0][0]
        assert second_call_offer_id == offer2.id

    async def test_retries_bounded_by_settings(self) -> None:
        service, mocks = _make_service(settings=_make_settings(offer_walk_depth=2))
        mocks["bundle_index"].resolve_bundle.return_value = _make_bundle_mapping()
        mocks["cf_client"].create_session_tunnel.return_value = _TUNNEL_RESULT
        offer1 = _make_offer(offer_id=1001)
        offer2 = _make_offer(offer_id=1002)
        mocks["vastai_client"].search_offers.return_value = [offer1, offer2]
        mocks["vastai_client"].create_instance.side_effect = OfferTakenError(
            "taken", status_code=409
        )

        with patch(_REPO_PATH) as MockRepo:
            mock_repo = AsyncMock()
            MockRepo.return_value = mock_repo
            mock_repo.get_non_terminal_for_model.return_value = None

            with pytest.raises(VastAIError):
                await service.start_session(
                    user_id=uuid4(),
                    product_id="vex",
                    model_type=ModelType.AISHA_IMAGE,
                    account_id=mocks["account_id"],
                )

        # Exhausted 2 offers (offer_walk_depth=2, 2 offers)
        assert mocks["vastai_client"].create_instance.call_count == 2
        # Tunnel was cleaned up
        mocks["cf_client"].delete_session_tunnel.assert_called_once_with(
            _TUNNEL_RESULT[0], _TUNNEL_RESULT[2]
        )

    async def test_non_offer_taken_vastai_error_cleans_up_and_no_db_write(self) -> None:
        """A non-OfferTaken VastAI error aborts the retry loop immediately.

        Must: (1) re-raise the original exception, (2) delete the CF tunnel,
        (3) never call repo.create.
        """
        service, mocks = _make_service()
        mocks["bundle_index"].resolve_bundle.return_value = _make_bundle_mapping()
        mocks["cf_client"].create_session_tunnel.return_value = _TUNNEL_RESULT
        mocks["vastai_client"].search_offers.return_value = [_make_offer()]
        # Non-OfferTaken error: generic VastAIError (could be network failure, 500, etc.)
        provisioning_error = VastAIError("unexpected upstream error")
        mocks["vastai_client"].create_instance.side_effect = provisioning_error

        with patch(_REPO_PATH) as MockRepo:
            mock_repo = AsyncMock()
            MockRepo.return_value = mock_repo
            mock_repo.get_non_terminal_for_model.return_value = None

            with pytest.raises(VastAIError) as exc_info:
                await service.start_session(
                    user_id=uuid4(),
                    product_id="vex",
                    model_type=ModelType.AISHA_IMAGE,
                    account_id=mocks["account_id"],
                )
            # The original exception is preserved (not replaced with a generic one)
            assert exc_info.value is provisioning_error

        # Only one create_instance attempt — loop aborted, no retry on non-OfferTaken
        assert mocks["vastai_client"].create_instance.call_count == 1
        # Tunnel was cleaned up
        mocks["cf_client"].delete_session_tunnel.assert_called_once_with(
            _TUNNEL_RESULT[0], _TUNNEL_RESULT[2]
        )
        # No DB session row was persisted
        mock_repo.create.assert_not_called()

    async def test_zero_offer_walk_depth_raises_config_error(self) -> None:
        """offer_walk_depth=0 is a misconfiguration, not 'all taken'.

        Settings enforces ge=1 via Pydantic, but defense-in-depth in the provisioning
        helper gives a clearer error if a misconstructed Settings slips through.
        """
        service, mocks = _make_service(settings=_make_settings(offer_walk_depth=0))
        mocks["bundle_index"].resolve_bundle.return_value = _make_bundle_mapping()
        mocks["cf_client"].create_session_tunnel.return_value = _TUNNEL_RESULT
        mocks["vastai_client"].search_offers.return_value = [_make_offer()]

        with patch(_REPO_PATH) as MockRepo:
            mock_repo = AsyncMock()
            MockRepo.return_value = mock_repo
            mock_repo.get_non_terminal_for_model.return_value = None

            with pytest.raises(VastAIError, match="offer_walk_depth must be >= 1"):
                await service.start_session(
                    user_id=uuid4(),
                    product_id="vex",
                    model_type=ModelType.AISHA_IMAGE,
                    account_id=mocks["account_id"],
                )

        # No create_instance attempts at all
        mocks["vastai_client"].create_instance.assert_not_called()
        # Tunnel still cleaned up
        mocks["cf_client"].delete_session_tunnel.assert_called_once_with(
            _TUNNEL_RESULT[0], _TUNNEL_RESULT[2]
        )
        mock_repo.create.assert_not_called()

    async def test_failure_log_uses_actual_exception_class(self) -> None:
        """error_class field in gpu_session.start.failed must reflect the actual exception."""
        service, mocks = _make_service()
        mocks["bundle_index"].resolve_bundle.return_value = _make_bundle_mapping()
        mocks["cf_client"].create_session_tunnel.return_value = _TUNNEL_RESULT
        mocks["vastai_client"].search_offers.return_value = [_make_offer()]

        class CustomVastAIError(VastAIError):
            pass

        mocks["vastai_client"].create_instance.side_effect = CustomVastAIError("custom")

        with (
            patch(_REPO_PATH) as MockRepo,
            patch("src.api.services.gpu_session.service.logger") as mock_logger,
        ):
            mock_repo = AsyncMock()
            MockRepo.return_value = mock_repo
            mock_repo.get_non_terminal_for_model.return_value = None

            with pytest.raises(CustomVastAIError):
                await service.start_session(
                    user_id=uuid4(),
                    product_id="vex",
                    model_type=ModelType.AISHA_IMAGE,
                    account_id=mocks["account_id"],
                )

        # Find the gpu_session.start.failed call and verify error_class matches
        failed_calls = [
            call
            for call in mock_logger.exception.call_args_list
            if call.args and call.args[0] == "gpu_session.start.failed"
        ]
        assert len(failed_calls) == 1
        assert failed_calls[0].kwargs["error_class"] == "CustomVastAIError"

    async def test_integrity_error_cleans_up_resources(self) -> None:
        service, mocks = _make_service()
        mocks["bundle_index"].resolve_bundle.return_value = _make_bundle_mapping()
        mocks["cf_client"].create_session_tunnel.return_value = _TUNNEL_RESULT
        mocks["vastai_client"].search_offers.return_value = [_make_offer()]
        mocks["vastai_client"].create_instance.return_value = 77777

        with patch(_REPO_PATH) as MockRepo:
            mock_repo = AsyncMock()
            MockRepo.return_value = mock_repo
            mock_repo.get_non_terminal_for_model.return_value = None
            mock_repo.create.side_effect = IntegrityError("INSERT", {}, Exception("duplicate key"))

            with pytest.raises(SessionAlreadyExistsError):
                await service.start_session(
                    user_id=uuid4(),
                    product_id="vex",
                    model_type=ModelType.AISHA_IMAGE,
                    account_id=mocks["account_id"],
                )

        mocks["vastai_client"].destroy_instance.assert_called_once_with(77777)
        mocks["cf_client"].delete_session_tunnel.assert_called_once_with(
            _TUNNEL_RESULT[0], _TUNNEL_RESULT[2]
        )

    async def test_persists_expected_env_vars(self) -> None:
        service, mocks = _make_service()
        bundle = _make_bundle_mapping(bundle_version="260201-01")
        mocks["bundle_index"].resolve_bundle.return_value = bundle
        mocks["cf_client"].create_session_tunnel.return_value = _TUNNEL_RESULT
        mocks["vastai_client"].search_offers.return_value = [_make_offer()]
        mocks["vastai_client"].create_instance.return_value = 55555

        with patch(_REPO_PATH) as MockRepo:
            mock_repo = AsyncMock()
            MockRepo.return_value = mock_repo
            mock_repo.get_non_terminal_for_model.return_value = None
            mock_repo.create.return_value = _make_gpu_session(status=GpuSessionStatus.pending)

            await service.start_session(
                user_id=uuid4(),
                product_id="vex",
                model_type=ModelType.AISHA_IMAGE,
                account_id=mocks["account_id"],
            )

        _, kwargs = mocks["vastai_client"].create_instance.call_args
        env = kwargs["env"]
        assert "ACS_BUNDLE" in env
        assert env["ACS_BUNDLE"] == "wan_2.2_i2v"
        assert "ACS_BUNDLE_VERSION" in env
        assert env["ACS_BUNDLE_VERSION"] == "260201-01"
        assert "ACS_CF_TUNNEL_TOKEN" in env
        assert "ACS_APEX_SESSION_ID" in env
        assert "ACS_APEX_CALLBACK_URL" in env
        assert "ACS_APEX_CALLBACK_TOKEN" in env
        assert "ACS_HF_TOKEN" in env
        assert "ACS_CIVITAI_API_TOKEN" in env
        assert f"-p {_DEFAULT_COMFYUI_PORT}:{_DEFAULT_COMFYUI_PORT}" in env
        assert env[f"-p {_DEFAULT_COMFYUI_PORT}:{_DEFAULT_COMFYUI_PORT}"] == "1"
        # New contract keys
        assert env["ACS_GITHUB_TOKEN"] == "ghp_test_token"
        assert env["ACS_BUNDLES_REPO"] == "https://github.com/gearbox/ai-bundles.git"
        assert env["ACS_BUNDLES_BRANCH"] == "master"
        assert env["ACS_AISHA_REPO"] == "https://github.com/gearbox/aisha.git"
        assert env["ACS_AISHA_BRANCH"] == "master"
        assert env["ACS_COMFYUI_PORT"] == str(_DEFAULT_COMFYUI_PORT)
        assert env["ACS_COMFYUI_PORT"] == str(bundle.hardware.comfyui_port)
        assert env["ACS_COMFYUI_HOST"] == "0.0.0.0"  # noqa: S104
        assert env["ACS_COMFYUI_EXTRA_ARGS"] == ""

    async def test_callback_token_is_random_per_call(self) -> None:
        service, mocks = _make_service()
        mocks["bundle_index"].resolve_bundle.return_value = _make_bundle_mapping()
        mocks["cf_client"].create_session_tunnel.return_value = _TUNNEL_RESULT
        mocks["vastai_client"].search_offers.return_value = [_make_offer()]
        mocks["vastai_client"].create_instance.return_value = 11111

        tokens = []

        def capture_env(*_args: Any, **kwargs: Any) -> int:
            tokens.append(kwargs["env"]["ACS_APEX_CALLBACK_TOKEN"])
            return 11111

        mocks["vastai_client"].create_instance.side_effect = capture_env

        with patch(_REPO_PATH) as MockRepo:
            mock_repo = AsyncMock()
            MockRepo.return_value = mock_repo
            mock_repo.get_non_terminal_for_model.return_value = None
            mock_repo.create.return_value = _make_gpu_session(status=GpuSessionStatus.pending)

            await service.start_session(
                user_id=uuid4(),
                product_id="vex",
                model_type=ModelType.AISHA_IMAGE,
                account_id=mocks["account_id"],
            )
            await service.start_session(
                user_id=uuid4(),
                product_id="vex",
                model_type=ModelType.AISHA_IMAGE,
                account_id=mocks["account_id"],
            )

        assert len(tokens) == 2
        assert tokens[0] != tokens[1]
        assert all(len(t) >= 64 for t in tokens)

    async def test_current_version_when_bundle_version_none(self) -> None:
        service, mocks = _make_service()
        bundle = _make_bundle_mapping(bundle_version=None)
        mocks["bundle_index"].resolve_bundle.return_value = bundle
        mocks["cf_client"].create_session_tunnel.return_value = _TUNNEL_RESULT
        mocks["vastai_client"].search_offers.return_value = [_make_offer()]
        mocks["vastai_client"].create_instance.return_value = 22222

        with patch(_REPO_PATH) as MockRepo:
            mock_repo = AsyncMock()
            MockRepo.return_value = mock_repo
            mock_repo.get_non_terminal_for_model.return_value = None
            mock_repo.create.return_value = _make_gpu_session(status=GpuSessionStatus.pending)

            await service.start_session(
                user_id=uuid4(),
                product_id="vex",
                model_type=ModelType.AISHA_IMAGE,
                account_id=mocks["account_id"],
            )

        _, kwargs = mocks["vastai_client"].create_instance.call_args
        assert kwargs["env"]["ACS_BUNDLE_VERSION"] == "current"

    async def test_start_session_insufficient_balance_raises_before_tunnel_creation(self) -> None:
        from src.api.services.billing_errors import InsufficientBalanceError

        service, mocks = _make_service()
        mocks["bundle_index"].resolve_bundle.return_value = _make_bundle_mapping()
        mocks["billing_service"].assert_sufficient_balance.side_effect = InsufficientBalanceError(
            balance=100, required=500
        )

        with patch(_REPO_PATH) as MockRepo:
            mock_repo = AsyncMock()
            MockRepo.return_value = mock_repo
            mock_repo.get_non_terminal_for_model.return_value = None

            with pytest.raises(InsufficientBalanceError):
                await service.start_session(
                    user_id=uuid4(),
                    product_id="vex",
                    model_type=ModelType.AISHA_IMAGE,
                    account_id=mocks["account_id"],
                )

        mocks["cf_client"].create_session_tunnel.assert_not_called()
        mocks["vastai_client"].search_offers.assert_not_called()
        mock_repo.create.assert_not_called()

    async def test_start_session_reserves_tokens_in_same_tx_as_insert(self) -> None:
        service, mocks = _make_service()
        mocks["bundle_index"].resolve_bundle.return_value = _make_bundle_mapping()
        mocks["cf_client"].create_session_tunnel.return_value = _TUNNEL_RESULT
        mocks["vastai_client"].search_offers.return_value = [_make_offer()]
        mocks["vastai_client"].create_instance.return_value = 99999
        expected_session = _make_gpu_session(status=GpuSessionStatus.pending)
        user_id = uuid4()

        with patch(_REPO_PATH) as MockRepo:
            mock_repo = AsyncMock()
            MockRepo.return_value = mock_repo
            mock_repo.get_non_terminal_for_model.return_value = None
            mock_repo.create.return_value = expected_session

            await service.start_session(
                user_id=user_id,
                product_id="vex",
                model_type=ModelType.AISHA_IMAGE,
                account_id=mocks["account_id"],
            )

        mocks["billing_service"].check_and_reserve.assert_called_once()
        call_args = mocks["billing_service"].check_and_reserve.call_args
        # Positional: (account_id, token_cost, session_id)
        assert call_args.args[0] == mocks["account_id"]
        assert call_args.args[1] == 500  # _MIN_BILLABLE_MINUTES (5) * tokens_per_minute (100)

        # The session_id passed to check_and_reserve must equal the id passed to repo.create
        # (both come from the same new_id() call inside start_session)
        create_kwargs = mock_repo.create.call_args.kwargs
        assert call_args.args[2] == create_kwargs["id"]

        # Keyword wiring — product_id, user_id, metadata all flow through
        assert call_args.kwargs["product_id"] == "vex"
        assert call_args.kwargs["user_id"] == user_id
        metadata = call_args.kwargs["metadata"]
        assert metadata["type"] == "gpu_session_reservation"
        assert metadata["model_type"] == ModelType.AISHA_IMAGE.value
        assert metadata["bundle_name"] == _make_bundle_mapping().bundle_name

    async def test_start_session_reservation_failure_cleans_up_instance_and_tunnel(self) -> None:
        from src.api.services.billing_errors import InsufficientBalanceError

        service, mocks = _make_service()
        mocks["bundle_index"].resolve_bundle.return_value = _make_bundle_mapping()
        mocks["cf_client"].create_session_tunnel.return_value = _TUNNEL_RESULT
        mocks["vastai_client"].search_offers.return_value = [_make_offer()]
        mocks["vastai_client"].create_instance.return_value = 77777
        mocks["billing_service"].check_and_reserve.side_effect = InsufficientBalanceError(
            balance=100, required=500
        )

        with patch(_REPO_PATH) as MockRepo:
            mock_repo = AsyncMock()
            MockRepo.return_value = mock_repo
            mock_repo.get_non_terminal_for_model.return_value = None
            mock_repo.create.return_value = _make_gpu_session(status=GpuSessionStatus.pending)

            with pytest.raises(InsufficientBalanceError):
                await service.start_session(
                    user_id=uuid4(),
                    product_id="vex",
                    model_type=ModelType.AISHA_IMAGE,
                    account_id=mocks["account_id"],
                )

        mocks["vastai_client"].destroy_instance.assert_called_once_with(77777)
        mocks["cf_client"].delete_session_tunnel.assert_called_once_with(
            _TUNNEL_RESULT[0], _TUNNEL_RESULT[2]
        )

    async def test_start_session_publishes_status_event(self) -> None:
        event_bus = AsyncMock()
        service, mocks = _make_service(event_bus=event_bus)
        mocks["bundle_index"].resolve_bundle.return_value = _make_bundle_mapping()
        mocks["cf_client"].create_session_tunnel.return_value = _TUNNEL_RESULT
        mocks["vastai_client"].search_offers.return_value = [_make_offer()]
        mocks["vastai_client"].create_instance.return_value = 11111
        expected_session = _make_gpu_session(status=GpuSessionStatus.pending)

        with patch(_REPO_PATH) as MockRepo:
            mock_repo = AsyncMock()
            MockRepo.return_value = mock_repo
            mock_repo.get_non_terminal_for_model.return_value = None
            mock_repo.create.return_value = expected_session

            await service.start_session(
                user_id=uuid4(),
                product_id="vex",
                model_type=ModelType.AISHA_IMAGE,
                account_id=mocks["account_id"],
            )

        event_bus.publish.assert_called_once()
        kwargs = event_bus.publish.call_args.kwargs
        assert kwargs["event_type"] == EventType.GPU_SESSION_STATUS_CHANGED
        assert kwargs["user_id"] == expected_session.user_id
        payload = kwargs["payload"]
        assert isinstance(payload, GpuSessionStatusPayload)
        assert payload.session_id == expected_session.id
        assert payload.status == GpuSessionStatus.pending.value
        assert payload.previous_status == "none"
        assert payload.model_type == expected_session.model_type
        assert not hasattr(payload, "bundle_name")
        assert "bundle_name" not in {f.name for f in msgspec.structs.fields(type(payload))}
        assert payload.error_message is None

    async def test_start_session_event_publish_failure_does_not_break(self) -> None:
        event_bus = AsyncMock()
        event_bus.publish.side_effect = Exception("redis down")
        service, mocks = _make_service(event_bus=event_bus)
        mocks["bundle_index"].resolve_bundle.return_value = _make_bundle_mapping()
        mocks["cf_client"].create_session_tunnel.return_value = _TUNNEL_RESULT
        mocks["vastai_client"].search_offers.return_value = [_make_offer()]
        mocks["vastai_client"].create_instance.return_value = 33333
        expected_session = _make_gpu_session(status=GpuSessionStatus.pending)

        with patch(_REPO_PATH) as MockRepo:
            mock_repo = AsyncMock()
            MockRepo.return_value = mock_repo
            mock_repo.get_non_terminal_for_model.return_value = None
            mock_repo.create.return_value = expected_session

            result = await service.start_session(
                user_id=uuid4(),
                product_id="vex",
                model_type=ModelType.AISHA_IMAGE,
                account_id=mocks["account_id"],
            )

        assert result is expected_session

    async def test_start_session_fails_fast_on_empty_github_token(self) -> None:
        """If ai_bundles_github_token is empty, raise before offer search; clean up tunnel."""
        from src.api.services.vastai.exceptions import NoCapacityError

        service, mocks = _make_service()
        mocks["settings"].ai_bundles_github_token = ""
        mocks["bundle_index"].resolve_bundle.return_value = _make_bundle_mapping()
        mocks["cf_client"].create_session_tunnel.return_value = _TUNNEL_RESULT

        with patch(_REPO_PATH) as MockRepo:
            mock_repo = AsyncMock()
            MockRepo.return_value = mock_repo
            mock_repo.get_non_terminal_for_model.return_value = None

            with pytest.raises(NoCapacityError, match="ai_bundles_github_token"):
                await service.start_session(
                    user_id=uuid4(),
                    product_id="vex",
                    model_type=ModelType.AISHA_IMAGE,
                    account_id=mocks["account_id"],
                )

        # Tunnel must have been cleaned up
        mocks["cf_client"].delete_session_tunnel.assert_called_once_with(
            _TUNNEL_RESULT[0], _TUNNEL_RESULT[2]
        )
        # No Vast.ai API calls
        mocks["vastai_client"].search_offers.assert_not_called()
        mocks["vastai_client"].create_instance.assert_not_called()


# ---------------------------------------------------------------------------
# pause_session tests
# ---------------------------------------------------------------------------


class TestPauseSession:
    async def test_from_active_calls_stop_instance(self) -> None:
        service, mocks = _make_service()
        user_id = uuid4()
        product_id = "vex"
        session = _make_gpu_session(user_id=user_id, product_id=product_id)
        session_id = session.id

        with patch(_REPO_PATH) as MockRepo, patch(_JOB_REPO_PATH) as MockJobRepo:
            mock_repo = AsyncMock()
            MockRepo.return_value = mock_repo
            mock_repo.get_by_id.return_value = session

            mock_job_repo = AsyncMock()
            MockJobRepo.return_value = mock_job_repo
            mock_job_repo.count_in_flight_for_session.return_value = 0

            result = await service.pause_session(
                session_id=session_id,
                user_id=user_id,
                product_id=product_id,
            )

        mocks["vastai_client"].stop_instance.assert_called_once_with(session.vastai_instance_id)
        assert result.status == GpuSessionStatus.paused

    async def test_sets_paused_at(self) -> None:
        service, _mocks = _make_service()
        user_id = uuid4()
        product_id = "vex"
        session = _make_gpu_session(user_id=user_id, product_id=product_id, paused_at=None)

        with patch(_REPO_PATH) as MockRepo, patch(_JOB_REPO_PATH) as MockJobRepo:
            mock_repo = AsyncMock()
            MockRepo.return_value = mock_repo
            mock_repo.get_by_id.return_value = session

            mock_job_repo = AsyncMock()
            MockJobRepo.return_value = mock_job_repo
            mock_job_repo.count_in_flight_for_session.return_value = 0

            result = await service.pause_session(
                session_id=session.id,
                user_id=user_id,
                product_id=product_id,
            )

        assert result.paused_at is not None

    @pytest.mark.parametrize(
        "bad_status",
        [
            GpuSessionStatus.pending,
            GpuSessionStatus.provisioning,
            GpuSessionStatus.paused,
            GpuSessionStatus.resuming,
            GpuSessionStatus.stale,
            GpuSessionStatus.stopping,
            GpuSessionStatus.stopped,
            GpuSessionStatus.failed,
        ],
    )
    async def test_from_non_active_raises(self, bad_status: GpuSessionStatus) -> None:
        service, _ = _make_service()
        user_id = uuid4()
        session = _make_gpu_session(user_id=user_id, status=bad_status)

        with patch(_REPO_PATH) as MockRepo:
            mock_repo = AsyncMock()
            MockRepo.return_value = mock_repo
            mock_repo.get_by_id.return_value = session

            with pytest.raises(InvalidSessionStateError) as exc_info:
                await service.pause_session(
                    session_id=session.id,
                    user_id=user_id,
                    product_id=session.product_id,
                )

        assert exc_info.value.current_status == bad_status
        assert exc_info.value.operation == "pause"

    async def test_vastai_failure_does_not_update_status(self) -> None:
        service, mocks = _make_service()
        user_id = uuid4()
        session = _make_gpu_session(user_id=user_id, status=GpuSessionStatus.active)
        mocks["vastai_client"].stop_instance.side_effect = VastAIError("timeout")

        with patch(_REPO_PATH) as MockRepo, patch(_JOB_REPO_PATH) as MockJobRepo:
            mock_repo = AsyncMock()
            MockRepo.return_value = mock_repo
            mock_repo.get_by_id.return_value = session

            mock_job_repo = AsyncMock()
            MockJobRepo.return_value = mock_job_repo
            mock_job_repo.count_in_flight_for_session.return_value = 0

            with pytest.raises(VastAIError):
                await service.pause_session(
                    session_id=session.id,
                    user_id=user_id,
                    product_id=session.product_id,
                )

        mock_repo.update_status.assert_not_called()


# ---------------------------------------------------------------------------
# resume_session tests
# ---------------------------------------------------------------------------


class TestResumeSession:
    async def test_from_paused_calls_start_instance(self) -> None:
        service, mocks = _make_service()
        user_id = uuid4()
        session = _make_gpu_session(user_id=user_id, status=GpuSessionStatus.paused)

        with patch(_REPO_PATH) as MockRepo:
            mock_repo = AsyncMock()
            MockRepo.return_value = mock_repo
            mock_repo.get_by_id.return_value = session

            result = await service.resume_session(
                session_id=session.id,
                user_id=user_id,
                product_id=session.product_id,
            )

        mocks["vastai_client"].start_instance.assert_called_once_with(session.vastai_instance_id)
        assert result.status == GpuSessionStatus.resuming

    async def test_sets_resumed_at(self) -> None:
        service, _mocks = _make_service()
        user_id = uuid4()
        session = _make_gpu_session(
            user_id=user_id, status=GpuSessionStatus.paused, resumed_at=None
        )

        with patch(_REPO_PATH) as MockRepo:
            mock_repo = AsyncMock()
            MockRepo.return_value = mock_repo
            mock_repo.get_by_id.return_value = session

            result = await service.resume_session(
                session_id=session.id,
                user_id=user_id,
                product_id=session.product_id,
            )

        assert result.resumed_at is not None

    @pytest.mark.parametrize(
        "bad_status",
        [
            GpuSessionStatus.pending,
            GpuSessionStatus.provisioning,
            GpuSessionStatus.active,
            GpuSessionStatus.resuming,
            GpuSessionStatus.stale,
            GpuSessionStatus.stopping,
            GpuSessionStatus.stopped,
            GpuSessionStatus.failed,
        ],
    )
    async def test_from_non_paused_raises(self, bad_status: GpuSessionStatus) -> None:
        service, _ = _make_service()
        user_id = uuid4()
        session = _make_gpu_session(user_id=user_id, status=bad_status)

        with patch(_REPO_PATH) as MockRepo:
            mock_repo = AsyncMock()
            MockRepo.return_value = mock_repo
            mock_repo.get_by_id.return_value = session

            with pytest.raises(InvalidSessionStateError) as exc_info:
                await service.resume_session(
                    session_id=session.id,
                    user_id=user_id,
                    product_id=session.product_id,
                )

        assert exc_info.value.current_status == bad_status
        assert exc_info.value.operation == "resume"

    async def test_vastai_failure_does_not_update_status(self) -> None:
        service, mocks = _make_service()
        user_id = uuid4()
        session = _make_gpu_session(user_id=user_id, status=GpuSessionStatus.paused)
        mocks["vastai_client"].start_instance.side_effect = VastAIError("timeout")

        with patch(_REPO_PATH) as MockRepo:
            mock_repo = AsyncMock()
            MockRepo.return_value = mock_repo
            mock_repo.get_by_id.return_value = session

            with pytest.raises(VastAIError):
                await service.resume_session(
                    session_id=session.id,
                    user_id=user_id,
                    product_id=session.product_id,
                )

        mock_repo.update_status.assert_not_called()


# ---------------------------------------------------------------------------
# stop_session tests
# ---------------------------------------------------------------------------


class TestStopSession:
    async def test_unconfirmed_returns_confirmation_no_state_changes(self) -> None:
        service, mocks = _make_service()
        user_id = uuid4()
        now = datetime.now(UTC)
        started = now - timedelta(hours=2)
        session = _make_gpu_session(
            user_id=user_id,
            status=GpuSessionStatus.active,
            started_at=started,
        )

        with patch(_REPO_PATH) as MockRepo:
            mock_repo = AsyncMock()
            MockRepo.return_value = mock_repo
            mock_repo.get_by_id_for_user.return_value = session

            result = await service.stop_session(
                session_id=session.id,
                user_id=user_id,
                product_id=session.product_id,
                confirmed=False,
            )

        assert isinstance(result, StopConfirmation)
        assert result.session_id == session.id
        assert result.model_type == session.model_type
        # No state changes
        mocks["vastai_client"].destroy_instance.assert_not_called()
        mocks["cf_client"].delete_session_tunnel.assert_not_called()
        mock_repo.update_status.assert_not_called()

    async def test_unconfirmed_duration_uses_started_at_when_present(self) -> None:
        service, _ = _make_service()
        user_id = uuid4()
        now = datetime.now(UTC)
        started = now - timedelta(hours=2)
        session = _make_gpu_session(
            user_id=user_id,
            status=GpuSessionStatus.active,
            started_at=started,
        )

        with patch(_REPO_PATH) as MockRepo:
            mock_repo = AsyncMock()
            MockRepo.return_value = mock_repo
            mock_repo.get_by_id_for_user.return_value = session

            result = await service.stop_session(
                session_id=session.id,
                user_id=user_id,
                product_id=session.product_id,
                confirmed=False,
            )

        assert isinstance(result, StopConfirmation)
        # Duration should be ~2 hours
        assert result.active_duration_seconds >= 2 * 3600 - 5  # allow 5s skew

    async def test_unconfirmed_for_active_shows_correct_duration(self) -> None:
        service, _ = _make_service()
        user_id = uuid4()
        now = datetime.now(UTC)
        started = now - timedelta(hours=3)
        session = _make_gpu_session(
            user_id=user_id,
            status=GpuSessionStatus.active,
            started_at=started,
        )

        with patch(_REPO_PATH) as MockRepo:
            mock_repo = AsyncMock()
            MockRepo.return_value = mock_repo
            mock_repo.get_by_id_for_user.return_value = session

            result = await service.stop_session(
                session_id=session.id,
                user_id=user_id,
                product_id=session.product_id,
                confirmed=False,
            )

        assert isinstance(result, StopConfirmation)
        assert result.active_duration_seconds >= 3 * 3600 - 5

    async def test_unconfirmed_for_paused_uses_paused_at_as_end(self) -> None:
        service, _ = _make_service()
        user_id = uuid4()
        base = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)
        started = base
        paused = base + timedelta(hours=2)
        session = _make_gpu_session(
            user_id=user_id,
            status=GpuSessionStatus.paused,
            started_at=started,
            paused_at=paused,
        )

        with patch(_REPO_PATH) as MockRepo:
            mock_repo = AsyncMock()
            MockRepo.return_value = mock_repo
            mock_repo.get_by_id_for_user.return_value = session

            result = await service.stop_session(
                session_id=session.id,
                user_id=user_id,
                product_id=session.product_id,
                confirmed=False,
            )

        assert isinstance(result, StopConfirmation)
        assert result.active_duration_seconds == 2 * 3600

    async def test_confirmed_happy_path_transitions_to_stopped(self) -> None:
        service, mocks = _make_service()
        user_id = uuid4()
        session = _make_gpu_session(user_id=user_id, status=GpuSessionStatus.active)

        with patch(_REPO_PATH) as MockRepo:
            mock_repo = AsyncMock()
            MockRepo.return_value = mock_repo
            mock_repo.get_by_id.return_value = session

            result = await service.stop_session(
                session_id=session.id,
                user_id=user_id,
                product_id=session.product_id,
                confirmed=True,
            )

        assert isinstance(result, GpuSession)
        assert result.status == GpuSessionStatus.stopped
        assert result.stopped_at is not None
        # Verify stopping → stopped via two update_status calls
        assert mock_repo.update_status.call_count == 2
        first_call_status = mock_repo.update_status.call_args_list[0][0][1]
        second_call_status = mock_repo.update_status.call_args_list[1][0][1]
        assert first_call_status == GpuSessionStatus.stopping
        assert second_call_status == GpuSessionStatus.stopped
        mocks["vastai_client"].destroy_instance.assert_called_once_with(session.vastai_instance_id)
        mocks["cf_client"].delete_session_tunnel.assert_called_once_with(
            session.cf_tunnel_id, session.cf_dns_record_id
        )

    async def test_confirmed_from_paused(self) -> None:
        service, _mocks = _make_service()
        user_id = uuid4()
        session = _make_gpu_session(user_id=user_id, status=GpuSessionStatus.paused)

        with patch(_REPO_PATH) as MockRepo:
            mock_repo = AsyncMock()
            MockRepo.return_value = mock_repo
            mock_repo.get_by_id.return_value = session

            result = await service.stop_session(
                session_id=session.id,
                user_id=user_id,
                product_id=session.product_id,
                confirmed=True,
            )

        assert isinstance(result, GpuSession)
        assert result.status == GpuSessionStatus.stopped

    async def test_confirmed_from_stale(self) -> None:
        service, _mocks = _make_service()
        user_id = uuid4()
        session = _make_gpu_session(user_id=user_id, status=GpuSessionStatus.stale)

        with patch(_REPO_PATH) as MockRepo:
            mock_repo = AsyncMock()
            MockRepo.return_value = mock_repo
            mock_repo.get_by_id.return_value = session

            result = await service.stop_session(
                session_id=session.id,
                user_id=user_id,
                product_id=session.product_id,
                confirmed=True,
            )

        assert isinstance(result, GpuSession)
        assert result.status == GpuSessionStatus.stopped

    async def test_confirmed_instance_destroy_fails_still_stops(self) -> None:
        service, mocks = _make_service()
        user_id = uuid4()
        session = _make_gpu_session(user_id=user_id, status=GpuSessionStatus.active)
        mocks["vastai_client"].destroy_instance.side_effect = VastAIError("destroy failed")

        with patch(_REPO_PATH) as MockRepo:
            mock_repo = AsyncMock()
            MockRepo.return_value = mock_repo
            mock_repo.get_by_id.return_value = session

            result = await service.stop_session(
                session_id=session.id,
                user_id=user_id,
                product_id=session.product_id,
                confirmed=True,
            )

        assert isinstance(result, GpuSession)
        assert result.status == GpuSessionStatus.stopped
        # Tunnel teardown still ran despite instance failure
        mocks["cf_client"].delete_session_tunnel.assert_called_once()

    async def test_confirmed_tunnel_delete_fails_still_stops(self) -> None:
        service, mocks = _make_service()
        user_id = uuid4()
        session = _make_gpu_session(user_id=user_id, status=GpuSessionStatus.active)
        mocks["cf_client"].delete_session_tunnel.side_effect = Exception("CF error")

        with patch(_REPO_PATH) as MockRepo:
            mock_repo = AsyncMock()
            MockRepo.return_value = mock_repo
            mock_repo.get_by_id.return_value = session

            result = await service.stop_session(
                session_id=session.id,
                user_id=user_id,
                product_id=session.product_id,
                confirmed=True,
            )

        assert isinstance(result, GpuSession)
        assert result.status == GpuSessionStatus.stopped

    @pytest.mark.parametrize(
        "terminal_status",
        [
            GpuSessionStatus.stopping,
            GpuSessionStatus.stopped,
            GpuSessionStatus.failed,
        ],
    )
    async def test_from_terminal_state_is_idempotent(
        self, terminal_status: GpuSessionStatus
    ) -> None:
        """Regression for 2026-05-12: stop on a terminal session must not raise."""
        service, mocks = _make_service()
        user_id = uuid4()
        session = _make_gpu_session(user_id=user_id, status=terminal_status)

        with patch(_REPO_PATH) as MockRepo:
            mock_repo = AsyncMock()
            MockRepo.return_value = mock_repo
            mock_repo.get_by_id_for_user.return_value = session

            result = await service.stop_session(
                session_id=session.id,
                user_id=user_id,
                product_id=session.product_id,
                confirmed=True,
            )

        assert result is session  # same row returned, no work done
        mocks["vastai_client"].destroy_instance.assert_not_called()
        mocks["cf_client"].delete_session_tunnel.assert_not_called()
        mock_repo.update_status.assert_not_called()


class TestBillableMinutesHelper:
    """Unit tests for billable_minutes_for_active_seconds.

    Per-started-minute billing: any partial second of a minute counts as a full
    minute. A 5-minute floor matches the base reservation.
    """

    @pytest.mark.parametrize(
        "active_seconds,expected_minutes",
        [
            (0, 5),  # zero usage still pays the minimum
            (1, 5),  # 1 second — floor wins
            (59, 5),  # under 1 min — floor wins
            (60, 5),  # exactly 1 min — floor wins
            (299, 5),  # 4m59s — floor wins
            (300, 5),  # exactly 5 min — exact match
            (301, 6),  # 5m01s — rounds up to 6 (per-started-minute)
            (359, 6),  # 5m59s — the specific case the reviewer flagged
            (360, 6),  # exactly 6 min
            (361, 7),  # 6m01s — rounds up
            (3600, 60),  # exactly 60 min
            (3601, 61),  # 60m01s — rounds up
        ],
    )
    def test_rounds_up_with_five_minute_floor(
        self, active_seconds: int, expected_minutes: int
    ) -> None:
        assert billable_minutes_for_active_seconds(active_seconds) == expected_minutes

    def test_negative_input_treated_as_zero(self) -> None:
        """Defensive: negative seconds (clock drift edge case) floor to the 5-min minimum."""
        assert billable_minutes_for_active_seconds(-100) == 5


# ---------------------------------------------------------------------------
# Billing finalization branch tests (_finalize_billing)
# ---------------------------------------------------------------------------


class TestFinalizeBilling:
    """Covers the three _finalize_billing branches: overage, underage, exact match.

    _finalize_billing runs AFTER _stop_confirmed's TX2 commits the 'stopped'
    status. It reads the session's timestamps + total_paused_seconds, computes
    billable minutes (5-min floor), multiplies by tokens_per_minute, and compares
    against the base reservation to decide overage/refund/no-op.
    """

    async def _run_stop_and_capture_billing(
        self,
        *,
        started_at: datetime,
        stopped_now: datetime,
        total_paused_seconds: int,
        tokens_per_minute: int = 100,
        existing_debit_amount: int = 500,
    ) -> tuple[MagicMock, GpuSession]:
        """Drive a confirmed stop and return the mock billing service + session.

        Mocks ``BillingRepository.get_debit_for_job`` to return a debit row of
        magnitude ``existing_debit_amount`` (stored as negative on the ledger).
        Round 4: ``_finalize_billing`` reads the debit amount as the authoritative
        cap for refund/overage math, decoupled from settings.
        """
        settings = _make_settings()
        settings.gpu_session_tokens_per_minute = tokens_per_minute

        service, mocks = _make_service(settings=settings)
        user_id = uuid4()
        session = _make_gpu_session(
            user_id=user_id,
            status=GpuSessionStatus.active,
            started_at=started_at,
            total_paused_seconds=total_paused_seconds,
        )
        # Round 4: _apply_finalize_billing reads debit.account_id (ledger
        # authority). Align the session row with the debit so test assertions
        # against either field hold.
        session.account_id = mocks["account_id"]

        # Freeze time so stopped_at is predictable
        with (
            patch(_REPO_PATH) as MockRepo,
            patch("src.api.services.gpu_session.service.datetime") as mock_dt,
            patch("src.api.services.gpu_session.service.BillingRepository") as MockBR,
        ):
            mock_dt.now.return_value = stopped_now
            # datetime.fromisoformat / UTC / timedelta still need to work
            mock_dt.side_effect = _datetime_utc

            mock_repo = AsyncMock()
            MockRepo.return_value = mock_repo
            mock_repo.get_by_id.return_value = session

            mock_billing_repo = AsyncMock()
            MockBR.return_value = mock_billing_repo
            mock_billing_repo.get_debit_for_job.return_value = MagicMock(
                amount=-existing_debit_amount,  # debits are stored negative
                account_id=mocks["account_id"],
            )
            # Simulate no prior metered debits — total settled equals base reservation.
            mock_billing_repo.get_settled_tokens_for_session.return_value = existing_debit_amount

            result = await service.stop_session(
                session_id=session.id,
                user_id=user_id,
                product_id=session.product_id,
                confirmed=True,
            )

        assert isinstance(result, GpuSession)
        return mocks["billing_service"], result

    async def test_overage_creates_additional_debit(self) -> None:
        """10 min active @ 100 tok/min = 1000 tokens, reserved 500 → 500 overage debit.

        The overage debit MUST use job_id=None to preserve the one-debit-per-job
        invariant on the base reservation (which keeps the session.id as job_id
        for refund-on-failure lookups). The session link is carried in metadata.
        """
        now = datetime.now(UTC)
        started = now - timedelta(minutes=10)

        billing, session = await self._run_stop_and_capture_billing(
            started_at=started,
            stopped_now=now,
            total_paused_seconds=0,
        )

        billing.settle_session_usage.assert_awaited_once()
        args = billing.settle_session_usage.await_args.args
        kwargs = billing.settle_session_usage.await_args.kwargs
        assert args[0] == session.account_id
        assert args[1] == 500  # overage = 1000 billable - 500 reserved
        # session_id is a direct kwarg; extra_metadata carries the charge-type metadata
        assert kwargs["session_id"] == session.id
        assert kwargs["model_type"] == session.model_type
        extra = kwargs["extra_metadata"]
        assert extra["type"] == "gpu_session_overage"
        assert extra["billable_minutes"] == 10
        assert extra["bundle_name"] == session.bundle_name
        # No refund in overage branch
        billing.partial_refund.assert_not_called()

    async def test_overage_5m59s_rounds_up_to_6_billable_minutes(self) -> None:
        """Per-started-minute billing: 5m59s must bill as 6 min, not 5 (floor bug).

        5m59s active @ 100 tok/min = 600 tokens; reserved = 500 → 100 overage.
        The prior floor-division logic returned 5 min = 500 tokens = no charge,
        undercharging the user by a full minute of the sixth-started minute.
        """
        now = datetime.now(UTC)
        started = now - timedelta(seconds=5 * 60 + 59)  # 5m59s

        billing, session = await self._run_stop_and_capture_billing(
            started_at=started,
            stopped_now=now,
            total_paused_seconds=0,
        )

        billing.settle_session_usage.assert_awaited_once()
        args = billing.settle_session_usage.await_args.args
        kwargs = billing.settle_session_usage.await_args.kwargs
        assert args[1] == 100  # overage = (6 * 100) - 500 reserved
        assert kwargs["session_id"] == session.id
        assert kwargs["extra_metadata"]["billable_minutes"] == 6

    async def test_partial_refund_when_billable_below_reservation(self) -> None:
        """5 min floor @ 80 tok/min = 400 tokens, reserved 500 → 100 refund.

        The partial refund uses the session.id as job_id (not None) because
        the base reservation debit is stamped with that same job_id, and
        partial_refund looks up the debit via get_debit_for_job(job_id).
        """
        now = datetime.now(UTC)
        # Actual runtime 3 min, but billable is floored to 5 min
        started = now - timedelta(minutes=3)

        billing, session = await self._run_stop_and_capture_billing(
            started_at=started,
            stopped_now=now,
            total_paused_seconds=0,
            tokens_per_minute=80,  # 5 * 80 = 400 < 500 reserved
        )

        billing.partial_refund.assert_awaited_once()
        args = billing.partial_refund.await_args.args
        assert args[0] == session.id  # partial refund IS linked to the base reservation
        assert args[1] == 100  # refund = 500 reserved - 400 billable
        kwargs = billing.partial_refund.await_args.kwargs
        assert kwargs["product_id"] == session.product_id
        assert kwargs["user_id"] == session.user_id
        assert "used 5min" in kwargs["description"]
        # No overage debit
        billing.settle_session_usage.assert_not_called()

    async def test_exact_match_creates_neither_debit_nor_refund(self) -> None:
        """5 min active @ 100 tok/min = 500 tokens, reserved 500 → no-op."""
        now = datetime.now(UTC)
        started = now - timedelta(minutes=5)

        billing, _ = await self._run_stop_and_capture_billing(
            started_at=started,
            stopped_now=now,
            total_paused_seconds=0,
        )

        billing.settle_session_usage.assert_not_called()
        billing.partial_refund.assert_not_called()

    async def test_paused_seconds_excluded_from_billable_duration(self) -> None:
        """Session ran 20 min wall-clock, paused 15 min → 5 min billable → exact match."""
        now = datetime.now(UTC)
        started = now - timedelta(minutes=20)

        billing, _ = await self._run_stop_and_capture_billing(
            started_at=started,
            stopped_now=now,
            total_paused_seconds=15 * 60,  # 15 min paused
        )

        # 20 min total - 15 min paused = 5 min active → 500 tokens = reservation → no-op
        billing.settle_session_usage.assert_not_called()
        billing.partial_refund.assert_not_called()

    async def test_paused_seconds_exclusion_plus_overage(self) -> None:
        """30 min wall-clock - 10 min paused = 20 min billable @ 100 = 2000 → 1500 overage."""
        now = datetime.now(UTC)
        started = now - timedelta(minutes=30)

        billing, session = await self._run_stop_and_capture_billing(
            started_at=started,
            stopped_now=now,
            total_paused_seconds=10 * 60,
        )

        billing.settle_session_usage.assert_awaited_once()
        args = billing.settle_session_usage.await_args.args
        kwargs = billing.settle_session_usage.await_args.kwargs
        assert args[1] == 1500  # 2000 billable - 500 reserved
        assert kwargs["session_id"] == session.id
        assert kwargs["extra_metadata"]["billable_minutes"] == 20
        assert kwargs["extra_metadata"]["type"] == "gpu_session_overage"


# ---------------------------------------------------------------------------
# Pause/resume duration tracking (total_paused_seconds accumulation)
# ---------------------------------------------------------------------------


class TestPauseResumeDuration:
    """Verifies that total_paused_seconds is updated on resume and consumed correctly.

    The architecture stores cumulative paused time on the gpu_sessions row via
    total_paused_seconds. On each resume we add (resumed_at - paused_at). At stop
    time, _finalize_billing subtracts this from wall-clock seconds to compute
    billable minutes.
    """

    async def test_resume_updates_total_paused_seconds_with_pause_duration(self) -> None:
        """resumed_at - paused_at must be persisted via repo.add_paused_seconds."""
        service, _mocks = _make_service()
        user_id = uuid4()
        paused_at = datetime.now(UTC) - timedelta(minutes=10)
        session = _make_gpu_session(
            user_id=user_id,
            status=GpuSessionStatus.paused,
            paused_at=paused_at,
            total_paused_seconds=0,
        )
        resume_now = paused_at + timedelta(minutes=10)  # exactly 600s paused

        with (
            patch(_REPO_PATH) as MockRepo,
            patch("src.api.services.gpu_session.service.datetime") as mock_dt,
        ):
            mock_dt.now.return_value = resume_now
            mock_dt.side_effect = _datetime_utc
            mock_repo = AsyncMock()
            MockRepo.return_value = mock_repo
            mock_repo.get_by_id.return_value = session

            await service.resume_session(
                session_id=session.id, user_id=user_id, product_id=session.product_id
            )

        # The repo must be told to add exactly 600s to total_paused_seconds
        mock_repo.add_paused_seconds.assert_awaited_once_with(session.id, 600)
        # Status updated to resuming with resumed_at
        mock_repo.update_status.assert_called_once()
        call_kwargs = mock_repo.update_status.call_args.kwargs
        assert "resumed_at" in call_kwargs
        # And the in-memory session_row is updated so downstream consumers see it
        assert session.total_paused_seconds == 600

    async def test_resume_accumulates_across_multiple_pause_cycles(self) -> None:
        """Second resume adds to the already-accumulated total_paused_seconds."""
        service, _ = _make_service()
        user_id = uuid4()
        paused_at = datetime.now(UTC) - timedelta(minutes=5)
        # Session already has 300s from a prior pause/resume cycle
        session = _make_gpu_session(
            user_id=user_id,
            status=GpuSessionStatus.paused,
            paused_at=paused_at,
            total_paused_seconds=300,
        )
        resume_now = paused_at + timedelta(minutes=5)  # 300s this pause

        with (
            patch(_REPO_PATH) as MockRepo,
            patch("src.api.services.gpu_session.service.datetime") as mock_dt,
        ):
            mock_dt.now.return_value = resume_now
            mock_dt.side_effect = _datetime_utc
            mock_repo = AsyncMock()
            MockRepo.return_value = mock_repo
            mock_repo.get_by_id.return_value = session

            await service.resume_session(
                session_id=session.id, user_id=user_id, product_id=session.product_id
            )

        # Repo receives just the delta (300), not the cumulative total
        mock_repo.add_paused_seconds.assert_awaited_once_with(session.id, 300)
        # In-memory session reflects 300 (prior) + 300 (this pause) = 600
        assert session.total_paused_seconds == 600


# ---------------------------------------------------------------------------
# compute_session_durations — pure helper tests
# ---------------------------------------------------------------------------


class TestComputeSessionDurations:
    """Unit tests for the active/paused split helper.

    Returns (active_seconds, paused_seconds). Non-negative. Authoritative
    consumer of the ``total_paused_seconds`` column; does not try to derive
    in-flight pause intervals — that's the write path's job.
    """

    def test_running_session_uses_now_as_end(self) -> None:
        started = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)
        now = started + timedelta(minutes=10)
        session = _make_gpu_session(
            status=GpuSessionStatus.active,
            started_at=started,
            stopped_at=None,
            total_paused_seconds=0,
        )

        result = compute_session_durations(session, now=now)

        assert result == SessionDurations(active_seconds=600, paused_seconds=0)

    def test_paused_session_uses_paused_at_as_end(self) -> None:
        """Preview path: while a session is paused, active time is frozen."""
        started = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)
        paused = started + timedelta(minutes=10)
        now = paused + timedelta(minutes=30)  # irrelevant — end is paused_at
        session = _make_gpu_session(
            status=GpuSessionStatus.paused,
            started_at=started,
            paused_at=paused,
            stopped_at=None,
            total_paused_seconds=0,
        )

        result = compute_session_durations(session, now=now)

        # end = paused_at → total wall-clock = 10 min, all active (no prior pauses)
        assert result.active_seconds == 600
        assert result.paused_seconds == 0

    def test_stopped_session_uses_stopped_at_as_end(self) -> None:
        """Post-stop: end is stopped_at, paused is what's in the column."""
        started = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)
        stopped = started + timedelta(minutes=60)
        session = _make_gpu_session(
            status=GpuSessionStatus.stopped,
            started_at=started,
            stopped_at=stopped,
            total_paused_seconds=15 * 60,  # 15 min paused
        )

        result = compute_session_durations(session, now=stopped + timedelta(hours=1))

        # 60 min wall-clock - 15 min paused = 45 min active
        assert result.active_seconds == 45 * 60
        assert result.paused_seconds == 15 * 60

    def test_paused_seconds_clamped_to_wall_clock(self) -> None:
        """Defensive: if total_paused_seconds somehow exceeds wall-clock, clamp."""
        started = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)
        stopped = started + timedelta(minutes=10)  # 600s wall-clock
        session = _make_gpu_session(
            status=GpuSessionStatus.stopped,
            started_at=started,
            stopped_at=stopped,
            total_paused_seconds=99999,  # bogus — larger than wall-clock
        )

        result = compute_session_durations(session, now=stopped)

        # Clamped so active never goes negative
        assert result.active_seconds == 0
        assert result.paused_seconds == 600

    def test_negative_total_paused_seconds_treated_as_zero(self) -> None:
        """Defensive: negative paused column value (corrupt data) treated as 0."""
        started = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)
        stopped = started + timedelta(minutes=10)
        session = _make_gpu_session(
            status=GpuSessionStatus.stopped,
            started_at=started,
            stopped_at=stopped,
            total_paused_seconds=-5,
        )

        result = compute_session_durations(session, now=stopped)

        assert result.active_seconds == 600
        assert result.paused_seconds == 0

    def test_session_with_no_started_at_falls_back_to_created_at(self) -> None:
        created = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)
        now = created + timedelta(minutes=5)
        session = _make_gpu_session(
            status=GpuSessionStatus.active,
            started_at=None,
            total_paused_seconds=0,
        )
        session.created_at = created

        result = compute_session_durations(session, now=now)

        assert result.active_seconds == 300


# ---------------------------------------------------------------------------
# Stop-from-paused rollup regression tests (Issue 1)
# ---------------------------------------------------------------------------


class TestStopFromPausedRollup:
    """Regression: stopping a paused session rolled the pause interval up into
    the authoritative column, rather than leaving it uncounted.

    Before the fix, ``total_paused_seconds`` was only updated on resume. Stopping
    a paused session therefore left the in-flight pause entirely uncounted, and
    ``_finalize_billing`` would bill the full wall-clock as active GPU time.
    Example: session active 10 min, paused 50 min, stopped → billed 60 min of
    GPU when the user only used 10.
    """

    async def test_stopping_from_paused_without_resume_rolls_up_pause(self) -> None:
        """repo.add_paused_seconds is called at TX1 with the in-flight pause delta."""
        settings = _make_settings()
        settings.gpu_session_tokens_per_minute = 100

        service, _mocks = _make_service(settings=settings)
        user_id = uuid4()
        started = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)
        paused = started + timedelta(minutes=10)
        stop_now = started + timedelta(minutes=60)  # 50 min paused

        session = _make_gpu_session(
            user_id=user_id,
            status=GpuSessionStatus.paused,
            started_at=started,
            paused_at=paused,
            total_paused_seconds=0,
        )

        with (
            patch(_REPO_PATH) as MockRepo,
            patch("src.api.services.gpu_session.service.datetime") as mock_dt,
        ):
            mock_dt.now.return_value = stop_now
            mock_dt.side_effect = _datetime_utc

            mock_repo = AsyncMock()
            MockRepo.return_value = mock_repo
            mock_repo.get_by_id.return_value = session

            await service.stop_session(
                session_id=session.id,
                user_id=user_id,
                product_id=session.product_id,
                confirmed=True,
            )

        # The in-flight pause (50 min = 3000s) must be rolled up into the column
        # in TX1 before the external teardown runs.
        mock_repo.add_paused_seconds.assert_awaited_once_with(session.id, 3000)
        # In-memory row is kept in sync so _finalize_billing reads the new total.
        assert session.total_paused_seconds == 3000

    async def test_stopping_from_paused_bills_only_active_duration(self) -> None:
        """The pure regression: 10 min active + 50 min paused → 10 min billed.

        Before the fix, _finalize_billing saw total_paused_seconds=0 and billed
        the full 60 min as active, leaving the user overcharged by 50 min.
        """
        settings = _make_settings()
        settings.gpu_session_tokens_per_minute = 100

        service, mocks = _make_service(settings=settings)
        user_id = uuid4()
        started = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)
        paused = started + timedelta(minutes=10)
        stop_now = started + timedelta(minutes=60)

        session = _make_gpu_session(
            user_id=user_id,
            status=GpuSessionStatus.paused,
            started_at=started,
            paused_at=paused,
            total_paused_seconds=0,
        )

        with (
            patch(_REPO_PATH) as MockRepo,
            patch("src.api.services.gpu_session.service.datetime") as mock_dt,
            patch("src.api.services.gpu_session.service.BillingRepository") as MockBR,
        ):
            mock_dt.now.return_value = stop_now
            mock_dt.side_effect = _datetime_utc

            mock_repo = AsyncMock()
            MockRepo.return_value = mock_repo
            mock_repo.get_by_id.return_value = session

            mock_billing_repo = AsyncMock()
            MockBR.return_value = mock_billing_repo
            # Original debit of 500 (5 min * 100 tok/min). Billable below covers
            # the same 500-token base reservation.
            mock_billing_repo.get_debit_for_job.return_value = MagicMock(
                amount=-500, account_id=mocks["account_id"]
            )
            mock_billing_repo.get_settled_tokens_for_session.return_value = 500

            await service.stop_session(
                session_id=session.id,
                user_id=user_id,
                product_id=session.product_id,
                confirmed=True,
            )

        billing = mocks["billing_service"]
        # 10 min active @ 100 tok/min = 1000 tokens billable
        # Original debit 500 → overage of 500 tokens
        billing.settle_session_usage.assert_awaited_once()
        args = billing.settle_session_usage.await_args.args
        assert args[1] == 500  # overage amount
        assert (
            billing.settle_session_usage.await_args.kwargs["extra_metadata"]["billable_minutes"]
            == 10
        )

    async def test_stop_from_paused_with_prior_pause_accumulates(self) -> None:
        """Session was previously paused+resumed, now paused again and stopped.

        Prior pauses live in total_paused_seconds; the in-flight pause is the
        delta to add. Billing should exclude the total of both.
        """
        settings = _make_settings()
        settings.gpu_session_tokens_per_minute = 100

        service, mocks = _make_service(settings=settings)
        user_id = uuid4()
        started = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)
        # Session ran 20 min (5 min before first pause, 5 min paused and rolled up,
        # 10 min active after resume), then paused again at minute 20.
        paused = started + timedelta(minutes=20)
        stop_now = started + timedelta(minutes=35)  # 15 min in this second pause

        session = _make_gpu_session(
            user_id=user_id,
            status=GpuSessionStatus.paused,
            started_at=started,
            paused_at=paused,
            total_paused_seconds=5 * 60,  # from a prior pause/resume cycle
        )

        with (
            patch(_REPO_PATH) as MockRepo,
            patch("src.api.services.gpu_session.service.datetime") as mock_dt,
            patch("src.api.services.gpu_session.service.BillingRepository") as MockBR,
        ):
            mock_dt.now.return_value = stop_now
            mock_dt.side_effect = _datetime_utc

            mock_repo = AsyncMock()
            MockRepo.return_value = mock_repo
            mock_repo.get_by_id.return_value = session

            mock_billing_repo = AsyncMock()
            MockBR.return_value = mock_billing_repo
            mock_billing_repo.get_debit_for_job.return_value = MagicMock(
                amount=-500, account_id=mocks["account_id"]
            )
            mock_billing_repo.get_settled_tokens_for_session.return_value = 500

            await service.stop_session(
                session_id=session.id,
                user_id=user_id,
                product_id=session.product_id,
                confirmed=True,
            )

        # Repo is told to add only the new pause delta (15 min = 900s), not the cumulative
        mock_repo.add_paused_seconds.assert_awaited_once_with(session.id, 900)
        # In-memory row shows 5 + 15 = 20 min total paused
        assert session.total_paused_seconds == 20 * 60

        billing = mocks["billing_service"]
        # 35 min wall - 20 min paused = 15 min active
        # 15 * 100 = 1500 billable, debit 500 → 1000 overage
        billing.settle_session_usage.assert_awaited_once()
        args = billing.settle_session_usage.await_args.args
        assert args[1] == 1000
        assert (
            billing.settle_session_usage.await_args.kwargs["extra_metadata"]["billable_minutes"]
            == 15
        )

    async def test_stopping_from_active_does_not_call_add_paused_seconds(self) -> None:
        """Regression guard: the rollup only triggers from 'paused'."""
        settings = _make_settings()
        settings.gpu_session_tokens_per_minute = 100

        service, _ = _make_service(settings=settings)
        user_id = uuid4()
        started = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)
        stop_now = started + timedelta(minutes=10)

        session = _make_gpu_session(
            user_id=user_id,
            status=GpuSessionStatus.active,
            started_at=started,
            paused_at=None,
            total_paused_seconds=0,
        )

        with (
            patch(_REPO_PATH) as MockRepo,
            patch("src.api.services.gpu_session.service.datetime") as mock_dt,
        ):
            mock_dt.now.return_value = stop_now
            mock_dt.side_effect = _datetime_utc

            mock_repo = AsyncMock()
            MockRepo.return_value = mock_repo
            mock_repo.get_by_id.return_value = session

            await service.stop_session(
                session_id=session.id,
                user_id=user_id,
                product_id=session.product_id,
                confirmed=True,
            )

        # No pause to roll up — the rollup path must not run
        mock_repo.add_paused_seconds.assert_not_awaited()

    async def test_stopped_at_equals_stop_now_no_teardown_drift(self) -> None:
        """stopped_at uses the same captured 'now' as the TX1 pause rollup.

        This is the guard against teardown-window drift: if _stop_confirmed
        called datetime.now() twice, the gap between TX1 and TX2 would land
        in the active bucket. We want the active bucket to stop at TX1.
        """
        service, _ = _make_service()
        user_id = uuid4()
        started = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)
        paused = started + timedelta(minutes=10)
        stop_now = started + timedelta(minutes=60)

        session = _make_gpu_session(
            user_id=user_id,
            status=GpuSessionStatus.paused,
            started_at=started,
            paused_at=paused,
            total_paused_seconds=0,
        )

        with (
            patch(_REPO_PATH) as MockRepo,
            patch("src.api.services.gpu_session.service.datetime") as mock_dt,
        ):
            mock_dt.now.return_value = stop_now
            mock_dt.side_effect = _datetime_utc

            mock_repo = AsyncMock()
            MockRepo.return_value = mock_repo
            mock_repo.get_by_id.return_value = session

            result = await service.stop_session(
                session_id=session.id,
                user_id=user_id,
                product_id=session.product_id,
                confirmed=True,
            )

        assert isinstance(result, GpuSession)
        assert result.stopped_at == stop_now


# ---------------------------------------------------------------------------
# Ledger-as-authority tests (Round 4 / Issue 8)
# ---------------------------------------------------------------------------


class TestLedgerAuthorityForBilling:
    """Round 4: _finalize_billing reads the actual debit from the ledger as the
    cap for refund/overage math, never a recomputed reservation. This makes
    over-refund impossible by construction even if ops changes
    ``gpu_session_tokens_per_minute`` while sessions are running.
    """

    async def test_partial_refund_capped_at_actual_debit_after_rate_drop(self) -> None:
        """Rate drops mid-session; refund cannot exceed what was actually debited.

        Setup:
          - Start: rate = 200/min → reservation 1000 tokens debited.
          - Session runs 4 min.
          - Stop: rate = 100/min (ops dropped it). Naive math would compute
            ``MIN(5)*100 - billable(5*100) = 0`` and refund nothing — fine.
            But the prior bug shape was: derive reservation from new rate
            (500), see billable(500) == reserved(500), refund nothing —
            actually correct here. The DANGEROUS case is the OTHER direction:
            new reservation > old debit. The code must use the LEDGER debit
            (1000) as cap, not a freshly-computed value.

        Construction: original debit = 500 tokens (lower than what new rate
        would imply). Session runs 1 min → billable = 5 min * 100 = 500.
        Refund = max(0, 500 - 500) = 0. No refund issued. Verifies the
        ledger value is read.
        """
        settings = _make_settings()
        settings.gpu_session_tokens_per_minute = 100  # current rate
        service, mocks = _make_service(settings=settings)

        now = datetime.now(UTC)
        started = now - timedelta(minutes=1)  # 1 min active → billable floor 5
        session = _make_gpu_session(
            user_id=uuid4(),
            status=GpuSessionStatus.active,
            started_at=started,
            total_paused_seconds=0,
        )
        session.account_id = mocks["account_id"]
        session.stopped_at = now

        billing = mocks["billing_service"]

        with (
            patch("src.api.services.gpu_session.service.BillingRepository") as MockBR,
            patch(_REPO_PATH) as MockSessionRepo,
        ):
            mock_billing_repo = AsyncMock()
            MockBR.return_value = mock_billing_repo
            # Ledger debit is 500. Billable = 5 * 100 = 500. No refund, no overage.
            mock_billing_repo.get_debit_for_job.return_value = MagicMock(
                amount=-500, account_id=mocks["account_id"]
            )
            mock_billing_repo.get_settled_tokens_for_session.return_value = 500
            MockSessionRepo.return_value = AsyncMock()

            await service._finalize_billing(session)

        billing.partial_refund.assert_not_awaited()
        billing.settle_session_usage.assert_not_awaited()

    async def test_refund_uses_ledger_debit_not_settings(self) -> None:
        """Original debit > what current settings would produce → refund the diff.

        Settings drift scenario: user reserved 800 (under old rate); current
        billable is 500. Refund must be exactly 300, capped by the actual
        ledger debit (800). Cannot over-refund.
        """
        settings = _make_settings()
        settings.gpu_session_tokens_per_minute = 100
        service, mocks = _make_service(settings=settings)

        now = datetime.now(UTC)
        started = now - timedelta(minutes=4)
        session = _make_gpu_session(
            user_id=uuid4(),
            status=GpuSessionStatus.active,
            started_at=started,
            total_paused_seconds=0,
        )
        session.account_id = mocks["account_id"]
        session.stopped_at = now

        billing = mocks["billing_service"]

        with (
            patch("src.api.services.gpu_session.service.BillingRepository") as MockBR,
            patch(_REPO_PATH) as MockSessionRepo,
        ):
            mock_billing_repo = AsyncMock()
            MockBR.return_value = mock_billing_repo
            mock_billing_repo.get_debit_for_job.return_value = MagicMock(
                amount=-800, account_id=mocks["account_id"]
            )
            mock_billing_repo.get_settled_tokens_for_session.return_value = 800
            MockSessionRepo.return_value = AsyncMock()

            await service._finalize_billing(session)

        # Billable = max(5, 4) * 100 = 500. Original = 800. Refund = 300.
        billing.partial_refund.assert_awaited_once()
        args = billing.partial_refund.await_args.args
        assert args[1] == 300, f"Expected refund of 300 (800-500), got {args[1]}"

    async def test_no_debit_found_marks_finalized_and_skips_billing(self) -> None:
        """No debit row (admin reversal, failed reservation step) → no-op finalize.

        The session is still marked billing_finalized_at so the reconciler
        doesn't loop. Logged at WARN for ops visibility.
        """
        service, mocks = _make_service()

        now = datetime.now(UTC)
        session = _make_gpu_session(
            user_id=uuid4(),
            status=GpuSessionStatus.active,
            started_at=now - timedelta(minutes=10),
            total_paused_seconds=0,
        )
        session.account_id = mocks["account_id"]
        session.stopped_at = now

        billing = mocks["billing_service"]

        with patch("src.api.services.gpu_session.service.BillingRepository") as MockBR:
            mock_billing_repo = AsyncMock()
            MockBR.return_value = mock_billing_repo
            mock_billing_repo.get_debit_for_job.return_value = None

            with patch(_REPO_PATH) as MockSessionRepo:
                mock_session_repo = AsyncMock()
                MockSessionRepo.return_value = mock_session_repo

                await service._finalize_billing(session)

        billing.partial_refund.assert_not_awaited()
        billing.check_and_reserve.assert_not_awaited()
        # Reconciler must not pick this up — finalized_at is stamped.
        mock_session_repo.mark_billing_finalized.assert_awaited_once()


# ---------------------------------------------------------------------------
# Finalize-billing retry + defer tests (Round 4 / Issue 7)
# ---------------------------------------------------------------------------


class TestFinalizeBillingRetry:
    """Round 4: _finalize_billing retries once on transient failure, then defers
    to the phase-2 reconciler if both attempts fail (billing_finalized_at NULL).
    """

    async def test_first_attempt_fails_second_succeeds_stamps_finalized_at(self) -> None:
        """Transient DB error on first try; second attempt stamps finalized_at."""
        service, mocks = _make_service()

        now = datetime.now(UTC)
        session = _make_gpu_session(
            user_id=uuid4(),
            status=GpuSessionStatus.active,
            started_at=now - timedelta(minutes=10),
            total_paused_seconds=0,
        )
        session.account_id = mocks["account_id"]
        session.stopped_at = now

        # Track call attempts
        call_count = 0

        async def flaky_apply(**kwargs: Any) -> None:  # noqa: ARG001
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("transient db hiccup")
            # Second call succeeds (no-op for test)
            return

        with (
            patch.object(service, "_apply_finalize_billing", new=flaky_apply),
            # Mock asyncio.sleep so the test doesn't actually wait 1s
            patch(
                "src.api.services.gpu_session.service.asyncio.sleep", new=AsyncMock()
            ) as mock_sleep,
        ):
            await service._finalize_billing(session)

        assert call_count == 2, "Expected 2 attempts (1 fail, 1 success)"
        mock_sleep.assert_awaited_once_with(1.0)

    async def test_both_attempts_fail_leaves_finalized_at_null(self) -> None:
        """Both attempts raise → log finalization_deferred, no stamp.

        The phase-2 reconciler picks up sessions where
        ``billing_finalized_at IS NULL AND status='stopped'`` and retries.
        """
        service, mocks = _make_service()

        now = datetime.now(UTC)
        session = _make_gpu_session(
            user_id=uuid4(),
            status=GpuSessionStatus.active,
            started_at=now - timedelta(minutes=10),
            total_paused_seconds=0,
        )
        session.account_id = mocks["account_id"]
        session.stopped_at = now

        call_count = 0

        async def always_fails(**kwargs: Any) -> None:  # noqa: ARG001
            nonlocal call_count
            call_count += 1
            raise RuntimeError("persistent failure")

        with (
            patch.object(service, "_apply_finalize_billing", new=always_fails),
            patch("src.api.services.gpu_session.service.asyncio.sleep", new=AsyncMock()),
        ):
            # Must NOT raise — finalize is best-effort. Failures are
            # handed off to the reconciler via the NULL column value.
            await service._finalize_billing(session)

        assert call_count == 2, "Expected 2 attempts before deferring"
        # mark_billing_finalized is NOT called — column stays NULL for reconciler.
        # (We can't directly assert that here without patching the inner repo,
        # but the absence of a successful _apply_finalize_billing call implies it.)


class TestFinalizeBillingForSession:
    """Public wrapper used by the BillingReconcilerWorker.

    Wraps ``_finalize_billing`` and re-checks the row's ``billing_finalized_at``
    column to return an explicit success/failure bool. Encapsulates the
    success check so the worker doesn't need to peek into private state or
    re-query independently.
    """

    async def test_returns_true_when_billing_finalized_at_set_after_call(self) -> None:
        service, _mocks = _make_service()
        session = _make_gpu_session(user_id=uuid4())
        refreshed = _make_gpu_session(billing_finalized_at=datetime.now(UTC))

        with (
            patch.object(service, "_finalize_billing", new=AsyncMock()),
            patch(_REPO_PATH) as MockRepo,
        ):
            mock_repo = AsyncMock()
            MockRepo.return_value = mock_repo
            mock_repo.get_by_id.return_value = refreshed

            result = await service.finalize_billing_for_session(session)

        assert result is True

    async def test_returns_false_when_column_still_null_after_call(self) -> None:
        service, _ = _make_service()
        session = _make_gpu_session(user_id=uuid4())
        refreshed = _make_gpu_session(billing_finalized_at=None)

        with (
            patch.object(service, "_finalize_billing", new=AsyncMock()),
            patch(_REPO_PATH) as MockRepo,
        ):
            mock_repo = AsyncMock()
            MockRepo.return_value = mock_repo
            mock_repo.get_by_id.return_value = refreshed

            result = await service.finalize_billing_for_session(session)

        assert result is False

    async def test_returns_false_when_session_disappeared(self) -> None:
        """If get_by_id returns None (deleted between calls), report failure."""
        service, _ = _make_service()
        session = _make_gpu_session(user_id=uuid4())

        with (
            patch.object(service, "_finalize_billing", new=AsyncMock()),
            patch(_REPO_PATH) as MockRepo,
        ):
            mock_repo = AsyncMock()
            MockRepo.return_value = mock_repo
            mock_repo.get_by_id.return_value = None

            result = await service.finalize_billing_for_session(session)

        assert result is False


class TestReadMethods:
    async def test_get_session_returns_owned_session(self) -> None:
        service, _ = _make_service()
        user_id = uuid4()
        session = _make_gpu_session(user_id=user_id)

        with patch(_REPO_PATH) as MockRepo:
            mock_repo = AsyncMock()
            MockRepo.return_value = mock_repo
            mock_repo.get_by_id_for_user.return_value = session

            result = await service.get_session(
                session_id=session.id, user_id=user_id, product_id="vex"
            )

        assert result is session
        mock_repo.get_by_id_for_user.assert_called_once_with(session.id, user_id, "vex")

    async def test_get_session_returns_none_for_wrong_user(self) -> None:
        service, _ = _make_service()

        with patch(_REPO_PATH) as MockRepo:
            mock_repo = AsyncMock()
            MockRepo.return_value = mock_repo
            mock_repo.get_by_id_for_user.return_value = None

            result = await service.get_session(
                session_id=uuid4(), user_id=uuid4(), product_id="vex"
            )

        assert result is None

    async def test_get_session_returns_none_for_wrong_product(self) -> None:
        service, _ = _make_service()

        with patch(_REPO_PATH) as MockRepo:
            mock_repo = AsyncMock()
            MockRepo.return_value = mock_repo
            mock_repo.get_by_id_for_user.return_value = None

            result = await service.get_session(
                session_id=uuid4(), user_id=uuid4(), product_id="synthara"
            )

        assert result is None

    async def test_get_active_session_for_model_returns_active_only(self) -> None:
        service, _ = _make_service()
        user_id = uuid4()
        session = _make_gpu_session(user_id=user_id, status=GpuSessionStatus.active)

        with patch(_REPO_PATH) as MockRepo:
            mock_repo = AsyncMock()
            MockRepo.return_value = mock_repo
            mock_repo.get_active_for_model.return_value = session

            result = await service.get_active_session_for_model(
                user_id=user_id,
                product_id="vex",
                model_type=ModelType.AISHA_IMAGE,
            )

        assert result is session
        mock_repo.get_active_for_model.assert_called_once_with(user_id, "vex", "aisha-image")

    async def test_get_active_session_for_model_returns_none_for_paused(self) -> None:
        service, _ = _make_service()

        with patch(_REPO_PATH) as MockRepo:
            mock_repo = AsyncMock()
            MockRepo.return_value = mock_repo
            mock_repo.get_active_for_model.return_value = None

            result = await service.get_active_session_for_model(
                user_id=uuid4(),
                product_id="vex",
                model_type=ModelType.AISHA_IMAGE,
            )

        assert result is None

    async def test_list_user_sessions_excludes_terminal_by_default(self) -> None:
        service, _ = _make_service()
        user_id = uuid4()
        sessions = [_make_gpu_session(user_id=user_id)]

        with patch(_REPO_PATH) as MockRepo:
            mock_repo = AsyncMock()
            MockRepo.return_value = mock_repo
            mock_repo.list_by_user.return_value = sessions

            result = await service.list_user_sessions(user_id=user_id, product_id="vex")

        mock_repo.list_by_user.assert_called_once_with(user_id, "vex", include_terminal=False)
        assert list(result) == sessions

    async def test_list_user_sessions_includes_terminal_when_requested(self) -> None:
        service, _ = _make_service()
        user_id = uuid4()

        with patch(_REPO_PATH) as MockRepo:
            mock_repo = AsyncMock()
            MockRepo.return_value = mock_repo
            mock_repo.list_by_user.return_value = []

            await service.list_user_sessions(
                user_id=user_id, product_id="vex", include_terminal=True
            )

        mock_repo.list_by_user.assert_called_once_with(user_id, "vex", include_terminal=True)


# ---------------------------------------------------------------------------
# Stop flow integration with JobSweepService (Blocker B)
# ---------------------------------------------------------------------------


def _make_sweep_service_mock() -> MagicMock:
    sweep = MagicMock()
    sweep.sweep_session_best_effort = AsyncMock(return_value=None)
    return sweep


def _make_service_with_sweep(**overrides: Any) -> tuple[GpuSessionService, dict[str, Any]]:
    sweep = _make_sweep_service_mock()
    service, mocks = _make_service(job_sweep_service=sweep, **overrides)
    mocks["job_sweep"] = sweep
    return service, mocks


class TestStopWithJobSweep:
    async def test_stop_confirmed_calls_sweep_after_tx1(self) -> None:
        """sweep_session_best_effort is called after TX1 commits status=stopping."""
        service, mocks = _make_service_with_sweep()
        user_id = uuid4()
        session = _make_gpu_session(user_id=user_id, status=GpuSessionStatus.active)

        with patch(_REPO_PATH) as MockRepo:
            mock_repo = AsyncMock()
            MockRepo.return_value = mock_repo
            mock_repo.get_by_id.return_value = session

            await service.stop_session(
                session_id=session.id,
                user_id=user_id,
                product_id=session.product_id,
                confirmed=True,
            )

        mocks["job_sweep"].sweep_session_best_effort.assert_awaited_once()
        call_kwargs = mocks["job_sweep"].sweep_session_best_effort.call_args.kwargs
        assert call_kwargs["session_id"] == session.id
        assert call_kwargs["product_id"] == session.product_id

    async def test_stop_confirmed_sweep_called_before_external_teardown(self) -> None:
        """sweep_session_best_effort is called before destroy_instance."""
        service, mocks = _make_service_with_sweep()
        user_id = uuid4()
        session = _make_gpu_session(user_id=user_id, status=GpuSessionStatus.active)
        call_order: list[str] = []

        async def record_sweep(**_kwargs: Any) -> None:
            call_order.append("sweep")

        async def record_destroy(*_args: Any, **_kwargs: Any) -> None:
            call_order.append("destroy")

        mocks["job_sweep"].sweep_session_best_effort.side_effect = record_sweep
        mocks["vastai_client"].destroy_instance.side_effect = record_destroy

        with patch(_REPO_PATH) as MockRepo:
            mock_repo = AsyncMock()
            MockRepo.return_value = mock_repo
            mock_repo.get_by_id.return_value = session

            await service.stop_session(
                session_id=session.id,
                user_id=user_id,
                product_id=session.product_id,
                confirmed=True,
            )

        assert call_order.index("sweep") < call_order.index("destroy")

    async def test_stop_confirmed_completes_normally_after_sweep(self) -> None:
        """Stop completes and returns stopped session after sweep call."""
        service, mocks = _make_service_with_sweep()
        user_id = uuid4()
        session = _make_gpu_session(user_id=user_id, status=GpuSessionStatus.active)

        with patch(_REPO_PATH) as MockRepo:
            mock_repo = AsyncMock()
            MockRepo.return_value = mock_repo
            mock_repo.get_by_id.return_value = session

            result = await service.stop_session(
                session_id=session.id,
                user_id=user_id,
                product_id=session.product_id,
                confirmed=True,
            )

        assert isinstance(result, GpuSession)
        assert result.status == GpuSessionStatus.stopped
        mocks["vastai_client"].destroy_instance.assert_called_once()

    async def test_stop_without_sweep_service_still_works(self) -> None:
        """Service with no sweep (job_sweep_service=None) stops normally."""
        service, _mocks = _make_service()
        assert service._job_sweep is None
        user_id = uuid4()
        session = _make_gpu_session(user_id=user_id, status=GpuSessionStatus.active)

        with patch(_REPO_PATH) as MockRepo:
            mock_repo = AsyncMock()
            MockRepo.return_value = mock_repo
            mock_repo.get_by_id.return_value = session

            result = await service.stop_session(
                session_id=session.id,
                user_id=user_id,
                product_id=session.product_id,
                confirmed=True,
            )

        assert isinstance(result, GpuSession)
        assert result.status == GpuSessionStatus.stopped


class TestPauseWithJobSweep:
    async def test_pause_rejects_when_in_flight_jobs_present(self) -> None:
        """pause_session raises SessionHasInFlightJobsError when count > 0."""
        from src.api.services.gpu_session.exceptions import SessionHasInFlightJobsError

        service, mocks = _make_service()
        user_id = uuid4()
        session = _make_gpu_session(user_id=user_id, status=GpuSessionStatus.active)

        with (
            patch(_REPO_PATH) as MockRepo,
            patch(_JOB_REPO_PATH) as MockJobRepo,
        ):
            mock_repo = AsyncMock()
            MockRepo.return_value = mock_repo
            mock_repo.get_by_id.return_value = session

            mock_job_repo = AsyncMock()
            MockJobRepo.return_value = mock_job_repo
            mock_job_repo.count_in_flight_for_session.return_value = 2

            with pytest.raises(SessionHasInFlightJobsError) as exc_info:
                await service.pause_session(
                    session_id=session.id,
                    user_id=user_id,
                    product_id=session.product_id,
                )

        assert exc_info.value.in_flight_count == 2
        # Pause was rejected — neither the VastAI client nor the repo update
        # should have been called.
        mocks["vastai_client"].stop_instance.assert_not_called()
        mock_repo.update_status.assert_not_called()

    async def test_pause_succeeds_when_zero_in_flight(self) -> None:
        """pause_session proceeds when count_in_flight_for_session returns 0."""
        service, mocks = _make_service()
        user_id = uuid4()
        session = _make_gpu_session(user_id=user_id, status=GpuSessionStatus.active)

        with (
            patch(_REPO_PATH) as MockRepo,
            patch(_JOB_REPO_PATH) as MockJobRepo,
        ):
            mock_repo = AsyncMock()
            MockRepo.return_value = mock_repo
            mock_repo.get_by_id.return_value = session

            mock_job_repo = AsyncMock()
            MockJobRepo.return_value = mock_job_repo
            mock_job_repo.count_in_flight_for_session.return_value = 0

            result = await service.pause_session(
                session_id=session.id,
                user_id=user_id,
                product_id=session.product_id,
            )

        assert result.status == GpuSessionStatus.paused
        mocks["vastai_client"].stop_instance.assert_called_once()

    async def test_pause_in_flight_error_carries_correct_count(self) -> None:
        from src.api.services.gpu_session.exceptions import SessionHasInFlightJobsError

        service, _ = _make_service()
        user_id = uuid4()
        session = _make_gpu_session(user_id=user_id, status=GpuSessionStatus.active)

        with (
            patch(_REPO_PATH) as MockRepo,
            patch(_JOB_REPO_PATH) as MockJobRepo,
        ):
            mock_repo = AsyncMock()
            MockRepo.return_value = mock_repo
            mock_repo.get_by_id.return_value = session

            mock_job_repo = AsyncMock()
            MockJobRepo.return_value = mock_job_repo
            mock_job_repo.count_in_flight_for_session.return_value = 5

            with pytest.raises(SessionHasInFlightJobsError) as exc_info:
                await service.pause_session(
                    session_id=session.id,
                    user_id=user_id,
                    product_id=session.product_id,
                )

        assert exc_info.value.in_flight_count == 5
        assert exc_info.value.operation == "pause"


# ---------------------------------------------------------------------------
# Regression tests for 2026-05-12: always-stoppable + refund on pre-active
# ---------------------------------------------------------------------------


class TestStopSessionLifecyclePolicy:
    """Regression for 2026-05-12 incident: stop must always succeed; pre-active → full refund."""

    async def test_stop_from_pending_destroys_instance_and_refunds(self) -> None:
        """Operator can stop a stuck pending session and gets a full refund."""
        service, mocks = _make_service()
        user_id = uuid4()
        session = _make_gpu_session(
            user_id=user_id,
            status=GpuSessionStatus.pending,
            started_at=None,
            vastai_instance_id=12345,
        )
        session.account_id = uuid4()

        with patch(_REPO_PATH) as MockRepo:
            mock_repo = AsyncMock()
            MockRepo.return_value = mock_repo
            mock_repo.get_by_id_for_user.return_value = session
            mock_repo.get_by_id.return_value = session

            result = await service.stop_session(
                session_id=session.id,
                user_id=user_id,
                product_id=session.product_id,
                confirmed=True,
            )

        assert isinstance(result, GpuSession)
        assert result.status == GpuSessionStatus.stopped
        # Vast.ai instance must be destroyed
        mocks["vastai_client"].destroy_instance.assert_called_once_with(12345)
        # Full refund (NOT finalize_billing which would charge the 5-min floor)
        mocks["billing_service"].refund.assert_called_once()
        # billing_finalized immediately (no outstanding amount)
        mock_repo.mark_billing_finalized.assert_called_once()

    async def test_stop_from_provisioning_also_refunds(self) -> None:
        """Provisioning sessions are also pre-active — same refund path."""
        service, mocks = _make_service()
        user_id = uuid4()
        session = _make_gpu_session(
            user_id=user_id,
            status=GpuSessionStatus.provisioning,
            started_at=None,
            vastai_instance_id=12345,
        )
        session.account_id = uuid4()

        with patch(_REPO_PATH) as MockRepo:
            mock_repo = AsyncMock()
            MockRepo.return_value = mock_repo
            mock_repo.get_by_id_for_user.return_value = session
            mock_repo.get_by_id.return_value = session

            result = await service.stop_session(
                session_id=session.id,
                user_id=user_id,
                product_id=session.product_id,
                confirmed=True,
            )

        assert isinstance(result, GpuSession)
        assert result.status == GpuSessionStatus.stopped
        mocks["billing_service"].refund.assert_called_once()

    async def test_stop_from_pending_without_account_skips_refund(self) -> None:
        """If no account_id (reservation never made), refund is skipped silently."""
        service, mocks = _make_service()
        user_id = uuid4()
        session = _make_gpu_session(
            user_id=user_id,
            status=GpuSessionStatus.pending,
            started_at=None,
        )
        session.account_id = None  # no billing reservation

        with patch(_REPO_PATH) as MockRepo:
            mock_repo = AsyncMock()
            MockRepo.return_value = mock_repo
            mock_repo.get_by_id_for_user.return_value = session
            mock_repo.get_by_id.return_value = session

            result = await service.stop_session(
                session_id=session.id,
                user_id=user_id,
                product_id=session.product_id,
                confirmed=True,
            )

        assert isinstance(result, GpuSession)
        assert result.status == GpuSessionStatus.stopped
        mocks["billing_service"].refund.assert_not_called()

    async def test_stop_from_active_uses_finalize_billing_not_refund(self) -> None:
        """Sessions that reached active pay usage-based bill, not full refund."""
        service, mocks = _make_service()
        user_id = uuid4()
        session = _make_gpu_session(
            user_id=user_id,
            status=GpuSessionStatus.active,
            started_at=datetime.now(UTC) - timedelta(minutes=10),
        )

        with patch(_REPO_PATH) as MockRepo:
            mock_repo = AsyncMock()
            MockRepo.return_value = mock_repo
            mock_repo.get_by_id_for_user.return_value = session
            mock_repo.get_by_id.return_value = session

            await service.stop_session(
                session_id=session.id,
                user_id=user_id,
                product_id=session.product_id,
                confirmed=True,
            )

        # Full refund must NOT fire for active sessions
        mocks["billing_service"].refund.assert_not_called()

    async def test_stop_from_stopped_is_idempotent_no_work(self) -> None:
        """Stopping an already-stopped session returns the row without any teardown."""
        service, mocks = _make_service()
        user_id = uuid4()
        session = _make_gpu_session(user_id=user_id, status=GpuSessionStatus.stopped)

        with patch(_REPO_PATH) as MockRepo:
            mock_repo = AsyncMock()
            MockRepo.return_value = mock_repo
            mock_repo.get_by_id_for_user.return_value = session

            result = await service.stop_session(
                session_id=session.id,
                user_id=user_id,
                product_id=session.product_id,
                confirmed=True,
            )

        assert result is session
        mocks["vastai_client"].destroy_instance.assert_not_called()
        mocks["cf_client"].delete_session_tunnel.assert_not_called()
        mocks["billing_service"].refund.assert_not_called()
        mock_repo.update_status.assert_not_called()

    async def test_stop_from_failed_is_idempotent(self) -> None:
        """Failed sessions also accept stop as a no-op."""
        service, mocks = _make_service()
        user_id = uuid4()
        session = _make_gpu_session(user_id=user_id, status=GpuSessionStatus.failed)

        with patch(_REPO_PATH) as MockRepo:
            mock_repo = AsyncMock()
            MockRepo.return_value = mock_repo
            mock_repo.get_by_id_for_user.return_value = session

            result = await service.stop_session(
                session_id=session.id,
                user_id=user_id,
                product_id=session.product_id,
                confirmed=True,
            )

        assert result is session
        mocks["vastai_client"].destroy_instance.assert_not_called()
        mock_repo.update_status.assert_not_called()

    async def test_stop_pre_active_refund_failure_does_not_block_transition(self) -> None:
        """Billing refund failure must not prevent the session from being marked stopped."""
        service, mocks = _make_service()
        user_id = uuid4()
        session = _make_gpu_session(
            user_id=user_id,
            status=GpuSessionStatus.pending,
            started_at=None,
        )
        session.account_id = uuid4()
        mocks["billing_service"].refund.side_effect = Exception("billing down")

        with patch(_REPO_PATH) as MockRepo:
            mock_repo = AsyncMock()
            MockRepo.return_value = mock_repo
            mock_repo.get_by_id_for_user.return_value = session
            mock_repo.get_by_id.return_value = session

            result = await service.stop_session(
                session_id=session.id,
                user_id=user_id,
                product_id=session.product_id,
                confirmed=True,
            )

        # Session must be stopped even though refund failed
        assert isinstance(result, GpuSession)
        assert result.status == GpuSessionStatus.stopped

    async def test_stop_pre_active_race_to_active_falls_through_to_normal_stop(self) -> None:
        """If provisioning worker promotes session to active between initial load and lock,
        fall through to the normal _stop_confirmed path — no double-refund."""
        service, mocks = _make_service()
        user_id = uuid4()
        # Initial load: started_at=None → routes to _stop_pre_active
        session_initial = _make_gpu_session(
            user_id=user_id,
            status=GpuSessionStatus.pending,
            started_at=None,
        )
        # Under lock: session raced to active (started_at now set)
        session_locked = _make_gpu_session(
            user_id=user_id,
            status=GpuSessionStatus.active,
            started_at=datetime.now(UTC) - timedelta(minutes=1),
        )
        session_locked.id = session_initial.id

        call_count = 0

        def get_by_id_side_effect(*_args: Any, **_kwargs: Any) -> GpuSession:
            nonlocal call_count
            call_count += 1
            # First call (in _stop_pre_active TX1 for_update) → locked version
            return session_locked

        with patch(_REPO_PATH) as MockRepo:
            mock_repo = AsyncMock()
            MockRepo.return_value = mock_repo
            mock_repo.get_by_id_for_user.return_value = session_initial
            mock_repo.get_by_id.side_effect = get_by_id_side_effect

            result = await service.stop_session(
                session_id=session_initial.id,
                user_id=user_id,
                product_id=session_initial.product_id,
                confirmed=True,
            )

        # Must end up stopped — no error, no double refund
        assert isinstance(result, GpuSession)
        assert result.status == GpuSessionStatus.stopped
        # Full refund must NOT fire — session DID become active
        mocks["billing_service"].refund.assert_not_called()

    async def test_stop_not_found_raises_gpu_session_error(self) -> None:
        """Session not found → GpuSessionError (not unhandled None)."""
        from src.api.services.gpu_session.exceptions import GpuSessionError

        service, _ = _make_service()

        with patch(_REPO_PATH) as MockRepo:
            mock_repo = AsyncMock()
            MockRepo.return_value = mock_repo
            mock_repo.get_by_id_for_user.return_value = None

            with pytest.raises(GpuSessionError):
                await service.stop_session(
                    session_id=uuid4(),
                    user_id=uuid4(),
                    product_id="vex",
                    confirmed=True,
                )

    async def test_stop_confirmed_raced_to_failed_returns_idempotently(self) -> None:
        """If the session transitions to 'failed' between stop_session's routing read
        and _stop_confirmed's lock acquisition, return the failed row idempotently
        rather than raising InvalidSessionStateError.

        Regression for Sourcery review: 'stop always succeeds' must hold even
        when an unrelated worker terminates the session during the race window."""
        service, mocks = _make_service()
        user_id = uuid4()
        routing_state = _make_gpu_session(
            user_id=user_id,
            status=GpuSessionStatus.active,
            started_at=datetime.now(UTC) - timedelta(minutes=10),
        )
        terminal_state = _make_gpu_session(
            user_id=user_id,
            status=GpuSessionStatus.failed,
            started_at=routing_state.started_at,
        )
        terminal_state.id = routing_state.id

        with patch(_REPO_PATH) as MockRepo:
            mock_repo = AsyncMock()
            MockRepo.return_value = mock_repo
            # Routing read returns active; locked read returns failed.
            mock_repo.get_by_id_for_user.return_value = routing_state
            mock_repo.get_by_id.return_value = terminal_state

            result = await service.stop_session(
                session_id=routing_state.id,
                user_id=user_id,
                product_id=routing_state.product_id,
                confirmed=True,
            )

        assert isinstance(result, GpuSession)
        assert result.status == GpuSessionStatus.failed
        # No teardown: the row was already terminal under lock.
        mocks["vastai_client"].destroy_instance.assert_not_called()
        mocks["cf_client"].delete_session_tunnel.assert_not_called()
        mocks["billing_service"].refund.assert_not_called()
        mocks["billing_service"].partial_refund.assert_not_called()

    async def test_stop_confirmed_raced_to_stopped_returns_idempotently(self) -> None:
        """Symmetric to the failed case: a concurrent stop completed before our lock.
        Return the already-stopped row; do no teardown and no billing."""
        service, mocks = _make_service()
        user_id = uuid4()
        routing_state = _make_gpu_session(
            user_id=user_id,
            status=GpuSessionStatus.active,
            started_at=datetime.now(UTC) - timedelta(minutes=10),
        )
        terminal_state = _make_gpu_session(
            user_id=user_id,
            status=GpuSessionStatus.stopped,
            started_at=routing_state.started_at,
        )
        terminal_state.id = routing_state.id

        with patch(_REPO_PATH) as MockRepo:
            mock_repo = AsyncMock()
            MockRepo.return_value = mock_repo
            mock_repo.get_by_id_for_user.return_value = routing_state
            mock_repo.get_by_id.return_value = terminal_state

            result = await service.stop_session(
                session_id=routing_state.id,
                user_id=user_id,
                product_id=routing_state.product_id,
                confirmed=True,
            )

        assert isinstance(result, GpuSession)
        assert result.status == GpuSessionStatus.stopped
        mocks["vastai_client"].destroy_instance.assert_not_called()
        mocks["cf_client"].delete_session_tunnel.assert_not_called()
        mocks["billing_service"].refund.assert_not_called()
        mocks["billing_service"].partial_refund.assert_not_called()

    async def test_stop_pre_active_raced_to_terminal_returns_idempotently(self) -> None:
        """If the provisioning worker marks a pending session 'failed' between
        stop_session's routing read and _stop_pre_active's lock, return the
        failed row idempotently — no refund, no TX2."""
        service, mocks = _make_service()
        user_id = uuid4()
        routing_state = _make_gpu_session(
            user_id=user_id,
            status=GpuSessionStatus.pending,
            started_at=None,
        )
        terminal_state = _make_gpu_session(
            user_id=user_id,
            status=GpuSessionStatus.failed,
            started_at=None,
        )
        terminal_state.id = routing_state.id

        with patch(_REPO_PATH) as MockRepo:
            mock_repo = AsyncMock()
            MockRepo.return_value = mock_repo
            mock_repo.get_by_id_for_user.return_value = routing_state
            mock_repo.get_by_id.return_value = terminal_state

            result = await service.stop_session(
                session_id=routing_state.id,
                user_id=user_id,
                product_id=routing_state.product_id,
                confirmed=True,
            )

        assert isinstance(result, GpuSession)
        assert result.status == GpuSessionStatus.failed
        # Pre-active terminal race: no refund, no external teardown.
        mocks["billing_service"].refund.assert_not_called()
        mocks["vastai_client"].destroy_instance.assert_not_called()


# ---------------------------------------------------------------------------
# _destroy_with_retry — regression for 2026-05-13 incident
# ---------------------------------------------------------------------------


class TestDestroyWithRetry:
    async def test_succeeds_on_first_attempt(self) -> None:
        service, mocks = _make_service()
        mocks["settings"].vastai_destroy_retry_attempts = 3
        mocks["vastai_client"].destroy_instance = AsyncMock(return_value=None)

        with patch(_REPO_PATH):
            result = await service._destroy_with_retry(12345, log_prefix="test", session_id=uuid4())

        assert result is True
        assert mocks["vastai_client"].destroy_instance.call_count == 1

    async def test_recovers_on_second_attempt(self) -> None:
        """Transient failure on first attempt recovers via in-flow retry."""
        service, mocks = _make_service()
        mocks["settings"].vastai_destroy_retry_attempts = 3
        mocks["vastai_client"].destroy_instance = AsyncMock(
            side_effect=[VastAIError("transient", status_code=500), None]
        )

        with patch("src.api.services.gpu_session.service.asyncio.sleep", new_callable=AsyncMock):
            result = await service._destroy_with_retry(12345, log_prefix="test", session_id=uuid4())

        assert result is True
        assert mocks["vastai_client"].destroy_instance.call_count == 2

    async def test_exhausts_retries_returns_false(self) -> None:
        """Sustained failure exhausts all attempts and returns False."""
        service, mocks = _make_service()
        mocks["settings"].vastai_destroy_retry_attempts = 3
        mocks["vastai_client"].destroy_instance = AsyncMock(
            side_effect=VastAIError("persistent", status_code=500)
        )

        with patch("src.api.services.gpu_session.service.asyncio.sleep", new_callable=AsyncMock):
            result = await service._destroy_with_retry(12345, log_prefix="test", session_id=uuid4())

        assert result is False
        assert mocks["vastai_client"].destroy_instance.call_count == 3

    async def test_teardown_marks_destroyed_on_success(self) -> None:
        """On successful destroy, vastai_instance_destroyed_at is stamped in DB."""
        service, mocks = _make_service()
        mocks["settings"].vastai_destroy_retry_attempts = 3
        session = _make_gpu_session(status=GpuSessionStatus.active)
        mocks["vastai_client"].destroy_instance = AsyncMock(return_value=None)
        mocks["cf_client"].delete_session_tunnel = AsyncMock(return_value=None)

        with patch(_REPO_PATH) as MockRepo:
            mock_repo = AsyncMock()
            MockRepo.return_value = mock_repo
            mock_repo.mark_instance_destroyed = AsyncMock(return_value=None)

            had_errors = await service._teardown_external_resources(session, log_prefix="test")

        assert had_errors is False
        mock_repo.mark_instance_destroyed.assert_called_once()

    async def test_teardown_does_not_mark_on_failure(self) -> None:
        """When destroy exhausts retries, DB is NOT stamped — orphan sweeper will retry."""
        service, mocks = _make_service()
        mocks["settings"].vastai_destroy_retry_attempts = 3
        session = _make_gpu_session(status=GpuSessionStatus.active)
        mocks["vastai_client"].destroy_instance = AsyncMock(
            side_effect=VastAIError("persistent", status_code=500)
        )

        with patch(_REPO_PATH) as MockRepo:
            mock_repo = AsyncMock()
            MockRepo.return_value = mock_repo
            mock_repo.mark_instance_destroyed = AsyncMock(return_value=None)

            with patch(
                "src.api.services.gpu_session.service.asyncio.sleep", new_callable=AsyncMock
            ):
                had_errors = await service._teardown_external_resources(session, log_prefix="test")

        assert had_errors is True
        mock_repo.mark_instance_destroyed.assert_not_called()

    async def test_retry_exception_binding_does_not_suppress_return_false(self) -> None:
        """Binding `except Exception as exc` must not change the return value on exhaustion."""
        service, mocks = _make_service()
        mocks["settings"].vastai_destroy_retry_attempts = 2
        error = VastAIError("auth failed", status_code=401)
        mocks["vastai_client"].destroy_instance = AsyncMock(side_effect=error)

        with patch("src.api.services.gpu_session.service.asyncio.sleep", new_callable=AsyncMock):
            result = await service._destroy_with_retry(12345, log_prefix="test", session_id=uuid4())

        assert result is False
        assert mocks["vastai_client"].destroy_instance.call_count == 2


# ---------------------------------------------------------------------------
# TestStartSessionReadinessMarkerAndTemplateHash — Step 14 tests
# ---------------------------------------------------------------------------


class TestStartSessionReadinessMarkerAndTemplateHash:
    """Verify that readiness_marker_node_class and template_hash_id flow through start_session."""

    async def _call_start_session(
        self,
        service: GpuSessionService,
        mocks: dict[str, Any],
        bundle: BundleMapping,
    ) -> GpuSession:
        mock_repo = AsyncMock()
        mock_repo.get_non_terminal_for_model.return_value = None
        session_row = _make_gpu_session(status=GpuSessionStatus.pending)
        mock_repo.create.return_value = session_row
        mocks["bundle_index"].resolve_bundle.return_value = bundle
        mocks["cf_client"].create_session_tunnel.return_value = _TUNNEL_RESULT
        mocks["vastai_client"].search_offers.return_value = [_make_offer()]
        mocks["vastai_client"].create_instance.return_value = 99999

        with patch(_REPO_PATH) as MockRepo:
            MockRepo.return_value = mock_repo
            result = await service.start_session(
                user_id=uuid4(),
                product_id="vex",
                model_type=ModelType.AISHA_IMAGE,
                account_id=mocks["account_id"],
            )
        return result, mock_repo  # type: ignore[return-value]

    async def test_persists_readiness_marker_when_bundle_has_one(self) -> None:
        service, mocks = _make_service()
        marker = ReadinessMarker(node_class="WanVideoSampler")
        bundle = _make_bundle_mapping(readiness_marker=marker)

        _, mock_repo = await self._call_start_session(service, mocks, bundle)  # type: ignore[misc]

        create_kwargs = mock_repo.create.call_args[1]
        assert create_kwargs["readiness_marker_node_class"] == "WanVideoSampler"

    async def test_persists_none_when_bundle_has_no_marker(self) -> None:
        service, mocks = _make_service()
        bundle = _make_bundle_mapping(readiness_marker=None)

        _, mock_repo = await self._call_start_session(service, mocks, bundle)  # type: ignore[misc]

        create_kwargs = mock_repo.create.call_args[1]
        assert create_kwargs["readiness_marker_node_class"] is None

    async def test_passes_template_hash_to_create_instance(self) -> None:
        service, mocks = _make_service()
        bundle = _make_bundle_mapping(template_hash_id="abc123templatehash")

        await self._call_start_session(service, mocks, bundle)  # type: ignore[misc]

        mocks["vastai_client"].create_instance.assert_called_once()
        call_kwargs = mocks["vastai_client"].create_instance.call_args[1]
        assert call_kwargs.get("template_hash_id") == "abc123templatehash"
        assert "image" not in call_kwargs
        assert "onstart_cmd" not in call_kwargs
