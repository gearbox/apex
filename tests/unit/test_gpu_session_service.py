"""Unit tests for GpuSessionService.

External clients (VastAIClient, CloudflareTunnelClient, BundleIndexService) are
mocked with AsyncMock/MagicMock. The session factory is mocked so no real DB is
required; GpuSessionRepository is patched at the service module level.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import ANY, AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from sqlalchemy.exc import IntegrityError

from src.api.schemas.events import EventType, GpuSessionStatusPayload
from src.api.services.bundle_index import BundleIndexService
from src.api.services.cloudflare.client import CloudflareTunnelClient
from src.api.services.gpu_session import (
    GpuSessionService,
    InvalidSessionStateError,
    SessionAlreadyExistsError,
    StopConfirmation,
    billable_minutes_for_active_seconds,
)
from src.api.services.vastai.client import VastAIClient
from src.api.services.vastai.exceptions import NoCapacityError, OfferTakenError, VastAIError
from src.api.services.vastai.schemas import VastAIOffer
from src.core.bundle_config import BundleMapping, HardwareRequirements
from src.core.enums import GpuSessionStatus, ModelType
from src.db.models.gpu_session import GpuSession

_REPO_PATH = "src.api.services.gpu_session.service.GpuSessionRepository"


# ---------------------------------------------------------------------------
# Helper builders
# ---------------------------------------------------------------------------


def _make_billing_mock() -> MagicMock:
    """Return a MagicMock whose async billing methods are explicit AsyncMocks.

    Plain AsyncMock() refuses attributes whose names start with "assert" (Python
    mock treats them as spy-assertion calls). By building the mock manually we
    avoid the AttributeError for assert_sufficient_balance.
    """
    m = MagicMock()
    m.assert_sufficient_balance = AsyncMock(return_value=None)
    m.check_and_reserve = AsyncMock(return_value=None)
    m.refund = AsyncMock(return_value=None)
    m.partial_refund = AsyncMock(return_value=None)
    return m


def _make_hardware() -> HardwareRequirements:
    return HardwareRequirements(
        gpu_whitelist=("RTX_4090",),
        min_disk_gb=100,
        min_network_upload_mbps=100,
        min_network_download_mbps=500,
        cuda_min_version="12.1",
        num_gpus=1,
        comfyui_port=18188,
    )


def _make_bundle_mapping(
    *,
    bundle_name: str = "wan_2.2_i2v",
    bundle_version: str | None = "260105-01",
) -> BundleMapping:
    return BundleMapping(
        bundle_name=bundle_name,
        bundle_version=bundle_version,
        hardware=_make_hardware(),
    )


def _make_offer(*, offer_id: int = 1001, dph_total: float = 0.5) -> VastAIOffer:
    return VastAIOffer(
        id=offer_id,
        gpu_name="RTX_4090",
        num_gpus=1,
        gpu_ram=24,
        disk_space=100.0,
        dph_total=dph_total,
        inet_up=100.0,
        inet_down=500.0,
        cuda_max_good=12.1,
        verified=True,
    )


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
    session.tunnel_hostname = "01234567.gpu.cloudin.space"
    session.callback_token = "test-callback-token"
    session.stale_detected_at = None
    session.stale_notified = False
    session.paused_at = None
    session.resumed_at = None
    session.stopped_at = None
    session.started_at = now - timedelta(hours=2)
    session.created_at = now - timedelta(hours=3)
    session.error_message = None
    session.node_host = None
    session.node_port = None
    session.provision_attempt = 1
    session.total_paused_seconds = 0
    session.account_id = uuid4()
    for k, v in kwargs.items():
        setattr(session, k, v)
    return session


def _make_settings(*, max_retries: int = 3) -> MagicMock:
    settings = MagicMock()
    settings.max_node_provisioning_retries = max_retries
    settings.apex_callback_url = "https://apex.example.com/callback"
    settings.hf_token = "test-hf-token"
    settings.civitai_api_token = "test-civitai-token"
    settings.gpu_session_base_reservation_tokens = 500
    settings.gpu_session_tokens_per_minute = 100
    return settings


def _make_mock_session_factory() -> tuple[MagicMock, MagicMock]:
    """Return (factory, session) mocks that support async with patterns."""
    mock_session = MagicMock()
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=None)

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
        "account_id": uuid4(),
        "event_bus": None,
    } | overrides
    service = GpuSessionService(
        vastai_client=mocks["vastai_client"],
        cf_client=mocks["cf_client"],
        bundle_index=mocks["bundle_index"],
        session_factory=mocks["session_factory"],
        settings=mocks["settings"],
        billing_service=mocks["billing_service"],
        event_bus=mocks["event_bus"],
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

        with patch(_REPO_PATH) as MockRepo:
            mock_repo = AsyncMock()
            MockRepo.return_value = mock_repo
            mock_repo.get_non_terminal_for_model.return_value = None
            mock_repo.create.return_value = expected_session

            result = await service.start_session(
                user_id=user_id,
                product_id="vex",
                model_type=ModelType.AISHA_IMAGE,
                account_id=mocks["account_id"],
                billing_service=mocks["billing_service"],
            )

        assert result is expected_session
        mocks["bundle_index"].resolve_bundle.assert_called_once_with("aisha-image")
        mock_repo.get_non_terminal_for_model.assert_called_once_with(user_id, "vex", "aisha-image")
        mocks["cf_client"].create_session_tunnel.assert_called_once()
        mocks["vastai_client"].search_offers.assert_called_once_with(bundle.hardware)
        mocks["vastai_client"].create_instance.assert_called_once_with(
            offer.id,
            image="vastai/comfy:latest",
            disk_gb=bundle.hardware.min_disk_gb,
            env=ANY,
            onstart_cmd=ANY,
        )
        mock_repo.create.assert_called_once()

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
                billing_service=mocks["billing_service"],
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
                billing_service=mocks["billing_service"],
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
                    billing_service=mocks["billing_service"],
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
                    billing_service=mocks["billing_service"],
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
                    billing_service=mocks["billing_service"],
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
                    billing_service=mocks["billing_service"],
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
        # Custom port — different from the default 18188
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
                billing_service=mocks["billing_service"],
            )

        # Inspect the env dict passed to create_instance
        env_arg = mocks["vastai_client"].create_instance.call_args.kwargs["env"]
        assert "-p 9999:9999" in env_arg
        assert "-p 18188:18188" not in env_arg
        # Also verify CF tunnel was configured for the same port
        mocks["cf_client"].create_session_tunnel.assert_called_once_with(ANY, 9999)

    async def test_offer_taken_retries_with_next_offer(self) -> None:
        service, mocks = _make_service(settings=_make_settings(max_retries=3))
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
                billing_service=mocks["billing_service"],
            )

        assert result.status == GpuSessionStatus.pending
        assert mocks["vastai_client"].create_instance.call_count == 2
        # Second offer was used
        second_call_offer_id = mocks["vastai_client"].create_instance.call_args_list[1][0][0]
        assert second_call_offer_id == offer2.id

    async def test_retries_bounded_by_settings(self) -> None:
        service, mocks = _make_service(settings=_make_settings(max_retries=2))
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
                    billing_service=mocks["billing_service"],
                )

        # Exhausted 2 retries (max_retries=2, 2 offers)
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
                    billing_service=mocks["billing_service"],
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

    async def test_zero_max_retries_raises_config_error(self) -> None:
        """max_node_provisioning_retries=0 is a misconfiguration, not 'all taken'.

        Settings enforces ge=1 via Pydantic, but defense-in-depth in the service
        gives a clearer error if a misconstructed Settings slips through.
        """
        service, mocks = _make_service(settings=_make_settings(max_retries=0))
        mocks["bundle_index"].resolve_bundle.return_value = _make_bundle_mapping()
        mocks["cf_client"].create_session_tunnel.return_value = _TUNNEL_RESULT
        mocks["vastai_client"].search_offers.return_value = [_make_offer()]

        with patch(_REPO_PATH) as MockRepo:
            mock_repo = AsyncMock()
            MockRepo.return_value = mock_repo
            mock_repo.get_non_terminal_for_model.return_value = None

            with pytest.raises(VastAIError, match="max_node_provisioning_retries must be >= 1"):
                await service.start_session(
                    user_id=uuid4(),
                    product_id="vex",
                    model_type=ModelType.AISHA_IMAGE,
                    account_id=mocks["account_id"],
                    billing_service=mocks["billing_service"],
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
                    billing_service=mocks["billing_service"],
                )

        # Find the gpu_session.start.failed call and verify error_class matches
        failed_calls = [
            call
            for call in mock_logger.error.call_args_list
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
                    billing_service=mocks["billing_service"],
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
                billing_service=mocks["billing_service"],
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
        assert "-p 18188:18188" in env
        assert env["-p 18188:18188"] == "1"

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
                billing_service=mocks["billing_service"],
            )
            await service.start_session(
                user_id=uuid4(),
                product_id="vex",
                model_type=ModelType.AISHA_IMAGE,
                account_id=mocks["account_id"],
                billing_service=mocks["billing_service"],
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
                billing_service=mocks["billing_service"],
            )

        _, kwargs = mocks["vastai_client"].create_instance.call_args
        assert kwargs["env"]["ACS_BUNDLE_VERSION"] == "current"

    async def test_start_session_insufficient_balance_raises_before_tunnel_creation(self) -> None:
        from src.api.services.billing_errors import InsufficientBalanceError

        service, mocks = _make_service()
        mocks["bundle_index"].resolve_bundle.return_value = _make_bundle_mapping()
        mocks["billing_service"].assert_sufficient_balance.side_effect = InsufficientBalanceError(  # type: ignore[union-attr]
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
                    billing_service=mocks["billing_service"],
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
                billing_service=mocks["billing_service"],
            )

        mocks["billing_service"].check_and_reserve.assert_called_once()
        call_args = mocks["billing_service"].check_and_reserve.call_args
        # Positional: (account_id, token_cost, session_id)
        assert call_args.args[0] == mocks["account_id"]
        assert call_args.args[1] == 500  # gpu_session_base_reservation_tokens

        # The session_id passed to check_and_reserve must equal the id passed to repo.create
        # (both come from the same new_id() call inside start_session)
        create_kwargs = mock_repo.create.call_args.kwargs
        assert call_args.args[2] == create_kwargs["id"]

        # Keyword wiring — product_id, user_id, metadata all flow through
        assert call_args.kwargs["product_id"] == "vex"
        assert call_args.kwargs["user_id"] == user_id
        metadata = call_args.kwargs["metadata"]
        assert metadata["type"] == "gpu_session_base"
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
                    billing_service=mocks["billing_service"],
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
                billing_service=mocks["billing_service"],
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
        assert payload.bundle_name == expected_session.bundle_name
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
                billing_service=mocks["billing_service"],
            )

        assert result is expected_session


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

        with patch(_REPO_PATH) as MockRepo:
            mock_repo = AsyncMock()
            MockRepo.return_value = mock_repo
            mock_repo.get_by_id.return_value = session

            result = await service.pause_session(
                session_id=session_id,
                user_id=user_id,
                product_id=product_id,
            )

        mocks["vastai_client"].stop_instance.assert_called_once_with(session.vastai_instance_id)
        assert result.status == GpuSessionStatus.paused

    async def test_sets_paused_at(self) -> None:
        service, mocks = _make_service()
        user_id = uuid4()
        product_id = "vex"
        session = _make_gpu_session(user_id=user_id, product_id=product_id, paused_at=None)

        with patch(_REPO_PATH) as MockRepo:
            mock_repo = AsyncMock()
            MockRepo.return_value = mock_repo
            mock_repo.get_by_id.return_value = session

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

        with patch(_REPO_PATH) as MockRepo:
            mock_repo = AsyncMock()
            MockRepo.return_value = mock_repo
            mock_repo.get_by_id.return_value = session

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
        service, mocks = _make_service()
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
        assert result.bundle_name == session.bundle_name
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

    async def test_unconfirmed_duration_uses_created_at_when_no_started_at(self) -> None:
        service, _ = _make_service()
        user_id = uuid4()
        now = datetime.now(UTC)
        created = now - timedelta(hours=3)
        session = _make_gpu_session(
            user_id=user_id,
            status=GpuSessionStatus.active,
            started_at=None,
            created_at=created,
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
        service, mocks = _make_service()
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
        service, mocks = _make_service()
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
        "bad_status",
        [
            GpuSessionStatus.pending,
            GpuSessionStatus.provisioning,
            GpuSessionStatus.resuming,
            GpuSessionStatus.stopping,
            GpuSessionStatus.stopped,
            GpuSessionStatus.failed,
        ],
    )
    async def test_from_non_stoppable_raises(self, bad_status: GpuSessionStatus) -> None:
        service, _ = _make_service()
        user_id = uuid4()
        session = _make_gpu_session(user_id=user_id, status=bad_status)

        with patch(_REPO_PATH) as MockRepo:
            mock_repo = AsyncMock()
            MockRepo.return_value = mock_repo
            # unconfirmed call uses get_by_id_for_user
            mock_repo.get_by_id_for_user.return_value = session

            with pytest.raises(InvalidSessionStateError) as exc_info:
                await service.stop_session(
                    session_id=session.id,
                    user_id=user_id,
                    product_id=session.product_id,
                    confirmed=False,
                )

        assert exc_info.value.current_status == bad_status
        assert exc_info.value.operation == "stop"


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
    ) -> tuple[MagicMock, GpuSession]:
        """Drive a confirmed stop and return the mock billing service + session."""
        settings = _make_settings()
        settings.gpu_session_base_reservation_tokens = 500
        settings.gpu_session_tokens_per_minute = tokens_per_minute

        service, mocks = _make_service(settings=settings)
        user_id = uuid4()
        session = _make_gpu_session(
            user_id=user_id,
            status=GpuSessionStatus.active,
            started_at=started_at,
            total_paused_seconds=total_paused_seconds,
        )

        # Freeze time so stopped_at is predictable
        with (
            patch(_REPO_PATH) as MockRepo,
            patch("src.api.services.gpu_session.service.datetime") as mock_dt,
        ):
            mock_dt.now.return_value = stopped_now
            # datetime.fromisoformat / UTC / timedelta still need to work
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)

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

        billing.check_and_reserve.assert_awaited_once()
        args = billing.check_and_reserve.await_args.args
        kwargs = billing.check_and_reserve.await_args.kwargs
        assert args[0] == session.account_id
        assert args[1] == 500  # overage = 1000 billable - 500 reserved
        # CRITICAL: overage debit must NOT reuse the session's job_id — that would
        # create two debits with the same job_id and break get_debit_for_job().
        assert args[2] is None
        metadata = kwargs["metadata"]
        assert metadata["type"] == "gpu_session_overage"
        assert metadata["session_id"] == str(session.id)  # link via metadata
        assert metadata["billable_minutes"] == 10
        assert metadata["bundle_name"] == session.bundle_name
        assert metadata["model_type"] == session.model_type
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

        billing.check_and_reserve.assert_awaited_once()
        args = billing.check_and_reserve.await_args.args
        assert args[1] == 100  # overage = (6 × 100) - 500 reserved
        metadata = billing.check_and_reserve.await_args.kwargs["metadata"]
        assert metadata["billable_minutes"] == 6
        # Overage debit not linked to a job
        assert args[2] is None
        assert metadata["session_id"] == str(session.id)

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
        billing.check_and_reserve.assert_not_called()

    async def test_exact_match_creates_neither_debit_nor_refund(self) -> None:
        """5 min active @ 100 tok/min = 500 tokens, reserved 500 → no-op."""
        now = datetime.now(UTC)
        started = now - timedelta(minutes=5)

        billing, _ = await self._run_stop_and_capture_billing(
            started_at=started,
            stopped_now=now,
            total_paused_seconds=0,
        )

        billing.check_and_reserve.assert_not_called()
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
        billing.check_and_reserve.assert_not_called()
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

        billing.check_and_reserve.assert_awaited_once()
        args = billing.check_and_reserve.await_args.args
        assert args[1] == 1500  # 2000 billable - 500 reserved
        assert args[2] is None  # overage debit still not linked by job_id
        kwargs = billing.check_and_reserve.await_args.kwargs
        assert kwargs["metadata"]["billable_minutes"] == 20
        assert kwargs["metadata"]["session_id"] == str(session.id)


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
        service, mocks = _make_service()
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
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
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
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
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
# Read method tests
# ---------------------------------------------------------------------------


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
