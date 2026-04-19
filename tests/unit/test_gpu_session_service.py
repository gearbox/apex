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

from src.api.services.bundle_index import BundleIndexService
from src.api.services.cloudflare.client import CloudflareTunnelClient
from src.api.services.gpu_session import (
    GpuSessionService,
    InvalidSessionStateError,
    SessionAlreadyExistsError,
    StopConfirmation,
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
    for k, v in kwargs.items():
        setattr(session, k, v)
    return session


def _make_settings(*, max_retries: int = 3) -> MagicMock:
    settings = MagicMock()
    settings.max_node_provisioning_retries = max_retries
    settings.apex_callback_url = "https://apex.example.com/callback"
    settings.hf_token = "test-hf-token"
    settings.civitai_api_token = "test-civitai-token"
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
    } | overrides
    service = GpuSessionService(
        vastai_client=mocks["vastai_client"],
        cf_client=mocks["cf_client"],
        bundle_index=mocks["bundle_index"],
        session_factory=mocks["session_factory"],
        settings=mocks["settings"],
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
                )

        mocks["cf_client"].delete_session_tunnel.assert_called_once_with(
            _TUNNEL_RESULT[0], _TUNNEL_RESULT[2]
        )

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
                )

        # Exhausted 2 retries (max_retries=2, 2 offers)
        assert mocks["vastai_client"].create_instance.call_count == 2
        # Tunnel was cleaned up
        mocks["cf_client"].delete_session_tunnel.assert_called_once_with(
            _TUNNEL_RESULT[0], _TUNNEL_RESULT[2]
        )

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
                user_id=uuid4(), product_id="vex", model_type=ModelType.AISHA_IMAGE
            )
            await service.start_session(
                user_id=uuid4(), product_id="vex", model_type=ModelType.AISHA_IMAGE
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
                user_id=uuid4(), product_id="vex", model_type=ModelType.AISHA_IMAGE
            )

        _, kwargs = mocks["vastai_client"].create_instance.call_args
        assert kwargs["env"]["ACS_BUNDLE_VERSION"] == "current"


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
