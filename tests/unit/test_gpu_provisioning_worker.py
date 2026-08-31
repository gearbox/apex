"""Unit tests for GpuProvisioningWorker."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import httpx

from src.api.schemas.events import EventType, GpuSessionStatusPayload
from src.api.services.bundle_index import BundleIndexService
from src.api.services.cloudflare.client import CloudflareTunnelClient
from src.api.services.gpu_session.node_cooldown import NodeCooldownStore, NullNodeCooldownStore
from src.api.services.gpu_session.provisioning_worker import (
    _REASON_PENDING_TIMEOUT,
    _REASON_PENDING_TIMEOUT_AFTER_ERRORS,
    _REASON_PROVISIONING_TIMEOUT,
    GpuProvisioningWorker,
    _classify_terminal_state,
    _match_checkpoint,
)
from src.api.services.vastai.client import VastAIClient
from src.api.services.vastai.exceptions import InstanceNotFoundError, VastAIError
from src.api.services.vastai.schemas import VastAIInstance, VastAIOffer
from src.core.bundle_config import BundleMapping, HardwareRequirements
from src.core.enums import GpuSessionStatus
from src.db.models.gpu_session import GpuSession

_DEFAULT_COMFYUI_PORT: int = HardwareRequirements.__dataclass_fields__["comfyui_port"].default

_REPO_PATH = "src.api.services.gpu_session.provisioning_worker.GpuSessionRepository"


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


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


def _make_bundle_mapping() -> BundleMapping:
    return BundleMapping(
        bundle_name="wan_2.2_i2v",
        bundle_version="260105-01",
        hardware=_make_hardware(),
    )


def _make_offer(*, offer_id: int = 1001, dph_total: float = 0.5) -> VastAIOffer:
    return VastAIOffer(id=offer_id, gpu_name="RTX_4090", num_gpus=1, dph_total=dph_total)


def _make_gpu_session(**kwargs: Any) -> GpuSession:
    import hashlib

    now = datetime.now(UTC)
    session = GpuSession()
    session.id = uuid4()
    session.user_id = uuid4()
    session.product_id = "vex"
    session.status = GpuSessionStatus.pending
    session.bundle_name = "wan_2.2_i2v"
    session.bundle_version = "260105-01"
    session.model_type = "aisha-image"
    session.vastai_instance_id = 12345
    session.vastai_offer_id = 67890
    session.vastai_cost_per_hour_micros = 500_000
    session.vastai_gpu_name = "RTX_4090"
    session.cf_tunnel_id = "tunnel-abc"
    session.cf_dns_record_id = "dns-abc"
    session.tunnel_hostname = "gpu-01234567.gpu-domain.com"
    session.callback_token_hash = hashlib.sha256(b"callback-tok").hexdigest()
    session.provision_attempt = 1
    session.provisioning_started_at = None
    session.started_at = None
    session.paused_at = None
    session.resumed_at = None
    session.stopped_at = None
    session.created_at = now - timedelta(minutes=2)
    session.error_message = None
    session.stale_detected_at = None
    session.stale_notified = False
    session.provisioning_phase = None
    session.provisioning_progress = None
    session.last_progress_at = None
    for k, v in kwargs.items():
        setattr(session, k, v)
    return session


def _make_settings(**overrides: Any) -> MagicMock:
    settings = MagicMock()
    settings.gpu_provision_poll_interval_seconds = 15
    settings.gpu_provision_timeout_seconds = (
        1200  # 20 min in seconds — matches timeout_minutes default
    )
    settings.gpu_provision_stall_timeout_seconds = 420
    settings.gpu_resume_timeout_minutes = 5
    settings.provisioning_offer_walk_depth = 10
    settings.provisioning_recreation_attempts = 1
    settings.vastai_offer_search_limit = 20
    settings.gpu_provision_worker_concurrency = 10
    settings.apex_callback_url = "https://apex.example.com/callback"
    settings.hf_token = "hf-tok"
    settings.civitai_api_token = "civitai-tok"
    settings.aisha_cf_tunnel_domain = "gpu-domain.com"
    settings.ai_bundles_github_token = "ghp_test_token"
    settings.ai_bundles_repo_url = "https://github.com/gearbox/ai-bundles.git"
    settings.ai_bundles_branch = "master"
    settings.aisha_repo_url = "https://github.com/gearbox/aisha.git"
    settings.aisha_branch = "master"
    settings.aisha_comfyui_host = "0.0.0.0"  # noqa: S104
    settings.aisha_comfyui_extra_args = ""
    for k, v in overrides.items():
        setattr(settings, k, v)
    return settings


def _make_mock_session_factory() -> tuple[MagicMock, MagicMock]:
    """Return (factory, db_session) mocks supporting 'async with factory() as db, db.begin()'."""
    mock_db = MagicMock()
    mock_db.__aenter__ = AsyncMock(return_value=mock_db)
    mock_db.__aexit__ = AsyncMock(return_value=None)

    mock_begin = MagicMock()
    mock_begin.__aenter__ = AsyncMock(return_value=None)
    mock_begin.__aexit__ = AsyncMock(return_value=None)
    mock_db.begin = MagicMock(return_value=mock_begin)

    mock_factory = MagicMock(return_value=mock_db)
    return mock_factory, mock_db


def _make_worker(**overrides: Any) -> tuple[GpuProvisioningWorker, dict[str, Any]]:
    mock_factory, mock_db = _make_mock_session_factory()
    mocks: dict[str, Any] = {
        "session_factory": mock_factory,
        "mock_db": mock_db,
        "vastai_client": AsyncMock(spec=VastAIClient),
        "cf_client": AsyncMock(spec=CloudflareTunnelClient),
        "bundle_index": MagicMock(spec=BundleIndexService),
        "http_client": AsyncMock(spec=httpx.AsyncClient),
        "settings": _make_settings(),
        "cooldown_store": NullNodeCooldownStore(),
        "billing_service": None,
        "event_bus": None,
        "job_sweep_service": None,
        "redis_client_factory": MagicMock(),
    } | overrides
    worker = GpuProvisioningWorker(
        session_factory=mocks["session_factory"],
        vastai_client=mocks["vastai_client"],
        cf_client=mocks["cf_client"],
        bundle_index=mocks["bundle_index"],
        http_client=mocks["http_client"],
        settings=mocks["settings"],
        cooldown_store=mocks["cooldown_store"],
        billing_service=mocks["billing_service"],
        event_bus=mocks["event_bus"],
        job_sweep_service=mocks["job_sweep_service"],
        redis_client_factory=mocks["redis_client_factory"],
    )
    return worker, mocks


# ---------------------------------------------------------------------------
# Lifecycle (start/stop/tick-error-recovery) is covered generically by
# tests/unit/test_periodic_worker.py — GpuProvisioningWorker has no lifecycle
# overrides beyond PeriodicWorker, so no worker-specific lifecycle tests here.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# TestSweep
# ---------------------------------------------------------------------------


class TestSweep:
    async def test_sweep_empty_list_is_noop(self) -> None:
        worker, mocks = _make_worker()

        with patch(_REPO_PATH) as MockRepo:
            mock_repo = AsyncMock()
            MockRepo.return_value = mock_repo
            mock_repo.list_by_status.return_value = []

            await worker.run_once()

        mocks["vastai_client"].get_instance.assert_not_called()

    async def test_sweep_processes_each_non_terminal_status(self) -> None:
        worker, _mocks = _make_worker()

        sessions = [
            _make_gpu_session(status=GpuSessionStatus.pending),
            _make_gpu_session(status=GpuSessionStatus.provisioning),
            _make_gpu_session(status=GpuSessionStatus.resuming),
        ]

        worker._advance_pending = AsyncMock()  # type: ignore[method-assign]
        worker._advance_provisioning = AsyncMock()  # type: ignore[method-assign]
        worker._advance_resuming = AsyncMock()  # type: ignore[method-assign]

        with patch(_REPO_PATH) as MockRepo:
            mock_repo = AsyncMock()
            MockRepo.return_value = mock_repo
            mock_repo.list_by_status.return_value = sessions

            await worker.run_once()

        worker._advance_pending.assert_called_once()
        worker._advance_provisioning.assert_called_once()
        worker._advance_resuming.assert_called_once()

    async def test_sweep_exception_in_one_session_does_not_abort_others(self) -> None:
        worker, _mocks = _make_worker()

        sessions = [
            _make_gpu_session(status=GpuSessionStatus.pending),
            _make_gpu_session(status=GpuSessionStatus.pending),
        ]
        call_count = 0

        async def flaky_advance(_session: GpuSession) -> None:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("first session fails")

        worker._advance = flaky_advance  # type: ignore[method-assign]

        with patch(_REPO_PATH) as MockRepo:
            mock_repo = AsyncMock()
            MockRepo.return_value = mock_repo
            mock_repo.list_by_status.return_value = sessions

            # Should not raise
            await worker.run_once()

        assert call_count == 2


# ---------------------------------------------------------------------------
# TestAdvancePending
# ---------------------------------------------------------------------------


class TestAdvancePending:
    async def test_vastai_not_running_yet_is_noop(self) -> None:
        worker, mocks = _make_worker()
        session = _make_gpu_session(status=GpuSessionStatus.pending)

        instance = MagicMock()
        instance.actual_status = "loading"
        mocks["vastai_client"].get_instance.return_value = instance

        with patch(_REPO_PATH) as MockRepo:
            mock_repo = AsyncMock()
            MockRepo.return_value = mock_repo

            await worker._advance_pending(session)

        mock_repo.update_status.assert_not_called()

    async def test_vastai_running_transitions_to_provisioning(self) -> None:
        worker, mocks = _make_worker()
        session = _make_gpu_session(status=GpuSessionStatus.pending)

        instance = MagicMock()
        instance.actual_status = "running"
        mocks["vastai_client"].get_instance.return_value = instance

        with patch(_REPO_PATH) as MockRepo:
            mock_repo = AsyncMock()
            MockRepo.return_value = mock_repo
            # _transition re-loads the session
            reloaded = _make_gpu_session(status=GpuSessionStatus.pending)
            mock_repo.get_by_id.return_value = reloaded

            await worker._advance_pending(session)

        mock_repo.update_status.assert_called_once_with(session.id, GpuSessionStatus.provisioning)

    async def test_vastai_instance_not_found_marks_failed(self) -> None:
        worker, mocks = _make_worker()
        session = _make_gpu_session(status=GpuSessionStatus.pending)
        mocks["vastai_client"].get_instance.side_effect = InstanceNotFoundError(
            "gone", status_code=404
        )

        with patch(_REPO_PATH) as MockRepo:
            mock_repo = AsyncMock()
            MockRepo.return_value = mock_repo
            reloaded = _make_gpu_session(status=GpuSessionStatus.pending)
            mock_repo.get_by_id.return_value = reloaded

            await worker._advance_pending(session)

        mock_repo.update_status.assert_called_once()
        call_args = mock_repo.update_status.call_args
        assert call_args[0][1] == GpuSessionStatus.failed

    async def test_get_instance_transient_error_skips_without_writing(self) -> None:
        worker, mocks = _make_worker()
        session = _make_gpu_session(status=GpuSessionStatus.pending)
        mocks["vastai_client"].get_instance.side_effect = VastAIError("timeout", status_code=503)

        with patch(_REPO_PATH) as MockRepo:
            mock_repo = AsyncMock()
            MockRepo.return_value = mock_repo

            await worker._advance_pending(session)

        mock_repo.update_status.assert_not_called()

    async def test_pending_timeout_triggers_retry(self) -> None:
        worker, mocks = _make_worker()
        old_created = datetime.now(UTC) - timedelta(minutes=25)
        session = _make_gpu_session(status=GpuSessionStatus.pending, created_at=old_created)

        instance = MagicMock()
        instance.actual_status = "loading"
        mocks["vastai_client"].get_instance.return_value = instance

        worker._retry_or_fail = AsyncMock()  # type: ignore[method-assign]

        await worker._advance_pending(session)

        worker._retry_or_fail.assert_called_once_with(session, reason=_REASON_PENDING_TIMEOUT)

    async def test_pending_without_instance_id_marks_failed(self) -> None:
        worker, _mocks = _make_worker()
        session = _make_gpu_session(status=GpuSessionStatus.pending, vastai_instance_id=None)

        with patch(_REPO_PATH) as MockRepo:
            mock_repo = AsyncMock()
            MockRepo.return_value = mock_repo
            reloaded = _make_gpu_session(status=GpuSessionStatus.pending)
            mock_repo.get_by_id.return_value = reloaded

            await worker._advance_pending(session)

        mock_repo.update_status.assert_called_once()
        assert mock_repo.update_status.call_args[0][1] == GpuSessionStatus.failed


# ---------------------------------------------------------------------------
# TestAdvanceProvisioning
# ---------------------------------------------------------------------------


class TestAdvanceProvisioning:
    def _mock_ok_response(self) -> MagicMock:
        resp = MagicMock(spec=httpx.Response)
        resp.status_code = 200
        return resp

    def _mock_bad_response(self, code: int = 500) -> MagicMock:
        resp = MagicMock(spec=httpx.Response)
        resp.status_code = code
        return resp

    async def test_probe_200_transitions_to_active_and_sets_started_at(self) -> None:
        """Unverifiable bundle (zero declared checkpoints) + probe 200 → active via degradation backstop."""
        worker, mocks = _make_worker()
        session = _make_gpu_session(status=GpuSessionStatus.provisioning, started_at=None)
        mocks["bundle_index"].get_model_filenames.return_value = []  # zero-checkpoint: unverifiable
        mocks["http_client"].get.return_value = self._mock_ok_response()

        with patch(_REPO_PATH) as MockRepo:
            mock_repo = AsyncMock()
            MockRepo.return_value = mock_repo
            reloaded = _make_gpu_session(status=GpuSessionStatus.provisioning)
            mock_repo.get_by_id.return_value = reloaded

            await worker._advance_provisioning(session)

        mock_repo.update_status.assert_called_once()
        call_kwargs = mock_repo.update_status.call_args[1]
        assert "started_at" in call_kwargs
        assert call_kwargs["started_at"] is not None

    async def test_probe_checkpoint_verified_transitions_to_active(self) -> None:
        """Checkpoint present in /object_info + matches bundle declaration → active via real readiness pass."""
        worker, mocks = _make_worker()
        session = _make_gpu_session(status=GpuSessionStatus.provisioning, started_at=None)
        ckpt = "Qwen-Rapid-AIO-NSFW-v19.safetensors"
        mocks["bundle_index"].get_model_filenames.return_value = [ckpt]
        mocks["http_client"].get.return_value = _make_object_info_with_checkpoint([ckpt])

        with patch(_REPO_PATH) as MockRepo:
            mock_repo = AsyncMock()
            MockRepo.return_value = mock_repo
            reloaded = _make_gpu_session(status=GpuSessionStatus.provisioning)
            mock_repo.get_by_id.return_value = reloaded

            await worker._advance_provisioning(session)

        mock_repo.update_status.assert_called_once()
        call_args = mock_repo.update_status.call_args[0]  # positional args
        assert call_args[1] == GpuSessionStatus.active

    async def test_probe_500_is_noop(self) -> None:
        worker, mocks = _make_worker()
        session = _make_gpu_session(status=GpuSessionStatus.provisioning)
        mocks["http_client"].get.return_value = self._mock_bad_response(500)

        with patch(_REPO_PATH) as MockRepo:
            mock_repo = AsyncMock()
            MockRepo.return_value = mock_repo

            await worker._advance_provisioning(session)

        mock_repo.update_status.assert_not_called()

    async def test_probe_connection_error_is_noop(self) -> None:
        worker, mocks = _make_worker()
        session = _make_gpu_session(status=GpuSessionStatus.provisioning)
        mocks["http_client"].get.side_effect = httpx.ConnectError("refused")

        with patch(_REPO_PATH) as MockRepo:
            mock_repo = AsyncMock()
            MockRepo.return_value = mock_repo

            await worker._advance_provisioning(session)

        mock_repo.update_status.assert_not_called()

    async def test_timeout_triggers_retry(self) -> None:
        worker, mocks = _make_worker()
        old_created = datetime.now(UTC) - timedelta(minutes=25)
        session = _make_gpu_session(status=GpuSessionStatus.provisioning, created_at=old_created)

        worker._retry_or_fail = AsyncMock()  # type: ignore[method-assign]

        await worker._advance_provisioning(session)

        worker._retry_or_fail.assert_called_once_with(session, reason=_REASON_PROVISIONING_TIMEOUT)
        mocks["http_client"].get.assert_not_called()

    async def test_retry_preserves_tunnel(self) -> None:
        """After retry, the session's tunnel ID and hostname must NOT change."""
        # provisioning_recreation_attempts=3 so new_attempt=2 < 3+1=4 → recreation fires
        worker, mocks = _make_worker(settings=_make_settings(provisioning_recreation_attempts=3))
        original_tunnel_id = "tunnel-abc"
        original_hostname = "gpu-01234567.gpu-domain.com"
        old_created = datetime.now(UTC) - timedelta(minutes=25)
        session = _make_gpu_session(
            status=GpuSessionStatus.pending,
            created_at=old_created,
            cf_tunnel_id=original_tunnel_id,
            tunnel_hostname=original_hostname,
        )

        bundle = _make_bundle_mapping()
        mocks["bundle_index"].resolve_bundle_override.return_value = bundle
        mocks["vastai_client"].search_offers.return_value = [_make_offer()]
        mocks["vastai_client"].destroy_instance = AsyncMock()
        mocks["cf_client"].get_tunnel_token.return_value = "new-token"

        with patch(_REPO_PATH) as MockRepo:
            mock_repo = AsyncMock()
            MockRepo.return_value = mock_repo
            mock_repo.increment_provision_attempt.return_value = 2
            reloaded = _make_gpu_session(
                status=GpuSessionStatus.pending,
                cf_tunnel_id=original_tunnel_id,
            )
            mock_repo.get_by_id.return_value = reloaded
            mocks["vastai_client"].create_instance.return_value = 99999

            await worker._retry_or_fail(session, reason=_REASON_PENDING_TIMEOUT)

        # Tunnel ID must be unchanged (not deleted/recreated)
        mocks["cf_client"].delete_session_tunnel.assert_not_called()
        mocks["cf_client"].create_session_tunnel.assert_not_called()

    async def test_retry_persists_machine_id_of_new_offer(self) -> None:
        """The re-persist block after a retry must write the new offer's machine_id."""
        # provisioning_recreation_attempts=3 so new_attempt=2 < 3+1=4 → recreation fires
        worker, mocks = _make_worker(settings=_make_settings(provisioning_recreation_attempts=3))
        session = _make_gpu_session(status=GpuSessionStatus.pending)
        mocks["vastai_client"].destroy_instance = AsyncMock()
        bundle = _make_bundle_mapping()
        mocks["bundle_index"].resolve_bundle_override.return_value = bundle
        offer = _make_offer(offer_id=42)
        offer = VastAIOffer(
            id=offer.id,
            gpu_name=offer.gpu_name,
            dph_total=offer.dph_total,
            num_gpus=offer.num_gpus,
            machine_id=777888,
        )
        mocks["vastai_client"].search_offers.return_value = [offer]
        mocks["cf_client"].get_tunnel_token.return_value = "new-token"
        mocks["vastai_client"].create_instance.return_value = 99999

        with patch(_REPO_PATH) as MockRepo:
            mock_repo = AsyncMock()
            MockRepo.return_value = mock_repo
            mock_repo.increment_provision_attempt.return_value = 2
            reloaded = _make_gpu_session(status=GpuSessionStatus.pending)
            mock_repo.get_by_id.return_value = reloaded

            await worker._retry_or_fail(session, reason=_REASON_PENDING_TIMEOUT)

        update_kwargs = mock_repo.update_instance.call_args.kwargs
        assert update_kwargs["vastai_machine_id"] == 777888

    async def test_retry_exhausted_marks_failed(self) -> None:
        # provisioning_recreation_attempts=2 → new_attempt=3 >= 2+1=3 → exhausted
        worker, mocks = _make_worker(settings=_make_settings(provisioning_recreation_attempts=2))
        session = _make_gpu_session(status=GpuSessionStatus.pending)
        mocks["vastai_client"].destroy_instance = AsyncMock()

        with patch(_REPO_PATH) as MockRepo:
            mock_repo = AsyncMock()
            MockRepo.return_value = mock_repo
            mock_repo.increment_provision_attempt.return_value = 3
            reloaded = _make_gpu_session(status=GpuSessionStatus.pending)
            mock_repo.get_by_id.return_value = reloaded

            await worker._retry_or_fail(session, reason="timeout")

        mock_repo.update_status.assert_called_once()
        assert mock_repo.update_status.call_args[0][1] == GpuSessionStatus.failed

    async def test_retry_reconstructs_env_vars_from_session_row(self) -> None:
        """The env dict passed to create_instance must contain all required vars.

        On retry, a fresh callback token is generated — it will NOT match any
        old session.callback_token_hash value (that's the security goal of D2).
        """
        import hashlib

        # provisioning_recreation_attempts=3 so new_attempt=2 < 3+1=4 → recreation fires
        worker, mocks = _make_worker(settings=_make_settings(provisioning_recreation_attempts=3))
        session = _make_gpu_session(
            status=GpuSessionStatus.pending,
            bundle_name="wan_2.2_i2v",
            bundle_version="260105-01",
        )
        mocks["vastai_client"].destroy_instance = AsyncMock()
        bundle = _make_bundle_mapping()
        mocks["bundle_index"].resolve_bundle_override.return_value = bundle
        mocks["vastai_client"].search_offers.return_value = [_make_offer()]
        mocks["cf_client"].get_tunnel_token.return_value = "fetched-tunnel-token"
        mocks["vastai_client"].create_instance.return_value = 88888

        with patch(_REPO_PATH) as MockRepo:
            mock_repo = AsyncMock()
            MockRepo.return_value = mock_repo
            mock_repo.increment_provision_attempt.return_value = 2
            reloaded = _make_gpu_session(status=GpuSessionStatus.pending)
            mock_repo.get_by_id.return_value = reloaded

            await worker._retry_or_fail(session, reason="timeout")

        call_kwargs = mocks["vastai_client"].create_instance.call_args[1]
        env = call_kwargs["env"]
        assert env["ACS_BUNDLE"] == "wan_2.2_i2v"
        assert env["ACS_BUNDLE_VERSION"] == "260105-01"
        assert env["ACS_CF_TUNNEL_TOKEN"] == "fetched-tunnel-token"
        # Fresh token generated per retry — must be a non-empty string
        fresh_callback_token = env["ACS_APEX_CALLBACK_TOKEN"]
        assert isinstance(fresh_callback_token, str) and len(fresh_callback_token) > 0
        # The hash written to DB must match the fresh token in the env
        expected_hash = hashlib.sha256(fresh_callback_token.encode()).hexdigest()
        mock_repo.update_callback_token_hash.assert_called_once_with(session.id, expected_hash)
        assert "ACS_HF_TOKEN" in env
        assert "ACS_CIVITAI_API_TOKEN" in env
        # New contract keys
        assert env["ACS_GITHUB_TOKEN"] == "ghp_test_token"
        assert env["ACS_BUNDLES_REPO"] == "https://github.com/gearbox/ai-bundles.git"
        assert env["ACS_BUNDLES_BRANCH"] == "master"
        assert env["ACS_AISHA_REPO"] == "https://github.com/gearbox/aisha.git"
        assert env["ACS_AISHA_BRANCH"] == "master"
        assert env["ACS_COMFYUI_PORT"] == str(_DEFAULT_COMFYUI_PORT)
        assert env["ACS_COMFYUI_HOST"] == "0.0.0.0"  # noqa: S104
        assert env["ACS_COMFYUI_EXTRA_ARGS"] == ""
        assert f"-p {_DEFAULT_COMFYUI_PORT}:{_DEFAULT_COMFYUI_PORT}" in env

    async def test_retry_fails_fast_on_empty_github_token(self) -> None:
        """If ai_bundles_github_token is empty, mark session failed before create_instance."""
        # provisioning_recreation_attempts=3 so new_attempt=2 allows recreation to proceed
        # far enough to hit the github_token check
        worker, mocks = _make_worker(settings=_make_settings(provisioning_recreation_attempts=3))
        mocks["settings"].ai_bundles_github_token = ""
        session = _make_gpu_session(
            status=GpuSessionStatus.pending,
            bundle_name="wan_2.2_i2v",
            bundle_version="260105-01",
            callback_token_hash="some-existing-hash",
        )
        mocks["vastai_client"].destroy_instance = AsyncMock()
        bundle = _make_bundle_mapping()
        mocks["bundle_index"].resolve_bundle_override.return_value = bundle
        mocks["vastai_client"].search_offers.return_value = [_make_offer()]
        mocks["cf_client"].get_tunnel_token.return_value = "fetched-tunnel-token"

        with patch(_REPO_PATH) as MockRepo:
            mock_repo = AsyncMock()
            MockRepo.return_value = mock_repo
            mock_repo.increment_provision_attempt.return_value = 2
            reloaded = _make_gpu_session(status=GpuSessionStatus.pending)
            mock_repo.get_by_id.return_value = reloaded

            await worker._retry_or_fail(session, reason="timeout")

        # No instance creation attempted
        mocks["vastai_client"].create_instance.assert_not_called()
        # Session must be marked failed
        mock_repo.update_status.assert_called()
        failed_call = mock_repo.update_status.call_args_list[-1]
        assert failed_call[0][1] == GpuSessionStatus.failed


# ---------------------------------------------------------------------------
# TestNodeCooldownRecording
# ---------------------------------------------------------------------------


class TestNodeCooldownRecording:
    async def test_retry_or_fail_records_cooldown_before_recreation(self) -> None:
        # provisioning_recreation_attempts=3 so new_attempt=2 < 3+1=4 → recreation fires
        cooldown = AsyncMock(spec=NodeCooldownStore)
        worker, mocks = _make_worker(
            settings=_make_settings(provisioning_recreation_attempts=3),
            cooldown_store=cooldown,
        )
        session = _make_gpu_session(status=GpuSessionStatus.pending, vastai_machine_id=555)
        mocks["vastai_client"].destroy_instance = AsyncMock()
        bundle = _make_bundle_mapping()
        mocks["bundle_index"].resolve_bundle_override.return_value = bundle
        mocks["vastai_client"].search_offers.return_value = [_make_offer()]
        mocks["cf_client"].get_tunnel_token.return_value = "new-token"
        mocks["vastai_client"].create_instance.return_value = 99999

        with patch(_REPO_PATH) as MockRepo:
            mock_repo = AsyncMock()
            MockRepo.return_value = mock_repo
            mock_repo.increment_provision_attempt.return_value = 2
            reloaded = _make_gpu_session(status=GpuSessionStatus.pending)
            mock_repo.get_by_id.return_value = reloaded

            await worker._retry_or_fail(session, reason="provisioning_stalled")

        cooldown.record_failure.assert_awaited_once_with(555, reason="provisioning_stalled")

    async def test_terminal_failure_still_records_cooldown(self) -> None:
        """Even when attempts are exhausted (terminal → _mark_failed), the machine
        is still cooled — the user's next manual session must avoid it."""
        cooldown = AsyncMock(spec=NodeCooldownStore)
        worker, mocks = _make_worker(cooldown_store=cooldown)  # default recreation_attempts=1
        session = _make_gpu_session(status=GpuSessionStatus.pending, vastai_machine_id=555)
        mocks["vastai_client"].destroy_instance = AsyncMock()

        with patch(_REPO_PATH) as MockRepo:
            mock_repo = AsyncMock()
            MockRepo.return_value = mock_repo
            mock_repo.increment_provision_attempt.return_value = 2  # exhausted
            reloaded = _make_gpu_session(status=GpuSessionStatus.pending)
            mock_repo.get_by_id.return_value = reloaded

            await worker._retry_or_fail(session, reason="provisioning_timeout")

        cooldown.record_failure.assert_awaited_once_with(555, reason="provisioning_timeout")

    async def test_missing_machine_id_skips_cooldown_recording(self) -> None:
        cooldown = AsyncMock(spec=NodeCooldownStore)
        worker, mocks = _make_worker(cooldown_store=cooldown)
        session = _make_gpu_session(status=GpuSessionStatus.pending, vastai_machine_id=None)
        mocks["vastai_client"].destroy_instance = AsyncMock()

        with patch(_REPO_PATH) as MockRepo:
            mock_repo = AsyncMock()
            MockRepo.return_value = mock_repo
            mock_repo.increment_provision_attempt.return_value = 2
            reloaded = _make_gpu_session(status=GpuSessionStatus.pending)
            mock_repo.get_by_id.return_value = reloaded

            await worker._retry_or_fail(session, reason="test")

        cooldown.record_failure.assert_not_awaited()

    async def test_apex_side_failures_do_not_record_cooldown(self) -> None:
        """A search failure mid-retry routes to _mark_failed via retry_search_failed,
        but must not trigger a second record_failure call for that internal reason —
        only the original top-level failure reason is ever recorded."""
        cooldown = AsyncMock(spec=NodeCooldownStore)
        worker, mocks = _make_worker(
            settings=_make_settings(provisioning_recreation_attempts=3),
            cooldown_store=cooldown,
        )
        session = _make_gpu_session(status=GpuSessionStatus.pending, vastai_machine_id=555)
        mocks["vastai_client"].destroy_instance = AsyncMock()
        bundle = _make_bundle_mapping()
        mocks["bundle_index"].resolve_bundle_override.return_value = bundle
        mocks["vastai_client"].search_offers.side_effect = RuntimeError("no capacity")

        with patch(_REPO_PATH) as MockRepo:
            mock_repo = AsyncMock()
            MockRepo.return_value = mock_repo
            mock_repo.increment_provision_attempt.return_value = 2
            reloaded = _make_gpu_session(status=GpuSessionStatus.pending)
            mock_repo.get_by_id.return_value = reloaded

            await worker._retry_or_fail(session, reason="node_reported_failure")

        cooldown.record_failure.assert_awaited_once_with(555, reason="node_reported_failure")
        mock_repo.update_status.assert_awaited_once()
        assert mock_repo.update_status.await_args.args[1] == GpuSessionStatus.failed


# ---------------------------------------------------------------------------
# TestAdvanceResuming
# ---------------------------------------------------------------------------


class TestAdvanceResuming:
    def _mock_ok_response(self) -> MagicMock:
        resp = MagicMock(spec=httpx.Response)
        resp.status_code = 200
        return resp

    async def test_probe_200_transitions_to_active_keeps_started_at(self) -> None:
        """Unverifiable bundle + probe 200 → active via degradation backstop; started_at preserved."""
        worker, mocks = _make_worker()
        original_start = datetime.now(UTC) - timedelta(hours=5)
        session = _make_gpu_session(
            status=GpuSessionStatus.resuming,
            started_at=original_start,
            resumed_at=datetime.now(UTC) - timedelta(minutes=1),
        )
        mocks["bundle_index"].get_model_filenames.return_value = []  # zero-checkpoint: unverifiable
        mocks["http_client"].get.return_value = self._mock_ok_response()

        with patch(_REPO_PATH) as MockRepo:
            mock_repo = AsyncMock()
            MockRepo.return_value = mock_repo
            reloaded = _make_gpu_session(
                status=GpuSessionStatus.resuming, started_at=original_start
            )
            mock_repo.get_by_id.return_value = reloaded

            await worker._advance_resuming(session)

        mock_repo.update_status.assert_called_once()
        # started_at should NOT be in extra_fields (session.started_at is not None)
        call_kwargs = mock_repo.update_status.call_args[1]
        assert "started_at" not in call_kwargs

    async def test_timeout_marks_failed_without_retry(self) -> None:
        worker, _mocks = _make_worker()
        old_resumed = datetime.now(UTC) - timedelta(minutes=10)
        session = _make_gpu_session(
            status=GpuSessionStatus.resuming,
            resumed_at=old_resumed,
        )

        worker._mark_failed = AsyncMock()  # type: ignore[method-assign]
        worker._retry_or_fail = AsyncMock()  # type: ignore[method-assign]

        await worker._advance_resuming(session)

        worker._mark_failed.assert_called_once()
        worker._retry_or_fail.assert_not_called()

    async def test_missing_resumed_at_marks_failed(self) -> None:
        """resumed_at=None triggers timeout logic (treat as timed out)."""
        worker, _mocks = _make_worker()
        session = _make_gpu_session(
            status=GpuSessionStatus.resuming,
            resumed_at=None,
        )

        worker._mark_failed = AsyncMock()  # type: ignore[method-assign]

        await worker._advance_resuming(session)

        worker._mark_failed.assert_called_once()


# ---------------------------------------------------------------------------
# TestStaleTransitionGuard
# ---------------------------------------------------------------------------


class TestStaleTransitionGuard:
    async def test_transition_skipped_if_user_stopped_session_mid_sweep(self) -> None:
        """If the session was stopped by the user during the sweep, no update_status call."""
        worker, _mocks = _make_worker()
        session = _make_gpu_session(status=GpuSessionStatus.pending)

        with patch(_REPO_PATH) as MockRepo:
            mock_repo = AsyncMock()
            MockRepo.return_value = mock_repo
            # Simulate user stopped the session before worker can write
            stopped = _make_gpu_session(status=GpuSessionStatus.stopping)
            mock_repo.get_by_id.return_value = stopped

            await worker._transition(
                session,
                new_status=GpuSessionStatus.provisioning,
                log_event="test.event",
            )

        mock_repo.update_status.assert_not_called()


# ---------------------------------------------------------------------------
# TestEnvReconstruction
# ---------------------------------------------------------------------------


class TestEnvReconstruction:
    async def test_rebuild_env_includes_tunnel_token_fetched_from_cf(self) -> None:
        # provisioning_recreation_attempts=3 so new_attempt=2 < 3+1=4 → recreation fires
        worker, mocks = _make_worker(settings=_make_settings(provisioning_recreation_attempts=3))
        session = _make_gpu_session(status=GpuSessionStatus.pending, cf_tunnel_id="tun-xyz")
        mocks["vastai_client"].destroy_instance = AsyncMock()
        mocks["cf_client"].get_tunnel_token.return_value = "fetched-token-secret"
        bundle = _make_bundle_mapping()
        mocks["bundle_index"].resolve_bundle_override.return_value = bundle
        mocks["vastai_client"].search_offers.return_value = [_make_offer()]
        mocks["vastai_client"].create_instance.return_value = 12345

        with patch(_REPO_PATH) as MockRepo:
            mock_repo = AsyncMock()
            MockRepo.return_value = mock_repo
            mock_repo.increment_provision_attempt.return_value = 2
            mock_repo.get_by_id.return_value = _make_gpu_session(status=GpuSessionStatus.pending)

            await worker._retry_or_fail(session, reason="timeout")

        mocks["cf_client"].get_tunnel_token.assert_called_once_with("tun-xyz")
        env = mocks["vastai_client"].create_instance.call_args[1]["env"]
        assert env["ACS_CF_TUNNEL_TOKEN"] == "fetched-token-secret"

    async def test_rebuild_env_uses_bundle_version_current_when_none(self) -> None:
        # provisioning_recreation_attempts=3 so new_attempt=2 < 3+1=4 → recreation fires
        worker, mocks = _make_worker(settings=_make_settings(provisioning_recreation_attempts=3))
        session = _make_gpu_session(
            status=GpuSessionStatus.pending,
            bundle_version=None,
        )
        mocks["vastai_client"].destroy_instance = AsyncMock()
        mocks["cf_client"].get_tunnel_token.return_value = "tok"
        bundle = _make_bundle_mapping()
        mocks["bundle_index"].resolve_bundle_override.return_value = bundle
        mocks["vastai_client"].search_offers.return_value = [_make_offer()]
        mocks["vastai_client"].create_instance.return_value = 12345

        with patch(_REPO_PATH) as MockRepo:
            mock_repo = AsyncMock()
            MockRepo.return_value = mock_repo
            mock_repo.increment_provision_attempt.return_value = 2
            mock_repo.get_by_id.return_value = _make_gpu_session(status=GpuSessionStatus.pending)

            await worker._retry_or_fail(session, reason="timeout")

        env = mocks["vastai_client"].create_instance.call_args[1]["env"]
        assert env["ACS_BUNDLE_VERSION"] == "current"

    async def test_rebuild_env_includes_port_mapping_from_hardware(self) -> None:
        # provisioning_recreation_attempts=3 so new_attempt=2 < 3+1=4 → recreation fires
        worker, mocks = _make_worker(settings=_make_settings(provisioning_recreation_attempts=3))
        session = _make_gpu_session(status=GpuSessionStatus.pending)
        mocks["vastai_client"].destroy_instance = AsyncMock()
        mocks["cf_client"].get_tunnel_token.return_value = "tok"
        bundle = _make_bundle_mapping()
        mocks["bundle_index"].resolve_bundle_override.return_value = bundle
        mocks["vastai_client"].search_offers.return_value = [_make_offer()]
        mocks["vastai_client"].create_instance.return_value = 12345

        with patch(_REPO_PATH) as MockRepo:
            mock_repo = AsyncMock()
            MockRepo.return_value = mock_repo
            mock_repo.increment_provision_attempt.return_value = 2
            mock_repo.get_by_id.return_value = _make_gpu_session(status=GpuSessionStatus.pending)

            await worker._retry_or_fail(session, reason="timeout")

        env = mocks["vastai_client"].create_instance.call_args[1]["env"]
        assert f"-p {_DEFAULT_COMFYUI_PORT}:{_DEFAULT_COMFYUI_PORT}" in env


class TestSweepConcurrency:
    async def test_sweep_bounds_concurrency_with_semaphore(self) -> None:
        """With gpu_provision_worker_concurrency=2 and 5 sessions, at most 2 _advance
        calls run in parallel at any instant.
        """
        worker, _mocks = _make_worker(settings=_make_settings(gpu_provision_worker_concurrency=2))
        sessions = [_make_gpu_session(status=GpuSessionStatus.pending) for _ in range(5)]

        # Track how many _advance calls are in-flight concurrently
        in_flight = 0
        max_in_flight = 0
        advance_lock = asyncio.Lock()

        async def fake_advance(_session: GpuSession) -> None:
            nonlocal in_flight, max_in_flight
            async with advance_lock:
                in_flight += 1
                max_in_flight = max(max_in_flight, in_flight)
            # Yield to the event loop so other tasks get a chance to start
            await asyncio.sleep(0.01)
            async with advance_lock:
                in_flight -= 1

        with (
            patch(_REPO_PATH) as MockRepo,
            patch.object(worker, "_advance", side_effect=fake_advance),
        ):
            mock_repo = AsyncMock()
            MockRepo.return_value = mock_repo
            mock_repo.list_by_status.return_value = sessions

            await worker.run_once()

        assert max_in_flight <= 2, f"Expected <=2 concurrent, observed {max_in_flight}"
        # Sanity: we actually exercised concurrency (otherwise the test is trivially true)
        assert max_in_flight >= 2

    async def test_sweep_concurrency_one_serializes(self) -> None:
        """With concurrency=1, _advance calls must be strictly serialized."""
        worker, _ = _make_worker(settings=_make_settings(gpu_provision_worker_concurrency=1))
        sessions = [_make_gpu_session(status=GpuSessionStatus.pending) for _ in range(3)]

        in_flight = 0
        max_in_flight = 0
        advance_lock = asyncio.Lock()

        async def fake_advance(_session: GpuSession) -> None:
            nonlocal in_flight, max_in_flight
            async with advance_lock:
                in_flight += 1
                max_in_flight = max(max_in_flight, in_flight)
            await asyncio.sleep(0.01)
            async with advance_lock:
                in_flight -= 1

        with (
            patch(_REPO_PATH) as MockRepo,
            patch.object(worker, "_advance", side_effect=fake_advance),
        ):
            mock_repo = AsyncMock()
            MockRepo.return_value = mock_repo
            mock_repo.list_by_status.return_value = sessions

            await worker.run_once()

        assert max_in_flight == 1


class TestMarkFailed:
    """Direct tests of _mark_failed cleanup behavior.

    The retry_exhausted path tests cover this indirectly, but these tests
    pin the contract of _mark_failed itself — specifically the skip-when-None
    branches — so cleanup regressions are caught precisely.
    """

    async def test_destroys_instance_and_deletes_tunnel_when_ids_present(self) -> None:
        worker, mocks = _make_worker()
        session = _make_gpu_session(
            status=GpuSessionStatus.pending,
            vastai_instance_id=12345,
            cf_tunnel_id="tun-abc",
            cf_dns_record_id="dns-abc",
        )

        with patch(_REPO_PATH) as MockRepo:
            mock_repo = AsyncMock()
            MockRepo.return_value = mock_repo
            mock_repo.get_by_id.return_value = session

            await worker._mark_failed(session, reason="test failure")

        mocks["vastai_client"].destroy_instance.assert_awaited_once_with(12345)
        mocks["cf_client"].delete_session_tunnel.assert_awaited_once_with("tun-abc", "dns-abc")
        # And the status transition was performed
        mock_repo.update_status.assert_awaited()
        update_kwargs = mock_repo.update_status.await_args.kwargs
        update_args = mock_repo.update_status.await_args.args
        # update_status(session_id, status, **extras) — second positional is status
        assert update_args[1] == GpuSessionStatus.failed
        assert update_kwargs.get("error_message") == "test failure"

    async def test_skips_destroy_when_instance_id_missing(self) -> None:
        worker, mocks = _make_worker()
        session = _make_gpu_session(
            status=GpuSessionStatus.pending,
            vastai_instance_id=None,
            cf_tunnel_id="tun-abc",
            cf_dns_record_id="dns-abc",
        )

        with patch(_REPO_PATH) as MockRepo:
            mock_repo = AsyncMock()
            MockRepo.return_value = mock_repo
            mock_repo.get_by_id.return_value = session

            await worker._mark_failed(session, reason="test")

        mocks["vastai_client"].destroy_instance.assert_not_awaited()
        mocks["cf_client"].delete_session_tunnel.assert_awaited_once()

    async def test_skips_tunnel_delete_when_dns_id_missing(self) -> None:
        """Tunnel delete requires both tunnel_id and dns_record_id — skip if either is None."""
        worker, mocks = _make_worker()
        session = _make_gpu_session(
            status=GpuSessionStatus.pending,
            vastai_instance_id=99,
            cf_tunnel_id="tun-abc",
            cf_dns_record_id=None,
        )

        with patch(_REPO_PATH) as MockRepo:
            mock_repo = AsyncMock()
            MockRepo.return_value = mock_repo
            mock_repo.get_by_id.return_value = session

            await worker._mark_failed(session, reason="test")

        mocks["vastai_client"].destroy_instance.assert_awaited_once_with(99)
        mocks["cf_client"].delete_session_tunnel.assert_not_awaited()

    async def test_instance_not_found_during_destroy_is_swallowed(self) -> None:
        """InstanceNotFoundError (404 from Vast) must not abort tunnel cleanup or status transition."""
        from src.api.services.vastai.exceptions import InstanceNotFoundError

        worker, mocks = _make_worker()
        session = _make_gpu_session(
            status=GpuSessionStatus.pending,
            vastai_instance_id=99,
            cf_tunnel_id="tun-abc",
            cf_dns_record_id="dns-abc",
        )
        mocks["vastai_client"].destroy_instance.side_effect = InstanceNotFoundError(
            "gone", status_code=404
        )

        with patch(_REPO_PATH) as MockRepo:
            mock_repo = AsyncMock()
            MockRepo.return_value = mock_repo
            mock_repo.get_by_id.return_value = session

            # Must not raise
            await worker._mark_failed(session, reason="test")

        # Tunnel cleanup still runs
        mocks["cf_client"].delete_session_tunnel.assert_awaited_once()
        # Status still transitions
        mock_repo.update_status.assert_awaited()

    async def test_tunnel_delete_failure_still_transitions_to_failed(self) -> None:
        """Best-effort tunnel cleanup: a CF error must not block the 'failed' transition."""
        worker, mocks = _make_worker()
        session = _make_gpu_session(
            status=GpuSessionStatus.pending,
            vastai_instance_id=None,
            cf_tunnel_id="tun-abc",
            cf_dns_record_id="dns-abc",
        )
        mocks["cf_client"].delete_session_tunnel.side_effect = RuntimeError("CF unreachable")

        with patch(_REPO_PATH) as MockRepo:
            mock_repo = AsyncMock()
            MockRepo.return_value = mock_repo
            mock_repo.get_by_id.return_value = session

            await worker._mark_failed(session, reason="test")

        mock_repo.update_status.assert_awaited()
        assert mock_repo.update_status.await_args.args[1] == GpuSessionStatus.failed

    async def test_error_message_truncated_to_500_chars(self) -> None:
        """Long reasons must be truncated so the DB column is never overflowed."""
        worker, _ = _make_worker()
        session = _make_gpu_session(
            status=GpuSessionStatus.pending,
            vastai_instance_id=None,
            cf_tunnel_id=None,
            cf_dns_record_id=None,
        )
        long_reason = "x" * 1000

        with patch(_REPO_PATH) as MockRepo:
            mock_repo = AsyncMock()
            MockRepo.return_value = mock_repo
            mock_repo.get_by_id.return_value = session

            await worker._mark_failed(session, reason=long_reason)

        stored = mock_repo.update_status.await_args.kwargs["error_message"]
        assert len(stored) == 500
        assert stored == "x" * 500

    async def test_mark_failed_issues_full_refund_via_billing_service(self) -> None:
        """billing_service.refund must be called after the 'failed' transition."""
        billing_mock = AsyncMock()
        worker, _mocks = _make_worker(billing_service=billing_mock)
        session = _make_gpu_session(
            status=GpuSessionStatus.pending,
            vastai_instance_id=None,
            cf_tunnel_id=None,
            cf_dns_record_id=None,
        )

        with patch(_REPO_PATH) as MockRepo:
            mock_repo = AsyncMock()
            MockRepo.return_value = mock_repo
            mock_repo.get_by_id.return_value = session

            await worker._mark_failed(session, reason="instance vanished")

        billing_mock.refund.assert_awaited_once()
        call_kwargs = billing_mock.refund.await_args.kwargs
        assert call_kwargs["job_id"] == session.id
        assert call_kwargs["product_id"] == session.product_id
        assert call_kwargs["user_id"] == session.user_id

    async def test_mark_failed_refund_failure_does_not_propagate(self) -> None:
        """A billing refund error must never surface from _mark_failed (fire-and-forget)."""
        billing_mock = AsyncMock()
        billing_mock.refund.side_effect = Exception("billing down")
        worker, _ = _make_worker(billing_service=billing_mock)
        session = _make_gpu_session(
            status=GpuSessionStatus.pending,
            vastai_instance_id=None,
            cf_tunnel_id=None,
            cf_dns_record_id=None,
        )

        with patch(_REPO_PATH) as MockRepo:
            mock_repo = AsyncMock()
            MockRepo.return_value = mock_repo
            mock_repo.get_by_id.return_value = session

            await worker._mark_failed(session, reason="timeout")  # must not raise

        mock_repo.update_status.assert_awaited()  # status transition still completed

    async def test_mark_failed_publishes_status_event(self) -> None:
        """GPU_SESSION_STATUS_CHANGED must be published with accurate payload."""
        event_bus_mock = AsyncMock()
        worker, _ = _make_worker(event_bus=event_bus_mock)
        session = _make_gpu_session(
            status=GpuSessionStatus.pending,
            vastai_instance_id=None,
            cf_tunnel_id=None,
            cf_dns_record_id=None,
        )

        with patch(_REPO_PATH) as MockRepo:
            mock_repo = AsyncMock()
            MockRepo.return_value = mock_repo
            mock_repo.get_by_id.return_value = session

            await worker._mark_failed(session, reason="boot timeout")

        event_bus_mock.publish.assert_awaited_once()
        kwargs = event_bus_mock.publish.call_args.kwargs
        assert kwargs["event_type"] == EventType.GPU_SESSION_STATUS_CHANGED
        assert kwargs["user_id"] == session.user_id
        payload = kwargs["payload"]
        assert isinstance(payload, GpuSessionStatusPayload)
        assert payload.session_id == session.id
        assert payload.status == GpuSessionStatus.failed.value
        assert payload.previous_status == GpuSessionStatus.pending.value
        assert payload.error_message == "boot timeout"
        assert payload.model_type == session.model_type
        assert not hasattr(payload, "bundle_name")

    async def test_mark_failed_truncates_error_message_in_status_event(self) -> None:
        """Long reasons must be truncated to 500 chars in the payload."""
        event_bus_mock = AsyncMock()
        worker, _ = _make_worker(event_bus=event_bus_mock)
        long_reason = "x" * 600  # 100 chars over the 500-char cap
        session = _make_gpu_session(
            status=GpuSessionStatus.pending,
            vastai_instance_id=None,
            cf_tunnel_id=None,
            cf_dns_record_id=None,
        )

        with patch(_REPO_PATH) as MockRepo:
            mock_repo = AsyncMock()
            MockRepo.return_value = mock_repo
            mock_repo.get_by_id.return_value = session

            await worker._mark_failed(session, reason=long_reason)

        event_bus_mock.publish.assert_awaited_once()
        payload = event_bus_mock.publish.call_args.kwargs["payload"]
        assert isinstance(payload, GpuSessionStatusPayload)
        assert payload.error_message is not None
        assert len(payload.error_message) == 500
        assert payload.error_message == "x" * 500

    async def test_mark_failed_event_publish_failure_does_not_propagate(self) -> None:
        """An SSE publish error inside _mark_failed must never surface."""
        event_bus_mock = AsyncMock()
        event_bus_mock.publish.side_effect = Exception("redis down")
        worker, _ = _make_worker(event_bus=event_bus_mock)
        session = _make_gpu_session(
            status=GpuSessionStatus.pending,
            vastai_instance_id=None,
            cf_tunnel_id=None,
            cf_dns_record_id=None,
        )

        with patch(_REPO_PATH) as MockRepo:
            mock_repo = AsyncMock()
            MockRepo.return_value = mock_repo
            mock_repo.get_by_id.return_value = session

            await worker._mark_failed(session, reason="test")  # must not raise

        mock_repo.update_status.assert_awaited()


# ---------------------------------------------------------------------------
# TestMarkFailedWithJobSweep (Blocker B)
# ---------------------------------------------------------------------------


def _make_worker_with_sweep(**overrides: Any) -> tuple[GpuProvisioningWorker, dict[str, Any]]:
    sweep = MagicMock()
    sweep.sweep_session_best_effort = AsyncMock(return_value=None)
    worker, mocks = _make_worker(job_sweep_service=sweep, **overrides)
    mocks["job_sweep"] = sweep
    return worker, mocks


class TestMarkFailedWithJobSweep:
    async def test_mark_failed_sweeps_jobs_before_status_transition(self) -> None:
        """sweep_session_best_effort is called before update_status(session_id, failed)."""
        worker, mocks = _make_worker_with_sweep()
        session = _make_gpu_session(
            status=GpuSessionStatus.pending,
            vastai_instance_id=None,
            cf_tunnel_id=None,
            cf_dns_record_id=None,
        )
        call_order: list[str] = []

        async def record_sweep(**_kwargs: Any) -> None:
            call_order.append("sweep")

        mocks["job_sweep"].sweep_session_best_effort.side_effect = record_sweep

        with patch(_REPO_PATH) as MockRepo:
            mock_repo = AsyncMock()

            async def record_update_status(*_args: Any, **_kwargs: Any) -> None:
                call_order.append("update_status")

            mock_repo.update_status.side_effect = record_update_status
            MockRepo.return_value = mock_repo
            mock_repo.get_by_id.return_value = session

            await worker._mark_failed(session, reason="test reason")

        assert "sweep" in call_order
        assert "update_status" in call_order
        assert call_order.index("sweep") < call_order.index("update_status")

    async def test_mark_failed_completes_normally_after_sweep(self) -> None:
        """Teardown + status transition complete after sweep_session_best_effort call."""
        worker, _mocks = _make_worker_with_sweep()
        session = _make_gpu_session(
            status=GpuSessionStatus.pending,
            vastai_instance_id=None,
            cf_tunnel_id=None,
            cf_dns_record_id=None,
        )

        with patch(_REPO_PATH) as MockRepo:
            mock_repo = AsyncMock()
            MockRepo.return_value = mock_repo
            mock_repo.get_by_id.return_value = session

            await worker._mark_failed(session, reason="provisioning timeout")

        mock_repo.update_status.assert_awaited()
        assert mock_repo.update_status.await_args.args[1] == GpuSessionStatus.failed

    async def test_mark_failed_passes_session_product_id_to_sweep(self) -> None:
        worker, mocks = _make_worker_with_sweep()
        session = _make_gpu_session(
            status=GpuSessionStatus.pending,
            vastai_instance_id=None,
            cf_tunnel_id=None,
            cf_dns_record_id=None,
            product_id="synthara",
        )

        with patch(_REPO_PATH) as MockRepo:
            mock_repo = AsyncMock()
            MockRepo.return_value = mock_repo
            mock_repo.get_by_id.return_value = session

            await worker._mark_failed(session, reason="test")

        mocks["job_sweep"].sweep_session_best_effort.assert_awaited_once()
        call_kwargs = mocks["job_sweep"].sweep_session_best_effort.call_args.kwargs
        assert call_kwargs["product_id"] == "synthara"

    async def test_mark_failed_without_sweep_service_still_works(self) -> None:
        """Worker with no sweep (job_sweep_service=None) behaves as before."""
        worker, _mocks = _make_worker()
        assert worker._job_sweep is None
        session = _make_gpu_session(
            status=GpuSessionStatus.pending,
            vastai_instance_id=None,
            cf_tunnel_id=None,
            cf_dns_record_id=None,
        )

        with patch(_REPO_PATH) as MockRepo:
            mock_repo = AsyncMock()
            MockRepo.return_value = mock_repo
            mock_repo.get_by_id.return_value = session

            await worker._mark_failed(session, reason="test")

        mock_repo.update_status.assert_awaited()
        assert mock_repo.update_status.await_args.args[1] == GpuSessionStatus.failed


# ---------------------------------------------------------------------------
# Regression tests for 2026-05-12 incident fixes
# ---------------------------------------------------------------------------


class TestAdvancePendingNullInstance:
    async def test_none_instance_is_noop_before_timeout(self) -> None:
        """get_instance returns None → fall through without transition (no infinite loop)."""
        worker, mocks = _make_worker()
        session = _make_gpu_session(status=GpuSessionStatus.pending)
        mocks["vastai_client"].get_instance.return_value = None

        worker._retry_or_fail = AsyncMock()  # type: ignore[method-assign]

        with patch(_REPO_PATH) as MockRepo:
            mock_repo = AsyncMock()
            MockRepo.return_value = mock_repo

            await worker._advance_pending(session)

        mock_repo.update_status.assert_not_called()
        worker._retry_or_fail.assert_not_called()

    async def test_none_instance_triggers_retry_on_timeout(self) -> None:
        """get_instance returns None + timeout exceeded → _retry_or_fail fires."""
        worker, mocks = _make_worker()
        old_created = datetime.now(UTC) - timedelta(minutes=25)
        session = _make_gpu_session(status=GpuSessionStatus.pending, created_at=old_created)
        mocks["vastai_client"].get_instance.return_value = None

        worker._retry_or_fail = AsyncMock()  # type: ignore[method-assign]

        await worker._advance_pending(session)

        worker._retry_or_fail.assert_called_once_with(session, reason=_REASON_PENDING_TIMEOUT)

    async def test_transient_error_triggers_retry_on_timeout(self) -> None:
        """Exception from get_instance + timeout exceeded → _retry_or_fail fires."""
        worker, mocks = _make_worker()
        old_created = datetime.now(UTC) - timedelta(minutes=25)
        session = _make_gpu_session(status=GpuSessionStatus.pending, created_at=old_created)
        mocks["vastai_client"].get_instance.side_effect = VastAIError("schema", status_code=200)

        worker._retry_or_fail = AsyncMock()  # type: ignore[method-assign]

        await worker._advance_pending(session)

        worker._retry_or_fail.assert_called_once_with(
            session, reason=_REASON_PENDING_TIMEOUT_AFTER_ERRORS
        )

    async def test_transient_error_before_timeout_is_noop(self) -> None:
        """Exception from get_instance before timeout → no transition, no retry."""
        worker, mocks = _make_worker()
        session = _make_gpu_session(status=GpuSessionStatus.pending)
        mocks["vastai_client"].get_instance.side_effect = VastAIError("timeout", status_code=503)

        worker._retry_or_fail = AsyncMock()  # type: ignore[method-assign]

        with patch(_REPO_PATH) as MockRepo:
            mock_repo = AsyncMock()
            MockRepo.return_value = mock_repo

            await worker._advance_pending(session)

        mock_repo.update_status.assert_not_called()
        worker._retry_or_fail.assert_not_called()


class TestRetryOrFailOneAttemptPolicy:
    async def test_default_one_attempt_fails_terminally(self) -> None:
        """With provisioning_recreation_attempts=1 (default), attempt=2 >= 1+1=2 → terminal."""
        worker, mocks = _make_worker()  # default: provisioning_recreation_attempts=1
        session = _make_gpu_session(status=GpuSessionStatus.pending)
        mocks["vastai_client"].destroy_instance = AsyncMock()

        with patch(_REPO_PATH) as MockRepo:
            mock_repo = AsyncMock()
            MockRepo.return_value = mock_repo
            mock_repo.increment_provision_attempt.return_value = 2  # first retry = attempt 2
            reloaded = _make_gpu_session(status=GpuSessionStatus.pending)
            mock_repo.get_by_id.return_value = reloaded

            await worker._retry_or_fail(session, reason="test")

        mock_repo.update_status.assert_called_once()
        assert mock_repo.update_status.call_args[0][1] == GpuSessionStatus.failed
        # No new instance created — recreation was not attempted
        mocks["vastai_client"].create_instance.assert_not_called()

    async def test_recreation_attempts_3_allows_second_attempt(self) -> None:
        """With provisioning_recreation_attempts=3, attempt=2 < 3+1=4 → recreation fires."""
        worker, mocks = _make_worker(settings=_make_settings(provisioning_recreation_attempts=3))
        session = _make_gpu_session(
            status=GpuSessionStatus.pending,
            bundle_name="wan_2.2_i2v",
            bundle_version="260105-01",
            callback_token_hash="some-existing-hash",
        )
        mocks["vastai_client"].destroy_instance = AsyncMock()
        bundle = _make_bundle_mapping()
        mocks["bundle_index"].resolve_bundle_override.return_value = bundle
        mocks["vastai_client"].search_offers.return_value = [_make_offer()]
        mocks["cf_client"].get_tunnel_token.return_value = "tunnel-token"
        mocks["vastai_client"].create_instance.return_value = 77777

        with patch(_REPO_PATH) as MockRepo:
            mock_repo = AsyncMock()
            MockRepo.return_value = mock_repo
            mock_repo.increment_provision_attempt.return_value = 2  # attempt 2 < max+1=4
            reloaded = _make_gpu_session(status=GpuSessionStatus.pending)
            mock_repo.get_by_id.return_value = reloaded

            await worker._retry_or_fail(session, reason="timeout")

        # Recreation must have fired — create_instance should be called
        mocks["vastai_client"].create_instance.assert_called_once()
        # Session must NOT be marked failed
        failed_calls = [
            c
            for c in mock_repo.update_status.call_args_list
            if len(c[0]) > 1 and c[0][1] == GpuSessionStatus.failed
        ]
        assert not failed_calls


# ---------------------------------------------------------------------------
# TestClassifyTerminalState
# ---------------------------------------------------------------------------


class TestClassifyTerminalState:
    """Pure-function tests for the terminal-state classifier."""

    def test_exited_is_terminal(self) -> None:
        instance = VastAIInstance(id=1, actual_status="exited", cur_state="exited")
        assert _classify_terminal_state(instance) == "exited"

    def test_unknown_is_terminal(self) -> None:
        instance = VastAIInstance(id=1, actual_status="unknown", cur_state="running")
        assert _classify_terminal_state(instance) == "unknown"

    def test_offline_is_terminal(self) -> None:
        instance = VastAIInstance(id=1, actual_status="offline", cur_state="running")
        assert _classify_terminal_state(instance) == "offline"

    def test_created_with_stopped_is_terminal(self) -> None:
        """Regression for 2026-05-12 incident: container failed to start on host."""
        instance = VastAIInstance(id=1, actual_status="created", cur_state="stopped")
        assert _classify_terminal_state(instance) == "stopped_before_running"

    def test_running_is_not_terminal(self) -> None:
        instance = VastAIInstance(id=1, actual_status="running", cur_state="running")
        assert _classify_terminal_state(instance) is None

    def test_created_with_running_is_not_terminal(self) -> None:
        """Still in normal provisioning — container is being scheduled."""
        instance = VastAIInstance(id=1, actual_status="created", cur_state="running")
        assert _classify_terminal_state(instance) is None

    def test_none_actual_status_is_not_terminal(self) -> None:
        """Pre-detail Vast.ai responses must NOT be terminal — keep waiting."""
        instance = VastAIInstance(id=1, actual_status=None, cur_state=None)
        assert _classify_terminal_state(instance) is None

    def test_actual_status_running_with_cur_state_stopped_is_not_terminal(self) -> None:
        """Trust actual_status='running'; Vast.ai may not have propagated cur_state yet."""
        instance = VastAIInstance(id=1, actual_status="running", cur_state="stopped")
        assert _classify_terminal_state(instance) is None


# ---------------------------------------------------------------------------
# TestAdvancePendingFastFail
# ---------------------------------------------------------------------------


class TestAdvancePendingFastFail:
    async def test_fast_fails_on_exited(self) -> None:
        """Container exited → mark failed immediately, don't wait for timeout."""
        worker, mocks = _make_worker()
        session = _make_gpu_session(status=GpuSessionStatus.pending)
        mocks["vastai_client"].get_instance.return_value = VastAIInstance(
            id=12345, actual_status="exited", cur_state="exited"
        )
        worker._retry_or_fail = AsyncMock()  # type: ignore[method-assign]

        await worker._advance_pending(session)

        worker._retry_or_fail.assert_called_once_with(
            session, reason="vastai_terminal_state:exited"
        )

    async def test_fast_fails_on_unknown(self) -> None:
        worker, mocks = _make_worker()
        session = _make_gpu_session(status=GpuSessionStatus.pending)
        mocks["vastai_client"].get_instance.return_value = VastAIInstance(
            id=12345, actual_status="unknown", cur_state="running"
        )
        worker._retry_or_fail = AsyncMock()  # type: ignore[method-assign]

        await worker._advance_pending(session)

        worker._retry_or_fail.assert_called_once_with(
            session, reason="vastai_terminal_state:unknown"
        )

    async def test_fast_fails_on_offline(self) -> None:
        worker, mocks = _make_worker()
        session = _make_gpu_session(status=GpuSessionStatus.pending)
        mocks["vastai_client"].get_instance.return_value = VastAIInstance(
            id=12345, actual_status="offline", cur_state="running"
        )
        worker._retry_or_fail = AsyncMock()  # type: ignore[method-assign]

        await worker._advance_pending(session)

        worker._retry_or_fail.assert_called_once_with(
            session, reason="vastai_terminal_state:offline"
        )

    async def test_fast_fails_on_created_stopped(self) -> None:
        """Regression for 2026-05-12: container failed to start on host (port collision)."""
        worker, mocks = _make_worker()
        session = _make_gpu_session(status=GpuSessionStatus.pending)
        mocks["vastai_client"].get_instance.return_value = VastAIInstance(
            id=12345, actual_status="created", cur_state="stopped"
        )
        worker._retry_or_fail = AsyncMock()  # type: ignore[method-assign]

        await worker._advance_pending(session)

        worker._retry_or_fail.assert_called_once_with(
            session, reason="vastai_terminal_state:stopped_before_running"
        )

    async def test_keeps_waiting_on_normal_provisioning(self) -> None:
        """actual_status='created' with cur_state='running' is normal — keep polling."""
        worker, mocks = _make_worker()
        session = _make_gpu_session(status=GpuSessionStatus.pending)
        mocks["vastai_client"].get_instance.return_value = VastAIInstance(
            id=12345, actual_status="created", cur_state="running"
        )
        worker._retry_or_fail = AsyncMock()  # type: ignore[method-assign]
        worker._transition = AsyncMock()  # type: ignore[method-assign]

        await worker._advance_pending(session)

        worker._retry_or_fail.assert_not_called()
        worker._transition.assert_not_called()

    async def test_fast_fail_does_not_wait_for_timeout(self) -> None:
        """Terminal state fires even when well within the 20-minute timeout window."""
        worker, mocks = _make_worker()
        # Session just created — nowhere near the 20-minute timeout
        session = _make_gpu_session(
            status=GpuSessionStatus.pending,
            created_at=datetime.now(UTC) - timedelta(seconds=30),
        )
        mocks["vastai_client"].get_instance.return_value = VastAIInstance(
            id=12345, actual_status="exited", cur_state="exited"
        )
        worker._retry_or_fail = AsyncMock()  # type: ignore[method-assign]

        await worker._advance_pending(session)

        worker._retry_or_fail.assert_called_once()


# ---------------------------------------------------------------------------
# TestAdvanceProvisioningFastFail
# ---------------------------------------------------------------------------


class TestAdvanceProvisioningFastFail:
    def _mock_ok_response(self) -> MagicMock:
        resp = MagicMock(spec=httpx.Response)
        resp.status_code = 200
        return resp

    async def test_fast_fails_on_terminal_state_mid_provisioning(self) -> None:
        """Container dies after reaching 'running' → fail fast, skip ComfyUI probe."""
        worker, mocks = _make_worker()
        session = _make_gpu_session(status=GpuSessionStatus.provisioning)
        mocks["vastai_client"].get_instance.return_value = VastAIInstance(
            id=12345, actual_status="exited", cur_state="exited"
        )
        worker._retry_or_fail = AsyncMock()  # type: ignore[method-assign]

        await worker._advance_provisioning(session)

        worker._retry_or_fail.assert_called_once_with(
            session, reason="vastai_terminal_state:exited"
        )
        mocks["http_client"].get.assert_not_called()

    async def test_get_instance_error_does_not_abort_probe(self) -> None:
        """If get_instance fails during provisioning, proceed with the ComfyUI probe."""
        worker, mocks = _make_worker()
        session = _make_gpu_session(status=GpuSessionStatus.provisioning)
        mocks["vastai_client"].get_instance.side_effect = VastAIError("timeout", status_code=503)
        mocks["http_client"].get.return_value = self._mock_ok_response()
        worker._mark_active = AsyncMock()  # type: ignore[method-assign]

        await worker._advance_provisioning(session)

        worker._mark_active.assert_called_once()

    async def test_provisioning_without_instance_id_marks_failed(self) -> None:
        """vastai_instance_id=None at provisioning stage → fail immediately."""
        worker, _mocks = _make_worker()
        session = _make_gpu_session(status=GpuSessionStatus.provisioning, vastai_instance_id=None)
        worker._mark_failed = AsyncMock()  # type: ignore[method-assign]

        await worker._advance_provisioning(session)

        worker._mark_failed.assert_called_once()
        args = worker._mark_failed.call_args[0]
        assert args[0] is session

    async def test_advance_provisioning_logs_when_get_instance_raises(self) -> None:
        """Defensive get_instance call must log a warning on exception, not swallow silently.

        Regression for Sourcery review on PR #51: bare except at line 302 used to
        silently set instance=None. The probe must still run after the exception.
        """
        from structlog.testing import capture_logs

        worker, mocks = _make_worker()
        session = _make_gpu_session(status=GpuSessionStatus.provisioning)
        mocks["vastai_client"].get_instance.side_effect = VastAIError("transient", status_code=500)
        mocks["http_client"].get.side_effect = httpx.ConnectError("refused")

        with capture_logs() as logs:
            await worker._advance_provisioning(session)

        warning_logs = [log for log in logs if "vastai_get_instance_failed" in log.get("event", "")]
        assert warning_logs, "expected gpu_session.provision.vastai_get_instance_failed log"
        # Probe still ran — broad catch did not abort the flow
        mocks["http_client"].get.assert_called_once()


# ---------------------------------------------------------------------------
# TestHandleTerminalStateIfAny
# ---------------------------------------------------------------------------


class TestHandleTerminalStateIfAny:
    async def test_none_instance_returns_false(self) -> None:
        worker, _mocks = _make_worker()
        session = _make_gpu_session(status=GpuSessionStatus.pending)
        worker._retry_or_fail = AsyncMock()  # type: ignore[method-assign]

        result = await worker._handle_terminal_state_if_any(session, None, stage="pending")

        assert result is False
        worker._retry_or_fail.assert_not_called()

    async def test_non_terminal_instance_returns_false(self) -> None:
        worker, _mocks = _make_worker()
        session = _make_gpu_session(status=GpuSessionStatus.pending)
        instance = VastAIInstance(id=1, actual_status="running", cur_state="running")
        worker._retry_or_fail = AsyncMock()  # type: ignore[method-assign]

        result = await worker._handle_terminal_state_if_any(session, instance, stage="pending")

        assert result is False
        worker._retry_or_fail.assert_not_called()

    async def test_terminal_instance_returns_true_and_calls_retry_or_fail(self) -> None:
        worker, _mocks = _make_worker()
        session = _make_gpu_session(status=GpuSessionStatus.pending, vastai_instance_id=1)
        instance = VastAIInstance(id=1, actual_status="exited", cur_state="exited")
        worker._retry_or_fail = AsyncMock()  # type: ignore[method-assign]

        result = await worker._handle_terminal_state_if_any(session, instance, stage="pending")

        assert result is True
        worker._retry_or_fail.assert_called_once_with(
            session, reason="vastai_terminal_state:exited"
        )

    async def test_stage_field_present_in_log(self) -> None:
        """The stage kwarg must appear as a structured field in the warning log."""
        from structlog.testing import capture_logs

        worker, _mocks = _make_worker()
        session = _make_gpu_session(status=GpuSessionStatus.provisioning, vastai_instance_id=1)
        instance = VastAIInstance(id=1, actual_status="exited", cur_state="exited")
        worker._retry_or_fail = AsyncMock()  # type: ignore[method-assign]

        with capture_logs() as logs:
            await worker._handle_terminal_state_if_any(session, instance, stage="provisioning")

        matching = [log for log in logs if log.get("stage") == "provisioning"]
        assert matching, "expected stage='provisioning' as a structured log field"


# ---------------------------------------------------------------------------
# TestProbeComfyui — unit tests for _probe_comfyui
# ---------------------------------------------------------------------------


def _make_object_info_response(classes: list[str]) -> MagicMock:
    """Build a mock httpx.Response whose content is a JSON dict of class names → {}."""
    import json

    resp = MagicMock(spec=httpx.Response)
    resp.status_code = 200
    resp.content = json.dumps({cls: {} for cls in classes}).encode()
    return resp


def _make_object_info_with_checkpoint(
    available_checkpoints: list[str],
    extra_classes: list[str] | None = None,
) -> MagicMock:
    """Build a mock /object_info response with a proper CheckpointLoaderSimple shape.

    Produces the nested structure ComfyUI actually returns:
        {"CheckpointLoaderSimple": {"input": {"required": {"ckpt_name": [<list>, {}]}}}, ...}
    """
    import json

    body: dict[str, Any] = {
        "CheckpointLoaderSimple": {
            "input": {
                "required": {
                    "ckpt_name": [available_checkpoints, {}],
                }
            }
        }
    }
    for cls in extra_classes or []:
        body[cls] = {}
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = 200
    resp.content = json.dumps(body).encode()
    return resp


class TestProbeComfyui:
    async def test_marker_present_class_in_response_returns_true(self) -> None:
        worker, mocks = _make_worker()
        session = _make_gpu_session(readiness_marker_node_class="WanVideoSampler")
        mocks["http_client"].get.return_value = _make_object_info_response(
            ["WanVideoSampler", "KSampler", "CLIPTextEncode"]
        )

        result = await worker._probe_comfyui(session)

        assert result is True

    async def test_marker_present_class_missing_returns_false(self) -> None:
        worker, mocks = _make_worker()
        session = _make_gpu_session(readiness_marker_node_class="WanVideoSampler")
        mocks["http_client"].get.return_value = _make_object_info_response(
            ["KSampler", "CLIPTextEncode"]
        )

        result = await worker._probe_comfyui(session)

        assert result is False

    async def test_no_checkpoint_no_marker_logs_unverifiable_and_returns_true(self) -> None:
        """Zero-checkpoint bundle + no marker → probe_unverifiable at WARNING, returns True."""
        from structlog.testing import capture_logs

        worker, mocks = _make_worker()
        mocks["bundle_index"].get_model_filenames.return_value = []
        session = _make_gpu_session(readiness_marker_node_class=None)
        mocks["http_client"].get.return_value = _make_object_info_response(["KSampler"])

        with capture_logs() as logs:
            result = await worker._probe_comfyui(session)

        assert result is True
        unverifiable_logs = [
            log for log in logs if log.get("event") == "gpu_session.provision.probe_unverifiable"
        ]
        assert unverifiable_logs, "expected probe_unverifiable log"
        assert unverifiable_logs[0].get("log_level") == "warning"

    async def test_checkpoint_lookup_failed_returns_false(self) -> None:
        """get_model_filenames returning None (lookup/read error) → return False, not unverifiable."""
        from structlog.testing import capture_logs

        worker, mocks = _make_worker()
        mocks["bundle_index"].get_model_filenames.return_value = None
        session = _make_gpu_session(readiness_marker_node_class=None)
        mocks["http_client"].get.return_value = _make_object_info_response(
            ["Qwen-Rapid-AIO-NSFW-v19.safetensors"]
        )

        with capture_logs() as logs:
            result = await worker._probe_comfyui(session)

        assert result is False
        assert any(
            log.get("event") == "gpu_session.provision.probe_checkpoint_lookup_failed"
            for log in logs
        ), "expected probe_checkpoint_lookup_failed warning"

    async def test_invalid_json_returns_false(self) -> None:
        worker, mocks = _make_worker()
        session = _make_gpu_session(readiness_marker_node_class="WanVideoSampler")
        resp = MagicMock(spec=httpx.Response)
        resp.status_code = 200
        resp.content = b"not valid json {"

        mocks["http_client"].get.return_value = resp

        result = await worker._probe_comfyui(session)

        assert result is False

    async def test_non_dict_json_returns_false(self) -> None:
        import json

        worker, mocks = _make_worker()
        session = _make_gpu_session(readiness_marker_node_class="WanVideoSampler")
        resp = MagicMock(spec=httpx.Response)
        resp.status_code = 200
        resp.content = json.dumps(["KSampler", "CLIPTextEncode"]).encode()

        mocks["http_client"].get.return_value = resp

        result = await worker._probe_comfyui(session)

        assert result is False

    async def test_non_200_returns_false(self) -> None:
        worker, mocks = _make_worker()
        session = _make_gpu_session(readiness_marker_node_class="WanVideoSampler")
        resp = MagicMock(spec=httpx.Response)
        resp.status_code = 503

        mocks["http_client"].get.return_value = resp

        result = await worker._probe_comfyui(session)

        assert result is False

    async def test_httpx_error_returns_false(self) -> None:
        worker, mocks = _make_worker()
        session = _make_gpu_session(readiness_marker_node_class="WanVideoSampler")
        mocks["http_client"].get.side_effect = httpx.ConnectError("refused")

        result = await worker._probe_comfyui(session)

        assert result is False

    async def test_logs_first_five_classes_when_more_present(self) -> None:
        from structlog.testing import capture_logs

        worker, mocks = _make_worker()
        session = _make_gpu_session(readiness_marker_node_class="MissingNode")
        all_classes = [f"Node{i}" for i in range(10)]
        mocks["http_client"].get.return_value = _make_object_info_response(all_classes)

        with capture_logs() as logs:
            result = await worker._probe_comfyui(session)

        assert result is False
        missing_logs = [
            log for log in logs if log.get("event") == "gpu_session.provision.probe_marker_missing"
        ]
        assert missing_logs, "expected probe_marker_missing log"
        log = missing_logs[0]
        assert log.get("total_classes_registered") == 10
        sample = log.get("sample_classes")
        assert isinstance(sample, list)
        assert len(sample) == 5

    # ------------------------------------------------------------------
    # Checkpoint-presence tests (the fix for the 2026-06 production incident)
    # ------------------------------------------------------------------

    async def test_checkpoint_present_no_marker_returns_true(self) -> None:
        """Single-checkpoint bundle, checkpoint in /object_info, no marker → True (happy path)."""
        from structlog.testing import capture_logs

        worker, mocks = _make_worker()
        mocks["bundle_index"].get_model_filenames.return_value = [
            "Qwen-Rapid-AIO-NSFW-v19.safetensors"
        ]
        session = _make_gpu_session(readiness_marker_node_class=None)
        mocks["http_client"].get.return_value = _make_object_info_with_checkpoint(
            available_checkpoints=["Qwen-Rapid-AIO-NSFW-v19.safetensors", "some_other.safetensors"]
        )

        with capture_logs() as logs:
            result = await worker._probe_comfyui(session)

        assert result is True
        assert all(
            log.get("event") != "gpu_session.provision.probe_checkpoint_path_mismatch"
            for log in logs
        ), "exact match must not fire probe_checkpoint_path_mismatch"

    async def test_checkpoint_subfolder_returns_true_and_warns(self) -> None:
        """Checkpoint exposed as 'sub/Model.safetensors', bundle declares 'Model.safetensors'
        → True (basename match), but probe_checkpoint_path_mismatch WARNING is logged."""
        from structlog.testing import capture_logs

        worker, mocks = _make_worker()
        mocks["bundle_index"].get_model_filenames.return_value = ["Model.safetensors"]
        session = _make_gpu_session(readiness_marker_node_class=None)
        mocks["http_client"].get.return_value = _make_object_info_with_checkpoint(
            available_checkpoints=["sub/Model.safetensors"]
        )

        with capture_logs() as logs:
            result = await worker._probe_comfyui(session)

        assert result is True
        mismatch_logs = [
            log
            for log in logs
            if log.get("event") == "gpu_session.provision.probe_checkpoint_path_mismatch"
        ]
        assert mismatch_logs, "expected probe_checkpoint_path_mismatch warning"
        assert mismatch_logs[0].get("log_level") == "warning"

    async def test_checkpoint_absent_returns_false_and_logs(self) -> None:
        """Regression: checkpoint declared but absent in ComfyUI → False (the incident root cause)."""
        from structlog.testing import capture_logs

        worker, mocks = _make_worker()
        mocks["bundle_index"].get_model_filenames.return_value = [
            "Qwen-Rapid-AIO-NSFW-v19.safetensors"
        ]
        session = _make_gpu_session(readiness_marker_node_class=None)
        # ComfyUI only has a stock SD1.5 ckpt, not the declared Qwen checkpoint
        mocks["http_client"].get.return_value = _make_object_info_with_checkpoint(
            available_checkpoints=["v1-5-pruned-emaonly.safetensors"]
        )

        with capture_logs() as logs:
            result = await worker._probe_comfyui(session)

        assert result is False
        missing_logs = [
            log
            for log in logs
            if log.get("event") == "gpu_session.provision.probe_checkpoint_missing"
        ]
        assert missing_logs, "expected probe_checkpoint_missing log"
        assert missing_logs[0]["expected"] == "Qwen-Rapid-AIO-NSFW-v19.safetensors"

    async def test_malformed_object_info_returns_false_and_logs_shape(self) -> None:
        """200 but /object_info missing CheckpointLoaderSimple shape → False, probe_object_info_shape at WARNING."""
        import json

        from structlog.testing import capture_logs

        worker, mocks = _make_worker()
        mocks["bundle_index"].get_model_filenames.return_value = ["Qwen.safetensors"]
        session = _make_gpu_session(readiness_marker_node_class=None)
        # Response has CheckpointLoaderSimple but with wrong shape (empty dict instead of nested)
        resp = MagicMock(spec=httpx.Response)
        resp.status_code = 200
        resp.content = json.dumps({"CheckpointLoaderSimple": {}}).encode()
        mocks["http_client"].get.return_value = resp

        with capture_logs() as logs:
            result = await worker._probe_comfyui(session)

        assert result is False
        shape_logs = [
            log
            for log in logs
            if log.get("event") == "gpu_session.provision.probe_object_info_shape"
        ]
        assert shape_logs, "expected probe_object_info_shape log"
        assert shape_logs[0].get("log_level") == "warning"

    async def test_checkpoint_and_marker_both_present_returns_true(self) -> None:
        """Both checkpoint and marker checks pass → True."""
        worker, mocks = _make_worker()
        mocks["bundle_index"].get_model_filenames.return_value = ["Qwen.safetensors"]
        session = _make_gpu_session(readiness_marker_node_class="WanVideoSampler")
        mocks["http_client"].get.return_value = _make_object_info_with_checkpoint(
            available_checkpoints=["Qwen.safetensors"],
            extra_classes=["WanVideoSampler"],
        )

        result = await worker._probe_comfyui(session)

        assert result is True

    async def test_checkpoint_present_but_marker_missing_returns_false(self) -> None:
        """Checkpoint passes, but marker class absent → False."""
        worker, mocks = _make_worker()
        mocks["bundle_index"].get_model_filenames.return_value = ["Qwen.safetensors"]
        session = _make_gpu_session(readiness_marker_node_class="WanVideoSampler")
        mocks["http_client"].get.return_value = _make_object_info_with_checkpoint(
            available_checkpoints=["Qwen.safetensors"]
            # WanVideoSampler NOT in extra_classes
        )

        result = await worker._probe_comfyui(session)

        assert result is False


# ---------------------------------------------------------------------------
# TestMatchCheckpoint — unit tests for _match_checkpoint helper
# ---------------------------------------------------------------------------


class TestMatchCheckpoint:
    def test_exact_match_returns_true_exact(self) -> None:
        matched, exact = _match_checkpoint("model.safetensors", ["model.safetensors", "other.ckpt"])
        assert matched is True
        assert exact is True

    def test_subfolder_exposure_basename_match(self) -> None:
        matched, exact = _match_checkpoint("model.safetensors", ["sub/model.safetensors"])
        assert matched is True
        assert exact is False

    def test_genuinely_absent_returns_false(self) -> None:
        matched, exact = _match_checkpoint("wanted.safetensors", ["other.safetensors"])
        assert matched is False
        assert exact is False

    def test_empty_available_returns_false(self) -> None:
        matched, exact = _match_checkpoint("model.safetensors", [])
        assert matched is False
        assert exact is False


# ---------------------------------------------------------------------------
# TestReadyProbeMismatch — _advance_provisioning warns when node says ready but probe disagrees
# ---------------------------------------------------------------------------


class TestReadyProbeMismatch:
    async def test_probe_false_with_ready_phase_logs_mismatch(self) -> None:
        """When provisioning_phase='ready' but probe returns False → log ready_probe_mismatch."""
        from structlog.testing import capture_logs

        worker, mocks = _make_worker()
        session = _make_gpu_session(
            status=GpuSessionStatus.provisioning,
            provisioning_phase="ready",
        )
        mocks["vastai_client"].get_instance.return_value = VastAIInstance(
            id=12345, actual_status="running", cur_state="running"
        )
        # Probe returns False (checkpoint not yet present or ComfyUI not up)
        mocks["http_client"].get.side_effect = httpx.ConnectError("refused")

        with patch(_REPO_PATH) as MockRepo:
            mock_repo = AsyncMock()
            MockRepo.return_value = mock_repo

            with capture_logs() as logs:
                await worker._advance_provisioning(session)

        # Must log the mismatch
        mismatch_logs = [
            log for log in logs if log.get("event") == "gpu_session.provision.ready_probe_mismatch"
        ]
        assert mismatch_logs, "expected ready_probe_mismatch warning log"
        assert mismatch_logs[0].get("log_level") == "warning"
        # Must NOT mark the session active
        mock_repo.update_status.assert_not_called()

    async def test_probe_false_with_non_ready_phase_no_mismatch_log(self) -> None:
        """When phase is 'downloading' (not 'ready'), probe=False is normal — no mismatch log."""
        from structlog.testing import capture_logs

        worker, mocks = _make_worker()
        session = _make_gpu_session(
            status=GpuSessionStatus.provisioning,
            provisioning_phase="downloading",
        )
        mocks["vastai_client"].get_instance.return_value = VastAIInstance(
            id=12345, actual_status="running", cur_state="running"
        )
        mocks["http_client"].get.side_effect = httpx.ConnectError("refused")

        with patch(_REPO_PATH) as MockRepo:
            mock_repo = AsyncMock()
            MockRepo.return_value = mock_repo

            with capture_logs() as logs:
                await worker._advance_provisioning(session)

        mismatch_logs = [
            log for log in logs if log.get("event") == "gpu_session.provision.ready_probe_mismatch"
        ]
        assert not mismatch_logs, "ready_probe_mismatch must not fire for non-ready phase"


# ---------------------------------------------------------------------------
# TestProvisionVastaiInstance — unit tests for provision_vastai_instance
# ---------------------------------------------------------------------------


class TestProvisionVastaiInstance:
    async def test_passes_template_hash_id_when_set(self) -> None:
        from src.api.services.gpu_session._provisioning import provision_vastai_instance

        vastai = AsyncMock(spec=VastAIClient)
        vastai.create_instance.return_value = 42
        offer = _make_offer(offer_id=7)

        instance_id, selected = await provision_vastai_instance(
            vastai_client=vastai,
            offers=[offer],
            disk_gb=100,
            env={},
            template_hash_id="abc123hash",
            cooldown_store=NullNodeCooldownStore(),
            offer_walk_depth=3,
        )

        assert instance_id == 42
        assert selected is offer
        vastai.create_instance.assert_called_once_with(
            7,
            disk_gb=100,
            env={},
            template_hash_id="abc123hash",
        )

    async def test_falls_back_to_image_when_template_hash_none(self) -> None:
        from src.api.services.gpu_session._provisioning import provision_vastai_instance

        vastai = AsyncMock(spec=VastAIClient)
        vastai.create_instance.return_value = 99
        offer = _make_offer(offer_id=5)

        await provision_vastai_instance(
            vastai_client=vastai,
            offers=[offer],
            disk_gb=100,
            env={},
            template_hash_id=None,
            cooldown_store=NullNodeCooldownStore(),
            image="vastai/comfy:v0.15.1-cuda-12.9-py312",
            onstart_cmd="bash /start.sh",
            offer_walk_depth=3,
        )

        call_kwargs = vastai.create_instance.call_args[1]
        assert call_kwargs.get("image") == "vastai/comfy:v0.15.1-cuda-12.9-py312"
        assert call_kwargs.get("onstart_cmd") == "bash /start.sh"
        assert "template_hash_id" not in call_kwargs


# ---------------------------------------------------------------------------
# TestProvisioningTimeoutSettings — Phase 2A timeout fix
# ---------------------------------------------------------------------------


class TestProvisioningTimeoutSettings:
    """Tests for GPU_PROVISION_TIMEOUT_SECONDS wiring and the no-relaunch guard."""

    def test_timeout_used_from_settings_small_value_triggers(self) -> None:
        """_provision_timeout_exceeded respects gpu_provision_timeout_seconds, not a literal."""
        elapsed_seconds = 90
        created = datetime.now(UTC) - timedelta(seconds=elapsed_seconds)
        session = _make_gpu_session(status=GpuSessionStatus.provisioning, created_at=created)

        worker_short, _ = _make_worker(settings=_make_settings(gpu_provision_timeout_seconds=50))
        assert worker_short._provision_timeout_exceeded(session) is True

    def test_timeout_used_from_settings_large_value_does_not_trigger(self) -> None:
        """_provision_timeout_exceeded with a large timeout: same elapsed time is not a timeout."""
        elapsed_seconds = 90
        created = datetime.now(UTC) - timedelta(seconds=elapsed_seconds)
        session = _make_gpu_session(status=GpuSessionStatus.provisioning, created_at=created)

        worker_long, _ = _make_worker(settings=_make_settings(gpu_provision_timeout_seconds=200))
        assert worker_long._provision_timeout_exceeded(session) is False

    async def test_within_timeout_window_is_noop(self) -> None:
        """Elapsed < timeout and probe still fails: no provisioning_timeout event, no destroy."""
        from structlog.testing import capture_logs

        worker, mocks = _make_worker(settings=_make_settings(gpu_provision_timeout_seconds=2000))
        session = _make_gpu_session(
            status=GpuSessionStatus.provisioning,
            created_at=datetime.now(UTC) - timedelta(seconds=30),
        )
        mocks["vastai_client"].get_instance.return_value = VastAIInstance(
            id=12345, actual_status="running", cur_state="running"
        )
        resp = MagicMock()
        resp.status_code = 502
        mocks["http_client"].get.return_value = resp

        with patch(_REPO_PATH) as MockRepo:
            mock_repo = AsyncMock()
            MockRepo.return_value = mock_repo

            with capture_logs() as logs:
                await worker._advance_provisioning(session)

        timeout_logs = [entry for entry in logs if "provisioning_timeout" in entry.get("event", "")]
        assert not timeout_logs
        mocks["vastai_client"].destroy_instance.assert_not_called()
        mock_repo.update_status.assert_not_called()

    async def test_over_timeout_emits_events_and_fails_terminally(self) -> None:
        """Elapsed >= timeout: provisioning_timeout + attempts_exhausted logged, session → failed."""
        from structlog.testing import capture_logs

        billing_mock = AsyncMock()
        # provisioning_recreation_attempts=3: without the guard, recreation would fire
        worker, mocks = _make_worker(
            settings=_make_settings(
                provisioning_recreation_attempts=3,
                gpu_provision_timeout_seconds=60,
            ),
            billing_service=billing_mock,
        )
        session = _make_gpu_session(
            status=GpuSessionStatus.provisioning,
            created_at=datetime.now(UTC) - timedelta(seconds=120),
        )
        mocks["vastai_client"].get_instance.return_value = VastAIInstance(
            id=12345, actual_status="running", cur_state="running"
        )
        mocks["vastai_client"].destroy_instance = AsyncMock()

        with patch(_REPO_PATH) as MockRepo:
            mock_repo = AsyncMock()
            MockRepo.return_value = mock_repo
            mock_repo.increment_provision_attempt.return_value = 2
            mock_repo.get_by_id.return_value = _make_gpu_session(
                status=GpuSessionStatus.provisioning
            )

            with capture_logs() as logs:
                await worker._advance_provisioning(session)

        event_names = [entry.get("event", "") for entry in logs]
        assert any("provisioning_timeout" in e for e in event_names)
        assert any("attempts_exhausted" in e for e in event_names)
        failed_calls = [
            c
            for c in mock_repo.update_status.call_args_list
            if len(c[0]) > 1 and c[0][1] == GpuSessionStatus.failed
        ]
        assert failed_calls, "session must be transitioned to failed"
        billing_mock.refund.assert_awaited_once()

    async def test_provisioning_timeout_never_calls_create_instance(self) -> None:
        """_retry_or_fail with reason='provisioning_timeout' must never call create_instance,
        even when provisioning_recreation_attempts > 1 would normally allow recreation."""
        from structlog.testing import capture_logs

        worker, mocks = _make_worker(settings=_make_settings(provisioning_recreation_attempts=3))
        session = _make_gpu_session(status=GpuSessionStatus.provisioning)
        mocks["vastai_client"].destroy_instance = AsyncMock()

        with patch(_REPO_PATH) as MockRepo:
            mock_repo = AsyncMock()
            MockRepo.return_value = mock_repo
            # attempt=2 < 3+1=4, so without the guard recreation would fire
            mock_repo.increment_provision_attempt.return_value = 2
            mock_repo.get_by_id.return_value = _make_gpu_session(
                status=GpuSessionStatus.provisioning
            )

            with capture_logs() as logs:
                await worker._retry_or_fail(session, reason=_REASON_PROVISIONING_TIMEOUT)

        mocks["vastai_client"].create_instance.assert_not_called()
        assert any("attempts_exhausted" in entry.get("event", "") for entry in logs)


# ---------------------------------------------------------------------------
# TestStallLiveness (D5)
# ---------------------------------------------------------------------------


class TestStallLiveness:
    """Stall-based liveness: stall when callbacks stop flowing, fall back to ceiling otherwise."""

    async def test_no_callbacks_no_stall_check(self) -> None:
        """last_progress_at=None → stall logic must not fire; only fixed ceiling applies."""
        worker, mocks = _make_worker()
        session = _make_gpu_session(
            status=GpuSessionStatus.provisioning,
            last_progress_at=None,
        )
        # Instance is healthy
        mocks["vastai_client"].get_instance.return_value = VastAIInstance(
            id=12345, actual_status="running", cur_state="running"
        )
        # Probe fails so we don't transition to active
        mocks["http_client"].get.side_effect = httpx.ConnectError("refused")
        worker._retry_or_fail = AsyncMock()  # type: ignore[method-assign]

        await worker._advance_provisioning(session)

        # _retry_or_fail should NOT have been called (session is recent, no stall)
        worker._retry_or_fail.assert_not_called()

    async def test_stall_timeout_exceeded_triggers_stalled_retry(self) -> None:
        """last_progress_at older than stall_timeout → provisioning_stalled retry."""
        from src.api.services.gpu_session.provisioning_worker import _REASON_PROVISIONING_STALLED

        worker, mocks = _make_worker()
        # last_progress_at is 500s ago; stall_timeout is 420s → stalled
        stale_progress = datetime.now(UTC) - timedelta(seconds=500)
        session = _make_gpu_session(
            status=GpuSessionStatus.provisioning,
            last_progress_at=stale_progress,
        )
        mocks["vastai_client"].get_instance.return_value = VastAIInstance(
            id=12345, actual_status="running", cur_state="running"
        )
        worker._retry_or_fail = AsyncMock()  # type: ignore[method-assign]

        await worker._advance_provisioning(session)

        worker._retry_or_fail.assert_called_once_with(session, reason=_REASON_PROVISIONING_STALLED)

    async def test_recent_progress_no_stall(self) -> None:
        """last_progress_at is recent → no stall, continue probing."""
        worker, mocks = _make_worker()
        recent_progress = datetime.now(UTC) - timedelta(seconds=30)
        session = _make_gpu_session(
            status=GpuSessionStatus.provisioning,
            last_progress_at=recent_progress,
        )
        mocks["vastai_client"].get_instance.return_value = VastAIInstance(
            id=12345, actual_status="running", cur_state="running"
        )
        # Probe fails so no active transition
        mocks["http_client"].get.side_effect = httpx.ConnectError("refused")
        worker._retry_or_fail = AsyncMock()  # type: ignore[method-assign]

        await worker._advance_provisioning(session)

        worker._retry_or_fail.assert_not_called()

    async def test_fixed_ceiling_terminal_even_with_recent_progress(self) -> None:
        """Fixed ceiling backstop: even if callbacks are flowing, ceiling-hit is terminal."""
        worker, mocks = _make_worker()
        # Created 25 minutes ago → ceiling exceeded (settings default 1200s = 20min)
        old_created = datetime.now(UTC) - timedelta(minutes=25)
        # recent progress (would normally prevent stall)
        recent_progress = datetime.now(UTC) - timedelta(seconds=10)
        session = _make_gpu_session(
            status=GpuSessionStatus.provisioning,
            created_at=old_created,
            last_progress_at=recent_progress,
        )
        mocks["vastai_client"].get_instance.return_value = VastAIInstance(
            id=12345, actual_status="running", cur_state="running"
        )
        worker._retry_or_fail = AsyncMock()  # type: ignore[method-assign]

        await worker._advance_provisioning(session)

        worker._retry_or_fail.assert_called_once_with(session, reason=_REASON_PROVISIONING_TIMEOUT)


# ---------------------------------------------------------------------------
# TestNodeReportedFailure (D4)
# ---------------------------------------------------------------------------


class TestNodeReportedFailure:
    """Worker picks up node-reported failure on next sweep via provisioning_phase column."""

    async def test_node_failed_phase_triggers_retry(self) -> None:
        from src.api.services.gpu_session.provisioning_worker import _REASON_NODE_REPORTED_FAILURE

        worker, mocks = _make_worker()
        session = _make_gpu_session(
            status=GpuSessionStatus.provisioning,
            provisioning_phase="failed",
            provisioning_progress={"error": "comfyui install failed", "ts": "2026-06-09T10:00:00Z"},
        )
        mocks["vastai_client"].get_instance.return_value = VastAIInstance(
            id=12345, actual_status="running", cur_state="running"
        )
        worker._retry_or_fail = AsyncMock()  # type: ignore[method-assign]

        await worker._advance_provisioning(session)

        worker._retry_or_fail.assert_called_once_with(session, reason=_REASON_NODE_REPORTED_FAILURE)

    async def test_non_failed_phase_does_not_trigger_retry(self) -> None:
        """Other phases (downloading, ready, etc.) should NOT trigger _retry_or_fail."""
        worker, mocks = _make_worker()
        session = _make_gpu_session(
            status=GpuSessionStatus.provisioning,
            provisioning_phase="downloading",
        )
        mocks["vastai_client"].get_instance.return_value = VastAIInstance(
            id=12345, actual_status="running", cur_state="running"
        )
        mocks["http_client"].get.side_effect = httpx.ConnectError("refused")
        worker._retry_or_fail = AsyncMock()  # type: ignore[method-assign]

        await worker._advance_provisioning(session)

        worker._retry_or_fail.assert_not_called()

    async def test_stalled_reason_is_retryable_not_terminal(self) -> None:
        """provisioning_stalled must NOT hit the terminal guard in _retry_or_fail."""
        from src.api.services.gpu_session.provisioning_worker import _REASON_PROVISIONING_STALLED

        # provisioning_recreation_attempts=2 → new_attempt=2 < 2+1=3 → should recreate
        worker, mocks = _make_worker(settings=_make_settings(provisioning_recreation_attempts=2))
        session = _make_gpu_session(status=GpuSessionStatus.provisioning)
        bundle = _make_bundle_mapping()
        mocks["bundle_index"].resolve_bundle_override.return_value = bundle
        mocks["vastai_client"].search_offers.return_value = [_make_offer()]
        mocks["cf_client"].get_tunnel_token.return_value = "token"
        mocks["vastai_client"].destroy_instance = AsyncMock()
        mocks["vastai_client"].create_instance.return_value = 11111

        with patch(_REPO_PATH) as MockRepo:
            mock_repo = AsyncMock()
            MockRepo.return_value = mock_repo
            mock_repo.increment_provision_attempt.return_value = 2
            mock_repo.get_by_id.return_value = _make_gpu_session(
                status=GpuSessionStatus.provisioning
            )

            await worker._retry_or_fail(session, reason=_REASON_PROVISIONING_STALLED)

        # Should create a new instance (retryable), NOT mark failed
        mocks["vastai_client"].create_instance.assert_called_once()
        # update_status should set pending, not failed
        update_calls = [str(c) for c in mock_repo.update_status.call_args_list]
        assert all("failed" not in c for c in update_calls)
