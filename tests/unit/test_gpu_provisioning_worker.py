"""Unit tests for GpuProvisioningWorker."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import httpx

from src.api.services.bundle_index import BundleIndexService
from src.api.services.cloudflare.client import CloudflareTunnelClient
from src.api.services.gpu_session.provisioning_worker import GpuProvisioningWorker
from src.api.services.vastai.client import VastAIClient
from src.api.services.vastai.exceptions import InstanceNotFoundError, VastAIError
from src.api.services.vastai.schemas import VastAIOffer
from src.core.bundle_config import BundleMapping, HardwareRequirements
from src.core.enums import GpuSessionStatus
from src.db.models.gpu_session import GpuSession

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
        comfyui_port=18188,
    )


def _make_bundle_mapping() -> BundleMapping:
    return BundleMapping(
        bundle_name="wan_2.2_i2v",
        bundle_version="260105-01",
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
    session.tunnel_hostname = "01234567.gpu.cloudin.space"
    session.callback_token = "callback-tok"
    session.provision_attempt = 1
    session.provisioning_started_at = None
    session.started_at = None
    session.paused_at = None
    session.resumed_at = None
    session.stopped_at = None
    session.created_at = now - timedelta(minutes=2)
    session.error_message = None
    session.node_host = None
    session.node_port = None
    session.stale_detected_at = None
    session.stale_notified = False
    for k, v in kwargs.items():
        setattr(session, k, v)
    return session


def _make_settings(**overrides: Any) -> MagicMock:
    settings = MagicMock()
    settings.gpu_provision_poll_interval_seconds = 15
    settings.gpu_provision_timeout_minutes = 20
    settings.gpu_resume_timeout_minutes = 5
    settings.max_node_provisioning_retries = 3
    settings.gpu_provision_worker_concurrency = 10
    settings.apex_callback_url = "https://apex.example.com/callback"
    settings.hf_token = "hf-tok"
    settings.civitai_api_token = "civitai-tok"
    settings.cf_tunnel_domain = "gpu.cloudin.space"
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
        "billing_service": None,
        "event_bus": None,
    } | overrides
    worker = GpuProvisioningWorker(
        session_factory=mocks["session_factory"],
        vastai_client=mocks["vastai_client"],
        cf_client=mocks["cf_client"],
        bundle_index=mocks["bundle_index"],
        http_client=mocks["http_client"],
        settings=mocks["settings"],
        billing_service=mocks["billing_service"],
        event_bus=mocks["event_bus"],
    )
    return worker, mocks


# ---------------------------------------------------------------------------
# TestLifecycle
# ---------------------------------------------------------------------------


class TestLifecycle:
    async def test_start_creates_task_and_sets_running(self) -> None:
        worker, _ = _make_worker()
        worker._run_loop = AsyncMock()  # type: ignore[method-assign]

        await worker.start()

        assert worker._running is True
        assert worker._task is not None
        await worker.stop()

    async def test_stop_cancels_task_and_clears_running(self) -> None:
        worker, _ = _make_worker()
        worker._run_loop = AsyncMock()  # type: ignore[method-assign]

        await worker.start()
        await worker.stop()

        assert worker._running is False
        assert worker._task is None

    async def test_start_is_idempotent(self) -> None:
        worker, _ = _make_worker()
        worker._run_loop = AsyncMock()  # type: ignore[method-assign]

        await worker.start()
        task_first = worker._task
        await worker.start()
        task_second = worker._task

        assert task_first is task_second
        await worker.stop()

    async def test_stop_without_start_is_noop(self) -> None:
        worker, _ = _make_worker()
        # Should not raise
        await worker.stop()
        assert worker._running is False

    async def test_loop_recovers_from_sweep_exception(self) -> None:
        """A sweep that raises must not kill the loop — next iteration still runs."""
        worker, _ = _make_worker()
        call_count = 0

        async def flaky_sweep() -> None:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("transient error")

        worker._sweep_once = flaky_sweep  # type: ignore[method-assign]
        worker._settings.gpu_provision_poll_interval_seconds = 0.01  # type: ignore[assignment]

        await worker.start()
        await asyncio.sleep(0.05)
        await worker.stop()

        assert call_count >= 2


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

            await worker._sweep_once()

        mocks["vastai_client"].get_instance.assert_not_called()

    async def test_sweep_processes_each_non_terminal_status(self) -> None:
        worker, mocks = _make_worker()

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

            await worker._sweep_once()

        worker._advance_pending.assert_called_once()
        worker._advance_provisioning.assert_called_once()
        worker._advance_resuming.assert_called_once()

    async def test_sweep_exception_in_one_session_does_not_abort_others(self) -> None:
        worker, mocks = _make_worker()

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
            await worker._sweep_once()

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

        worker._retry_or_fail.assert_called_once_with(session, reason="pending_timeout")

    async def test_pending_without_instance_id_marks_failed(self) -> None:
        worker, mocks = _make_worker()
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
        worker, mocks = _make_worker()
        session = _make_gpu_session(status=GpuSessionStatus.provisioning, started_at=None)
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

        worker._retry_or_fail.assert_called_once_with(session, reason="provisioning_timeout")
        mocks["http_client"].get.assert_not_called()

    async def test_retry_preserves_tunnel(self) -> None:
        """After retry, the session's tunnel ID and hostname must NOT change."""
        worker, mocks = _make_worker()
        original_tunnel_id = "tunnel-abc"
        original_hostname = "01234567.gpu.cloudin.space"
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

            await worker._retry_or_fail(session, reason="pending_timeout")

        # Tunnel ID must be unchanged (not deleted/recreated)
        mocks["cf_client"].delete_session_tunnel.assert_not_called()
        mocks["cf_client"].create_session_tunnel.assert_not_called()

    async def test_retry_exhausted_marks_failed(self) -> None:
        worker, mocks = _make_worker(settings=_make_settings(max_node_provisioning_retries=2))
        session = _make_gpu_session(status=GpuSessionStatus.pending)
        mocks["vastai_client"].destroy_instance = AsyncMock()

        with patch(_REPO_PATH) as MockRepo:
            mock_repo = AsyncMock()
            MockRepo.return_value = mock_repo
            # new_attempt = 3 > max_retries=2 → exhausted
            mock_repo.increment_provision_attempt.return_value = 3
            reloaded = _make_gpu_session(status=GpuSessionStatus.pending)
            mock_repo.get_by_id.return_value = reloaded

            await worker._retry_or_fail(session, reason="timeout")

        # Should mark failed after exhausting retries
        mock_repo.update_status.assert_called_once()
        assert mock_repo.update_status.call_args[0][1] == GpuSessionStatus.failed

    async def test_retry_reconstructs_env_vars_from_session_row(self) -> None:
        """The env dict passed to create_instance must contain all required vars."""
        worker, mocks = _make_worker()
        session = _make_gpu_session(
            status=GpuSessionStatus.pending,
            bundle_name="wan_2.2_i2v",
            bundle_version="260105-01",
            callback_token="cb-tok",
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
        assert env["ACS_APEX_CALLBACK_TOKEN"] == "cb-tok"
        assert "ACS_HF_TOKEN" in env
        assert "ACS_CIVITAI_API_TOKEN" in env


# ---------------------------------------------------------------------------
# TestAdvanceResuming
# ---------------------------------------------------------------------------


class TestAdvanceResuming:
    def _mock_ok_response(self) -> MagicMock:
        resp = MagicMock(spec=httpx.Response)
        resp.status_code = 200
        return resp

    async def test_probe_200_transitions_to_active_keeps_started_at(self) -> None:
        """started_at must NOT be overwritten when transitioning resuming → active."""
        worker, mocks = _make_worker()
        original_start = datetime.now(UTC) - timedelta(hours=5)
        session = _make_gpu_session(
            status=GpuSessionStatus.resuming,
            started_at=original_start,
            resumed_at=datetime.now(UTC) - timedelta(minutes=1),
        )
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
        worker, mocks = _make_worker()
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
        worker, mocks = _make_worker()
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
        worker, mocks = _make_worker()
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
        worker, mocks = _make_worker()
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
        worker, mocks = _make_worker()
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
        worker, mocks = _make_worker()
        session = _make_gpu_session(status=GpuSessionStatus.pending)
        mocks["vastai_client"].destroy_instance = AsyncMock()
        mocks["cf_client"].get_tunnel_token.return_value = "tok"
        bundle = _make_bundle_mapping()  # comfyui_port = 18188
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
        assert "-p 18188:18188" in env


class TestSweepConcurrency:
    async def test_sweep_bounds_concurrency_with_semaphore(self) -> None:
        """With gpu_provision_worker_concurrency=2 and 5 sessions, at most 2 _advance
        calls run in parallel at any instant.
        """
        worker, mocks = _make_worker(settings=_make_settings(gpu_provision_worker_concurrency=2))
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

            await worker._sweep_once()

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

            await worker._sweep_once()

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
        worker, mocks = _make_worker(billing_service=billing_mock)
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
        """GPU_SESSION_STATUS_CHANGED must be published when event_bus is present."""
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

            await worker._mark_failed(session, reason="test")

        event_bus_mock.publish.assert_awaited_once()

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
