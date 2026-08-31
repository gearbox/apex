"""Unit tests for OrphanedTunnelCleanupWorker."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

from src.api.services.cloudflare.client import CloudflareTunnelClient
from src.api.services.cloudflare.schemas import TunnelListEntry
from src.api.services.gpu_session.cleanup_worker import OrphanedTunnelCleanupWorker
from src.api.services.vastai.client import VastAIClient
from src.api.services.vastai.exceptions import InstanceNotFoundError, VastAIRateLimitError
from src.api.services.vastai.schemas import VastAIInstance
from src.core.enums import GpuSessionStatus
from src.db.models.gpu_session import GpuSession

_REPO_PATH = "src.api.services.gpu_session.cleanup_worker.GpuSessionRepository"


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def _make_tunnel(
    *,
    tunnel_id: str = "tun-abc",
    name: str = "gpu-session-01234567",
    created_at: datetime | None = None,
    deleted_at: datetime | None = None,
) -> TunnelListEntry:
    if created_at is None:
        created_at = datetime.now(UTC) - timedelta(hours=2)
    return TunnelListEntry(
        id=tunnel_id,
        name=name,
        status="healthy",
        created_at=created_at,
        deleted_at=deleted_at,
    )


def _make_gpu_session(
    *,
    cf_tunnel_id: str | None = "tun-abc",
    status: GpuSessionStatus = GpuSessionStatus.active,
) -> GpuSession:
    now = datetime.now(UTC)
    session = GpuSession()
    session.id = uuid4()
    session.user_id = uuid4()
    session.product_id = "vex"
    session.status = status
    session.bundle_name = "wan_2.2_i2v"
    session.bundle_version = "260105-01"
    session.model_type = "aisha-image"
    session.cf_tunnel_id = cf_tunnel_id
    session.cf_dns_record_id = "dns-abc"
    session.tunnel_hostname = "gpu-01234567.gpu-domain.com"
    session.vastai_instance_id = 12345
    session.vastai_offer_id = 67890
    session.vastai_cost_per_hour_micros = 500_000
    session.vastai_gpu_name = "RTX_4090"
    session.callback_token_hash = "tok"
    session.provision_attempt = 1
    session.provisioning_started_at = None
    session.started_at = now - timedelta(hours=1)
    session.paused_at = None
    session.resumed_at = None
    session.stopped_at = None
    session.created_at = now - timedelta(hours=2)
    session.error_message = None
    session.stale_detected_at = None
    session.stale_notified = False
    return session


def _make_settings(**overrides: Any) -> MagicMock:
    settings = MagicMock()
    settings.orphaned_tunnel_cleanup_interval_minutes = 60
    settings.orphaned_tunnel_cleanup_grace_period_minutes = 10
    settings.orphaned_instance_cleanup_grace_period_minutes = 5
    settings.orphaned_instance_cleanup_horizon_minutes = 1440
    settings.aisha_cf_tunnel_domain = "gpu-domain.com"
    for k, v in overrides.items():
        setattr(settings, k, v)
    return settings


def _make_mock_session_factory() -> tuple[MagicMock, MagicMock]:
    mock_db = MagicMock()
    mock_db.__aenter__ = AsyncMock(return_value=mock_db)
    mock_db.__aexit__ = AsyncMock(return_value=None)

    mock_begin = MagicMock()
    mock_begin.__aenter__ = AsyncMock(return_value=None)
    mock_begin.__aexit__ = AsyncMock(return_value=None)
    mock_db.begin = MagicMock(return_value=mock_begin)

    mock_factory = MagicMock(return_value=mock_db)
    return mock_factory, mock_db


def _make_worker(**overrides: Any) -> tuple[OrphanedTunnelCleanupWorker, dict[str, Any]]:
    mock_factory, mock_db = _make_mock_session_factory()
    mocks: dict[str, Any] = {
        "session_factory": mock_factory,
        "mock_db": mock_db,
        "cf_client": AsyncMock(spec=CloudflareTunnelClient),
        "vastai_client": AsyncMock(spec=VastAIClient),
        "settings": _make_settings(),
        "redis_client_factory": MagicMock(),
    } | overrides
    worker = OrphanedTunnelCleanupWorker(
        session_factory=mocks["session_factory"],
        cf_client=mocks["cf_client"],
        vastai_client=mocks["vastai_client"],
        settings=mocks["settings"],
        redis_client_factory=mocks["redis_client_factory"],
    )
    return worker, mocks


def _make_orphan_session(
    *,
    vastai_instance_id: int = 12345,
    status: GpuSessionStatus = GpuSessionStatus.stopped,
) -> GpuSession:
    now = datetime.now(UTC)
    session = GpuSession()
    session.id = uuid4()
    session.user_id = uuid4()
    session.product_id = "vex"
    session.status = status
    session.bundle_name = "wan_2.2_i2v"
    session.bundle_version = "260105-01"
    session.model_type = "aisha-image"
    session.vastai_instance_id = vastai_instance_id
    session.vastai_offer_id = 67890
    session.vastai_cost_per_hour_micros = 500_000
    session.vastai_gpu_name = "RTX_4090"
    session.vastai_instance_destroyed_at = None
    session.cf_tunnel_id = None
    session.cf_dns_record_id = None
    session.tunnel_hostname = None
    session.callback_token_hash = "tok"
    session.provision_attempt = 1
    session.provisioning_started_at = None
    session.started_at = now - timedelta(hours=2)
    session.paused_at = None
    session.resumed_at = None
    session.stopped_at = now - timedelta(minutes=30)
    session.created_at = now - timedelta(hours=3)
    session.error_message = None
    session.stale_detected_at = None
    session.stale_notified = False
    return session


# ---------------------------------------------------------------------------
# Lifecycle (start/stop/tick-error-recovery) is covered generically by
# tests/unit/test_periodic_worker.py — OrphanedTunnelCleanupWorker has no
# lifecycle overrides beyond PeriodicWorker.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# TestSweep
# ---------------------------------------------------------------------------


class TestSweep:
    async def test_tunnel_with_live_session_is_kept(self) -> None:
        """Tunnel claimed by a non-terminal session must NOT be deleted."""
        worker, mocks = _make_worker()

        tunnel = _make_tunnel(tunnel_id="tun-live")
        live_session = _make_gpu_session(cf_tunnel_id="tun-live", status=GpuSessionStatus.active)

        mocks["cf_client"].list_tunnels.return_value = [tunnel]

        with patch(_REPO_PATH) as MockRepo:
            mock_repo = AsyncMock()
            MockRepo.return_value = mock_repo
            mock_repo.list_by_status.return_value = [live_session]
            mock_repo.list_orphaned_instance_candidates.return_value = []

            await worker.run_once()

        mocks["cf_client"].delete_tunnel.assert_not_called()
        mocks["cf_client"].delete_session_tunnel.assert_not_called()

    async def test_tunnel_with_terminal_session_row_is_deleted(self) -> None:
        """Tunnel whose session is stopped (terminal) must be treated as orphaned."""
        worker, mocks = _make_worker()

        tunnel = _make_tunnel(tunnel_id="tun-term")
        # Only non-terminal sessions are fetched — terminal sessions aren't in live list
        mocks["cf_client"].list_tunnels.return_value = [tunnel]
        mocks["cf_client"].find_dns_record_by_hostname.return_value = "dns-term"

        with patch(_REPO_PATH) as MockRepo:
            mock_repo = AsyncMock()
            MockRepo.return_value = mock_repo
            # No live sessions claiming this tunnel
            mock_repo.list_by_status.return_value = []
            mock_repo.list_orphaned_instance_candidates.return_value = []

            await worker.run_once()

        mocks["cf_client"].delete_session_tunnel.assert_called_once_with("tun-term", "dns-term")

    async def test_tunnel_within_grace_period_is_kept(self) -> None:
        """Tunnel created recently must be skipped (grace period)."""
        worker, mocks = _make_worker()

        # Tunnel created 2 minutes ago, grace period is 10 minutes
        young_tunnel = _make_tunnel(
            tunnel_id="tun-young",
            created_at=datetime.now(UTC) - timedelta(minutes=2),
        )
        mocks["cf_client"].list_tunnels.return_value = [young_tunnel]

        with patch(_REPO_PATH) as MockRepo:
            mock_repo = AsyncMock()
            MockRepo.return_value = mock_repo
            mock_repo.list_by_status.return_value = []
            mock_repo.list_orphaned_instance_candidates.return_value = []

            await worker.run_once()

        mocks["cf_client"].delete_tunnel.assert_not_called()
        mocks["cf_client"].delete_session_tunnel.assert_not_called()

    async def test_tunnel_without_our_prefix_is_ignored(self) -> None:
        """Tunnels not named 'gpu-session-*' must be ignored."""
        worker, mocks = _make_worker()

        foreign_tunnel = _make_tunnel(tunnel_id="tun-foreign", name="not-our-tunnel")
        mocks["cf_client"].list_tunnels.return_value = [foreign_tunnel]

        with patch(_REPO_PATH) as MockRepo:
            mock_repo = AsyncMock()
            MockRepo.return_value = mock_repo
            mock_repo.list_by_status.return_value = []
            mock_repo.list_orphaned_instance_candidates.return_value = []

            await worker.run_once()

        mocks["cf_client"].delete_tunnel.assert_not_called()

    async def test_soft_deleted_tunnel_is_ignored(self) -> None:
        """Tunnels with deleted_at set (soft-deleted) must not be re-deleted.

        Defense-in-depth: list_tunnels already passes is_deleted=false to CF,
        but if CF ever surfaces a soft-deleted entry we skip it to avoid noise.
        """
        worker, mocks = _make_worker()

        orphan_age = datetime.now(UTC) - timedelta(hours=2)
        soft_deleted_tunnel = _make_tunnel(
            tunnel_id="tun-soft-deleted",
            name="gpu-session-deadbeef",
            created_at=orphan_age,
            deleted_at=datetime.now(UTC) - timedelta(minutes=30),
        )
        mocks["cf_client"].list_tunnels.return_value = [soft_deleted_tunnel]

        with patch(_REPO_PATH) as MockRepo:
            mock_repo = AsyncMock()
            MockRepo.return_value = mock_repo
            mock_repo.list_by_status.return_value = []
            mock_repo.list_orphaned_instance_candidates.return_value = []

            await worker.run_once()

        # Despite being orphaned + past the grace period, the soft-deleted
        # tunnel must not be touched.
        mocks["cf_client"].delete_tunnel.assert_not_called()
        mocks["cf_client"].find_dns_record_by_hostname.assert_not_called()

    async def test_delete_failure_continues_to_next_candidate(self) -> None:
        """If deleting one orphan fails, the worker must continue to the next."""
        worker, mocks = _make_worker()

        tunnel_a = _make_tunnel(tunnel_id="tun-a", name="gpu-session-aaaaaaaa")
        tunnel_b = _make_tunnel(tunnel_id="tun-b", name="gpu-session-bbbbbbbb")
        mocks["cf_client"].list_tunnels.return_value = [tunnel_a, tunnel_b]
        mocks["cf_client"].find_dns_record_by_hostname.return_value = "dns-x"

        call_count = 0

        async def flaky_delete(tunnel_id: str, _dns_id: str) -> None:
            nonlocal call_count
            call_count += 1
            if tunnel_id == "tun-a":
                raise RuntimeError("CF API error")

        mocks["cf_client"].delete_session_tunnel.side_effect = flaky_delete

        with patch(_REPO_PATH) as MockRepo:
            mock_repo = AsyncMock()
            MockRepo.return_value = mock_repo
            mock_repo.list_by_status.return_value = []
            mock_repo.list_orphaned_instance_candidates.return_value = []

            await worker.run_once()

        # Both tunnels were attempted
        assert call_count == 2

    async def test_multiple_orphans_all_deleted_independently(self) -> None:
        worker, mocks = _make_worker()

        tunnels = [
            _make_tunnel(tunnel_id=f"tun-{i}", name=f"gpu-session-0000000{i}") for i in range(3)
        ]
        mocks["cf_client"].list_tunnels.return_value = tunnels
        mocks["cf_client"].find_dns_record_by_hostname.return_value = "dns-x"

        with patch(_REPO_PATH) as MockRepo:
            mock_repo = AsyncMock()
            MockRepo.return_value = mock_repo
            mock_repo.list_by_status.return_value = []
            mock_repo.list_orphaned_instance_candidates.return_value = []

            await worker.run_once()

        assert mocks["cf_client"].delete_session_tunnel.call_count == 3

    async def test_empty_tunnel_list_is_noop(self) -> None:
        worker, mocks = _make_worker()
        mocks["cf_client"].list_tunnels.return_value = []

        with patch(_REPO_PATH) as MockRepo:
            mock_repo = AsyncMock()
            MockRepo.return_value = mock_repo
            mock_repo.list_by_status.return_value = []
            mock_repo.list_orphaned_instance_candidates.return_value = []

            await worker.run_once()

        mocks["cf_client"].delete_tunnel.assert_not_called()

    async def test_dns_record_not_found_tunnel_still_deleted(self) -> None:
        """If DNS record lookup returns None, delete the tunnel only (log warning)."""
        worker, mocks = _make_worker()

        tunnel = _make_tunnel(tunnel_id="tun-nodns")
        mocks["cf_client"].list_tunnels.return_value = [tunnel]
        mocks["cf_client"].find_dns_record_by_hostname.return_value = None

        with patch(_REPO_PATH) as MockRepo:
            mock_repo = AsyncMock()
            MockRepo.return_value = mock_repo
            mock_repo.list_by_status.return_value = []
            mock_repo.list_orphaned_instance_candidates.return_value = []

            await worker.run_once()

        mocks["cf_client"].delete_session_tunnel.assert_not_called()
        mocks["cf_client"].delete_tunnel.assert_called_once_with("tun-nodns")


# ---------------------------------------------------------------------------
# TestInstanceSweep — regression for 2026-05-13 incident
# ---------------------------------------------------------------------------

_INSTANCE_REPO_PATH = "src.api.services.gpu_session.cleanup_worker.GpuSessionRepository"


class TestInstanceSweep:
    async def test_destroys_alive_instance(self) -> None:
        """Regression for 2026-05-13: stopped session whose Vast.ai instance
        is still alive must be destroyed by the sweeper."""
        worker, mocks = _make_worker()
        session = _make_orphan_session(status=GpuSessionStatus.stopped)

        instance = MagicMock(spec=VastAIInstance)
        instance.actual_status = "running"
        instance.cur_state = "running"
        mocks["vastai_client"].get_instance = AsyncMock(return_value=instance)
        mocks["vastai_client"].destroy_instance = AsyncMock(return_value=None)

        with patch(_INSTANCE_REPO_PATH) as MockRepo:
            mock_repo = AsyncMock()
            MockRepo.return_value = mock_repo
            mock_repo.list_orphaned_instance_candidates.return_value = [session]
            mock_repo.mark_instance_destroyed = AsyncMock(return_value=None)

            await worker._sweep_instances()

        mocks["vastai_client"].destroy_instance.assert_called_once_with(session.vastai_instance_id)
        mock_repo.mark_instance_destroyed.assert_called_once()

    async def test_already_gone_404_just_marks_db(self) -> None:
        """If Vast.ai 404s, the instance is gone; just record in DB without destroying."""
        worker, mocks = _make_worker()
        session = _make_orphan_session(status=GpuSessionStatus.failed)

        mocks["vastai_client"].get_instance = AsyncMock(
            side_effect=InstanceNotFoundError("gone", status_code=404)
        )

        with patch(_INSTANCE_REPO_PATH) as MockRepo:
            mock_repo = AsyncMock()
            MockRepo.return_value = mock_repo
            mock_repo.list_orphaned_instance_candidates.return_value = [session]
            mock_repo.mark_instance_destroyed = AsyncMock(return_value=None)

            await worker._sweep_instances()

        mocks["vastai_client"].destroy_instance.assert_not_called()
        mock_repo.mark_instance_destroyed.assert_called_once()

    async def test_null_envelope_marks_db_without_destroy(self) -> None:
        """Vast.ai null envelope means instance is gone — mark DB, don't call destroy."""
        worker, mocks = _make_worker()
        session = _make_orphan_session()

        mocks["vastai_client"].get_instance = AsyncMock(return_value=None)

        with patch(_INSTANCE_REPO_PATH) as MockRepo:
            mock_repo = AsyncMock()
            MockRepo.return_value = mock_repo
            mock_repo.list_orphaned_instance_candidates.return_value = [session]
            mock_repo.mark_instance_destroyed = AsyncMock(return_value=None)

            await worker._sweep_instances()

        mocks["vastai_client"].destroy_instance.assert_not_called()
        mock_repo.mark_instance_destroyed.assert_called_once()

    async def test_rate_limited_skips_session(self) -> None:
        """VastAIRateLimitError during get_instance skips session for next sweep."""
        worker, mocks = _make_worker()
        session = _make_orphan_session()

        mocks["vastai_client"].get_instance = AsyncMock(
            side_effect=VastAIRateLimitError("rate limited", retry_after_seconds=5.0)
        )

        with patch(_INSTANCE_REPO_PATH) as MockRepo:
            mock_repo = AsyncMock()
            MockRepo.return_value = mock_repo
            mock_repo.list_orphaned_instance_candidates.return_value = [session]
            mock_repo.mark_instance_destroyed = AsyncMock(return_value=None)

            await worker._sweep_instances()

        mocks["vastai_client"].destroy_instance.assert_not_called()
        mock_repo.mark_instance_destroyed.assert_not_called()

    async def test_get_instance_error_skips_session(self) -> None:
        """Generic get_instance error skips session for next sweep (no crash)."""
        worker, mocks = _make_worker()
        session = _make_orphan_session()

        mocks["vastai_client"].get_instance = AsyncMock(
            side_effect=Exception("unexpected network error")
        )

        with patch(_INSTANCE_REPO_PATH) as MockRepo:
            mock_repo = AsyncMock()
            MockRepo.return_value = mock_repo
            mock_repo.list_orphaned_instance_candidates.return_value = [session]

            # Must not raise
            await worker._sweep_instances()

        mocks["vastai_client"].destroy_instance.assert_not_called()

    async def test_no_candidates_is_noop(self) -> None:
        """Empty candidate list → no API calls."""
        worker, mocks = _make_worker()

        with patch(_INSTANCE_REPO_PATH) as MockRepo:
            mock_repo = AsyncMock()
            MockRepo.return_value = mock_repo
            mock_repo.list_orphaned_instance_candidates.return_value = []

            await worker._sweep_instances()

        mocks["vastai_client"].get_instance.assert_not_called()
        mocks["vastai_client"].destroy_instance.assert_not_called()

    async def test_sweep_aborts_on_rate_limit(self) -> None:
        """Regression: sweep must abort when any candidate hits VastAIRateLimitError,
        not continue burning rate-limit budget on remaining candidates."""
        worker, mocks = _make_worker()
        sessions = [_make_orphan_session(vastai_instance_id=i) for i in range(1, 6)]

        # First candidate succeeds (instance alive → destroy), second rate-limits.
        alive = MagicMock(spec=VastAIInstance)
        alive.actual_status = "running"
        alive.cur_state = "running"
        mocks["vastai_client"].get_instance = AsyncMock(
            side_effect=[alive, VastAIRateLimitError("rate limited", retry_after_seconds=1.0)]
        )
        mocks["vastai_client"].destroy_instance = AsyncMock(return_value=None)

        with patch(_INSTANCE_REPO_PATH) as MockRepo:
            mock_repo = AsyncMock()
            MockRepo.return_value = mock_repo
            mock_repo.list_orphaned_instance_candidates.return_value = sessions
            mock_repo.mark_instance_destroyed = AsyncMock(return_value=None)

            # Must not raise — sweep aborts cleanly.
            await worker._sweep_instances()

        # First candidate: get_instance + destroy; second: get_instance (raises); rest: not called.
        assert mocks["vastai_client"].get_instance.call_count == 2
        assert mocks["vastai_client"].destroy_instance.call_count == 1

    async def test_sweep_aborts_marks_only_completed_candidates(self) -> None:
        """On rate-limit abort, only candidates processed before the error are stamped in DB."""
        worker, mocks = _make_worker()
        sessions = [_make_orphan_session(vastai_instance_id=i) for i in range(1, 4)]

        # All three candidates 404 (already gone), but the third rate-limits on get_instance.
        mocks["vastai_client"].get_instance = AsyncMock(
            side_effect=[
                InstanceNotFoundError("gone", status_code=404),
                InstanceNotFoundError("gone", status_code=404),
                VastAIRateLimitError("rate limited", retry_after_seconds=1.0),
            ]
        )

        with patch(_INSTANCE_REPO_PATH) as MockRepo:
            mock_repo = AsyncMock()
            MockRepo.return_value = mock_repo
            mock_repo.list_orphaned_instance_candidates.return_value = sessions
            mock_repo.mark_instance_destroyed = AsyncMock(return_value=None)

            await worker._sweep_instances()

        # Only the two successful 404s stamped in DB; third candidate aborted the sweep.
        assert mock_repo.mark_instance_destroyed.call_count == 2
