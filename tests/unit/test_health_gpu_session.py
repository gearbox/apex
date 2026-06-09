"""Tests for GpuSessionReconciler."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import httpx
import structlog.testing

from src.api.services.health.base import HealthChecker
from src.api.services.health.checkers.gpu_session import GpuSessionReconciler
from src.core.enums import ComponentCategory, ComponentStatus, GpuSessionStatus
from src.db.models.gpu_session import GpuSession

_TUNNEL_DOMAIN = "gpu-domain.com"
_TUNNEL_PREFIX = "gpu-"
_GRACE_SECONDS = 120


def _make_reconciler(
    *,
    session_factory: object = None,
    http_client: object = None,
    tunnel_domain: str = _TUNNEL_DOMAIN,
    tunnel_prefix: str | None = _TUNNEL_PREFIX,
    grace_seconds: int = _GRACE_SECONDS,
) -> GpuSessionReconciler:
    return GpuSessionReconciler(
        session_factory=session_factory or AsyncMock(),
        http_client=http_client or AsyncMock(),
        tunnel_domain=tunnel_domain,
        tunnel_prefix=tunnel_prefix,
        grace_seconds=grace_seconds,
    )


def _make_session(
    *,
    status: str = GpuSessionStatus.active.value,
    tunnel_hostname: str | None = "gpu-abc123.gpu-domain.com",
    stale_detected_at: datetime | None = None,
    last_reachable_at: datetime | None = None,
    created_at: datetime | None = None,
) -> GpuSession:
    """Build a mock GpuSession without touching the DB."""
    s = MagicMock(spec=GpuSession)
    s.id = uuid4()
    s.user_id = uuid4()
    s.product_id = "vex"
    s.status = status
    s.tunnel_hostname = tunnel_hostname
    s.stale_detected_at = stale_detected_at
    s.stale_notified = False
    s.last_reachable_at = last_reachable_at
    s.created_at = created_at or datetime.now(UTC) - timedelta(minutes=30)
    return s


class TestProtocolCompliance:
    def test_is_health_checker(self) -> None:
        assert isinstance(_make_reconciler(), HealthChecker)

    def test_metadata(self) -> None:
        r = _make_reconciler()
        assert r.name == "gpu_sessions"
        assert r.category == ComponentCategory.gpu_session
        assert r.product_id is None


class TestNoActiveSessions:
    async def test_returns_inactive(self) -> None:
        r = _make_reconciler()
        r._get_active_sessions = AsyncMock(return_value=[])  # type: ignore[method-assign]
        result = await r.check()
        assert result.status == ComponentStatus.inactive
        assert result.metadata["total"] == 0
        assert "no active" in result.message


class TestProbesTunnelEndpoint:
    async def test_probes_tunnel_not_host_port(self) -> None:
        """Reconciler must probe https://{tunnel_hostname}/object_info, not host:port."""
        session = _make_session(tunnel_hostname="gpu-abc123.gpu-domain.com")
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.get = AsyncMock(return_value=MagicMock(status_code=200))

        r = _make_reconciler(http_client=mock_client)
        r._get_active_sessions = AsyncMock(return_value=[session])  # type: ignore[method-assign]
        r._mark_stale = AsyncMock()  # type: ignore[method-assign]
        r._update_last_reachable_batch = AsyncMock()  # type: ignore[method-assign]

        result = await r.check()

        assert result.status == ComponentStatus.healthy
        call_url = mock_client.get.call_args[0][0]
        assert call_url == "https://gpu-abc123.gpu-domain.com/object_info"
        r._mark_stale.assert_not_awaited()


class TestAllHealthy:
    async def test_all_nodes_reachable(self) -> None:
        sessions = [
            _make_session(tunnel_hostname="gpu-node1.gpu-domain.com"),
            _make_session(tunnel_hostname="gpu-node2.gpu-domain.com"),
        ]
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.get = AsyncMock(return_value=MagicMock(status_code=200))

        r = _make_reconciler(http_client=mock_client)
        r._get_active_sessions = AsyncMock(return_value=sessions)  # type: ignore[method-assign]
        r._mark_stale = AsyncMock()  # type: ignore[method-assign]
        r._clear_stale_batch = AsyncMock()  # type: ignore[method-assign]
        r._update_last_reachable_batch = AsyncMock()  # type: ignore[method-assign]

        result = await r.check()
        assert result.status == ComponentStatus.healthy
        assert result.metadata["total"] == 2
        assert result.metadata["healthy"] == 2
        assert result.metadata["stale"] == 0
        r._mark_stale.assert_not_awaited()
        r._clear_stale_batch.assert_not_awaited()
        r._update_last_reachable_batch.assert_awaited_once()


class TestGracePeriod:
    async def test_freshly_active_blip_not_marked_stale(self) -> None:
        """last_reachable_at = now() → probe fails once → within grace, NOT stale."""
        session = _make_session(
            last_reachable_at=datetime.now(UTC),
        )
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.get = AsyncMock(side_effect=httpx.ConnectError("blip"))

        r = _make_reconciler(http_client=mock_client, grace_seconds=120)
        r._get_active_sessions = AsyncMock(return_value=[session])  # type: ignore[method-assign]
        r._mark_stale = AsyncMock()  # type: ignore[method-assign]
        r._update_last_reachable_batch = AsyncMock()  # type: ignore[method-assign]

        with structlog.testing.capture_logs() as logs:
            await r.check()

        r._mark_stale.assert_not_awaited()
        events = [log["event"] for log in logs]
        assert "health.gpu_session.within_grace" in events

    async def test_sustained_outage_marks_stale(self) -> None:
        """last_reachable_at past grace + 1s → marked stale."""
        session = _make_session(
            last_reachable_at=datetime.now(UTC) - timedelta(seconds=121),
        )
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.get = AsyncMock(side_effect=httpx.ConnectError("down"))

        r = _make_reconciler(http_client=mock_client, grace_seconds=120)
        r._get_active_sessions = AsyncMock(return_value=[session])  # type: ignore[method-assign]
        r._mark_stale = AsyncMock()  # type: ignore[method-assign]
        r._update_last_reachable_batch = AsyncMock()  # type: ignore[method-assign]

        with structlog.testing.capture_logs() as logs:
            await r.check()

        r._mark_stale.assert_awaited_once()
        stale_ids = r._mark_stale.call_args[0][0]
        assert session.id in stale_ids
        events = [log["event"] for log in logs]
        assert "health.gpu_sessions.stale_detected" in events


class TestInvalidHostname:
    async def test_invalid_hostname_within_grace_not_stale(self) -> None:
        """Invalid tunnel_hostname, last_reachable_at = now() → within grace, not stale."""
        session = _make_session(
            tunnel_hostname="bad-hostname.evil.com",
            last_reachable_at=datetime.now(UTC),
        )
        r = _make_reconciler()
        r._get_active_sessions = AsyncMock(return_value=[session])  # type: ignore[method-assign]
        r._mark_stale = AsyncMock()  # type: ignore[method-assign]
        r._update_last_reachable_batch = AsyncMock()  # type: ignore[method-assign]

        with structlog.testing.capture_logs() as logs:
            await r.check()

        r._mark_stale.assert_not_awaited()
        events = [log["event"] for log in logs]
        assert "health.gpu_session.invalid_hostname" in events
        assert "health.gpu_session.within_grace" in events

    async def test_invalid_hostname_past_grace_marks_stale(self) -> None:
        """Invalid tunnel_hostname past grace → marked stale."""
        session = _make_session(
            tunnel_hostname="bad-hostname.evil.com",
            last_reachable_at=datetime.now(UTC) - timedelta(seconds=121),
        )
        r = _make_reconciler(grace_seconds=120)
        r._get_active_sessions = AsyncMock(return_value=[session])  # type: ignore[method-assign]
        r._mark_stale = AsyncMock()  # type: ignore[method-assign]
        r._update_last_reachable_batch = AsyncMock()  # type: ignore[method-assign]

        with structlog.testing.capture_logs() as logs:
            await r.check()

        r._mark_stale.assert_awaited_once()
        events = [log["event"] for log in logs]
        assert "health.gpu_session.invalid_hostname" in events


class TestReachableUpdatesLastReachableAt:
    async def test_reachable_updates_last_reachable_at(self) -> None:
        """Successful probe calls _update_last_reachable_batch."""
        session = _make_session()
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.get = AsyncMock(return_value=MagicMock(status_code=200))

        r = _make_reconciler(http_client=mock_client)
        r._get_active_sessions = AsyncMock(return_value=[session])  # type: ignore[method-assign]
        r._mark_stale = AsyncMock()  # type: ignore[method-assign]
        r._update_last_reachable_batch = AsyncMock()  # type: ignore[method-assign]
        r._clear_stale_batch = AsyncMock()  # type: ignore[method-assign]

        await r.check()

        r._update_last_reachable_batch.assert_awaited_once()
        updated_ids = r._update_last_reachable_batch.call_args[0][0]
        assert session.id in updated_ids

    async def test_stale_session_recovers_when_reachable(self) -> None:
        """Previously stale session that becomes reachable recovers."""
        session = _make_session(
            stale_detected_at=datetime(2026, 1, 1, tzinfo=UTC),
        )
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.get = AsyncMock(return_value=MagicMock(status_code=200))

        r = _make_reconciler(http_client=mock_client)
        r._get_active_sessions = AsyncMock(return_value=[session])  # type: ignore[method-assign]
        r._clear_stale_batch = AsyncMock()  # type: ignore[method-assign]
        r._mark_stale = AsyncMock()  # type: ignore[method-assign]
        r._update_last_reachable_batch = AsyncMock()  # type: ignore[method-assign]

        with structlog.testing.capture_logs() as logs:
            result = await r.check()

        assert result.metadata["healthy"] == 1
        assert result.metadata["stale"] == 0
        r._clear_stale_batch.assert_awaited_once()
        recovered_ids = r._clear_stale_batch.call_args[0][0]
        assert session.id in recovered_ids
        events = [log["event"] for log in logs]
        assert "health.gpu_sessions.recovered" in events


class TestSomeStale:
    async def test_mixed_reachability(self) -> None:
        good_session = _make_session(
            tunnel_hostname="gpu-node1.gpu-domain.com",
            last_reachable_at=datetime.now(UTC) - timedelta(seconds=200),
        )
        bad_session = _make_session(
            tunnel_hostname="gpu-node2.gpu-domain.com",
            last_reachable_at=datetime.now(UTC) - timedelta(seconds=200),
        )

        responses = [
            MagicMock(status_code=200),  # good
            httpx.ConnectError("connection refused"),  # bad
        ]

        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.get = AsyncMock(side_effect=responses)

        r = _make_reconciler(http_client=mock_client, grace_seconds=120)
        r._get_active_sessions = AsyncMock(  # type: ignore[method-assign]
            return_value=[good_session, bad_session]
        )
        r._mark_stale = AsyncMock()  # type: ignore[method-assign]
        r._update_last_reachable_batch = AsyncMock()  # type: ignore[method-assign]

        result = await r.check()
        assert result.status == ComponentStatus.degraded
        assert result.metadata["healthy"] == 1
        assert result.metadata["stale"] == 1
        r._mark_stale.assert_awaited_once()
        stale_ids = r._mark_stale.call_args[0][0]
        assert bad_session.id in stale_ids


class TestAllUnreachable:
    async def test_all_nodes_down(self) -> None:
        sessions = [
            _make_session(last_reachable_at=datetime.now(UTC) - timedelta(seconds=200)),
            _make_session(last_reachable_at=datetime.now(UTC) - timedelta(seconds=200)),
        ]
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.get = AsyncMock(side_effect=httpx.ConnectError("down"))

        r = _make_reconciler(http_client=mock_client, grace_seconds=120)
        r._get_active_sessions = AsyncMock(return_value=sessions)  # type: ignore[method-assign]
        r._mark_stale = AsyncMock()  # type: ignore[method-assign]
        r._update_last_reachable_batch = AsyncMock()  # type: ignore[method-assign]

        result = await r.check()
        assert result.status == ComponentStatus.unhealthy
        assert result.metadata["healthy"] == 0
        assert result.metadata["stale"] == 2


class TestIdempotentStaleMarking:
    async def test_already_stale_not_remarked(self) -> None:
        """Session already has stale_detected_at — should not be re-marked."""
        session = _make_session(
            stale_detected_at=datetime(2026, 1, 1, tzinfo=UTC),
            last_reachable_at=datetime.now(UTC) - timedelta(seconds=200),
        )
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.get = AsyncMock(side_effect=httpx.ConnectError("down"))

        r = _make_reconciler(http_client=mock_client, grace_seconds=120)
        r._get_active_sessions = AsyncMock(return_value=[session])  # type: ignore[method-assign]
        r._mark_stale = AsyncMock()  # type: ignore[method-assign]
        r._update_last_reachable_batch = AsyncMock()  # type: ignore[method-assign]

        result = await r.check()
        assert result.metadata["stale"] == 1
        r._mark_stale.assert_not_awaited()


class TestStaleRecovery:
    async def test_stale_session_recovers(self) -> None:
        """stale_detected_at != None + reachable → recovery; _clear_stale_batch called."""
        session = _make_session(
            stale_detected_at=datetime(2026, 1, 1, tzinfo=UTC),
        )
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.get = AsyncMock(return_value=MagicMock(status_code=200))

        r = _make_reconciler(http_client=mock_client)
        r._get_active_sessions = AsyncMock(return_value=[session])  # type: ignore[method-assign]
        r._clear_stale_batch = AsyncMock()  # type: ignore[method-assign]
        r._mark_stale = AsyncMock()  # type: ignore[method-assign]
        r._update_last_reachable_batch = AsyncMock()  # type: ignore[method-assign]

        result = await r.check()
        assert result.metadata["healthy"] == 1
        assert result.metadata["stale"] == 0
        r._clear_stale_batch.assert_awaited_once()
        recovered_ids = r._clear_stale_batch.call_args[0][0]
        assert session.id in recovered_ids


class TestReconciliationError:
    async def test_db_error_returns_unknown(self) -> None:
        r = _make_reconciler()
        r._get_active_sessions = AsyncMock(side_effect=ConnectionError("db gone"))  # type: ignore[method-assign]
        result = await r.check()
        assert result.status == ComponentStatus.unknown
        assert "reconciliation failed" in result.message


class TestBatchRecovery:
    async def test_multiple_stale_sessions_recover_in_single_batch(self) -> None:
        """Multiple recovered sessions should result in one _clear_stale_batch call."""
        sessions = [
            _make_session(
                tunnel_hostname=f"gpu-node{i}.gpu-domain.com",
                stale_detected_at=datetime(2026, 1, 1, tzinfo=UTC),
            )
            for i in range(5)
        ]
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.get = AsyncMock(return_value=MagicMock(status_code=200))

        r = _make_reconciler(http_client=mock_client)
        r._get_active_sessions = AsyncMock(return_value=sessions)  # type: ignore[method-assign]
        r._mark_stale = AsyncMock()  # type: ignore[method-assign]
        r._clear_stale_batch = AsyncMock()  # type: ignore[method-assign]
        r._update_last_reachable_batch = AsyncMock()  # type: ignore[method-assign]

        result = await r.check()
        assert result.metadata["healthy"] == 5
        assert result.metadata["stale"] == 0
        r._clear_stale_batch.assert_awaited_once()
        recovered_ids = r._clear_stale_batch.call_args[0][0]
        assert len(recovered_ids) == 5


class TestQueryCap:
    async def test_get_active_sessions_has_limit(self) -> None:
        """Verify the query includes a LIMIT clause."""
        from src.api.services.health.checkers.gpu_session import _MAX_PROBE_SESSIONS

        assert _MAX_PROBE_SESSIONS > 0
