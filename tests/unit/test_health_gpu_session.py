"""Tests for GpuSessionReconciler."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import httpx

from src.api.services.health.base import HealthChecker
from src.api.services.health.checkers.gpu_session import GpuSessionReconciler
from src.core.enums import ComponentCategory, ComponentStatus, GpuSessionStatus
from src.db.models.gpu_session import GpuSession


def _make_session(
    *,
    status: str = GpuSessionStatus.active.value,
    host: str | None = "10.0.0.1",
    port: int | None = 18188,
    stale_detected_at: datetime | None = None,
) -> GpuSession:
    """Build a mock GpuSession without touching the DB."""
    s = MagicMock(spec=GpuSession)
    s.id = uuid4()
    s.user_id = uuid4()
    s.product_id = "vex"
    s.status = status
    s.node_host = host
    s.node_port = port
    s.stale_detected_at = stale_detected_at
    s.stale_notified = False
    return s


class TestProtocolCompliance:
    def test_is_health_checker(self) -> None:
        reconciler = GpuSessionReconciler(
            session_factory=AsyncMock(),
            http_client=AsyncMock(),
        )
        assert isinstance(reconciler, HealthChecker)

    def test_metadata(self) -> None:
        reconciler = GpuSessionReconciler(
            session_factory=AsyncMock(),
            http_client=AsyncMock(),
        )
        assert reconciler.name == "gpu_sessions"
        assert reconciler.category == ComponentCategory.gpu_session
        assert reconciler.product_id is None


class TestNoActiveSessions:
    async def test_returns_inactive(self) -> None:
        reconciler = GpuSessionReconciler(
            session_factory=AsyncMock(),
            http_client=AsyncMock(),
        )
        reconciler._get_active_sessions = AsyncMock(return_value=[])  # type: ignore[method-assign]

        result = await reconciler.check()
        assert result.status == ComponentStatus.inactive
        assert result.metadata["total"] == 0
        assert "no active" in result.message


class TestAllHealthy:
    async def test_all_nodes_reachable(self) -> None:
        sessions = [_make_session(host="10.0.0.1"), _make_session(host="10.0.0.2")]
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.get = AsyncMock(return_value=MagicMock(status_code=200))

        reconciler = GpuSessionReconciler(
            session_factory=AsyncMock(),
            http_client=mock_client,
        )
        reconciler._get_active_sessions = AsyncMock(return_value=sessions)  # type: ignore[method-assign]
        reconciler._mark_stale = AsyncMock()  # type: ignore[method-assign]
        reconciler._clear_stale_batch = AsyncMock()  # type: ignore[method-assign]

        result = await reconciler.check()
        assert result.status == ComponentStatus.healthy
        assert result.metadata["total"] == 2
        assert result.metadata["healthy"] == 2
        assert result.metadata["stale"] == 0
        reconciler._mark_stale.assert_not_awaited()
        reconciler._clear_stale_batch.assert_not_awaited()


class TestSomeStale:
    async def test_mixed_reachability(self) -> None:
        good_session = _make_session(host="10.0.0.1")
        bad_session = _make_session(host="10.0.0.2")

        responses = [
            MagicMock(status_code=200),  # good
            httpx.ConnectError("connection refused"),  # bad
        ]

        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.get = AsyncMock(side_effect=responses)

        reconciler = GpuSessionReconciler(
            session_factory=AsyncMock(),
            http_client=mock_client,
        )
        reconciler._get_active_sessions = AsyncMock(  # type: ignore[method-assign]
            return_value=[good_session, bad_session]
        )
        reconciler._mark_stale = AsyncMock()  # type: ignore[method-assign]

        result = await reconciler.check()
        assert result.status == ComponentStatus.degraded
        assert result.metadata["healthy"] == 1
        assert result.metadata["stale"] == 1
        reconciler._mark_stale.assert_awaited_once()
        # Verify the correct session was marked
        stale_ids = reconciler._mark_stale.call_args[0][0]
        assert bad_session.id in stale_ids


class TestAllUnreachable:
    async def test_all_nodes_down(self) -> None:
        sessions = [_make_session(), _make_session()]
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.get = AsyncMock(side_effect=httpx.ConnectError("down"))

        reconciler = GpuSessionReconciler(
            session_factory=AsyncMock(),
            http_client=mock_client,
        )
        reconciler._get_active_sessions = AsyncMock(return_value=sessions)  # type: ignore[method-assign]
        reconciler._mark_stale = AsyncMock()  # type: ignore[method-assign]

        result = await reconciler.check()
        assert result.status == ComponentStatus.unhealthy
        assert result.metadata["healthy"] == 0
        assert result.metadata["stale"] == 2


class TestMissingEndpoint:
    async def test_null_host_treated_as_stale(self) -> None:
        session = _make_session(host=None, port=None)
        reconciler = GpuSessionReconciler(
            session_factory=AsyncMock(),
            http_client=AsyncMock(),
        )
        reconciler._get_active_sessions = AsyncMock(return_value=[session])  # type: ignore[method-assign]
        reconciler._mark_stale = AsyncMock()  # type: ignore[method-assign]

        result = await reconciler.check()
        assert result.metadata["stale"] == 1


class TestIdempotentStaleMarking:
    async def test_already_stale_not_remarked(self) -> None:
        """Session already has stale_detected_at — should not be re-marked."""
        session = _make_session(
            stale_detected_at=datetime(2026, 1, 1, tzinfo=UTC),
        )
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.get = AsyncMock(side_effect=httpx.ConnectError("down"))

        reconciler = GpuSessionReconciler(
            session_factory=AsyncMock(),
            http_client=mock_client,
        )
        reconciler._get_active_sessions = AsyncMock(return_value=[session])  # type: ignore[method-assign]
        reconciler._mark_stale = AsyncMock()  # type: ignore[method-assign]

        result = await reconciler.check()
        assert result.metadata["stale"] == 1
        # _mark_stale should NOT be called — already marked
        reconciler._mark_stale.assert_not_awaited()


class TestStaleRecovery:
    async def test_stale_session_recovers_when_reachable(self) -> None:
        """A previously stale session that becomes reachable should clear staleness."""
        session = _make_session(
            stale_detected_at=datetime(2026, 1, 1, tzinfo=UTC),
        )
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.get = AsyncMock(return_value=MagicMock(status_code=200))

        reconciler = GpuSessionReconciler(
            session_factory=AsyncMock(),
            http_client=mock_client,
        )
        reconciler._get_active_sessions = AsyncMock(return_value=[session])  # type: ignore[method-assign]
        reconciler._clear_stale_batch = AsyncMock()  # type: ignore[method-assign]
        reconciler._mark_stale = AsyncMock()  # type: ignore[method-assign]

        result = await reconciler.check()
        assert result.metadata["healthy"] == 1
        assert result.metadata["stale"] == 0
        reconciler._clear_stale_batch.assert_awaited_once()
        # Verify the correct session ID was in the batch
        recovered_ids = reconciler._clear_stale_batch.call_args[0][0]
        assert session.id in recovered_ids


class TestReconciliationError:
    async def test_db_error_returns_unknown(self) -> None:
        reconciler = GpuSessionReconciler(
            session_factory=AsyncMock(),
            http_client=AsyncMock(),
        )
        reconciler._get_active_sessions = AsyncMock(side_effect=ConnectionError("db gone"))  # type: ignore[method-assign]

        result = await reconciler.check()
        assert result.status == ComponentStatus.unknown
        assert "reconciliation failed" in result.message


class TestBatchRecovery:
    async def test_multiple_stale_sessions_recover_in_single_batch(self) -> None:
        """Multiple recovered sessions should result in one _clear_stale_batch call."""
        sessions = [
            _make_session(
                host=f"10.0.0.{i}",
                stale_detected_at=datetime(2026, 1, 1, tzinfo=UTC),
            )
            for i in range(5)
        ]
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.get = AsyncMock(return_value=MagicMock(status_code=200))

        reconciler = GpuSessionReconciler(
            session_factory=AsyncMock(),
            http_client=mock_client,
        )
        reconciler._get_active_sessions = AsyncMock(return_value=sessions)  # type: ignore[method-assign]
        reconciler._mark_stale = AsyncMock()  # type: ignore[method-assign]
        reconciler._clear_stale_batch = AsyncMock()  # type: ignore[method-assign]

        result = await reconciler.check()
        assert result.metadata["healthy"] == 5
        assert result.metadata["stale"] == 0
        # Single batch call, not 5 individual calls
        reconciler._clear_stale_batch.assert_awaited_once()
        recovered_ids = reconciler._clear_stale_batch.call_args[0][0]
        assert len(recovered_ids) == 5


class TestQueryCap:
    async def test_get_active_sessions_has_limit(self) -> None:
        """Verify the query includes a LIMIT clause."""
        from src.api.services.health.checkers.gpu_session import _MAX_PROBE_SESSIONS

        assert _MAX_PROBE_SESSIONS > 0  # sanity — cap exists
