"""Tests for HealthSnapshotRepository."""

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest  # noqa: F401

from src.db.models.health import HealthSnapshot
from src.db.repositories.health import HealthSnapshotRepository


def _make_session() -> AsyncMock:
    """Create a session mock with synchronous add() and async flush/execute."""
    session = AsyncMock()
    session.add = MagicMock()  # add() is synchronous in SQLAlchemy
    return session


class TestInsert:
    async def test_insert_creates_snapshot(self) -> None:
        mock_session = _make_session()
        repo = HealthSnapshotRepository(mock_session)
        await repo.insert(
            checked_at=datetime.now(UTC),
            overall_status="healthy",
            snapshot_data={"status": "healthy"},
        )
        mock_session.add.assert_called_once()
        mock_session.flush.assert_awaited_once()

    async def test_insert_sets_fields(self) -> None:
        mock_session = _make_session()
        repo = HealthSnapshotRepository(mock_session)
        now = datetime.now(UTC)
        data = {"status": "degraded", "checked_at": now.isoformat()}
        await repo.insert(
            checked_at=now,
            overall_status="degraded",
            snapshot_data=data,
        )
        added = mock_session.add.call_args[0][0]
        assert isinstance(added, HealthSnapshot)
        assert added.overall_status == "degraded"
        assert added.snapshot_data == data


class TestListRange:
    async def test_returns_list(self) -> None:
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = []
        mock_session = AsyncMock()
        mock_session.execute = AsyncMock(return_value=mock_result)

        repo = HealthSnapshotRepository(mock_session)
        result = await repo.list_range()
        assert result == []


class TestCleanup:
    async def test_cleanup_returns_count(self) -> None:
        mock_result = MagicMock()
        mock_result.rowcount = 42
        mock_session = AsyncMock()
        mock_session.execute = AsyncMock(return_value=mock_result)

        repo = HealthSnapshotRepository(mock_session)
        count = await repo.cleanup(retention_days=30)
        assert count == 42
        mock_session.flush.assert_awaited_once()
