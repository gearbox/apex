"""Health snapshot repository — insert, query, cleanup."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any, cast

import structlog
from sqlalchemy import delete, select
from sqlalchemy.engine import CursorResult

from src.db.models.health import HealthSnapshot

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

logger = structlog.get_logger()


class HealthSnapshotRepository:
    """Data access for health_snapshots table."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def insert(
        self,
        *,
        checked_at: datetime,
        overall_status: str,
        snapshot_data: dict[str, Any],
    ) -> None:
        """Insert a health snapshot row.

        Args:
            checked_at: When the check cycle ran.
            overall_status: Aggregate status string (healthy/degraded/unhealthy).
            snapshot_data: Full DetailedHealthResponse as dict.
        """
        snapshot = HealthSnapshot(
            checked_at=checked_at,
            overall_status=overall_status,
            snapshot_data=snapshot_data,
        )
        self._session.add(snapshot)
        await self._session.flush()

    async def list_range(
        self,
        *,
        after: datetime | None = None,
        before: datetime | None = None,
        limit: int = 60,
    ) -> list[HealthSnapshot]:
        """Query snapshots in a time range for dashboard charts.

        Args:
            after: Only snapshots after this time (inclusive).
            before: Only snapshots before this time (inclusive).
            limit: Maximum results (default 60 = 1 hour at 1/min interval).

        Returns:
            List of HealthSnapshot ordered by checked_at DESC.
        """
        stmt = select(HealthSnapshot).order_by(HealthSnapshot.checked_at.desc())
        if after is not None:
            stmt = stmt.where(HealthSnapshot.checked_at >= after)
        if before is not None:
            stmt = stmt.where(HealthSnapshot.checked_at <= before)
        stmt = stmt.limit(limit)
        result = await self._session.execute(stmt)
        return list(result.scalars().all())

    async def cleanup(self, *, retention_days: int) -> int:
        """Delete snapshots older than retention period.

        Args:
            retention_days: Delete snapshots older than this many days.

        Returns:
            Number of rows deleted.
        """
        cutoff = datetime.now(UTC) - timedelta(days=retention_days)
        stmt = delete(HealthSnapshot).where(HealthSnapshot.checked_at < cutoff)
        result = cast("CursorResult[Any]", await self._session.execute(stmt))
        await self._session.flush()
        return int(result.rowcount)
