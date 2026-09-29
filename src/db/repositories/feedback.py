"""Repository for in-product problem reports (``feedback_reports``).

Every query is scoped by ``product_id``. Never commits or flushes — callers
own the transaction (the route commits, then publishes the ops event).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from sqlalchemy import Select, literal, select, tuple_

from src.db.models.feedback import FeedbackReport
from src.db.models.user import User

if TYPE_CHECKING:
    from collections.abc import Sequence
    from datetime import datetime
    from uuid import UUID

    from sqlalchemy.ext.asyncio import AsyncSession

    from src.core.enums import FeedbackCategory, FeedbackStatus


class FeedbackReportRepository:
    """Data access layer for feedback_reports."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    def add(self, report: FeedbackReport) -> None:
        """Stage a new report. The id is app-side, so no flush is needed."""
        self._session.add(report)

    async def get(self, report_id: UUID, *, product_id: str) -> FeedbackReport | None:
        """Fetch one report of this product. No email join, no lock."""
        result = await self._session.execute(
            select(FeedbackReport).where(
                FeedbackReport.id == report_id, FeedbackReport.product_id == product_id
            )
        )
        return result.scalar_one_or_none()

    async def list_page(
        self,
        *,
        product_id: str,
        status: FeedbackStatus | None,
        category: FeedbackCategory | None,
        limit: int,
        cursor_ts: datetime | None = None,
        cursor_id: UUID | None = None,
    ) -> Sequence[tuple[FeedbackReport, str | None]]:
        """List reports with the reporter's email, newest first, keyset-paginated.

        Uses the limit+1 fetch pattern — the caller checks
        ``len(result) > limit`` to derive ``has_more``.
        """
        query = (
            select(FeedbackReport, User.email)
            .outerjoin(User, User.id == FeedbackReport.user_id)
            .where(FeedbackReport.product_id == product_id)
        )
        if status is not None:
            query = query.where(FeedbackReport.status == status)
        if category is not None:
            query = query.where(FeedbackReport.category == category)
        if cursor_ts is not None and cursor_id is not None:
            query = query.where(
                tuple_(FeedbackReport.created_at, FeedbackReport.id)
                < tuple_(literal(cursor_ts), literal(cursor_id))
            )

        result = await self._session.execute(
            query.order_by(FeedbackReport.created_at.desc(), FeedbackReport.id.desc()).limit(
                limit + 1
            )
        )
        return result.tuples().all()

    @staticmethod
    def _select_with_email(report_id: UUID, product_id: str) -> Select[tuple[FeedbackReport, str]]:
        """One report + the reporter's email (``None`` once the user is purged).

        Typed ``str`` because ``User.email`` is non-null; the outer join yields
        ``None`` at runtime, which callers' ``str | None`` return types cover.
        """
        return (
            select(FeedbackReport, User.email)
            .outerjoin(User, User.id == FeedbackReport.user_id)
            .where(FeedbackReport.id == report_id, FeedbackReport.product_id == product_id)
        )

    async def get_with_email(
        self,
        report_id: UUID,
        *,
        product_id: str,
    ) -> tuple[FeedbackReport, str | None] | None:
        """Fetch one report with the reporter's email (``None`` once the user is purged)."""
        result = await self._session.execute(self._select_with_email(report_id, product_id))
        return result.tuples().one_or_none()

    async def get_for_update(
        self, report_id: UUID, *, product_id: str
    ) -> tuple[FeedbackReport, str | None] | None:
        """Fetch one report and its reporter's email, locking only the report row.

        ``FOR UPDATE OF feedback_reports``: Postgres rejects FOR UPDATE on the
        nullable side of an outer join, and the users row must not be locked.
        ``populate_existing`` so a row already in the identity map is refreshed
        with the post-lock state rather than served stale.
        """
        result = await self._session.execute(
            self._select_with_email(report_id, product_id)
            .with_for_update(of=FeedbackReport)
            .execution_options(populate_existing=True)
        )
        return result.tuples().one_or_none()
