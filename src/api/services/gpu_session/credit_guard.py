"""Credit-aware GPU session guard.

Runs on the HealthSnapshotWorker cadence. Each cycle:
1. Loads active/stale sessions.
2. Settles consumed tokens since the last cycle (metered debit, clamped).
3. Evaluates the warning/terminate ladder.
4. Emits gpu_session.credit_warning SSE events on upward transitions.
5. Terminates sessions whose balance falls to the floor.

Key invariants:
- Never raises InsufficientBalanceError (settle_session_usage is clamped).
- Emits each level at most once per upward transition (state stored on DB row).
- De-escalates (clears warning) when balance rises above warning threshold
  (e.g. user tops up).
- Termination is dispatched via asyncio.create_task (fire-and-forget) so the
  guard cycle completes quickly regardless of stop latency.
"""

from __future__ import annotations

import asyncio
import dataclasses
from datetime import UTC, datetime
from math import ceil
from typing import TYPE_CHECKING
from uuid import UUID

import structlog
from sqlalchemy import select

from src.api.schemas.events import EventType, GpuSessionCreditWarningPayload
from src.core.enums import GpuSessionStatus, NotificationLevel
from src.db.models.gpu_session import GpuSession
from src.db.repositories.billing import BillingRepository
from src.db.repositories.gpu_session import GpuSessionRepository

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

    from src.api.services.billing import BillingService
    from src.api.services.event_bus import EventBus
    from src.api.services.gpu_session.service import GpuSessionService
    from src.core.config import Settings

logger = structlog.get_logger(__name__)

_MAX_GUARD_SESSIONS = 200


@dataclasses.dataclass(frozen=True)
class CreditGuardOutcome:
    """Result of evaluating a single session."""

    session_id: UUID
    balance: int
    floor_tokens: int
    warning_level: NotificationLevel | None
    should_terminate: bool
    settled_tokens: int


class SessionCreditGuard:
    """Per-cycle credit evaluation and metered settlement for active GPU sessions.

    Designed to run on the same cadence as HealthSnapshotWorker._run_once(),
    outside the health check timed section so its latency doesn't inflate the
    health check duration.
    """

    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        billing_service: BillingService,
        gpu_session_service: GpuSessionService,
        settings: Settings,
        event_bus: EventBus | None = None,
    ) -> None:
        self._session_factory = session_factory
        self._billing_service = billing_service
        self._gpu_session_service = gpu_session_service
        self._settings = settings
        self._event_bus = event_bus

    async def run_cycle(self) -> None:
        """Execute one guard cycle. Never raises — all errors are logged."""
        try:
            await self._run_cycle_inner()
        except Exception:
            logger.exception("credit_guard.cycle_error")

    async def _run_cycle_inner(self) -> None:
        sessions = await self._load_active_sessions()
        if not sessions:
            return

        logger.debug("credit_guard.cycle_start", count=len(sessions))

        for session in sessions:
            try:
                await self._evaluate_session(session)
            except Exception:
                logger.exception("credit_guard.session_error", session_id=str(session.id))

    async def _load_active_sessions(self) -> list[GpuSession]:
        """Load sessions that should be evaluated (active or stale)."""
        async with self._session_factory() as db:
            stmt = (
                select(GpuSession)
                .where(
                    GpuSession.status.in_(
                        [GpuSessionStatus.active.value, GpuSessionStatus.stale.value]
                    )
                )
                .order_by(GpuSession.created_at.desc())
                .limit(_MAX_GUARD_SESSIONS)
            )
            result = await db.execute(stmt)
            return list(result.scalars().all())

    def _compute_floor_tokens(self) -> int:
        """Derived floor: ceil(interval_minutes) × rate × safety_factor.

        The floor is the minimum balance required to cover one more full monitor
        cycle. Sessions below this threshold cannot afford even one more cycle
        and are terminated to prevent billing in arrears.
        """
        interval_minutes = self._settings.health_snapshot_interval_seconds / 60.0
        rate = self._settings.gpu_session_tokens_per_minute
        safety = self._settings.gpu_session_credit_safety_factor
        return int(ceil(interval_minutes * rate * safety))

    def _compute_owed_since_last_cycle(self, session: GpuSession) -> int:  # noqa: ARG002
        """Compute tokens owed for the current monitor interval.

        Returns the per-cycle token cost — a fixed amount regardless of
        when exactly we last settled. The guard settles this amount each
        cycle for any active/stale session that has been running.
        """
        interval_minutes = self._settings.health_snapshot_interval_seconds / 60.0
        rate = self._settings.gpu_session_tokens_per_minute
        return int(ceil(interval_minutes * rate))

    def _classify_level(self, balance: int, floor_tokens: int) -> NotificationLevel | None:
        """Return the warning level for the given balance, or None if no warning needed."""
        rate = self._settings.gpu_session_tokens_per_minute
        warning_tokens = self._settings.gpu_session_credit_warning_minutes * rate
        critical_tokens = self._settings.gpu_session_credit_critical_minutes * rate

        if balance <= floor_tokens:
            return NotificationLevel.CRITICAL
        if balance <= critical_tokens:
            return NotificationLevel.CRITICAL
        if balance <= warning_tokens:
            return NotificationLevel.WARNING
        return None

    def _should_emit(
        self,
        new_level: NotificationLevel | None,
        current_level: str | None,
    ) -> bool:
        """Determine if we should emit a new warning event.

        Emit on upward transitions only (INFO→WARNING→CRITICAL).
        De-escalation (balance recovered) clears the stored level without emitting.
        """
        if new_level is None:
            return False
        level_order = {
            NotificationLevel.WARNING: 1,
            NotificationLevel.CRITICAL: 2,
        }
        current_order = level_order.get(NotificationLevel(current_level), 0) if current_level else 0
        new_order = level_order.get(new_level, 0)
        return new_order > current_order

    async def _evaluate_session(self, session: GpuSession) -> None:
        """Evaluate and settle one session. Core logic of the guard."""
        if session.account_id is None:
            return

        # Only bill sessions that have actually started.
        if session.started_at is None:
            return

        now = datetime.now(UTC)
        floor_tokens = self._compute_floor_tokens()
        owed = self._compute_owed_since_last_cycle(session)
        rate = self._settings.gpu_session_tokens_per_minute

        # --- Metered settlement ---
        settled, new_balance, _ = await self._settle_metered(session, owed)

        # --- Classify warning level ---
        new_level = self._classify_level(new_balance, floor_tokens)
        current_level = session.credit_warning_level

        # --- Terminate if at/below floor ---
        if new_balance <= floor_tokens:
            logger.warning(
                "credit_guard.terminating_session",
                session_id=str(session.id),
                balance=new_balance,
                floor_tokens=floor_tokens,
            )
            # Publish terminal credit warning before terminating
            await self._publish_warning(
                session,
                level=NotificationLevel.CRITICAL,
                balance=new_balance,
                floor_tokens=floor_tokens,
                rate=rate,
                now=now,
            )
            await self._clear_warning(session)
            # Fire-and-forget: stop the session asynchronously
            asyncio.create_task(
                self._terminate_session(session),
                name=f"credit_guard_terminate_{session.id}",
            )
            return

        # --- Warning ladder ---
        if new_level is not None and self._should_emit(new_level, current_level):
            await self._publish_warning(
                session,
                level=new_level,
                balance=new_balance,
                floor_tokens=floor_tokens,
                rate=rate,
                now=now,
            )
            await self._persist_warning(session, new_level, now)
        elif new_level is None and current_level is not None:
            # Balance recovered above warning threshold — de-escalate
            await self._clear_warning(session)

    async def _settle_metered(self, session: GpuSession, owed: int) -> tuple[int, int, bool]:
        """Run settle_session_usage inside its own transaction."""
        async with self._session_factory() as db, db.begin():
            return await self._billing_service.settle_session_usage(
                session.account_id,  # type: ignore[arg-type]  # guarded by caller
                owed,
                session_id=session.id,
                model_type=session.model_type,
                session=db,
                product_id=session.product_id,
                user_id=session.user_id,
            )

    async def _persist_warning(
        self,
        session: GpuSession,
        level: NotificationLevel,
        warned_at: datetime,
    ) -> None:
        """Write the new credit warning level to the DB row."""
        async with self._session_factory() as db, db.begin():
            repo = GpuSessionRepository(db)
            await repo.update_credit_warning(session.id, level.value, warned_at)
        session.credit_warning_level = level.value
        session.credit_warned_at = warned_at

    async def _clear_warning(self, session: GpuSession) -> None:
        """Clear the credit warning level (de-escalate)."""
        async with self._session_factory() as db, db.begin():
            repo = GpuSessionRepository(db)
            await repo.update_credit_warning(session.id, None, None)
        session.credit_warning_level = None
        session.credit_warned_at = None

    async def _publish_warning(
        self,
        session: GpuSession,
        *,
        level: NotificationLevel,
        balance: int,
        floor_tokens: int,
        rate: int,
        now: datetime,
    ) -> None:
        """Emit gpu_session.credit_warning SSE event to the session owner."""
        if self._event_bus is None:
            return

        # minutes_remaining: how many more full minutes the session can run
        # terminate_at: estimated stop time if balance doesn't change
        minutes_remaining = max(0, (balance - floor_tokens) // rate) if rate > 0 else 0
        terminate_at: datetime | None = None
        if minutes_remaining > 0 and minutes_remaining < 60 * 24:
            from datetime import timedelta

            terminate_at = now + timedelta(minutes=minutes_remaining)

        try:
            await self._event_bus.publish(
                user_id=session.user_id,
                event_type=EventType.GPU_SESSION_CREDIT_WARNING,
                payload=GpuSessionCreditWarningPayload(
                    session_id=session.id,
                    level=level,
                    minutes_remaining=minutes_remaining,
                    terminate_at=terminate_at,
                    balance=balance,
                ),
            )
            logger.info(
                "credit_guard.warning_emitted",
                session_id=str(session.id),
                level=level.value,
                balance=balance,
                minutes_remaining=minutes_remaining,
            )
        except Exception:
            logger.exception("credit_guard.warning_publish_failed", session_id=str(session.id))

    async def _terminate_session(self, session: GpuSession) -> None:
        """Terminate the session due to insufficient credits.

        Called as asyncio.create_task so it doesn't block the guard cycle.
        """
        try:
            await self._gpu_session_service.stop_session(
                session_id=session.id,
                user_id=session.user_id,
                product_id=session.product_id,
                confirmed=True,
                reason="insufficient_credits",
            )
            logger.info(
                "credit_guard.session_terminated",
                session_id=str(session.id),
            )
        except Exception:
            logger.exception("credit_guard.terminate_failed", session_id=str(session.id))

    async def get_settled_tokens(self, session_id: UUID) -> int:
        """Return total tokens settled for a session (delegation to billing repo)."""
        async with self._session_factory() as db:
            repo = BillingRepository(db)
            return await repo.get_settled_tokens_for_session(session_id)
