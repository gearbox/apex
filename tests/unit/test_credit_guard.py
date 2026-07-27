"""Tests for SessionCreditGuard and settle_session_usage."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID, uuid4

import pytest

from src.api.services.billing import BillingService, SettleUsageResult
from src.api.services.billing_errors import AccountNotFoundError, InsufficientBalanceError
from src.api.services.gpu_session.credit_guard import SessionCreditGuard
from src.core.config import Settings
from src.core.enums import GpuSessionStatus, NotificationLevel, TransactionType
from src.core.product import PaymentProvider

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_settings(
    *,
    interval_seconds: int = 60,
    tokens_per_minute: int = 100,
    safety_factor: float = 1.5,
    warning_minutes: int = 20,
    critical_minutes: int = 10,
) -> Settings:
    return Settings(
        jwt_secret_key="test-secret-key-that-is-definitely-long-enough-32bytes",
        health_snapshot_interval_seconds=interval_seconds,
        gpu_session_tokens_per_minute=tokens_per_minute,
        gpu_session_credit_safety_factor=safety_factor,
        gpu_session_credit_warning_minutes=warning_minutes,
        gpu_session_credit_critical_minutes=critical_minutes,
    )


def _make_session(
    *,
    status: str = GpuSessionStatus.active,
    warning_level: str | None = None,
) -> MagicMock:
    s = MagicMock()
    s.id = uuid4()
    s.user_id = uuid4()
    s.product_id = "vex"
    s.account_id = uuid4()
    s.model_type = "aisha-image"
    s.bundle_name = "wan_2.2_i2v"
    s.status = status
    s.started_at = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)
    s.stopped_at = None
    s.paused_at = None
    s.total_paused_seconds = 0
    s.credit_warning_level = warning_level
    s.credit_warned_at = None
    return s


def _make_account(is_active: bool = True) -> MagicMock:
    a = MagicMock()
    a.is_active = is_active
    return a


def _make_db_factory() -> tuple[AsyncMock, MagicMock]:
    """Return (db_mock, session_factory) supporting 'async with factory() as db, db.begin()'."""
    begin_cm = AsyncMock()
    db_mock = AsyncMock()
    db_mock.begin = MagicMock(return_value=begin_cm)

    session_cm = MagicMock()
    session_cm.__aenter__ = AsyncMock(return_value=db_mock)
    session_cm.__aexit__ = AsyncMock(return_value=False)

    factory = MagicMock()
    factory.return_value = session_cm
    return db_mock, factory


class _FakeBillingRepo:
    """Stateful billing ledger for integration tests."""

    def __init__(self, initial_balance: int, initial_settled: int) -> None:
        self.balance = initial_balance
        self.settled = initial_settled
        self.transaction_count = 0

    async def get_account_for_update(self, account_id: UUID) -> MagicMock:  # noqa: ARG002
        a = MagicMock()
        a.is_active = True
        return a

    async def get_balance(self, account_id: UUID) -> int:  # noqa: ARG002
        return self.balance

    async def get_settled_tokens_for_session(self, session_id: UUID) -> int:  # noqa: ARG002
        return self.settled

    async def create_transaction(self, *, amount: int, **kwargs: Any) -> MagicMock:  # noqa: ARG002
        self.balance += amount
        self.settled += abs(amount)
        self.transaction_count += 1
        return MagicMock()


# ---------------------------------------------------------------------------
# Tests: settle_session_usage — no-debt invariant
# ---------------------------------------------------------------------------


class TestSettleSessionUsage:
    """BillingService.settle_session_usage records full incurred cost; may drive balance negative."""

    def _make_repo(self, balance: int, is_active: bool = True) -> MagicMock:
        repo = MagicMock()
        repo.get_account_for_update = AsyncMock(return_value=_make_account(is_active))
        repo.get_balance = AsyncMock(return_value=balance)
        repo.create_transaction = AsyncMock(return_value=MagicMock())
        return repo

    @pytest.mark.asyncio
    async def test_settle_normal(self) -> None:
        """When balance > owed, settle full owed amount."""
        svc = BillingService()
        mock_session = AsyncMock()
        mock_session.flush = AsyncMock()

        with patch(
            "src.api.services.billing.BillingRepository",
            return_value=self._make_repo(1000),
        ):
            settle_result = await svc.settle_session_usage(
                uuid4(),
                100,
                session_id=uuid4(),
                model_type="aisha-image",
                session=mock_session,
                product_id="vex",
            )

        assert settle_result.settled_tokens == 100
        assert settle_result.new_balance == 900

    @pytest.mark.asyncio
    async def test_settle_records_debt_when_owed_exceeds_balance(self) -> None:
        """When owed > balance, settle full owed — balance goes negative (debt recorded)."""
        svc = BillingService()
        mock_session = AsyncMock()
        mock_session.flush = AsyncMock()
        repo = self._make_repo(30)

        with patch(
            "src.api.services.billing.BillingRepository",
            return_value=repo,
        ):
            settle_result = await svc.settle_session_usage(
                uuid4(),
                100,
                session_id=uuid4(),
                model_type="aisha-image",
                session=mock_session,
                product_id="vex",
            )

        assert settle_result.settled_tokens == 100
        assert settle_result.new_balance == -70  # 30 - 100
        call_kwargs = repo.create_transaction.call_args.kwargs
        assert call_kwargs["amount"] == -100
        assert call_kwargs["balance_after"] == -70

    @pytest.mark.asyncio
    async def test_settle_zero_balance(self) -> None:
        """When balance is 0, settle full owed — balance goes negative (debt recorded)."""
        svc = BillingService()
        mock_session = AsyncMock()
        repo = self._make_repo(0)

        with patch("src.api.services.billing.BillingRepository", return_value=repo):
            settle_result = await svc.settle_session_usage(
                uuid4(),
                50,
                session_id=uuid4(),
                model_type="aisha-image",
                session=mock_session,
                product_id="vex",
            )

        assert settle_result.settled_tokens == 50
        assert settle_result.new_balance == -50
        repo.create_transaction.assert_called_once()
        assert repo.create_transaction.call_args.kwargs["balance_after"] == -50

    @pytest.mark.asyncio
    async def test_settle_owed_zero_is_noop(self) -> None:
        """When owed=0, return immediately without touching account."""
        svc = BillingService()
        mock_session = AsyncMock()
        repo = self._make_repo(500)

        with patch("src.api.services.billing.BillingRepository", return_value=repo):
            settle_result = await svc.settle_session_usage(
                uuid4(),
                0,
                session_id=uuid4(),
                model_type="aisha-image",
                session=mock_session,
                product_id="vex",
            )

        assert settle_result.settled_tokens == 0
        repo.get_account_for_update.assert_not_called()

    @pytest.mark.asyncio
    async def test_settle_account_not_found_raises(self) -> None:
        """AccountNotFoundError still propagates (guard handles it above)."""
        svc = BillingService()
        mock_session = AsyncMock()
        repo = MagicMock()
        repo.get_account_for_update = AsyncMock(return_value=None)

        with (
            patch("src.api.services.billing.BillingRepository", return_value=repo),
            pytest.raises(AccountNotFoundError),
        ):
            await svc.settle_session_usage(
                uuid4(),
                50,
                session_id=uuid4(),
                model_type="aisha-image",
                session=mock_session,
                product_id="vex",
            )

    @pytest.mark.asyncio
    async def test_settle_metadata_type(self) -> None:
        """Settled transaction carries type='gpu_session_metered' in metadata."""
        svc = BillingService()
        mock_session = AsyncMock()
        mock_session.flush = AsyncMock()
        repo = self._make_repo(500)
        session_id = uuid4()

        with patch("src.api.services.billing.BillingRepository", return_value=repo):
            await svc.settle_session_usage(
                uuid4(),
                50,
                session_id=session_id,
                model_type="aisha-image",
                session=mock_session,
                product_id="vex",
            )

        call_kwargs = repo.create_transaction.call_args.kwargs
        assert call_kwargs["metadata"]["type"] == "gpu_session_metered"
        assert call_kwargs["metadata"]["session_id"] == str(session_id)
        assert call_kwargs["transaction_type"] == TransactionType.DEBIT.value
        assert call_kwargs["job_id"] is None


# ---------------------------------------------------------------------------
# Tests: SessionCreditGuard — warning ladder + terminate-at-floor
# ---------------------------------------------------------------------------


class TestSessionCreditGuardFloorComputation:
    """Verify floor_tokens derivation from settings."""

    def test_floor_tokens_default(self) -> None:
        settings = _make_settings(interval_seconds=60, tokens_per_minute=100, safety_factor=1.5)
        guard = SessionCreditGuard(
            session_factory=AsyncMock(),
            billing_service=MagicMock(),
            gpu_session_service=MagicMock(),
            settings=settings,
        )
        # ceil(60/60 * 100 * 1.5) = ceil(150) = 150
        assert guard._compute_floor_tokens() == 150

    def test_floor_tokens_partial_interval(self) -> None:
        settings = _make_settings(interval_seconds=90, tokens_per_minute=100, safety_factor=2.0)
        guard = SessionCreditGuard(
            session_factory=AsyncMock(),
            billing_service=MagicMock(),
            gpu_session_service=MagicMock(),
            settings=settings,
        )
        # ceil(90/60 * 100 * 2.0) = ceil(300) = 300
        assert guard._compute_floor_tokens() == 300


class TestSessionCreditGuardClassifyLevel:
    """Verify warning level classification."""

    def _guard(self, warning_minutes: int = 20, critical_minutes: int = 10) -> SessionCreditGuard:
        settings = _make_settings(
            tokens_per_minute=100,
            warning_minutes=warning_minutes,
            critical_minutes=critical_minutes,
        )
        return SessionCreditGuard(
            session_factory=AsyncMock(),
            billing_service=MagicMock(),
            gpu_session_service=MagicMock(),
            settings=settings,
        )

    def test_no_warning_well_funded(self) -> None:
        guard = self._guard()
        # 2100 tokens > warning threshold (20*100=2000)
        assert guard._classify_level(2100) is None

    def test_warning_level(self) -> None:
        guard = self._guard()
        # 1500 tokens <= 2000 (warning) but > 1000 (critical) and > floor
        assert guard._classify_level(1500) == NotificationLevel.WARNING

    def test_critical_level(self) -> None:
        guard = self._guard()
        # 800 tokens <= 1000 (critical) but > floor
        assert guard._classify_level(800) == NotificationLevel.CRITICAL

    def test_at_floor_is_critical(self) -> None:
        guard = self._guard()
        # Exactly at floor → terminate path, but classify returns CRITICAL (150 <= critical=1000)
        assert guard._classify_level(150) == NotificationLevel.CRITICAL


class TestSessionCreditGuardShouldEmit:
    """Verify emit-once/de-escalate logic."""

    def _guard(self) -> SessionCreditGuard:
        return SessionCreditGuard(
            session_factory=AsyncMock(),
            billing_service=MagicMock(),
            gpu_session_service=MagicMock(),
            settings=_make_settings(),
        )

    def test_emit_on_first_warning(self) -> None:
        guard = self._guard()
        assert guard._should_emit(NotificationLevel.WARNING, current_level=None) is True

    def test_emit_on_escalation_warning_to_critical(self) -> None:
        guard = self._guard()
        assert (
            guard._should_emit(
                NotificationLevel.CRITICAL, current_level=NotificationLevel.WARNING.value
            )
            is True
        )

    def test_no_emit_when_same_level(self) -> None:
        guard = self._guard()
        assert (
            guard._should_emit(
                NotificationLevel.WARNING, current_level=NotificationLevel.WARNING.value
            )
            is False
        )

    def test_no_emit_when_null_level(self) -> None:
        guard = self._guard()
        assert guard._should_emit(None, current_level=None) is False

    def test_no_emit_on_de_escalation(self) -> None:
        # De-escalation is handled separately by clearing the warning, not emitting
        guard = self._guard()
        assert (
            guard._should_emit(
                NotificationLevel.WARNING,
                current_level=NotificationLevel.CRITICAL.value,
            )
            is False
        )


class TestSessionCreditGuardTerminateAtFloor:
    """Guard terminates session and dispatches fire-and-forget stop task."""

    @pytest.mark.asyncio
    async def test_terminates_when_balance_at_floor(self) -> None:
        settings = _make_settings(interval_seconds=60, tokens_per_minute=100, safety_factor=1.5)
        # floor = 150; session balance after settle = 100 (below floor)
        mock_billing = MagicMock()
        mock_billing.settle_session_usage = AsyncMock(
            return_value=SettleUsageResult(settled_tokens=50, new_balance=100, event=None)
        )

        mock_gpu_svc = MagicMock()
        mock_gpu_svc.stop_session = AsyncMock()

        mock_event_bus = MagicMock()
        mock_event_bus.publish = AsyncMock()
        mock_event_bus.publish_balance = AsyncMock()

        session_factory = AsyncMock()
        db_ctx = AsyncMock()
        db_ctx.__aenter__ = AsyncMock(return_value=db_ctx)
        db_ctx.__aexit__ = AsyncMock(return_value=None)
        session_factory.return_value = db_ctx

        guard = SessionCreditGuard(
            session_factory=session_factory,
            billing_service=mock_billing,
            gpu_session_service=mock_gpu_svc,
            settings=settings,
            event_bus=mock_event_bus,
        )

        session = _make_session()

        # Patch _settle_metered to return (settled, balance_after)
        guard._settle_metered = AsyncMock(
            return_value=SettleUsageResult(settled_tokens=50, new_balance=100, event=None)
        )  # type: ignore[method-assign]
        guard._publish_warning = AsyncMock()  # type: ignore[method-assign]
        guard._clear_warning = AsyncMock()  # type: ignore[method-assign]
        guard._persist_warning = AsyncMock()  # type: ignore[method-assign]

        # Use asyncio.create_task; patch the terminate function to track calls
        terminate_calls: list[object] = []

        async def mock_terminate(s: object) -> None:
            terminate_calls.append(s)

        guard._terminate_session = mock_terminate  # type: ignore[assignment]

        # Run evaluate and let pending tasks flush
        await guard._evaluate_session(session)
        # Let the created task run
        await asyncio.sleep(0)

        assert len(terminate_calls) == 1
        guard._clear_warning.assert_called_once()
        guard._publish_warning.assert_called_once()

    @pytest.mark.asyncio
    async def test_no_termination_when_balance_above_floor(self) -> None:
        settings = _make_settings(interval_seconds=60, tokens_per_minute=100, safety_factor=1.5)
        # floor = 150; balance after settle = 2000

        guard = SessionCreditGuard(
            session_factory=AsyncMock(),
            billing_service=MagicMock(),
            gpu_session_service=MagicMock(),
            settings=settings,
        )

        session = _make_session()
        guard._settle_metered = AsyncMock(
            return_value=SettleUsageResult(settled_tokens=100, new_balance=2000, event=None)
        )  # type: ignore[method-assign]
        guard._publish_warning = AsyncMock()  # type: ignore[method-assign]
        guard._clear_warning = AsyncMock()  # type: ignore[method-assign]
        guard._persist_warning = AsyncMock()  # type: ignore[method-assign]
        guard._terminate_session = AsyncMock()  # type: ignore[method-assign]

        await guard._evaluate_session(session)

        guard._terminate_session.assert_not_called()


class TestSessionCreditGuardDeEscalation:
    """When balance recovers above warning threshold, clear the stored level."""

    @pytest.mark.asyncio
    async def test_de_escalates_when_balance_recovered(self) -> None:
        settings = _make_settings(tokens_per_minute=100, warning_minutes=20, critical_minutes=10)
        guard = SessionCreditGuard(
            session_factory=AsyncMock(),
            billing_service=MagicMock(),
            gpu_session_service=MagicMock(),
            settings=settings,
        )

        # Session previously warned (WARNING level stored)
        session = _make_session(warning_level=NotificationLevel.WARNING.value)
        # After settle, balance is 4900 — well above warning (2000) + hysteresis
        # minutes_remaining = (4900-150)//100 = 47 > 20+1 = 21 → de-escalate
        guard._settle_metered = AsyncMock(
            return_value=SettleUsageResult(settled_tokens=100, new_balance=4900, event=None)
        )  # type: ignore[method-assign]
        guard._publish_warning = AsyncMock()  # type: ignore[method-assign]
        guard._clear_warning = AsyncMock()  # type: ignore[method-assign]
        guard._persist_warning = AsyncMock()  # type: ignore[method-assign]
        guard._terminate_session = AsyncMock()  # type: ignore[method-assign]

        await guard._evaluate_session(session)

        guard._clear_warning.assert_called_once_with(session)
        guard._publish_warning.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_clear_when_no_prior_warning(self) -> None:
        settings = _make_settings()
        guard = SessionCreditGuard(
            session_factory=AsyncMock(),
            billing_service=MagicMock(),
            gpu_session_service=MagicMock(),
            settings=settings,
        )

        session = _make_session(warning_level=None)
        guard._settle_metered = AsyncMock(
            return_value=SettleUsageResult(settled_tokens=100, new_balance=4900, event=None)
        )  # type: ignore[method-assign]
        guard._publish_warning = AsyncMock()  # type: ignore[method-assign]
        guard._clear_warning = AsyncMock()  # type: ignore[method-assign]
        guard._persist_warning = AsyncMock()  # type: ignore[method-assign]
        guard._terminate_session = AsyncMock()  # type: ignore[method-assign]

        await guard._evaluate_session(session)

        guard._clear_warning.assert_not_called()


class TestSessionCreditGuardSkips:
    """Guard skips sessions without account_id or started_at."""

    @pytest.mark.asyncio
    async def test_skips_when_account_id_none(self) -> None:
        guard = SessionCreditGuard(
            session_factory=AsyncMock(),
            billing_service=MagicMock(),
            gpu_session_service=MagicMock(),
            settings=_make_settings(),
        )
        session = _make_session()
        session.account_id = None
        guard._settle_metered = AsyncMock()  # type: ignore[method-assign]

        await guard._evaluate_session(session)
        guard._settle_metered.assert_not_called()

    @pytest.mark.asyncio
    async def test_skips_when_not_started(self) -> None:
        guard = SessionCreditGuard(
            session_factory=AsyncMock(),
            billing_service=MagicMock(),
            gpu_session_service=MagicMock(),
            settings=_make_settings(),
        )
        session = _make_session()
        session.started_at = None
        guard._settle_metered = AsyncMock()  # type: ignore[method-assign]

        await guard._evaluate_session(session)
        guard._settle_metered.assert_not_called()


# ---------------------------------------------------------------------------
# Tests: F1 — proportional metering (fails pre-fix, passes post-fix)
# ---------------------------------------------------------------------------


class TestProportionalMetering:
    """Guard meters proportional to active runtime, netting the base reservation."""

    def _guard(self, tokens_per_minute: int = 100) -> SessionCreditGuard:
        return SessionCreditGuard(
            session_factory=AsyncMock(),
            billing_service=MagicMock(),
            gpu_session_service=MagicMock(),
            settings=_make_settings(tokens_per_minute=tokens_per_minute),
        )

    def test_compute_owed_6min_active(self) -> None:
        """6 minutes of active runtime → 600 consumed tokens (6 * 100)."""
        guard = self._guard()
        session = _make_session()
        now = datetime(2026, 1, 1, 12, 6, 0, tzinfo=UTC)
        assert guard._compute_owed(session, now=now) == 600

    def test_compute_owed_5min_active(self) -> None:
        """5 minutes → 500 consumed tokens (floor=5 min matches base reservation)."""
        guard = self._guard()
        session = _make_session()
        now = datetime(2026, 1, 1, 12, 5, 0, tzinfo=UTC)
        assert guard._compute_owed(session, now=now) == 500

    @pytest.mark.asyncio
    async def test_settle_metered_nets_base_reservation(self) -> None:
        """consumed=600, settled=500 (base) → owed=100 passed to settle_session_usage."""
        _, factory = _make_db_factory()

        billing_repo = MagicMock()
        billing_repo.get_settled_tokens_for_session = AsyncMock(return_value=500)
        billing_repo.get_account_for_update = AsyncMock(return_value=_make_account())
        billing_repo.get_balance = AsyncMock(return_value=1000)
        billing_repo.create_transaction = AsyncMock(return_value=MagicMock())

        guard = SessionCreditGuard(
            session_factory=factory,
            billing_service=BillingService(),
            gpu_session_service=MagicMock(),
            settings=_make_settings(tokens_per_minute=100),
        )
        session = _make_session()

        with (
            patch(
                "src.api.services.gpu_session.credit_guard.BillingRepository",
                return_value=billing_repo,
            ),
            patch("src.api.services.billing.BillingRepository", return_value=billing_repo),
        ):
            settle_result = await guard._settle_metered(session, 600)

        # owed = max(0, 600 - 500) = 100; debit amount = -100
        billing_repo.create_transaction.assert_called_once()
        assert billing_repo.create_transaction.call_args.kwargs["amount"] == -100
        assert settle_result.settled_tokens == 100

    @pytest.mark.asyncio
    async def test_settle_metered_5min_no_additional_owed(self) -> None:
        """consumed=500 == settled=500 → owed=0 → no create_transaction."""
        _, factory = _make_db_factory()

        billing_repo = MagicMock()
        billing_repo.get_settled_tokens_for_session = AsyncMock(return_value=500)
        billing_repo.get_balance = AsyncMock(return_value=1000)

        guard = SessionCreditGuard(
            session_factory=factory,
            billing_service=BillingService(),
            gpu_session_service=MagicMock(),
            settings=_make_settings(tokens_per_minute=100),
        )
        session = _make_session()

        with (
            patch(
                "src.api.services.gpu_session.credit_guard.BillingRepository",
                return_value=billing_repo,
            ),
            patch("src.api.services.billing.BillingRepository", return_value=billing_repo),
        ):
            settle_result = await guard._settle_metered(session, 500)

        billing_repo.create_transaction.assert_not_called()
        assert settle_result.settled_tokens == 0


# ---------------------------------------------------------------------------
# Tests: F1 — no premature termination
# ---------------------------------------------------------------------------


class TestNoPrematureTermination:
    """Session funded for N minutes is not terminated before N - floor/rate minutes."""

    @pytest.mark.asyncio
    async def test_funded_15min_not_terminated_at_8min(self) -> None:
        """Funded for 15 min (balance well above floor); session not terminated at 8 min."""
        # floor = ceil(1 * 100 * 1.5) = 150; 15 min = 1500 tokens reserve
        # After 8 min guard cycle: balance = 1650 (well above floor)
        settings = _make_settings(interval_seconds=60, tokens_per_minute=100, safety_factor=1.5)

        gpu_svc = MagicMock()
        gpu_svc.stop_session = AsyncMock()

        guard = SessionCreditGuard(
            session_factory=AsyncMock(),
            billing_service=MagicMock(),
            gpu_session_service=gpu_svc,
            settings=settings,
        )
        session = _make_session()
        guard._settle_metered = AsyncMock(
            return_value=SettleUsageResult(settled_tokens=300, new_balance=1650, event=None)
        )  # type: ignore[method-assign]
        guard._publish_warning = AsyncMock()  # type: ignore[method-assign]
        guard._clear_warning = AsyncMock()  # type: ignore[method-assign]
        guard._persist_warning = AsyncMock()  # type: ignore[method-assign]
        guard._terminate_session = AsyncMock()  # type: ignore[method-assign]

        await guard._evaluate_session(session)

        guard._terminate_session.assert_not_called()
        assert session.id not in guard._terminating


# ---------------------------------------------------------------------------
# Tests: F1a — idempotent settle
# ---------------------------------------------------------------------------


class TestIdempotentSettle:
    """Two evaluate passes at same active_seconds → second settles 0 (create_transaction once)."""

    @pytest.mark.asyncio
    async def test_two_evaluations_same_consumed_settle_once(self) -> None:
        """_settle_metered called twice with same consumed_tokens → debit issued once."""
        _, factory = _make_db_factory()

        # First call: total_settled=500 → owed=100
        # Second call: total_settled=600 (500 base + 100 metered) → owed=0
        billing_repo = MagicMock()
        billing_repo.get_settled_tokens_for_session = AsyncMock(side_effect=[500, 600])
        billing_repo.get_account_for_update = AsyncMock(return_value=_make_account())
        billing_repo.get_balance = AsyncMock(return_value=1000)
        billing_repo.create_transaction = AsyncMock(return_value=MagicMock())

        guard = SessionCreditGuard(
            session_factory=factory,
            billing_service=BillingService(),
            gpu_session_service=MagicMock(),
            settings=_make_settings(tokens_per_minute=100),
        )
        session = _make_session()

        with (
            patch(
                "src.api.services.gpu_session.credit_guard.BillingRepository",
                return_value=billing_repo,
            ),
            patch("src.api.services.billing.BillingRepository", return_value=billing_repo),
        ):
            await guard._settle_metered(session, 600)
            await guard._settle_metered(session, 600)

        # Only the first pass issues a transaction
        billing_repo.create_transaction.assert_called_once()


# ---------------------------------------------------------------------------
# Tests: F1 — pause awareness
# ---------------------------------------------------------------------------


class TestPauseNotBilled:
    """Paused time is excluded from billable consumption."""

    def test_pause_excluded_from_owed(self) -> None:
        """12 min wall-clock, 6 min paused → billed for 6 active min only."""
        guard = SessionCreditGuard(
            session_factory=AsyncMock(),
            billing_service=MagicMock(),
            gpu_session_service=MagicMock(),
            settings=_make_settings(tokens_per_minute=100),
        )
        session = _make_session()
        session.total_paused_seconds = 360  # 6 min paused
        now = datetime(2026, 1, 1, 12, 12, 0, tzinfo=UTC)  # 12 min after started_at

        consumed = guard._compute_owed(session, now=now)
        # active = 720 - 360 = 360 s → 6 billable min → 600 tokens
        assert consumed == 600

    def test_no_pause_bills_full_runtime(self) -> None:
        """Without pause, 12 min active → 1200 tokens consumed."""
        guard = SessionCreditGuard(
            session_factory=AsyncMock(),
            billing_service=MagicMock(),
            gpu_session_service=MagicMock(),
            settings=_make_settings(tokens_per_minute=100),
        )
        session = _make_session()
        now = datetime(2026, 1, 1, 12, 12, 0, tzinfo=UTC)

        consumed = guard._compute_owed(session, now=now)
        assert consumed == 1200


# ---------------------------------------------------------------------------
# Tests: F2 — no re-termination
# ---------------------------------------------------------------------------


class TestNoReTerminate:
    """Guard dispatches stop_session at most once per session lifetime."""

    @pytest.mark.asyncio
    async def test_two_cycles_below_floor_stop_dispatched_once(self) -> None:
        """When balance is at floor across two evaluation cycles, stop is dispatched once."""
        settings = _make_settings(interval_seconds=60, tokens_per_minute=100, safety_factor=1.5)

        stop_calls: list[object] = []

        async def mock_terminate(s: object) -> None:
            stop_calls.append(getattr(s, "id", s))

        guard = SessionCreditGuard(
            session_factory=AsyncMock(),
            billing_service=MagicMock(),
            gpu_session_service=MagicMock(),
            settings=settings,
        )
        session = _make_session()
        guard._settle_metered = AsyncMock(
            return_value=SettleUsageResult(settled_tokens=50, new_balance=100, event=None)
        )  # type: ignore[method-assign]
        guard._publish_warning = AsyncMock()  # type: ignore[method-assign]
        guard._clear_warning = AsyncMock()  # type: ignore[method-assign]
        guard._persist_warning = AsyncMock()  # type: ignore[method-assign]
        guard._terminate_session = mock_terminate  # type: ignore[assignment]

        # First cycle — triggers termination
        await guard._evaluate_session(session)
        await asyncio.sleep(0)

        # Second cycle — session in _terminating, should be skipped entirely
        await guard._evaluate_session(session)
        await asyncio.sleep(0)

        assert len(stop_calls) == 1, "stop_session must be dispatched exactly once"
        assert guard._publish_warning.call_count == 1, "terminal CRITICAL emitted exactly once"


# ---------------------------------------------------------------------------
# Tests: F4 — de-escalation hysteresis
# ---------------------------------------------------------------------------


class TestDeEscalationHysteresis:
    """Balance oscillating near warning threshold does not flap emit/clear."""

    @pytest.mark.asyncio
    async def test_balance_just_above_warning_does_not_clear(self) -> None:
        """Balance above warning_tokens but within hysteresis band → no clear."""
        # warning=20 min, rate=100, interval=60s → hysteresis=1 min
        # de-escalate only when minutes_remaining > 21
        # minutes_remaining at balance=2050: (2050-150)//100 = 19 → NOT cleared
        settings = _make_settings(
            interval_seconds=60,
            tokens_per_minute=100,
            safety_factor=1.5,
            warning_minutes=20,
            critical_minutes=10,
        )
        guard = SessionCreditGuard(
            session_factory=AsyncMock(),
            billing_service=MagicMock(),
            gpu_session_service=MagicMock(),
            settings=settings,
        )
        session = _make_session(warning_level=NotificationLevel.WARNING.value)
        # Balance just above warning (2050 > 2000) but not past hysteresis band
        guard._settle_metered = AsyncMock(
            return_value=SettleUsageResult(settled_tokens=0, new_balance=2050, event=None)
        )  # type: ignore[method-assign]
        guard._publish_warning = AsyncMock()  # type: ignore[method-assign]
        guard._clear_warning = AsyncMock()  # type: ignore[method-assign]
        guard._persist_warning = AsyncMock()  # type: ignore[method-assign]
        guard._terminate_session = AsyncMock()  # type: ignore[method-assign]

        await guard._evaluate_session(session)

        guard._clear_warning.assert_not_called()
        guard._publish_warning.assert_not_called()

    @pytest.mark.asyncio
    async def test_balance_well_above_warning_clears(self) -> None:
        """Balance well past warning+hysteresis → de-escalate."""
        # minutes_remaining at 4900: (4900-150)//100 = 47 > 21 → cleared
        settings = _make_settings(
            interval_seconds=60,
            tokens_per_minute=100,
            safety_factor=1.5,
            warning_minutes=20,
            critical_minutes=10,
        )
        guard = SessionCreditGuard(
            session_factory=AsyncMock(),
            billing_service=MagicMock(),
            gpu_session_service=MagicMock(),
            settings=settings,
        )
        session = _make_session(warning_level=NotificationLevel.WARNING.value)
        guard._settle_metered = AsyncMock(
            return_value=SettleUsageResult(settled_tokens=0, new_balance=4900, event=None)
        )  # type: ignore[method-assign]
        guard._publish_warning = AsyncMock()  # type: ignore[method-assign]
        guard._clear_warning = AsyncMock()  # type: ignore[method-assign]
        guard._persist_warning = AsyncMock()  # type: ignore[method-assign]
        guard._terminate_session = AsyncMock()  # type: ignore[method-assign]

        await guard._evaluate_session(session)

        guard._clear_warning.assert_called_once_with(session)


# ---------------------------------------------------------------------------
# Tests: integration — session outruns funding
# ---------------------------------------------------------------------------


class TestIntegrationSessionOutrunsFunding:
    """End-to-end billing: guard metering + settle invariants hold across cycles."""

    @pytest.mark.asyncio
    async def test_session_outruns_funding_records_debt_no_double_charge(self) -> None:
        """Full owed cost is recorded; re-run is idempotent; balance goes negative.

        Scenario:
            initial_balance=200 (remaining after base reservation of 500)
            Guard cycle (consumed=800): owed=300, recorded in full → balance=-100
            Re-run (idempotent, consumed=800): owed=0 → no additional charge
            Final: total_settled=800, balance=-100 < 0.
        """
        _, factory = _make_db_factory()
        fake_repo = _FakeBillingRepo(initial_balance=200, initial_settled=500)

        guard = SessionCreditGuard(
            session_factory=factory,
            billing_service=BillingService(),
            gpu_session_service=MagicMock(),
            settings=_make_settings(tokens_per_minute=100),
        )
        session = _make_session()

        with (
            patch(
                "src.api.services.gpu_session.credit_guard.BillingRepository",
                return_value=fake_repo,
            ),
            patch("src.api.services.billing.BillingRepository", return_value=fake_repo),
        ):
            # First guard cycle: consumed=800, settled=500, owed=300, recorded in full
            result1 = await guard._settle_metered(session, 800)
            assert result1.settled_tokens == 300
            assert result1.new_balance == -100
            assert fake_repo.balance < 0  # debt recorded

            # Second guard cycle (idempotent): consumed=800, settled=800, owed=0
            result2 = await guard._settle_metered(session, 800)
            assert result2.settled_tokens == 0

        # Only one transaction created (first cycle only)
        assert fake_repo.transaction_count == 1
        assert fake_repo.balance == -100
        # total settled = 500 (base reservation) + 300 (metered, full owed)
        assert fake_repo.settled == 800


# ---------------------------------------------------------------------------
# Tests: debt visibility and recovery via top-up
# ---------------------------------------------------------------------------


class TestTopupNetsAgainstDebt:
    """credit() uses SUM(ledger) netting — no special debt-payment logic needed."""

    def _make_repo(self, balance: int) -> MagicMock:
        repo = MagicMock()
        a = MagicMock()
        a.is_active = True
        a.user_id = None
        a.organization_id = None
        repo.get_account_for_update = AsyncMock(return_value=a)
        repo.get_balance = AsyncMock(return_value=balance)
        repo.create_transaction = AsyncMock(return_value=MagicMock())
        return repo

    @pytest.mark.asyncio
    async def test_topup_nets_against_debt_to_zero(self) -> None:
        """From -2500 with credit +2500 → balance 0."""
        svc = BillingService()
        mock_session = AsyncMock()
        repo = self._make_repo(-2500)

        with patch("src.api.services.billing.BillingRepository", return_value=repo):
            await svc.credit(
                uuid4(),
                2500,
                uuid4(),
                description="top-up",
                payment_provider=PaymentProvider.STRIPE,
                session=mock_session,
                product_id="vex",
            )

        assert repo.create_transaction.call_args.kwargs["balance_after"] == 0

    @pytest.mark.asyncio
    async def test_topup_nets_against_debt_to_positive(self) -> None:
        """From -2500 with credit +3000 → balance 500."""
        svc = BillingService()
        mock_session = AsyncMock()
        repo = self._make_repo(-2500)

        with patch("src.api.services.billing.BillingRepository", return_value=repo):
            await svc.credit(
                uuid4(),
                3000,
                uuid4(),
                description="top-up",
                payment_provider=PaymentProvider.STRIPE,
                session=mock_session,
                product_id="vex",
            )

        assert repo.create_transaction.call_args.kwargs["balance_after"] == 500


# ---------------------------------------------------------------------------
# Tests: negative balance blocks new work
# ---------------------------------------------------------------------------


class TestDebtBlocksNewWork:
    """Negative balance → check_and_reserve raises InsufficientBalanceError."""

    @pytest.mark.asyncio
    async def test_debt_blocks_new_work(self) -> None:
        """check_and_reserve refuses when balance is negative (debtor locked out)."""
        svc = BillingService()
        mock_session = AsyncMock()

        repo = MagicMock()
        a = MagicMock()
        a.is_active = True
        repo.get_account_for_update = AsyncMock(return_value=a)
        repo.get_balance = AsyncMock(return_value=-500)

        with (
            patch("src.api.services.billing.BillingRepository", return_value=repo),
            pytest.raises(InsufficientBalanceError),
        ):
            await svc.check_and_reserve(
                uuid4(),
                100,
                None,
                metadata={},
                session=mock_session,
                product_id="vex",
            )


# ---------------------------------------------------------------------------
# Tests: idempotent settle after negative balance
# ---------------------------------------------------------------------------


class TestMeteredSettleIdempotentAfterNegative:
    """Re-running _settle_metered with same consumed_tokens after a debt-recording settle → owed=0."""

    @pytest.mark.asyncio
    async def test_metered_settle_idempotent_after_negative(self) -> None:
        """Re-run at same consumed_tokens after negative-recording settle → owed=0, no new DEBIT."""
        _, factory = _make_db_factory()

        billing_repo = MagicMock()
        # First call: settled=500; second call: settled=800 (500 base + 300 metered)
        billing_repo.get_settled_tokens_for_session = AsyncMock(side_effect=[500, 800])
        billing_repo.get_account_for_update = AsyncMock(return_value=_make_account())
        # First call: balance=200; second call (owed=0 early-return): balance=-100
        billing_repo.get_balance = AsyncMock(side_effect=[200, -100])
        billing_repo.create_transaction = AsyncMock(return_value=MagicMock())

        guard = SessionCreditGuard(
            session_factory=factory,
            billing_service=BillingService(),
            gpu_session_service=MagicMock(),
            settings=_make_settings(tokens_per_minute=100),
        )
        session = _make_session()

        with (
            patch(
                "src.api.services.gpu_session.credit_guard.BillingRepository",
                return_value=billing_repo,
            ),
            patch("src.api.services.billing.BillingRepository", return_value=billing_repo),
        ):
            # First pass: consumed=800, settled=500 → owed=300; balance=200 → balance_after=-100
            result1 = await guard._settle_metered(session, 800)
            assert result1.settled_tokens == 300
            assert result1.new_balance == -100

            # Second pass: consumed=800, settled=800 → owed=0 (idempotent, no new debit)
            result2 = await guard._settle_metered(session, 800)
            assert result2.settled_tokens == 0

        billing_repo.create_transaction.assert_called_once()


# ---------------------------------------------------------------------------
# Tests: balance surface returns signed sum (regression guard against max(0, …))
# ---------------------------------------------------------------------------


class TestBalanceSignedSum:
    """BillingService.get_balance must return the raw signed ledger SUM."""

    @pytest.mark.asyncio
    async def test_balance_route_returns_signed_sum(self) -> None:
        """get_balance() returns negative when ledger SUM is negative — no max(0, …) re-hiding."""
        svc = BillingService()
        mock_session = AsyncMock()

        repo = MagicMock()
        repo.get_balance = AsyncMock(return_value=-3000)

        with patch("src.api.services.billing.BillingRepository", return_value=repo):
            balance = await svc.get_balance(uuid4(), session=mock_session)

        assert balance == -3000
        assert balance < 0


# ---------------------------------------------------------------------------
# Tests: guard integration — session outruns balance live → negative, terminates
# ---------------------------------------------------------------------------


class TestIntegrationGuardNegativeBalance:
    """Guard terminates session after metered settle drives balance negative."""

    @pytest.mark.asyncio
    async def test_guard_terminates_after_negative_settle(self) -> None:
        """Session outruns balance: settle records debt, guard terminates, balance stays negative."""
        settings = _make_settings(interval_seconds=60, tokens_per_minute=100, safety_factor=1.5)
        # floor = 150; balance after settle = -100 (below floor) → terminate

        terminate_calls: list[object] = []

        async def mock_terminate(s: object) -> None:
            terminate_calls.append(s)

        guard = SessionCreditGuard(
            session_factory=AsyncMock(),
            billing_service=MagicMock(),
            gpu_session_service=MagicMock(),
            settings=settings,
        )
        session = _make_session()
        # _settle_metered returns negative balance: debt was recorded
        guard._settle_metered = AsyncMock(
            return_value=SettleUsageResult(settled_tokens=300, new_balance=-100, event=None)
        )  # type: ignore[method-assign]
        guard._publish_warning = AsyncMock()  # type: ignore[method-assign]
        guard._clear_warning = AsyncMock()  # type: ignore[method-assign]
        guard._persist_warning = AsyncMock()  # type: ignore[method-assign]
        guard._terminate_session = mock_terminate  # type: ignore[assignment]

        await guard._evaluate_session(session)
        await asyncio.sleep(0)

        # Session terminated, warning published, balance NOT clamped by guard
        assert len(terminate_calls) == 1
        guard._clear_warning.assert_called_once()
        guard._publish_warning.assert_called_once()
        # Confirm the recorded (negative) balance was used for the termination decision
        publish_call = guard._publish_warning.call_args
        assert publish_call.kwargs["balance"] == -100
