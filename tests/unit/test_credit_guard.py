"""Tests for SessionCreditGuard and settle_session_usage."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from src.api.services.billing import BillingService
from src.api.services.billing_errors import AccountNotFoundError
from src.api.services.gpu_session.credit_guard import SessionCreditGuard
from src.core.config import Settings
from src.core.enums import GpuSessionStatus, NotificationLevel, TransactionType

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
    s.credit_warning_level = warning_level
    s.credit_warned_at = None
    return s


def _make_account(is_active: bool = True) -> MagicMock:
    a = MagicMock()
    a.is_active = is_active
    return a


# ---------------------------------------------------------------------------
# Tests: settle_session_usage — no-debt invariant
# ---------------------------------------------------------------------------


class TestSettleSessionUsage:
    """BillingService.settle_session_usage never goes negative."""

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
            settled, new_balance, fully = await svc.settle_session_usage(
                uuid4(),
                100,
                session_id=uuid4(),
                model_type="aisha-image",
                session=mock_session,
                product_id="vex",
            )

        assert settled == 100
        assert new_balance == 900
        assert fully is True

    @pytest.mark.asyncio
    async def test_settle_clamped_to_balance(self) -> None:
        """When balance < owed, settle only what is available (no-debt invariant)."""
        svc = BillingService()
        mock_session = AsyncMock()
        mock_session.flush = AsyncMock()

        with patch(
            "src.api.services.billing.BillingRepository",
            return_value=self._make_repo(30),
        ):
            settled, new_balance, fully = await svc.settle_session_usage(
                uuid4(),
                100,
                session_id=uuid4(),
                model_type="aisha-image",
                session=mock_session,
                product_id="vex",
            )

        assert settled == 30
        assert new_balance == 0
        assert fully is False

    @pytest.mark.asyncio
    async def test_settle_zero_balance(self) -> None:
        """When balance is 0, settle 0 tokens — no transaction created."""
        svc = BillingService()
        mock_session = AsyncMock()
        repo = self._make_repo(0)

        with patch("src.api.services.billing.BillingRepository", return_value=repo):
            settled, new_balance, fully = await svc.settle_session_usage(
                uuid4(),
                50,
                session_id=uuid4(),
                model_type="aisha-image",
                session=mock_session,
                product_id="vex",
            )

        assert settled == 0
        assert new_balance == 0
        assert fully is False
        repo.create_transaction.assert_not_called()

    @pytest.mark.asyncio
    async def test_settle_owed_zero_is_noop(self) -> None:
        """When owed=0, return immediately without touching account."""
        svc = BillingService()
        mock_session = AsyncMock()
        repo = self._make_repo(500)

        with patch("src.api.services.billing.BillingRepository", return_value=repo):
            settled, new_balance, fully = await svc.settle_session_usage(
                uuid4(),
                0,
                session_id=uuid4(),
                model_type="aisha-image",
                session=mock_session,
                product_id="vex",
            )

        assert settled == 0
        assert fully is True
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
        assert guard._classify_level(2100, floor_tokens=150) is None

    def test_warning_level(self) -> None:
        guard = self._guard()
        # 1500 tokens <= 2000 (warning) but > 1000 (critical) and > floor
        assert guard._classify_level(1500, floor_tokens=150) == NotificationLevel.WARNING

    def test_critical_level(self) -> None:
        guard = self._guard()
        # 800 tokens <= 1000 (critical) but > floor
        assert guard._classify_level(800, floor_tokens=150) == NotificationLevel.CRITICAL

    def test_at_floor_is_critical(self) -> None:
        guard = self._guard()
        # Exactly at floor → terminate path, but classify returns CRITICAL
        assert guard._classify_level(150, floor_tokens=150) == NotificationLevel.CRITICAL


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
        mock_billing.settle_session_usage = AsyncMock(return_value=(50, 100, False))

        mock_gpu_svc = MagicMock()
        mock_gpu_svc.stop_session = AsyncMock()

        mock_event_bus = MagicMock()
        mock_event_bus.publish = AsyncMock()

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

        # Patch _settle_metered to return (settled, balance_after, fully_settled)
        guard._settle_metered = AsyncMock(return_value=(50, 100, False))  # type: ignore[method-assign]
        guard._publish_warning = AsyncMock()  # type: ignore[method-assign]
        guard._clear_warning = AsyncMock()  # type: ignore[method-assign]
        guard._persist_warning = AsyncMock()  # type: ignore[method-assign]

        # Use asyncio.create_task; patch the terminate function to track calls
        terminate_calls: list[MagicMock] = []

        async def mock_terminate(s: MagicMock) -> None:
            terminate_calls.append(s)

        guard._terminate_session = mock_terminate  # type: ignore[method-assign]

        import asyncio

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
        guard._settle_metered = AsyncMock(return_value=(100, 2000, True))  # type: ignore[method-assign]
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
        # After settle, balance is still 4900 — well above warning (2000)
        guard._settle_metered = AsyncMock(return_value=(100, 4900, True))  # type: ignore[method-assign]
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
        guard._settle_metered = AsyncMock(return_value=(100, 4900, True))  # type: ignore[method-assign]
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
