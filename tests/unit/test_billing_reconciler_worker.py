"""Unit tests for BillingReconcilerWorker."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import ANY, AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from src.api.services.gpu_session.billing_reconciler_worker import (
    BillingReconcilerWorker,
    compute_next_attempt_at,
)
from src.db.models.gpu_session import GpuSession

_REPO_PATH = "src.api.services.gpu_session.billing_reconciler_worker.GpuSessionRepository"


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def _make_gpu_session(
    *,
    billing_finalized_at: datetime | None = None,
    billing_finalization_attempts: int = 0,
    stopped_at: datetime | None = None,
) -> GpuSession:
    now = datetime.now(UTC)
    session = GpuSession()
    session.id = uuid4()
    session.user_id = uuid4()
    session.product_id = "vex"
    session.status = "stopped"
    session.bundle_name = "wan_2.2_i2v"
    session.bundle_version = "260105-01"
    session.model_type = "aisha-image"
    session.cf_tunnel_id = None
    session.cf_dns_record_id = None
    session.tunnel_hostname = None
    session.vastai_instance_id = 12345
    session.vastai_offer_id = 67890
    session.vastai_cost_per_hour_micros = 500_000
    session.vastai_gpu_name = "RTX_4090"
    session.callback_token_hash = "tok"
    session.provision_attempt = 1
    session.provisioning_started_at = None
    session.account_id = uuid4()
    session.total_paused_seconds = 0
    session.started_at = now - timedelta(hours=1)
    session.paused_at = None
    session.resumed_at = None
    session.stopped_at = stopped_at or (now - timedelta(hours=1))
    session.created_at = now - timedelta(hours=2)
    session.billing_finalized_at = billing_finalized_at
    session.billing_finalization_attempts = billing_finalization_attempts
    session.error_message = None
    session.stale_detected_at = None
    session.stale_notified = False
    return session


def _make_settings(**overrides: Any) -> MagicMock:
    settings = MagicMock()
    settings.billing_reconciler_interval_minutes = 10
    settings.billing_reconciler_grace_period_minutes = 2
    settings.billing_reconciler_quarantine_threshold = 10
    settings.billing_reconciler_backoff_base_minutes = 5
    settings.billing_reconciler_backoff_cap_hours = 24
    settings.billing_reconciler_max_per_sweep = 50
    for k, v in overrides.items():
        setattr(settings, k, v)
    return settings


def _make_mock_session_factory() -> tuple[MagicMock, MagicMock]:
    mock_db = MagicMock()
    mock_db.__aenter__ = AsyncMock(return_value=mock_db)
    mock_db.__aexit__ = AsyncMock(return_value=None)
    mock_db.commit = AsyncMock()

    mock_factory = MagicMock(return_value=mock_db)
    return mock_factory, mock_db


def _make_worker(**overrides: Any) -> tuple[BillingReconcilerWorker, dict[str, Any]]:
    mock_factory, mock_db = _make_mock_session_factory()
    mocks: dict[str, Any] = {
        "session_factory": mock_factory,
        "mock_db": mock_db,
        "gpu_session_service": AsyncMock(),
        "settings": _make_settings(),
        "redis_client_factory": MagicMock(),
    } | overrides
    worker = BillingReconcilerWorker(
        session_factory=mocks["session_factory"],
        gpu_session_service=mocks["gpu_session_service"],
        settings=mocks["settings"],
        redis_client_factory=mocks["redis_client_factory"],
    )
    return worker, mocks


# ---------------------------------------------------------------------------
# Lifecycle (start/stop/tick-error-recovery) is covered generically by
# tests/unit/test_periodic_worker.py — BillingReconcilerWorker has no
# lifecycle overrides beyond PeriodicWorker.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# TestSweep
# ---------------------------------------------------------------------------


class TestSweep:
    async def test_no_candidates_logs_debug_and_returns(self) -> None:
        """Empty candidate list → no calls to _finalize_billing."""
        worker, mocks = _make_worker()

        with patch(_REPO_PATH) as MockRepo:
            mock_repo = AsyncMock()
            MockRepo.return_value = mock_repo
            mock_repo.list_pending_billing_finalization.return_value = []

            await worker.run_once()

        mocks["gpu_session_service"].finalize_billing_for_session.assert_not_called()

    async def test_candidate_reconciles_on_first_sweep(self) -> None:
        """Successful finalization: service reports True → logged as reconciled."""
        worker, mocks = _make_worker()

        candidate = _make_gpu_session()
        # Service's public wrapper returns True on success.
        mocks["gpu_session_service"].finalize_billing_for_session.return_value = True

        with patch(_REPO_PATH) as MockRepo:
            mock_repo = AsyncMock()
            MockRepo.return_value = mock_repo
            mock_repo.list_pending_billing_finalization.return_value = [candidate]

            await worker.run_once()

        mocks["gpu_session_service"].finalize_billing_for_session.assert_called_once_with(candidate)
        mock_repo.increment_billing_finalization_attempts.assert_not_called()

    async def test_candidate_still_failing_bumps_attempts_and_logs(self) -> None:
        """Failed finalization: service reports False → attempt counter bumped (no quarantine yet)."""
        worker, mocks = _make_worker()

        candidate = _make_gpu_session(billing_finalization_attempts=0)
        mocks["gpu_session_service"].finalize_billing_for_session.return_value = False

        with patch(_REPO_PATH) as MockRepo:
            mock_repo = AsyncMock()
            MockRepo.return_value = mock_repo
            mock_repo.list_pending_billing_finalization.return_value = [candidate]
            # Returns 1 — below the quarantine threshold of 10
            mock_repo.increment_billing_finalization_attempts.return_value = 1

            await worker.run_once()

        mock_repo.increment_billing_finalization_attempts.assert_called_once_with(candidate.id)
        mock_repo.set_billing_reconciliation_next_attempt_at.assert_called_once_with(
            candidate.id, next_attempt_at=ANY
        )
        # No quarantine error — just still_failing
        mocks["gpu_session_service"].finalize_billing_for_session.assert_called_once_with(candidate)

    async def test_backoff_uses_the_atomic_updates_returned_count(self) -> None:
        """Y2: a stale candidate count must not choose the backoff exponent.

        The in-memory row says this is the first failure, while the atomic SQL
        update returns four. The persisted timestamp must therefore use the
        fourth-failure (40 minute) schedule rather than five minutes.
        """
        worker, mocks = _make_worker()
        candidate = _make_gpu_session(billing_finalization_attempts=0)
        mocks["gpu_session_service"].finalize_billing_for_session.return_value = False
        fixed_now = datetime(2026, 4, 27, 12, 0, tzinfo=UTC)

        with (
            patch(_REPO_PATH) as MockRepo,
            patch(
                "src.api.services.gpu_session.billing_reconciler_worker.datetime"
            ) as mock_datetime,
        ):
            mock_repo = AsyncMock()
            MockRepo.return_value = mock_repo
            mock_repo.list_pending_billing_finalization.return_value = [candidate]
            mock_repo.increment_billing_finalization_attempts.return_value = 4
            mock_datetime.now.return_value = fixed_now

            await worker.run_once()

        mock_repo.increment_billing_finalization_attempts.assert_called_once_with(candidate.id)
        mock_repo.set_billing_reconciliation_next_attempt_at.assert_called_once_with(
            candidate.id,
            next_attempt_at=fixed_now + timedelta(minutes=40),
        )

    async def test_quarantine_threshold_triggers_error_log_once_on_crossing(self) -> None:
        """Attempt counter crosses threshold → quarantine log at ERROR; session NOT mutated."""
        worker, mocks = _make_worker(
            settings=_make_settings(billing_reconciler_quarantine_threshold=5)
        )

        # previous attempts=4 (below threshold) -> new count=5 is the crossing sweep.
        candidate = _make_gpu_session(billing_finalization_attempts=4)
        mocks["gpu_session_service"].finalize_billing_for_session.return_value = False

        with (
            patch(_REPO_PATH) as MockRepo,
            patch("src.api.services.gpu_session.billing_reconciler_worker.logger") as mock_logger,
        ):
            mock_repo = AsyncMock()
            MockRepo.return_value = mock_repo
            mock_repo.list_pending_billing_finalization.return_value = [candidate]
            # Returns 5 — equals the quarantine threshold
            mock_repo.increment_billing_finalization_attempts.return_value = 5

            await worker.run_once()

        # Must log quarantine=True at ERROR exactly once; no WARNING for this sweep.
        mock_logger.error.assert_called_once()
        call_kwargs = mock_logger.error.call_args
        assert call_kwargs[0][0] == "billing_reconciler.session_quarantined"
        assert call_kwargs[1].get("quarantine") is True
        mock_logger.warning.assert_not_called()

    async def test_quarantine_already_crossed_logs_warning_not_error(self) -> None:
        """Attempt counter already past threshold on a prior sweep → WARNING, not ERROR.

        X1, round-5 remediation: the ERROR quarantine log fires once, on
        crossing. A session that keeps failing after that must not re-trigger
        the ops alert every sweep forever.
        """
        worker, mocks = _make_worker(
            settings=_make_settings(billing_reconciler_quarantine_threshold=5)
        )

        # previous attempts=7 (already past threshold) -> new count=8.
        candidate = _make_gpu_session(billing_finalization_attempts=7)
        mocks["gpu_session_service"].finalize_billing_for_session.return_value = False

        with (
            patch(_REPO_PATH) as MockRepo,
            patch("src.api.services.gpu_session.billing_reconciler_worker.logger") as mock_logger,
        ):
            mock_repo = AsyncMock()
            MockRepo.return_value = mock_repo
            mock_repo.list_pending_billing_finalization.return_value = [candidate]
            mock_repo.increment_billing_finalization_attempts.return_value = 8

            await worker.run_once()

        mock_logger.error.assert_not_called()
        mock_logger.warning.assert_called_once()
        call_kwargs = mock_logger.warning.call_args
        assert call_kwargs[0][0] == "billing_reconciler.session_quarantined"
        assert call_kwargs[1].get("quarantine") is True
        assert call_kwargs[1].get("attempts") == 8

    async def test_refund_quarantine_threshold_triggers_error_log_once_on_crossing(self) -> None:
        """Same crossing-vs-repeat rule as the finalization sweep, for the
        refund-reconciliation sweep (_process_refund_session)."""
        worker, mocks = _make_worker(
            settings=_make_settings(billing_reconciler_quarantine_threshold=5)
        )

        candidate = _make_gpu_session(billing_finalization_attempts=4)
        mocks["gpu_session_service"].reconcile_pending_refund.return_value = False

        with (
            patch(_REPO_PATH) as MockRepo,
            patch("src.api.services.gpu_session.billing_reconciler_worker.logger") as mock_logger,
        ):
            mock_repo = AsyncMock()
            MockRepo.return_value = mock_repo
            mock_repo.list_pending_billing_finalization.return_value = []
            mock_repo.list_pending_refund_reconciliation.return_value = [candidate]
            mock_repo.increment_billing_finalization_attempts.return_value = 5

            await worker.run_once()

        mock_logger.error.assert_called_once()
        call_kwargs = mock_logger.error.call_args
        assert call_kwargs[0][0] == "billing_reconciler.refund_session_quarantined"
        assert call_kwargs[1].get("quarantine") is True
        mock_logger.warning.assert_not_called()

    async def test_refund_quarantine_already_crossed_logs_warning_not_error(self) -> None:
        worker, mocks = _make_worker(
            settings=_make_settings(billing_reconciler_quarantine_threshold=5)
        )

        candidate = _make_gpu_session(billing_finalization_attempts=7)
        mocks["gpu_session_service"].reconcile_pending_refund.return_value = False

        with (
            patch(_REPO_PATH) as MockRepo,
            patch("src.api.services.gpu_session.billing_reconciler_worker.logger") as mock_logger,
        ):
            mock_repo = AsyncMock()
            MockRepo.return_value = mock_repo
            mock_repo.list_pending_billing_finalization.return_value = []
            mock_repo.list_pending_refund_reconciliation.return_value = [candidate]
            mock_repo.increment_billing_finalization_attempts.return_value = 8

            await worker.run_once()

        mock_logger.error.assert_not_called()
        mock_logger.warning.assert_called_once()
        call_kwargs = mock_logger.warning.call_args
        assert call_kwargs[0][0] == "billing_reconciler.refund_session_quarantined"
        assert call_kwargs[1].get("quarantine") is True

    async def test_grace_period_skips_freshly_stopped_sessions(self) -> None:
        """Repository query receives grace_cutoff = now - grace_minutes."""
        worker, _mocks = _make_worker(
            settings=_make_settings(billing_reconciler_grace_period_minutes=3)
        )

        with (
            patch(_REPO_PATH) as MockRepo,
            patch("src.api.services.gpu_session.billing_reconciler_worker.datetime") as mock_dt,
        ):
            fixed_now = datetime(2026, 4, 27, 12, 0, 0, tzinfo=UTC)
            mock_dt.now.return_value = fixed_now

            mock_repo = AsyncMock()
            MockRepo.return_value = mock_repo
            mock_repo.list_pending_billing_finalization.return_value = []

            await worker.run_once()

        expected_cutoff = fixed_now - timedelta(minutes=3)
        # Z2: finalization queries ceil(budget / 2) — the budget is shared across passes.
        mock_repo.list_pending_billing_finalization.assert_called_once_with(
            grace_cutoff=expected_cutoff,
            limit=25,
            now=fixed_now,
        )

    async def test_max_per_sweep_caps_query_limit(self) -> None:
        """Finalization queries ceil(budget / 2); refunds get the rest (Z2)."""
        worker, _mocks = _make_worker(settings=_make_settings(billing_reconciler_max_per_sweep=7))

        with patch(_REPO_PATH) as MockRepo:
            mock_repo = AsyncMock()
            MockRepo.return_value = mock_repo
            mock_repo.list_pending_billing_finalization.return_value = []
            mock_repo.list_pending_refund_reconciliation.return_value = []

            await worker.run_once()

        mock_repo.list_pending_billing_finalization.assert_called_once()
        _, call_kwargs = mock_repo.list_pending_billing_finalization.call_args
        assert call_kwargs["limit"] == 4

    async def test_both_sweeps_pass_now_through_for_backoff_filtering(self) -> None:
        """X1: both sweeps pass `now` to their query so a session past its
        backoff window is re-selected, not just re-logged — replaces the
        removed quarantine_threshold-passthrough test (round-3 T7)."""
        worker, _mocks = _make_worker()

        with patch(_REPO_PATH) as MockRepo:
            mock_repo = AsyncMock()
            MockRepo.return_value = mock_repo
            mock_repo.list_pending_billing_finalization.return_value = []
            mock_repo.list_pending_refund_reconciliation.return_value = []

            await worker.run_once()

        mock_repo.list_pending_refund_reconciliation.assert_called_once()
        _, refund_kwargs = mock_repo.list_pending_refund_reconciliation.call_args
        assert isinstance(refund_kwargs["now"], datetime)
        _, finalize_kwargs = mock_repo.list_pending_billing_finalization.call_args
        assert isinstance(finalize_kwargs["now"], datetime)

    async def test_per_session_exception_does_not_break_sweep(self) -> None:
        """First candidate raises RuntimeError; second candidate is still processed."""
        worker, mocks = _make_worker()

        candidate_a = _make_gpu_session()
        candidate_b = _make_gpu_session()

        finalize_calls: list[Any] = []

        async def flaky_finalize(row: Any) -> bool:
            finalize_calls.append(row)
            if row is candidate_a:
                raise RuntimeError("billing service down")
            return True  # candidate_b succeeds

        mocks["gpu_session_service"].finalize_billing_for_session.side_effect = flaky_finalize

        with patch(_REPO_PATH) as MockRepo:
            mock_repo = AsyncMock()
            MockRepo.return_value = mock_repo
            mock_repo.list_pending_billing_finalization.return_value = [candidate_a, candidate_b]
            # candidate_a fails → bump path; configure increment to return below threshold
            mock_repo.increment_billing_finalization_attempts.return_value = 1

            await worker.run_once()

        assert len(finalize_calls) == 2
        assert finalize_calls[0] is candidate_a
        assert finalize_calls[1] is candidate_b

    # Tick-error recovery (a failing run_once must not kill the loop) is
    # covered generically by test_periodic_worker.py::TestTickErrorLogged.


# ---------------------------------------------------------------------------
# TestSharedSweepBudget — one billing_reconciler_max_per_sweep across both passes (Z2)
# ---------------------------------------------------------------------------


def _limited_backlog(rows: list[GpuSession]) -> AsyncMock:
    """A repository query that honours ``limit`` like the real ORDER BY ... LIMIT."""

    async def query(*, grace_cutoff: datetime, limit: int, now: datetime) -> list[GpuSession]:  # noqa: ARG001
        return rows[:limit]

    return AsyncMock(side_effect=query)


class TestSharedSweepBudget:
    async def _run(
        self, *, budget: int, finalization_backlog: int, refund_backlog: int
    ) -> tuple[AsyncMock, AsyncMock, dict[str, Any]]:
        worker, mocks = _make_worker(
            settings=_make_settings(billing_reconciler_max_per_sweep=budget)
        )
        mocks["gpu_session_service"].finalize_billing_for_session.return_value = True
        mocks["gpu_session_service"].reconcile_pending_refund.return_value = True

        with patch(_REPO_PATH) as MockRepo:
            mock_repo = AsyncMock()
            MockRepo.return_value = mock_repo
            mock_repo.list_pending_billing_finalization = _limited_backlog(
                [_make_gpu_session() for _ in range(finalization_backlog)]
            )
            mock_repo.list_pending_refund_reconciliation = _limited_backlog(
                [_make_gpu_session() for _ in range(refund_backlog)]
            )

            await worker.run_once()

        return (
            mock_repo.list_pending_billing_finalization,
            mock_repo.list_pending_refund_reconciliation,
            mocks,
        )

    @pytest.mark.parametrize(
        ("budget", "finalized", "refunded"), [(2, 1, 1), (7, 4, 3), (50, 25, 25)]
    )
    async def test_both_backlogs_larger_than_budget_process_exactly_budget(
        self, budget: int, finalized: int, refunded: int
    ) -> None:
        """Total across both passes == budget, split ceil / floor — not 2x budget."""
        _, _, mocks = await self._run(
            budget=budget, finalization_backlog=budget * 3, refund_backlog=budget * 3
        )

        service = mocks["gpu_session_service"]
        assert service.finalize_billing_for_session.await_count == finalized
        assert service.reconcile_pending_refund.await_count == refunded
        assert finalized + refunded == budget

    async def test_empty_finalization_backlog_gives_refunds_the_full_budget(self) -> None:
        finalization_query, refund_query, mocks = await self._run(
            budget=7, finalization_backlog=0, refund_backlog=20
        )

        assert refund_query.call_args.kwargs["limit"] == 7
        assert finalization_query.call_args.kwargs["limit"] == 4
        assert mocks["gpu_session_service"].reconcile_pending_refund.await_count == 7

    async def test_small_finalization_backlog_hands_its_remainder_to_refunds(self) -> None:
        """Finalization < ceil(budget / 2) → the refund pass inherits the unused part."""
        _, refund_query, mocks = await self._run(
            budget=10, finalization_backlog=2, refund_backlog=20
        )

        assert refund_query.call_args.kwargs["limit"] == 8  # 10 - 2, more than floor(10 / 2)
        assert mocks["gpu_session_service"].finalize_billing_for_session.await_count == 2
        assert mocks["gpu_session_service"].reconcile_pending_refund.await_count == 8

    async def test_finalization_backlog_cannot_starve_refunds(self) -> None:
        """The half-reservation: a persistent finalization backlog leaves refunds >= floor(b/2)."""
        _, refund_query, _ = await self._run(budget=9, finalization_backlog=100, refund_backlog=100)

        assert refund_query.call_args.kwargs["limit"] == 4  # floor(9 / 2)

    async def test_done_events_log_the_budget_split(self) -> None:
        worker, mocks = _make_worker(settings=_make_settings(billing_reconciler_max_per_sweep=7))
        mocks["gpu_session_service"].finalize_billing_for_session.return_value = True
        mocks["gpu_session_service"].reconcile_pending_refund.return_value = True

        with (
            patch(_REPO_PATH) as MockRepo,
            patch("src.api.services.gpu_session.billing_reconciler_worker.logger") as mock_logger,
        ):
            mock_repo = AsyncMock()
            MockRepo.return_value = mock_repo
            mock_repo.list_pending_billing_finalization = _limited_backlog(
                [_make_gpu_session() for _ in range(9)]
            )
            mock_repo.list_pending_refund_reconciliation = _limited_backlog(
                [_make_gpu_session() for _ in range(9)]
            )

            await worker.run_once()

        events = {call.args[0]: call.kwargs for call in mock_logger.info.call_args_list}
        finalization_done = events["billing_reconciler.finalization_sweep.done"]
        refund_done = events["billing_reconciler.refund_sweep.done"]
        assert (finalization_done["budget"], finalization_done["limit"]) == (7, 4)
        assert (refund_done["budget"], refund_done["limit"]) == (7, 3)
        assert refund_done["finalization_candidates"] == 4


# ---------------------------------------------------------------------------
# TestComputeNextAttemptAt — pure backoff function (X1)
# ---------------------------------------------------------------------------


class TestComputeNextAttemptAt:
    def test_first_failure_backs_off_by_exactly_base(self) -> None:
        now = datetime(2026, 1, 1, tzinfo=UTC)
        result = compute_next_attempt_at(1, now=now, base_minutes=5, cap_hours=24)
        assert result == now + timedelta(minutes=5)

    def test_backoff_doubles_with_each_attempt(self) -> None:
        now = datetime(2026, 1, 1, tzinfo=UTC)
        assert compute_next_attempt_at(2, now=now, base_minutes=5, cap_hours=24) == now + timedelta(
            minutes=10
        )
        assert compute_next_attempt_at(3, now=now, base_minutes=5, cap_hours=24) == now + timedelta(
            minutes=20
        )
        assert compute_next_attempt_at(4, now=now, base_minutes=5, cap_hours=24) == now + timedelta(
            minutes=40
        )

    def test_backoff_is_capped(self) -> None:
        """10 failures at base=5min would be 5*2**9=2560min (~42.7h); capped to 24h."""
        now = datetime(2026, 1, 1, tzinfo=UTC)
        result = compute_next_attempt_at(10, now=now, base_minutes=5, cap_hours=24)
        assert result == now + timedelta(hours=24)

    def test_backoff_never_exceeds_cap_even_for_very_large_attempt_counts(self) -> None:
        now = datetime(2026, 1, 1, tzinfo=UTC)
        result = compute_next_attempt_at(1000, now=now, base_minutes=5, cap_hours=24)
        assert result == now + timedelta(hours=24)
