"""Unit tests for BillingReconcilerWorker."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

from src.api.services.gpu_session.billing_reconciler_worker import BillingReconcilerWorker
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
    } | overrides
    worker = BillingReconcilerWorker(
        session_factory=mocks["session_factory"],
        gpu_session_service=mocks["gpu_session_service"],
        settings=mocks["settings"],
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
        # No quarantine error — just still_failing
        mocks["gpu_session_service"].finalize_billing_for_session.assert_called_once_with(candidate)

    async def test_quarantine_threshold_triggers_error_log(self) -> None:
        """Attempt counter at threshold → quarantine log at ERROR; session NOT mutated."""
        worker, mocks = _make_worker(
            settings=_make_settings(billing_reconciler_quarantine_threshold=5)
        )

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

        # Must log quarantine=True at ERROR
        mock_logger.error.assert_called_once()
        call_kwargs = mock_logger.error.call_args
        assert call_kwargs[0][0] == "billing_reconciler.session_quarantined"
        assert call_kwargs[1].get("quarantine") is True

    async def test_grace_period_skips_freshly_stopped_sessions(self) -> None:
        """Repository query receives grace_cutoff = now - grace_minutes."""
        worker, mocks = _make_worker(
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
        mock_repo.list_pending_billing_finalization.assert_called_once_with(
            grace_cutoff=expected_cutoff,
            limit=mocks["settings"].billing_reconciler_max_per_sweep,
        )

    async def test_max_per_sweep_caps_query_limit(self) -> None:
        """Repository query receives limit=billing_reconciler_max_per_sweep."""
        worker, mocks = _make_worker(settings=_make_settings(billing_reconciler_max_per_sweep=7))

        with patch(_REPO_PATH) as MockRepo:
            mock_repo = AsyncMock()
            MockRepo.return_value = mock_repo
            mock_repo.list_pending_billing_finalization.return_value = []

            await worker.run_once()

        mock_repo.list_pending_billing_finalization.assert_called_once()
        _, call_kwargs = mock_repo.list_pending_billing_finalization.call_args
        assert call_kwargs["limit"] == 7

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
