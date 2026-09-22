"""CLI wiring and reporting for the empty-completion refund backfill.

The refund/status logic is covered against a real database in
``tests/integration/test_refund_empty_completions.py``; this file covers what
surrounds it: the Typer surface (which once rejected the documented ``run``
subcommand), argument normalisation, report rendering, and ``_run_impl``'s
resource setup and teardown.
"""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from rich.console import Console
from typer.testing import CliRunner

from src.cli import refund_empty_completions as cli
from src.db.repositories.job import EmptyCompletion

if TYPE_CHECKING:
    from pathlib import Path

runner = CliRunner()


def _kwargs(mock: AsyncMock) -> dict[str, Any]:
    """The keyword arguments of a mock's most recent await (asserting there was one)."""
    assert mock.await_args is not None
    return dict(mock.await_args.kwargs)


def _row(*, debit: int | None = 20, account: bool = True) -> EmptyCompletion:
    return EmptyCompletion(
        job_id=uuid4(),
        user_id=uuid4(),
        product_id="vex",
        created_at=datetime(2026, 9, 21, 10, 30, tzinfo=UTC),
        account_id=uuid4() if account else None,
        debit_amount=debit,
    )


def _render(report: cli.RefundReport) -> str:
    recording = Console(record=True, width=200)
    with patch.object(cli, "console", recording):
        cli._print_report(report)
    return recording.export_text()


# ---------------------------------------------------------------------------
# Typer surface
# ---------------------------------------------------------------------------


class TestCommandSurface:
    def test_run_is_a_real_subcommand(self) -> None:
        """The documented ``... refund_empty_completions run`` invocation must parse."""
        with patch.object(cli, "_run_impl", new=AsyncMock()) as run_impl:
            result = runner.invoke(cli.app, ["run"])

        assert result.exit_code == 0, result.output
        run_impl.assert_awaited_once()

    def test_dry_run_is_the_default(self) -> None:
        with patch.object(cli, "_run_impl", new=AsyncMock()) as run_impl:
            runner.invoke(cli.app, ["run"])

        assert _kwargs(run_impl)["apply"] is False

    def test_apply_requires_reviewed_ids_before_database_setup(self) -> None:
        with patch.object(cli, "_run_impl", new=AsyncMock()) as run_impl:
            result = runner.invoke(cli.app, ["run", "--apply"])

        assert result.exit_code != 0
        assert "requires explicit reviewed ids" in result.output
        # Validation happens in Typer's synchronous entrypoint, before
        # _run_impl can create a database connection or issue a refund.
        run_impl.assert_not_awaited()

    def test_apply_with_repeated_job_ids_is_allowed(self) -> None:
        job_a, job_b = uuid4(), uuid4()
        with patch.object(cli, "_run_impl", new=AsyncMock()) as run_impl:
            result = runner.invoke(
                cli.app,
                ["run", "--apply", "--job-id", str(job_a), "--job-id", str(job_b)],
            )

        assert result.exit_code == 0, result.output
        assert _kwargs(run_impl)["apply"] is True
        assert _kwargs(run_impl)["job_ids"] == [job_a, job_b]

    def test_job_ids_file_is_parsed_before_apply(self, tmp_path: Path) -> None:
        job_a, job_b = uuid4(), uuid4()
        job_ids_file = tmp_path / "reviewed-job-ids.txt"
        job_ids_file.write_text(f"# sourced from completion logs\n{job_a}\n\n{job_b} # verified\n")

        with patch.object(cli, "_run_impl", new=AsyncMock()) as run_impl:
            result = runner.invoke(cli.app, ["run", "--apply", "--job-ids-file", str(job_ids_file)])

        assert result.exit_code == 0, result.output
        assert _kwargs(run_impl)["job_ids"] == [job_a, job_b]

    def test_malformed_job_ids_file_fails_before_processing(self, tmp_path: Path) -> None:
        job_ids_file = tmp_path / "reviewed-job-ids.txt"
        job_ids_file.write_text(f"{uuid4()}\nnot-a-uuid\n")

        with patch.object(cli, "_run_impl", new=AsyncMock()) as run_impl:
            result = runner.invoke(cli.app, ["run", "--apply", "--job-ids-file", str(job_ids_file)])

        assert result.exit_code != 0
        assert "line 2" in result.output
        run_impl.assert_not_awaited()

    def test_filters_are_normalised(self) -> None:
        job_a, job_b = uuid4(), uuid4()
        with patch.object(cli, "_run_impl", new=AsyncMock()) as run_impl:
            result = runner.invoke(
                cli.app,
                [
                    "run",
                    "--product",
                    "vex",
                    "--since",
                    "2026-01-01",
                    "--job-id",
                    str(job_a),
                    "--job-id",
                    str(job_b),
                    "--limit",
                    "5",
                ],
            )

        assert result.exit_code == 0, result.output
        kwargs = _kwargs(run_impl)
        assert kwargs["product"] == "vex"
        assert kwargs["limit"] == 5
        assert kwargs["job_ids"] == [job_a, job_b]
        # A naive --since is read as UTC, never as the operator's local zone.
        assert kwargs["since"] == datetime(2026, 1, 1, tzinfo=UTC)

    def test_no_job_id_means_no_restriction_not_an_empty_restriction(self) -> None:
        """``job_ids=[]`` would select nothing; absence must be ``None``."""
        with patch.object(cli, "_run_impl", new=AsyncMock()) as run_impl:
            runner.invoke(cli.app, ["run"])

        assert _kwargs(run_impl)["job_ids"] is None
        assert _kwargs(run_impl)["since"] is None

    def test_non_positive_limit_is_rejected(self) -> None:
        with patch.object(cli, "_run_impl", new=AsyncMock()) as run_impl:
            result = runner.invoke(cli.app, ["run", "--limit", "0"])

        assert result.exit_code != 0
        run_impl.assert_not_awaited()


# ---------------------------------------------------------------------------
# Report rendering
# ---------------------------------------------------------------------------


class TestReportRendering:
    def test_dry_run_lists_jobs_totals_and_warns(self) -> None:
        rows = [_row(debit=20), _row(debit=30)]
        report = cli.RefundReport(apply=False, found=rows)

        text = _render(report)

        assert "dry-run" in text
        for row in rows:
            assert str(row.job_id) in text
        assert "Jobs: 2" in text
        assert "Total debit: 50 tokens" in text
        assert "candidates" in text
        # The user-deleted-outputs caveat travels with every unfiltered dry-run.
        assert "owners later deleted" in text
        assert "--job-ids-file" in text
        assert "Refunded:" not in text

    def test_job_without_a_debit_is_shown_as_such(self) -> None:
        report = cli.RefundReport(apply=False, found=[_row(debit=None, account=False)])

        text = _render(report)

        assert "no debit" in text
        assert "Total debit: 0 tokens" in text

    def test_apply_report_summarises_outcomes_and_names_skips(self) -> None:
        skipped_id = uuid4()
        report = cli.RefundReport(
            apply=True,
            found=[_row(), _row(), _row()],
            refunded=1,
            already_refunded=1,
            skipped=[cli._Skip(skipped_id, cli._Outcome.NOT_ELIGIBLE, "no_debit_found: x")],
        )

        text = _render(report)

        assert "dry-run" not in text
        assert "Refunded: 1" in text
        assert "Already refunded (status corrected): 1" in text
        assert "Skipped: 1" in text
        assert str(skipped_id) in text
        assert "no_debit_found" in text

    def test_total_debit_ignores_jobs_without_a_debit(self) -> None:
        report = cli.RefundReport(apply=False, found=[_row(debit=20), _row(debit=None)])

        assert report.total_debit_tokens == 20


# ---------------------------------------------------------------------------
# Resource setup and teardown
# ---------------------------------------------------------------------------


def _settings(*, redis_url: str | None) -> SimpleNamespace:
    return SimpleNamespace(
        database_url="postgresql+asyncpg://x:x@localhost/x",
        redis_url=redis_url,
        redis_socket_connect_timeout_seconds=0.25,
        redis_socket_timeout_seconds=0.05,
        redis_health_check_interval_seconds=30.0,
        redis_max_connections=5,
    )


class _Harness:
    """Patches every external dependency of ``_run_impl`` and records how it was used."""

    def __init__(self, *, redis_url: str | None) -> None:
        self.session = AsyncMock()
        self.db = MagicMock()
        self.db.session_factory.return_value = self.session
        self.db.close = AsyncMock()
        self.report = cli.RefundReport(apply=False)
        self.run_refund = AsyncMock(return_value=self.report)
        self.print_report = MagicMock()
        self.init_pool = MagicMock()
        self.close_pool = AsyncMock()
        self._redis_url = redis_url

    def patches(self) -> list[object]:
        return [
            patch.object(cli, "Settings", return_value=_settings(redis_url=self._redis_url)),
            patch.object(cli, "DatabaseManager", return_value=self.db),
            patch.object(cli, "run_refund", new=self.run_refund),
            patch.object(cli, "_print_report", new=self.print_report),
            patch("src.core.redis.init_redis_pool", new=self.init_pool),
            patch("src.core.redis.close_redis_pool", new=self.close_pool),
        ]


async def _run_impl(harness: _Harness, *, apply: bool) -> None:
    from contextlib import ExitStack

    with ExitStack() as stack:
        for p in harness.patches():
            stack.enter_context(p)  # type: ignore[arg-type]
        await cli._run_impl(product="vex", since=None, job_ids=None, limit=None, apply=apply)


class TestRunImpl:
    async def test_dry_run_never_touches_redis(self) -> None:
        harness = _Harness(redis_url="redis://localhost:6379")

        await _run_impl(harness, apply=False)

        harness.init_pool.assert_not_called()
        harness.close_pool.assert_not_awaited()
        assert _kwargs(harness.run_refund)["event_bus"] is None
        assert _kwargs(harness.run_refund)["apply"] is False

    async def test_apply_with_redis_publishes_live_balance_events_and_closes_the_pool(
        self,
    ) -> None:
        harness = _Harness(redis_url="redis://localhost:6379")

        await _run_impl(harness, apply=True)

        harness.init_pool.assert_called_once()
        bus = _kwargs(harness.run_refund)["event_bus"]
        assert bus is not None
        assert bus._enabled is True
        harness.close_pool.assert_awaited_once()

    async def test_apply_without_redis_still_runs_but_publishes_nothing(self) -> None:
        harness = _Harness(redis_url=None)

        await _run_impl(harness, apply=True)

        harness.init_pool.assert_not_called()
        assert _kwargs(harness.run_refund)["event_bus"] is None
        harness.run_refund.assert_awaited_once()

    async def test_session_and_database_are_closed_and_the_report_printed(self) -> None:
        harness = _Harness(redis_url=None)

        await _run_impl(harness, apply=False)

        harness.session.close.assert_awaited_once()
        harness.db.close.assert_awaited_once()
        harness.print_report.assert_called_once_with(harness.report)

    async def test_resources_are_released_when_the_run_fails(self) -> None:
        harness = _Harness(redis_url="redis://localhost:6379")
        harness.run_refund.side_effect = RuntimeError("boom")

        with pytest.raises(RuntimeError, match="boom"):
            await _run_impl(harness, apply=True)

        harness.session.close.assert_awaited_once()
        harness.db.close.assert_awaited_once()
        harness.close_pool.assert_awaited_once()
        harness.print_report.assert_not_called()

    async def test_filters_are_forwarded_to_the_run(self) -> None:
        harness = _Harness(redis_url=None)
        job_ids = [uuid4()]
        since = datetime(2026, 1, 1, tzinfo=UTC)

        from contextlib import ExitStack

        with ExitStack() as stack:
            for p in harness.patches():
                stack.enter_context(p)  # type: ignore[arg-type]
            await cli._run_impl(product="vex", since=since, job_ids=job_ids, limit=3, apply=False)

        kwargs = _kwargs(harness.run_refund)
        assert kwargs["product"] == "vex"
        assert kwargs["since"] == since
        assert kwargs["job_ids"] == job_ids
        assert kwargs["limit"] == 3
