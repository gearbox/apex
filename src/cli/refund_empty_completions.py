"""CLI command: refund Aisha jobs that were completed with no outputs.

Before the empty-completion fix, ``AishaJobPoller`` marked a job ``completed``
whenever ComfyUI's history entry carried an ``outputs`` key — including the
empty one ComfyUI writes when execution *errors*. The user was debited and never
refunded. This command finds those jobs, refunds the debit and corrects the
status to ``failed``.

Usage:
    python -m src.cli.refund_empty_completions run [--product vex] [--since 2026-01-01]
           [--job-id UUID ...] [--limit N] [--apply]

Dry-run by default: without ``--apply`` it prints what it would do and changes
nothing. Re-runnable: a corrected job is no longer ``completed``, so a second
``--apply`` finds nothing to do.

⚠️ Review the dry-run before applying. Selection is "Aisha, ``completed``, not
soft-deleted, zero output rows". A job whose owner later deleted all of its
outputs matches too — the database cannot tell it from a defective completion.
Cross-check against ``job.transition.completed output_count=0`` log lines and
narrow with ``--job-id`` / ``--since`` when in doubt.
"""

from __future__ import annotations

import asyncio
import dataclasses
import enum
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Annotated
from uuid import UUID  # noqa: TC003 - Typer resolves the --job-id annotation at runtime

import structlog
import typer
from rich.console import Console
from rich.table import Table

from src.api.services.billing import BillingService
from src.api.services.billing_errors import RefundNotEligibleError, RefundNotEligibleReason
from src.api.services.event_bus import EventBus
from src.api.services.generation.aisha_failures import AishaFailure
from src.core.config import Settings
from src.db.repositories.job import EmptyCompletion, JobRepository
from src.db.session import DatabaseManager

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from src.api.services.billing import BalanceEvent

logger = structlog.get_logger(__name__)
console = Console()

app = typer.Typer(
    name="refund-empty-completions",
    help="Refund Aisha jobs that were completed with no outputs",
    no_args_is_help=True,
)

_FAILURE_CODE = AishaFailure.PROVIDER_EXECUTION_FAILED
_ERROR_MESSAGE = (
    "Marked failed by the empty-completion backfill: the job was completed with no "
    "outputs (ComfyUI execution error reported as success)."
)
_REFUND_DESCRIPTION = "Job produced no output — tokens refunded"


# ---------------------------------------------------------------------------
# Internal types
# ---------------------------------------------------------------------------


class _Outcome(enum.StrEnum):
    REFUNDED = "refunded"
    ALREADY_REFUNDED = "already_refunded"
    NOT_ELIGIBLE = "not_eligible"
    NO_LONGER_EMPTY = "no_longer_empty"
    FAILED = "failed"


@dataclasses.dataclass
class _Skip:
    job_id: UUID
    outcome: _Outcome
    detail: str


@dataclasses.dataclass
class RefundReport:
    """What a run found and did. In dry-run only ``found`` is populated."""

    apply: bool
    found: list[EmptyCompletion] = dataclasses.field(default_factory=list)
    refunded: int = 0
    already_refunded: int = 0
    skipped: list[_Skip] = dataclasses.field(default_factory=list)

    @property
    def total_debit_tokens(self) -> int:
        return sum(row.debit_amount or 0 for row in self.found)


# ---------------------------------------------------------------------------
# Per-job logic
# ---------------------------------------------------------------------------


async def _settle_one(
    session: AsyncSession,
    billing: BillingService,
    row: EmptyCompletion,
) -> tuple[_Outcome, BalanceEvent | None, str]:
    """Correct one job and refund its debit inside ONE transaction.

    The predicate-guarded status change runs first: if the job has outputs or is
    no longer ``completed`` nothing is refunded at all. It also takes the row
    locks in the same order as ``transition_to_failed`` (job, then debit), so the
    two cannot deadlock. Any exception rolls the whole job back and leaves the
    caller free to continue with the next one.

    Returns:
        ``(outcome, balance_event, detail)``. ``balance_event`` is non-None only
        for ``REFUNDED`` and must be published by the caller *after* the commit
        this function performs.
    """
    try:
        changed = await JobRepository(session).mark_empty_completion_failed(
            row.job_id,
            failure_code=_FAILURE_CODE.value,
            error_message=_ERROR_MESSAGE,
            public_error_message=_FAILURE_CODE.public_message,
        )
        if not changed:
            await session.rollback()
            return _Outcome.NO_LONGER_EMPTY, None, "job has outputs or is no longer completed"

        try:
            refund = await billing.refund(
                row.job_id,
                description=_REFUND_DESCRIPTION,
                session=session,
                product_id=row.product_id,
                user_id=row.user_id,
            )
        except RefundNotEligibleError as exc:
            if exc.reason is not RefundNotEligibleReason.ALREADY_REFUNDED:
                await session.rollback()
                return _Outcome.NOT_ELIGIBLE, None, f"{exc.reason.value}: {exc}"
            # Debit was already compensated (e.g. by a manual adjustment): the
            # status is still wrong and still gets corrected, with no second refund.
            await session.commit()
            return _Outcome.ALREADY_REFUNDED, None, "debit already refunded"

        await session.commit()
    except Exception as exc:
        await session.rollback()
        logger.exception("refund_empty_completions.job_failed", job_id=str(row.job_id))
        return _Outcome.FAILED, None, f"{type(exc).__name__}: {exc}"
    else:
        return _Outcome.REFUNDED, refund.event, ""


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


async def run_refund(
    session: AsyncSession,
    *,
    billing: BillingService,
    event_bus: EventBus | None = None,
    product: str | None = None,
    since: datetime | None = None,
    job_ids: list[UUID] | None = None,
    limit: int | None = None,
    apply: bool = False,
) -> RefundReport:
    """Find empty completions and, with ``apply``, refund and correct each one.

    Each job is its own transaction, committed (or rolled back) before the next
    begins, so one bad row never undoes the rest. Balance events are published
    strictly after their job's commit.
    """
    rows = await JobRepository(session).list_empty_completions(
        product_id=product,
        since=since,
        job_ids=job_ids,
        limit=limit,
    )

    report = RefundReport(apply=apply, found=list(rows))
    if not apply:
        return report

    for row in rows:
        outcome, event, detail = await _settle_one(session, billing, row)
        match outcome:
            case _Outcome.REFUNDED:
                report.refunded += 1
                if event_bus is not None:
                    await event_bus.publish_balance(event)
            case _Outcome.ALREADY_REFUNDED:
                report.already_refunded += 1
            case _:
                report.skipped.append(_Skip(row.job_id, outcome, detail))
        logger.info(
            "refund_empty_completions.job",
            job_id=str(row.job_id),
            outcome=outcome.value,
            debit_amount=row.debit_amount,
        )
    return report


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _print_report(report: RefundReport) -> None:
    title = "Empty completions" + ("" if report.apply else " (dry-run — nothing changed)")
    table = Table(title=title)
    table.add_column("Job", style="bold")
    table.add_column("Product")
    table.add_column("Account")
    table.add_column("Created")
    table.add_column("Debit (tokens)", justify="right")

    for row in report.found:
        table.add_row(
            str(row.job_id),
            row.product_id,
            str(row.account_id) if row.account_id is not None else "— no debit —",
            row.created_at.astimezone(UTC).strftime("%Y-%m-%d %H:%M:%S"),
            str(row.debit_amount) if row.debit_amount is not None else "—",
        )
    console.print(table)
    console.print(
        f"Jobs: [bold]{len(report.found)}[/bold]   "
        f"Total debit: [bold]{report.total_debit_tokens}[/bold] tokens"
    )

    if not report.apply:
        console.print(
            "[yellow]Dry-run. Re-run with --apply to refund and mark these jobs failed. "
            "A job whose owner deleted its outputs looks identical to a defective one — "
            "review the list (or pass --job-id) first.[/yellow]"
        )
        return

    console.print(
        f"Refunded: [green]{report.refunded}[/green]   "
        f"Already refunded (status corrected): [cyan]{report.already_refunded}[/cyan]   "
        f"Skipped: [red]{len(report.skipped)}[/red]"
    )
    for skip in report.skipped:
        console.print(f"  [red]skipped[/red] {skip.job_id}  {skip.outcome.value}: {skip.detail}")


# ---------------------------------------------------------------------------
# CLI wiring
# ---------------------------------------------------------------------------


async def _run_impl(
    *,
    product: str | None,
    since: datetime | None,
    job_ids: list[UUID] | None,
    limit: int | None,
    apply: bool,
) -> None:
    settings = Settings()

    event_bus: EventBus | None = None
    if apply and settings.redis_url:
        from src.core.redis import init_redis_pool

        init_redis_pool(
            settings.redis_url,
            socket_connect_timeout=settings.redis_socket_connect_timeout_seconds,
            socket_timeout=settings.redis_socket_timeout_seconds,
            health_check_interval=settings.redis_health_check_interval_seconds,
            max_connections=settings.redis_max_connections,
        )
        event_bus = EventBus(enabled=True)

    db = DatabaseManager(settings.database_url)
    session = db.session_factory()
    try:
        report = await run_refund(
            session,
            billing=BillingService(),
            event_bus=event_bus,
            product=product,
            since=since,
            job_ids=job_ids,
            limit=limit,
            apply=apply,
        )
    finally:
        await session.close()
        await db.close()
        if event_bus is not None:
            from src.core.redis import close_redis_pool

            await close_redis_pool()

    _print_report(report)


@app.callback()
def _root() -> None:
    """Refund Aisha jobs that were completed with no outputs.

    The callback keeps ``run`` a real subcommand: Typer otherwise collapses a
    single-command app into the root command and rejects ``run`` as an
    unexpected argument.
    """


@app.command("run")
def run(
    product: Annotated[
        str | None,
        typer.Option("--product", help="Product slug filter (e.g. vex)"),
    ] = None,
    since: Annotated[
        datetime | None,
        typer.Option("--since", help="Only jobs created at or after this date (e.g. 2026-01-01)"),
    ] = None,
    job_id: Annotated[
        list[UUID] | None,
        typer.Option(
            "--job-id",
            help="Restrict to this job (repeatable) — apply a reviewed subset only",
        ),
    ] = None,
    limit: Annotated[
        int | None,
        typer.Option("--limit", min=1, help="Max jobs to process (for incremental runs)"),
    ] = None,
    apply: Annotated[
        bool,
        typer.Option("--apply", help="Refund and correct the jobs. Default is a dry-run."),
    ] = False,
) -> None:
    """Refund Aisha jobs that were completed with no outputs (dry-run unless --apply)."""
    if since is not None and since.tzinfo is None:
        since = since.replace(tzinfo=UTC)
    asyncio.run(
        _run_impl(product=product, since=since, job_ids=job_id or None, limit=limit, apply=apply)
    )


if __name__ == "__main__":
    app()
