"""Planner runs, worker side: claim, execute without a connection, finish, sweep.

``StudioRunExecutor`` is a process singleton that owns its short transactions: the claim, each
stage change and the final write are separate commits, so **no database connection is held while a
model call is awaited**. The success write saves the result version and marks the run ``succeeded``
atomically; a crashed or timed-out run is failed and refunded by the sweep and never executed
twice. Lock order is track, then run.

It is **generic over the run kind** (P2a-48): it is given one handler per kind
(``run_kinds/*.py``) and names none. Request decoding, loading the working documents, the graph and
which version is saved are the handler's; claim, stages, deadline, metering, the atomic finish,
failure, refund and sweep are here, once. The handlers - not this module - load ``langgraph``, so
processes that never run the planner never load it.

Logs carry ids, kinds, stages, model keys, token counts, durations and error codes only - never the
brief, the artist, a prompt or model output; an unexpected exception is logged as its type and
traceback frames, **without its message** (messages can quote user data).
"""

from __future__ import annotations

import asyncio
import time
import traceback
from typing import TYPE_CHECKING

import msgspec
import structlog

from src.core.enums import StudioRunKind, StudioRunStage
from src.db.repositories.studio import StudioRunRepository
from src.llm.port import LlmReplyError
from src.studio.errors import StudioError
from src.studio.run_failures import (
    ERR_INTERNAL,
    ERR_TIMEOUT,
    RunFailure,
    failure_of_error,
    failure_of_outcome,
)
from src.studio.services.run_context import RunContextBuilder
from src.studio.services.run_kinds.base import ClaimedRun

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from sqlalchemy.ext.asyncio import AsyncSession

    from src.api.services.billing import BalanceEvent, BillingService
    from src.core.config import Settings
    from src.db.models.studio import StudioRun
    from src.db.session import DatabaseManager
    from src.llm.port import LlmMessage, LlmResult, LlmUsage, StructuredLlmPort
    from src.llm.registry import LlmModelRegistry
    from src.studio.planner.context import ArtistContextProvider
    from src.studio.planner.outcome import RunOutcome
    from src.studio.services.run_events import StudioRunEvents
    from src.studio.services.run_kinds.base import RunHandler

__all__ = ["ClaimedRun", "RunSwept", "StudioRunExecutor"]

logger = structlog.get_logger(__name__)

_LEASE_GRACE_SECONDS = 30
"""Added to the deadline: the lease outlives the run's own timeout, so only a dead worker's run
is ever swept."""


class RunSwept(Exception):
    """The run is no longer ``running`` (the sweep failed it): stop and write nothing."""


class _MeteredLlm:
    """Counts the calls and tokens that pass through a port, so a failed run still reports them.

    A reply that arrives but is unusable (truncated, undecodable) has been paid for: the port's
    error carries its token counts, and they are added before the error goes on.
    """

    def __init__(self, inner: StructuredLlmPort) -> None:
        self._inner = inner
        self.calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0

    async def complete[T: msgspec.Struct](
        self, messages: Sequence[LlmMessage], *, schema: type[T], schema_name: str
    ) -> LlmResult[T]:
        self.calls += 1
        try:
            result = await self._inner.complete(messages, schema=schema, schema_name=schema_name)
        except LlmReplyError as exc:
            if exc.usage is not None:
                self._add(exc.usage)
            raise
        self._add(result.usage)
        return result

    def _add(self, usage: LlmUsage) -> None:
        self.prompt_tokens += usage.prompt_tokens
        self.completion_tokens += usage.completion_tokens


def _frames(exc: BaseException) -> str:
    """The traceback of ``exc`` without its message (a message can quote user data)."""
    return "".join(traceback.format_tb(exc.__traceback__))


class StudioRunExecutor:
    """Claims queued runs, executes them without holding a connection, and sweeps dead ones."""

    def __init__(
        self,
        *,
        db_manager: DatabaseManager,
        billing: BillingService,
        registry: LlmModelRegistry,
        llm_factory: Callable[[str], StructuredLlmPort],
        context_provider: ArtistContextProvider[AsyncSession],
        events: StudioRunEvents,
        settings: Settings,
        handlers: Mapping[StudioRunKind, RunHandler],
    ) -> None:
        self._db = db_manager
        self._billing = billing
        self._registry = registry
        self._llm_factory = llm_factory
        self._context = RunContextBuilder(context_provider)
        self._events = events
        self._deadline = settings.studio_run_deadline_seconds
        self._handlers = handlers

    # -- claim -------------------------------------------------------------------------------

    async def claim(self, free_slots: int) -> list[ClaimedRun]:
        """Move up to ``free_slots`` queued runs (oldest first) to ``running``.

        One short transaction: it locks the rows (``SKIP LOCKED``, so workers never contend),
        freezes each run's context snapshot and stamps its lease. A run whose context cannot be
        built is failed and refunded on the spot, so one bad run never blocks the others.
        """
        if free_slots <= 0:
            return []
        claimed: list[ClaimedRun] = []
        changed: list[StudioRun] = []
        balance_events: list[BalanceEvent | None] = []
        async with self._db.session() as session:
            runs = StudioRunRepository(session)
            for row in await runs.claim_queued(free_slots):
                try:
                    async with session.begin_nested():
                        item, started = await self._start(session, runs, row)
                except Exception as exc:  # the run guard of the claim: one run must not stop others
                    # Not ``logger.exception``: an exception message can quote user data.
                    logger.error(  # noqa: TRY400
                        "studio.run.claim_failed",
                        run_id=str(row.id),
                        exc_type=type(exc).__name__,
                        frames=_frames(exc),
                    )
                    failed = await runs.fail_queued(row.id, error_code=ERR_INTERNAL)
                    if failed is not None:
                        changed.append(failed)
                        balance_events.append(await self._refund(session, failed, ERR_INTERNAL))
                    continue
                claimed.append(item)
                changed.append(started)
        for event in balance_events:
            await self._events.balance(event)
        for run in changed:
            await self._events.run_updated(run)
        return claimed

    async def _start(
        self, session: AsyncSession, runs: StudioRunRepository, row: StudioRun
    ) -> tuple[ClaimedRun, StudioRun]:
        kind = StudioRunKind(row.kind)
        handler = self._handlers[kind]
        request = handler.decode_request(row.request)
        context = await self._context.build(session, row, request)
        # The working documents are loaded here, inside the claim transaction, from the ids the
        # run was accepted with - never from the track's current pointers.
        loaded = await handler.load(session, row, request, context)
        entry = self._registry.entry(row.model_key)
        started = await runs.mark_running(
            row.id,
            lease_seconds=self._deadline + _LEASE_GRACE_SECONDS,
            model_id=entry.model,
            prompt_version=handler.prompt_version,
            context_snapshot=msgspec.to_builtins(loaded.context),
        )
        if started is None:  # pragma: no cover - the row is locked by this transaction
            raise RuntimeError("claimed run was not queued")
        return (
            ClaimedRun(
                run_id=row.id,
                user_id=row.user_id,
                product_id=row.product_id,
                track_id=row.track_id,
                kind=kind,
                request=request,
                context=loaded.context,
                model_key=row.model_key,
                price_tokens=row.price_tokens,
                inputs=loaded.inputs,
                documents=loaded.documents,
            ),
            started,
        )

    # -- execute -----------------------------------------------------------------------------

    async def execute(self, claimed: ClaimedRun) -> None:
        """Run a claimed run to a terminal state. Never raises (except cancellation).

        The graph runs with **no database connection checked out**; every stage change, and the
        final write, is its own short transaction. A swept run (its lease expired and it was
        failed) is abandoned silently; a cancelled task leaves the run to the sweep.
        """
        started = time.monotonic()
        meter = _MeteredLlm(self._llm_factory(claimed.model_key))
        failure: RunFailure
        try:
            handler = self._handlers[claimed.kind]
            try:
                outcome = await self._run_graph(claimed, handler, meter)
            except TimeoutError:
                # The run's own deadline. Only the graph is bounded by it: a timeout raised by the
                # save step below is not a ``run_timeout`` and must not be reported as one.
                failure = RunFailure(ERR_TIMEOUT)
            else:
                unsuccessful = failure_of_outcome(outcome)
                if unsuccessful is None:
                    await self._on_stage(claimed, StudioRunStage.SAVING)
                    if await self._finish_success(claimed, handler, outcome, meter):
                        self._log_finished(claimed, "succeeded", None, meter, started)
                    return
                failure = unsuccessful
        except RunSwept:
            logger.warning("studio.run.swept", run_id=str(claimed.run_id))
            return
        except StudioError as exc:  # the save step's own refusals (e.g. VersionLimitReached)
            failure = failure_of_error(exc)
        except Exception as exc:  # the run guard: whatever else happens, the run ends and refunds
            failure = failure_of_error(exc)
            if failure.code == ERR_INTERNAL:
                # Not ``logger.exception``: an exception message can quote user data.
                logger.error(  # noqa: TRY400
                    "studio.run.internal_error",
                    run_id=str(claimed.run_id),
                    exc_type=type(exc).__name__,
                    frames=_frames(exc),
                )
        await self._finish_failure(claimed, failure, meter)
        self._log_finished(claimed, "failed", failure.code, meter, started)

    async def _run_graph(
        self, claimed: ClaimedRun, handler: RunHandler, meter: _MeteredLlm
    ) -> RunOutcome:
        async with asyncio.timeout(self._deadline):
            return await handler.run(
                claimed,
                llm=meter,
                on_stage=lambda stage: self._on_stage(claimed, stage),
            )

    async def _on_stage(self, claimed: ClaimedRun, stage: StudioRunStage) -> None:
        """Record a stage change (short transaction, then publish); stop if the run was swept."""
        async with self._db.session() as session:
            run = await StudioRunRepository(session).set_stage(claimed.run_id, stage)
        if run is None:
            raise RunSwept
        await self._events.run_updated(run)

    # -- finish ------------------------------------------------------------------------------

    async def _finish_success(
        self,
        claimed: ClaimedRun,
        handler: RunHandler,
        outcome: RunOutcome,
        meter: _MeteredLlm,
    ) -> bool:
        """Save the result as an ordinary version **and** mark the run ``succeeded``, atomically.

        *Which* version is saved - its kind, parent, fallback and warnings - is the handler's
        (``origin = llm``, never made current; if the claim-time parent was deleted the result goes
        under the track's current version of that kind with ``parent_version_replaced``: a paid,
        valid result is never thrown away because its lineage anchor is gone). If the run is no
        longer ``running`` the whole transaction is rolled back, so no version outlives a swept
        run.

        Returns:
            ``True`` when the run is now ``succeeded``; ``False`` when it had been swept.
        """
        async with self._db.session() as session:
            saved = await handler.save(session, claimed, outcome)
            run = await StudioRunRepository(session).finish_succeeded(
                claimed.run_id,
                result_plan_version_id=saved.result_plan_version_id,
                result_lyrics_version_id=saved.result_lyrics_version_id,
                warnings=[warning.to_json() for warning in saved.warnings],
                llm_calls=meter.calls,
                prompt_tokens=meter.prompt_tokens,
                completion_tokens=meter.completion_tokens,
            )
            if run is None:
                await session.rollback()
                logger.warning("studio.run.finish_lost", run_id=str(claimed.run_id))
                return False
        await self._events.run_updated(run)
        return True

    async def _finish_failure(
        self, claimed: ClaimedRun, failure: RunFailure, meter: _MeteredLlm
    ) -> None:
        """``running → failed`` and the refund, in one transaction."""
        balance_event: BalanceEvent | None = None
        async with self._db.session() as session:
            run = await StudioRunRepository(session).finish_failed(
                claimed.run_id,
                error_code=failure.code,
                error_detail=failure.detail,
                llm_calls=meter.calls,
                prompt_tokens=meter.prompt_tokens,
                completion_tokens=meter.completion_tokens,
            )
            if run is not None:
                balance_event = await self._refund(session, run, failure.code)
        if run is None:  # swept (and refunded) in the meantime
            logger.warning("studio.run.finish_lost", run_id=str(claimed.run_id))
            return
        await self._events.balance(balance_event)
        await self._events.run_updated(run)

    async def _refund(
        self, session: AsyncSession, run: StudioRun, reason: str
    ) -> BalanceEvent | None:
        """Refund a run's reservation in full (nothing is written for a free run)."""
        if run.price_tokens <= 0:
            return None
        refunded = await self._billing.refund(
            run.id,
            description="Studio run failed",
            metadata={"studio_run_id": str(run.id), "reason": reason},
            session=session,
            product_id=run.product_id,
            user_id=run.user_id,
        )
        return refunded.event

    # -- sweep -------------------------------------------------------------------------------

    async def sweep(self) -> int:
        """Fail every ``running`` run past its lease as ``run_timeout`` and refund it.

        Takes only run locks. A run is failed once; a worker that was merely slow finds its run
        already terminal and stops.

        Returns:
            How many runs were swept.
        """
        events: list[BalanceEvent | None] = []
        async with self._db.session() as session:
            swept = list(await StudioRunRepository(session).sweep_expired(error_code=ERR_TIMEOUT))
            for run in swept:
                events.append(await self._refund(session, run, ERR_TIMEOUT))
        for event in events:
            await self._events.balance(event)
        for run in swept:
            logger.warning("studio.run.timed_out", run_id=str(run.id), kind=run.kind)
            await self._events.run_updated(run)
        return len(swept)

    # -- logging -----------------------------------------------------------------------------

    @staticmethod
    def _log_finished(
        claimed: ClaimedRun,
        status: str,
        error_code: str | None,
        meter: _MeteredLlm,
        started: float,
    ) -> None:
        logger.info(
            "studio.run.finished",
            run_id=str(claimed.run_id),
            track_id=str(claimed.track_id),
            kind=claimed.kind.value,
            status=status,
            error_code=error_code,
            model_key=claimed.model_key,
            llm_calls=meter.calls,
            prompt_tokens=meter.prompt_tokens,
            completion_tokens=meter.completion_tokens,
            duration_ms=int((time.monotonic() - started) * 1000),
        )
