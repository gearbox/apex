"""Accepted-task receipts: store retries/shielding, immediate cancel, compensators, wiring."""

from __future__ import annotations

import ast
import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest
from sqlalchemy.exc import OperationalError
from structlog.testing import capture_logs

from src.api.schemas.ops_events import OpsEventType
from src.api.services.grok import (
    GrokAPIError,
    GrokDeferredExpiredError,
    GrokDeferredTerminalError,
    GrokModerationError,
    GrokRateLimitError,
    GrokVideoResult,
)
from src.api.services.grok.job_service import GrokJobService
from src.api.services.grok.orphan_compensator import GrokOrphanCompensator
from src.api.services.model_studio import (
    ModelStudioAPIError,
    ModelStudioFailureClassifier,
    ModelStudioTask,
    ModelStudioTaskStatus,
    ModelStudioTransientError,
)
from src.api.services.model_studio.orphan_compensator import ModelStudioOrphanCompensator
from src.api.services.model_studio.types import ModelStudioCancelResult
from src.api.services.model_studio.wan_job_service import WanJobService
from src.api.services.video_jobs import receipts as receipts_module
from src.api.services.video_jobs.protocol import (
    OrphanCancelOutcome,
    OrphanObservation,
    OrphanTaskState,
)
from src.api.services.video_jobs.receipts import (
    AcceptedTask,
    AcceptedTaskReceipts,
    cancel_orphan_immediately,
    record_acceptance,
)
from src.core.enums import AcceptedTaskReceiptStatus, Provider
from tests.video_receipt_support import stub_compensator, stub_receipts

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

pytestmark = pytest.mark.unit

_classifier = ModelStudioFailureClassifier()


def _task() -> AcceptedTask:
    return AcceptedTask(
        provider=Provider.ALIBABA,
        external_task_id="task-1",
        job_id=uuid4(),
        user_id=uuid4(),
        product_id="vex",
        model="wan3.0-video",
        token_cost=40,
    )


class _FakeReceiptDb:
    """``DatabaseManager`` stand-in whose sessions record inserts or fail."""

    def __init__(self, *, fail_times: int = 0, gate: asyncio.Event | None = None) -> None:
        self.fail_times = fail_times
        self.gate = gate
        self.sessions = 0
        self.committed = 0

    @asynccontextmanager
    async def session(self) -> AsyncIterator[Any]:
        self.sessions += 1
        if self.gate is not None:
            await self.gate.wait()
        if self.sessions <= self.fail_times:
            raise OperationalError("INSERT", {}, Exception("connection reset"))
        session = MagicMock()
        session.execute = AsyncMock()
        yield session
        self.committed += 1


@pytest.fixture(autouse=True)
def _no_backoff(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(receipts_module, "_BACKOFF_SECONDS", (0.0, 0.0, 0.0))


def _store(db: _FakeReceiptDb, *, attempts: int = 3) -> tuple[AcceptedTaskReceipts, MagicMock]:
    ops = MagicMock()
    ops.publish = AsyncMock()
    return AcceptedTaskReceipts(db, ops, record_attempts=attempts), ops  # type: ignore[arg-type]


# ---- record -----------------------------------------------------------------------------


async def test_record_retries_then_succeeds() -> None:
    db = _FakeReceiptDb(fail_times=2)
    store, ops = _store(db)

    assert await store.record(_task()) is True

    assert db.sessions == 3
    ops.publish.assert_not_awaited()


async def test_record_gives_up_without_raising_and_alerts_with_the_task_id() -> None:
    db = _FakeReceiptDb(fail_times=99)
    store, ops = _store(db, attempts=3)
    task = _task()

    with capture_logs() as logs:
        recorded = await store.record(task)

    assert recorded is False
    assert db.sessions == 3
    errors = [log for log in logs if log["event"] == "video_receipt.unrecorded"]
    assert len(errors) == 1
    assert errors[0]["external_task_id"] == "task-1"
    assert errors[0]["log_level"] == "error"
    kwargs = ops.publish.await_args.kwargs
    assert kwargs["event_type"] is OpsEventType.VIDEO_TASK_RECEIPT_UNRECORDED
    assert kwargs["payload"].external_task_id == "task-1"


async def test_record_finishes_the_write_when_the_caller_is_cancelled() -> None:
    gate = asyncio.Event()
    db = _FakeReceiptDb(gate=gate)
    store, _ = _store(db)

    caller = asyncio.ensure_future(store.record(_task()))
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    caller.cancel()
    gate.set()
    with pytest.raises(asyncio.CancelledError):
        await caller

    # The shielded write ran to its commit despite the cancellation.
    for _ in range(5):
        await asyncio.sleep(0)
    assert db.committed == 1


# ---- record_acceptance ------------------------------------------------------------------


async def test_record_acceptance_claims_on_the_request_session_after_recording() -> None:
    receipts = stub_receipts()
    session = MagicMock()
    task = _task()

    assert await record_acceptance(receipts, session, task) is True

    receipts.record.assert_awaited_once_with(task)
    receipts.claim_for_request.assert_awaited_once_with(
        session, provider=Provider.ALIBABA, external_task_id="task-1"
    )


async def test_record_acceptance_issues_no_delete_when_the_receipt_was_not_recorded() -> None:
    receipts = stub_receipts(recorded=False)

    assert await record_acceptance(receipts, MagicMock(), _task()) is False

    receipts.claim_for_request.assert_not_awaited()


# ---- immediate cancel -------------------------------------------------------------------


async def test_immediate_cancel_rolls_back_before_resolving() -> None:
    order: list[str] = []
    receipts = stub_receipts()
    receipts.resolve.side_effect = lambda **_: order.append("resolve")
    session = MagicMock()
    session.rollback = AsyncMock(side_effect=lambda: order.append("rollback"))

    await cancel_orphan_immediately(
        receipts=receipts, compensator=stub_compensator(), session=session, task=_task()
    )

    assert order == ["rollback", "resolve"]
    assert receipts.resolve.await_args.kwargs["status"] is AcceptedTaskReceiptStatus.CANCELLED


@pytest.mark.parametrize(
    "outcome", [OrphanCancelOutcome.NOT_CANCELLABLE, OrphanCancelOutcome.UNSUPPORTED]
)
async def test_immediate_cancel_leaves_a_refused_task_pending(
    outcome: OrphanCancelOutcome,
) -> None:
    receipts = stub_receipts()

    await cancel_orphan_immediately(
        receipts=receipts,
        compensator=stub_compensator(outcome),
        session=MagicMock(rollback=AsyncMock()),
        task=_task(),
    )

    receipts.resolve.assert_not_awaited()


async def test_immediate_cancel_times_out_without_raising(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(receipts_module, "_IMMEDIATE_CANCEL_TIMEOUT_SECONDS", 0.01)
    receipts = stub_receipts()
    compensator = stub_compensator()

    async def hang(_: str) -> OrphanCancelOutcome:
        await asyncio.Event().wait()
        return OrphanCancelOutcome.CANCELLED

    compensator.cancel = hang

    await cancel_orphan_immediately(
        receipts=receipts,
        compensator=compensator,
        session=MagicMock(rollback=AsyncMock()),
        task=_task(),
    )

    receipts.resolve.assert_not_awaited()


# ---- compensators -----------------------------------------------------------------------


def _ms_task(status: ModelStudioTaskStatus) -> ModelStudioTask:
    return ModelStudioTask(task_id="task-1", task_status=status)


@pytest.mark.parametrize(
    ("status", "state"),
    [
        (ModelStudioTaskStatus.PENDING, OrphanTaskState.PENDING),
        (ModelStudioTaskStatus.CANCELED, OrphanTaskState.CANCELLED),
        (ModelStudioTaskStatus.RUNNING, OrphanTaskState.OTHER),
        (ModelStudioTaskStatus.SUCCEEDED, OrphanTaskState.OTHER),
        (ModelStudioTaskStatus.FAILED, OrphanTaskState.OTHER),
        (ModelStudioTaskStatus.UNKNOWN, OrphanTaskState.GONE),
    ],
)
async def test_model_studio_observe_maps_task_status(
    status: ModelStudioTaskStatus, state: OrphanTaskState
) -> None:
    client = MagicMock()
    client.get_task = AsyncMock(return_value=_ms_task(status))

    observed = await ModelStudioOrphanCompensator(client).observe("task-1")

    assert observed.state is state
    expected_status = None if state is OrphanTaskState.GONE else status.value
    assert observed.upstream_status == expected_status


async def test_model_studio_observe_404_is_gone_and_transient_propagates() -> None:
    client = MagicMock()
    client.get_task = AsyncMock(
        side_effect=ModelStudioAPIError(
            _classifier.classify(code=None, http_status=404, provider_request_accepted=True)
        )
    )
    compensator = ModelStudioOrphanCompensator(client)

    assert (await compensator.observe("task-1")).state is OrphanTaskState.GONE

    client.get_task = AsyncMock(
        side_effect=ModelStudioTransientError(_classifier.timeout(provider_request_accepted=True))
    )
    with pytest.raises(ModelStudioTransientError):
        await compensator.observe("task-1")


@pytest.mark.parametrize(
    ("result", "outcome"),
    [
        (ModelStudioCancelResult.CANCELLED, OrphanCancelOutcome.CANCELLED),
        (ModelStudioCancelResult.NOT_CANCELLABLE, OrphanCancelOutcome.NOT_CANCELLABLE),
    ],
)
async def test_model_studio_cancel_maps_the_client_result(
    result: ModelStudioCancelResult, outcome: OrphanCancelOutcome
) -> None:
    client = MagicMock()
    client.cancel_task = AsyncMock(return_value=result)

    assert await ModelStudioOrphanCompensator(client).cancel("task-1") is outcome


async def test_grok_cancel_is_unsupported() -> None:
    assert await GrokOrphanCompensator(MagicMock()).cancel("req-1") is (
        OrphanCancelOutcome.UNSUPPORTED
    )


@pytest.mark.parametrize(
    ("result", "state", "upstream"),
    [
        (None, OrphanTaskState.PENDING, "pending"),
        (GrokVideoResult(url="https://x/v.mp4"), OrphanTaskState.OTHER, "done"),
    ],
)
async def test_grok_observe_maps_results_without_fetching_the_video(
    result: GrokVideoResult | None, state: OrphanTaskState, upstream: str
) -> None:
    client = MagicMock()
    client.get_video_result = AsyncMock(return_value=result)

    observed = await GrokOrphanCompensator(client).observe("req-1")

    assert observed == OrphanObservation(state=state, upstream_status=upstream)
    client.get_video_result.assert_awaited_once_with("req-1")


@pytest.mark.parametrize(
    ("error", "state"),
    [
        (GrokDeferredExpiredError("expired"), OrphanTaskState.GONE),
        (GrokDeferredTerminalError("failed"), OrphanTaskState.OTHER),
        (GrokModerationError("blocked"), OrphanTaskState.OTHER),
    ],
)
async def test_grok_observe_maps_terminal_errors(error: Exception, state: OrphanTaskState) -> None:
    client = MagicMock()
    client.get_video_result = AsyncMock(side_effect=error)

    assert (await GrokOrphanCompensator(client).observe("req-1")).state is state


@pytest.mark.parametrize("error", [GrokRateLimitError("slow"), GrokAPIError("5xx")])
async def test_grok_observe_lets_transient_errors_propagate(error: Exception) -> None:
    client = MagicMock()
    client.get_video_result = AsyncMock(side_effect=error)

    with pytest.raises(type(error)):
        await GrokOrphanCompensator(client).observe("req-1")
    assert isinstance(error, GrokOrphanCompensator.transient_errors)


# ---- wiring invariants ------------------------------------------------------------------


def test_receipt_reconciler_lease_names_are_pinned_and_distinct() -> None:
    assert WanJobService.RECEIPT_RECONCILER_NAME == "alibaba_video_receipts"
    assert GrokJobService.RECEIPT_RECONCILER_NAME == "grok_video_receipts"
    names = {
        WanJobService.RECEIPT_RECONCILER_NAME,
        GrokJobService.RECEIPT_RECONCILER_NAME,
        WanJobService.VIDEO_POLLER_NAME,
        GrokJobService.VIDEO_POLLER_NAME,
    }
    assert len(names) == 4


def test_the_reconciler_never_touches_billing() -> None:
    source = Path("src/api/services/video_jobs/reconciler.py").read_text()
    imported = [
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Import | ast.ImportFrom)
    ]
    modules = {
        (node.module or "") if isinstance(node, ast.ImportFrom) else alias.name
        for node in imported
        for alias in (node.names if isinstance(node, ast.Import) else [SimpleNamespace(name="")])
    }
    assert not any("billing" in module for module in modules)
    assert "TokenTransaction" not in source
    assert "BillingService" not in source
