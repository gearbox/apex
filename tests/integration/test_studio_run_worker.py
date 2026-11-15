"""``StudioRunWorker``: ticks that claim, launch and sweep - driven by ``run_once``, never by sleeps.

Every wait here is on an ``asyncio.Event`` the test controls.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pytest

from src.workers.studio_run import StudioRunTickError, StudioRunWorker
from tests.studio_llm import FakeLlm, make_draft

if TYPE_CHECKING:
    from collections.abc import Callable
    from uuid import UUID

    from tests.studio_runs_support import Actor, RunWorld

TIMEOUT = 10


def _worker(
    world: RunWorld, llm: FakeLlm, *, max_concurrent: int = 4, drain: float = 30.0
) -> StudioRunWorker:
    return StudioRunWorker(
        executor=world.executor(llm),
        interval_seconds=1,
        max_concurrent=max_concurrent,
        drain_timeout_seconds=drain,
        redis_client_factory=MagicMock,
    )


async def _statuses(world: RunWorld, runs: list[UUID]) -> list[str]:
    return sorted([(await world.run(r)).status for r in runs])


async def _submit_many(world: RunWorld, count: int) -> tuple[Actor, list[UUID]]:
    actor = await world.add_actor(balance=20)
    runs = []
    for index in range(count):
        track_id = await world.new_track(actor, f"T{index}")
        runs.append((await world.submit(actor, track_id)).run.id)
    return actor, runs


async def test_the_lease_name_is_stable(run_world: RunWorld) -> None:
    assert _worker(run_world, FakeLlm()).name == "studio_run"


async def test_a_tick_starts_the_runs_and_returns_before_they_finish(
    make_run_world: Callable[..., RunWorld],
) -> None:
    world = make_run_world(studio_run_max_active_per_user=5)
    _, runs = await _submit_many(world, 3)
    gate = asyncio.Event()
    entered = asyncio.Event()
    calls = 0

    async def block() -> None:
        nonlocal calls
        calls += 1
        entered.set()
        await gate.wait()

    worker = _worker(world, FakeLlm(make_draft(), make_draft(), make_draft(), before_call=block))

    await asyncio.wait_for(worker.run_once(), TIMEOUT)  # returns although the model is "thinking"
    await asyncio.wait_for(entered.wait(), TIMEOUT)

    assert worker.in_flight == 3
    assert {(await world.run(r)).status for r in runs} == {"running"}

    gate.set()
    await asyncio.wait_for(worker.stop(), TIMEOUT)  # drains the in-flight runs
    assert worker.in_flight == 0
    assert calls == 3
    assert {(await world.run(r)).status for r in runs} == {"succeeded"}


async def test_the_tick_never_claims_more_than_the_free_slots(
    make_run_world: Callable[..., RunWorld],
) -> None:
    world = make_run_world(studio_run_max_active_per_user=5)
    _, runs = await _submit_many(world, 3)
    gate = asyncio.Event()

    async def block() -> None:
        await gate.wait()

    worker = _worker(
        world,
        FakeLlm(make_draft(), make_draft(), make_draft(), before_call=block),
        max_concurrent=2,
    )

    await worker.run_once()
    assert worker.in_flight == 2
    assert await _statuses(world, runs) == ["queued", "running", "running"]

    await worker.run_once()  # no free slot: nothing more is claimed
    assert worker.in_flight == 2
    assert await _statuses(world, runs) == ["queued", "running", "running"]

    gate.set()
    await asyncio.wait_for(worker.stop(), TIMEOUT)
    assert await _statuses(world, runs) == ["queued", "succeeded", "succeeded"]

    await worker.run_once()  # a slot is free again: the third run is picked up
    await asyncio.wait_for(worker.stop(), TIMEOUT)
    assert {(await world.run(r)).status for r in runs} == {"succeeded"}


async def test_a_tick_sweeps_before_it_claims(run_world: RunWorld) -> None:
    actor = await run_world.add_actor()
    track_id = await run_world.new_track(actor)
    write = await run_world.submit(actor, track_id)
    await run_world.executor(FakeLlm()).claim(1)  # a worker that then died
    await run_world.expire_lease(write.run.id)

    worker = _worker(run_world, FakeLlm())
    await worker.run_once()

    run = await run_world.run(write.run.id)
    assert (run.status, run.error_code) == ("failed", "run_timeout")
    assert await run_world.refunds_and_debits(actor) == (1, 1)
    assert worker.in_flight == 0


async def test_runs_left_after_the_drain_budget_are_swept_never_re_executed(
    run_world: RunWorld,
) -> None:
    actor = await run_world.add_actor()
    track_id = await run_world.new_track(actor)
    write = await run_world.submit(actor, track_id)
    started = asyncio.Event()

    async def hang() -> None:
        started.set()
        await asyncio.sleep(60)

    worker = _worker(run_world, FakeLlm(make_draft(), before_call=hang), drain=0.1)
    await worker.run_once()
    await asyncio.wait_for(started.wait(), TIMEOUT)

    await asyncio.wait_for(
        worker.stop(), TIMEOUT
    )  # the drain budget runs out: the task is cancelled
    assert worker.in_flight == 0
    run = await run_world.run(write.run.id)
    assert run.status == "running"  # abandoned, not failed by the dying process
    assert await run_world.versions(track_id) == []

    # The next process's sweep finishes it - once - and it is not claimed again.
    await run_world.expire_lease(write.run.id)
    successor = _worker(run_world, FakeLlm())
    await successor.run_once()
    run = await run_world.run(write.run.id)
    assert (run.status, run.error_code) == ("failed", "run_timeout")
    await successor.run_once()
    assert successor.in_flight == 0
    assert await run_world.refunds_and_debits(actor) == (1, 1)


async def test_a_failing_run_does_not_stop_the_worker(run_world: RunWorld) -> None:
    actor = await run_world.add_actor(balance=20)
    first = await run_world.new_track(actor, "A")
    second = await run_world.new_track(actor, "B")
    bad = await run_world.submit(actor, first)
    good = await run_world.submit(actor, second)

    worker = _worker(run_world, FakeLlm(RuntimeError("boom"), make_draft()), max_concurrent=1)
    await worker.run_once()
    await asyncio.wait_for(worker.stop(), TIMEOUT)
    await worker.run_once()
    await asyncio.wait_for(worker.stop(), TIMEOUT)

    statuses = {(await run_world.run(r.run.id)).status for r in (bad, good)}
    assert statuses == {"failed", "succeeded"}


async def test_a_failing_tick_surfaces_without_the_causes_message() -> None:
    class Boom:
        async def sweep(self) -> int:
            raise RuntimeError("statement parameters: SECRET-PERSONA")

    worker = StudioRunWorker(
        executor=Boom(),  # type: ignore[arg-type]
        interval_seconds=1,
        max_concurrent=1,
        redis_client_factory=MagicMock,
    )
    with pytest.raises(StudioRunTickError) as caught:
        await worker.run_once()
    assert "SECRET" not in str(caught.value)
    assert str(caught.value) == "RuntimeError"
    assert caught.value.__cause__ is None
    assert caught.value.__suppress_context__ is True
