"""Studio planner runs through the services, on committed rows and a real pool.

Covers the run lifecycle invariants (terminal-once, one active run per track, idempotency, atomic
reserve + insert, refunds, cancel, the saved version, events). The executor opens its own short
sessions, so nothing here can use the rolled-back ``db_session`` fixture.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import pytest
from sqlalchemy import select, text

from src.api.schemas.events import EventType
from src.api.services.billing_errors import InsufficientBalanceError
from src.core.enums import StudioDocumentOrigin, StudioRunKind, StudioRunStage
from src.core.uid import new_id
from src.db.models.studio import StudioRun
from src.db.repositories.studio import StudioRunRepository
from src.llm.port import (
    LlmOutputInvalid,
    LlmOutputTruncated,
    LlmRejected,
    LlmUnavailable,
)
from src.studio.errors import (
    IdempotencyConflict,
    IdempotencyKeyRequired,
    MemoriesNotAvailable,
    RunInProgress,
    RunLimitReached,
    RunNotCancellable,
    RunNotFound,
    RunRateLimited,
    StudioDocumentInvalid,
    StudioNotFound,
    VersionLimitReached,
)
from src.studio.planner.drafts import DraftRefusal
from src.studio.run_requests import PlanRunRequest
from src.studio.services.version import DocumentVersionService
from tests.llm_support import FakeLlm
from tests.studio_llm import make_draft, refusal_draft
from tests.studio_runs_support import PRODUCT, RunWorld
from tests.studio_support import make_plan

if TYPE_CHECKING:
    from collections.abc import Callable
    from uuid import UUID


async def _submit_and_drive(world: RunWorld, actor, track_id: UUID, llm: FakeLlm) -> StudioRun:  # type: ignore[no-untyped-def]
    write = await world.submit(actor, track_id)
    await world.drive(llm)
    return await world.run(write.run.id)


# ---------------------------------------------------------------------------
# Submit
# ---------------------------------------------------------------------------


class TestSubmit:
    async def test_a_submit_queues_the_run_and_reserves_its_price(
        self, run_world: RunWorld
    ) -> None:
        actor = await run_world.add_actor(balance=5)
        track_id = await run_world.new_track(actor)

        write = await run_world.submit(actor, track_id, key="k1")

        run = await run_world.run(write.run.id)
        assert write.changed is True
        assert (run.status, run.stage, run.revision) == ("queued", None, 1)
        assert run.kind == StudioRunKind.PLAN.value
        assert (run.price_tokens, run.model_key) == (1, "grok")
        assert run.billing_account_id == actor.account_id
        assert run.idempotency_key == "k1"
        assert run.request["brief"] == "a song about rain"
        assert run.request["kind"] == "plan"
        ledger = await run_world.ledger(actor)
        debit = ledger[-1]
        assert (debit.transaction_type, debit.amount, debit.job_id) == ("debit", -1, run.id)
        assert debit.metadata_["studio_run_id"] == str(run.id)
        assert await run_world.balance(actor) == 4

    async def test_the_events_are_published_after_commit(self, run_world: RunWorld) -> None:
        actor = await run_world.add_actor()
        track_id = await run_world.new_track(actor)

        async def check(_user: UUID, _type: EventType, payload: object) -> None:
            # By the time anything is published the row must already be visible to others.
            assert (await run_world.run(payload.run_id)).revision == payload.revision  # type: ignore[attr-defined]

        run_world.bus.on_publish = check
        write = await run_world.submit(actor, track_id)

        (event,) = run_world.bus.run_events
        assert (event.run_id, event.track_id) == (write.run.id, track_id)
        assert (event.kind, event.status, event.stage, event.revision) == (
            "plan",
            "queued",
            None,
            1,
        )
        assert len(run_world.bus.balance_events) == 1
        assert run_world.bus.balance_events[0].delta == -1

    async def test_memories_are_not_available_yet(self, run_world: RunWorld) -> None:
        from uuid import uuid4

        actor = await run_world.add_actor()
        track_id = await run_world.new_track(actor)
        request = PlanRunRequest(brief="x", memory_ids=(uuid4(),))
        with pytest.raises(MemoriesNotAvailable):
            await run_world.submit(actor, track_id, request=request)
        assert await run_world.count_runs(actor) == 0

    @pytest.mark.parametrize("key", ["", "   ", "k" * 129])
    async def test_a_usable_idempotency_key_is_required(
        self, run_world: RunWorld, key: str
    ) -> None:
        actor = await run_world.add_actor()
        track_id = await run_world.new_track(actor)
        with pytest.raises(IdempotencyKeyRequired):
            await run_world.submit(actor, track_id, key=key)
        async with run_world.factory() as session:
            with pytest.raises(IdempotencyKeyRequired):
                await run_world.service(session).submit(
                    track_id,
                    PlanRunRequest(brief="x"),
                    idempotency_key=None,
                    user_id=actor.user_id,
                    product_id=PRODUCT,
                )

    async def test_a_blank_brief_is_refused(self, run_world: RunWorld) -> None:
        actor = await run_world.add_actor()
        track_id = await run_world.new_track(actor)
        with pytest.raises(StudioDocumentInvalid):
            await run_world.submit(actor, track_id, brief="   \n  ")

    async def test_another_users_track_is_not_found(self, run_world: RunWorld) -> None:
        owner = await run_world.add_actor()
        intruder = await run_world.add_actor()
        track_id = await run_world.new_track(owner)
        with pytest.raises(StudioNotFound):
            await run_world.submit(intruder, track_id)
        assert await run_world.balance(intruder) == 10

    async def test_the_brief_is_stored_normalized(self, run_world: RunWorld) -> None:
        actor = await run_world.add_actor()
        track_id = await run_world.new_track(actor)
        write = await run_world.submit(actor, track_id, brief="  Café  ")
        assert (await run_world.run(write.run.id)).request["brief"] == "Café"


# ---------------------------------------------------------------------------
# Idempotency (invariant 3)
# ---------------------------------------------------------------------------


class TestIdempotency:
    async def test_the_same_key_and_body_is_the_same_run_with_one_reservation(
        self, run_world: RunWorld
    ) -> None:
        actor = await run_world.add_actor()
        track_id = await run_world.new_track(actor)

        first = await run_world.submit(actor, track_id, key="same")
        second = await run_world.submit(actor, track_id, key="same")

        assert (first.changed, second.changed) == (True, False)
        assert second.run.id == first.run.id
        assert await run_world.count_runs(actor) == 1
        assert await run_world.refunds_and_debits(actor) == (1, 0)
        assert len(run_world.bus.run_events) == 1  # a replay publishes nothing
        assert await run_world.balance(actor) == 9

    async def test_a_replay_survives_the_run_having_moved_on(self, run_world: RunWorld) -> None:
        actor = await run_world.add_actor()
        track_id = await run_world.new_track(actor)
        first = await run_world.submit(actor, track_id, key="same")
        await run_world.drive(FakeLlm(make_draft()))

        replay = await run_world.submit(actor, track_id, key="same")

        assert replay.changed is False
        assert replay.run.id == first.run.id
        assert replay.run.status == "succeeded"

    async def test_the_same_key_with_another_body_conflicts(self, run_world: RunWorld) -> None:
        actor = await run_world.add_actor()
        track_id = await run_world.new_track(actor)
        await run_world.submit(actor, track_id, key="same", brief="one")
        with pytest.raises(IdempotencyConflict):
            await run_world.submit(actor, track_id, key="same", brief="two")
        assert await run_world.count_runs(actor) == 1
        assert await run_world.refunds_and_debits(actor) == (1, 0)

    async def test_the_same_key_for_another_track_conflicts(self, run_world: RunWorld) -> None:
        actor = await run_world.add_actor()
        first = await run_world.new_track(actor, "A")
        second = await run_world.new_track(actor, "B")
        await run_world.submit(actor, first, key="same")
        with pytest.raises(IdempotencyConflict):
            await run_world.submit(actor, second, key="same")
        assert await run_world.count_runs(actor) == 1

    async def test_keys_are_per_user(self, run_world: RunWorld) -> None:
        alice = await run_world.add_actor()
        bob = await run_world.add_actor()
        a = await run_world.submit(alice, await run_world.new_track(alice), key="shared")
        b = await run_world.submit(bob, await run_world.new_track(bob), key="shared")
        assert a.run.id != b.run.id

    async def test_a_concurrent_duplicate_submit_creates_one_run(self, run_world: RunWorld) -> None:
        actor = await run_world.add_actor()
        track_id = await run_world.new_track(actor)

        writes = await asyncio.gather(
            *(run_world.submit(actor, track_id, key="race") for _ in range(4))
        )

        assert {w.run.id for w in writes} == {writes[0].run.id}
        assert sum(w.changed for w in writes) == 1
        assert await run_world.count_runs(actor) == 1
        assert await run_world.refunds_and_debits(actor) == (1, 0)


# ---------------------------------------------------------------------------
# Limits (invariant 2 and P2a-19)
# ---------------------------------------------------------------------------


class TestLimits:
    async def test_one_active_run_per_track(self, run_world: RunWorld) -> None:
        actor = await run_world.add_actor()
        track_id = await run_world.new_track(actor)
        await run_world.submit(actor, track_id)
        with pytest.raises(RunInProgress):
            await run_world.submit(actor, track_id)
        assert await run_world.count_runs(actor) == 1
        assert await run_world.refunds_and_debits(actor) == (
            1,
            0,
        )  # the refused one reserved nothing

    async def test_concurrent_submits_on_one_track_admit_exactly_one(
        self, run_world: RunWorld
    ) -> None:
        actor = await run_world.add_actor(balance=20)
        track_id = await run_world.new_track(actor)

        results = await asyncio.gather(
            *(run_world.submit(actor, track_id) for _ in range(5)), return_exceptions=True
        )

        assert sum(not isinstance(r, Exception) for r in results) == 1
        assert all(isinstance(r, RunInProgress) for r in results if isinstance(r, Exception))
        assert await run_world.count_runs(actor) == 1
        assert await run_world.refunds_and_debits(actor) == (1, 0)

    async def test_a_new_run_is_possible_once_the_active_one_finished(
        self, run_world: RunWorld
    ) -> None:
        actor = await run_world.add_actor()
        track_id = await run_world.new_track(actor)
        await run_world.submit(actor, track_id)
        await run_world.drive(FakeLlm(make_draft()))
        again = await run_world.submit(actor, track_id)
        assert again.changed is True

    async def test_the_per_user_active_limit(self, make_run_world: Callable[..., RunWorld]) -> None:
        world = make_run_world(studio_run_max_active_per_user=2)
        actor = await world.add_actor(balance=20)
        tracks = [await world.new_track(actor, f"T{i}") for i in range(3)]
        await world.submit(actor, tracks[0])
        await world.submit(actor, tracks[1])
        with pytest.raises(RunLimitReached):
            await world.submit(actor, tracks[2])
        assert await world.count_runs(actor) == 2
        assert await world.refunds_and_debits(actor) == (2, 0)

    async def test_the_hourly_limit_counts_runs_in_the_database(
        self, make_run_world: Callable[..., RunWorld]
    ) -> None:
        world = make_run_world(studio_runs_per_user_per_hour=2)
        actor = await world.add_actor(balance=20)
        tracks = [await world.new_track(actor, f"T{i}") for i in range(3)]
        for track_id in tracks[:2]:
            await world.submit(actor, track_id)
        await world.drive(FakeLlm(make_draft(), make_draft()))  # both finished: not "active"
        with pytest.raises(RunRateLimited):
            await world.submit(actor, tracks[2])
        assert await world.count_runs(actor) == 2

    async def test_old_runs_do_not_count_towards_the_hourly_limit(
        self, make_run_world: Callable[..., RunWorld]
    ) -> None:
        world = make_run_world(studio_runs_per_user_per_hour=1)
        actor = await world.add_actor(balance=20)
        track_id = await world.new_track(actor)
        first = await world.submit(actor, track_id)
        await world.drive(FakeLlm(make_draft()))
        async with world.factory() as session, session.begin():
            await session.execute(
                text(
                    "UPDATE studio_runs SET created_at = now() - interval '2 hours' WHERE id = :id"
                ),
                {"id": first.run.id},
            )
        assert (await world.submit(actor, track_id)).changed is True


# ---------------------------------------------------------------------------
# Billing (invariants 4 and 5)
# ---------------------------------------------------------------------------


class TestBilling:
    async def test_insufficient_balance_leaves_no_run_and_no_ledger_row(
        self, run_world: RunWorld
    ) -> None:
        actor = await run_world.add_actor(balance=0)
        track_id = await run_world.new_track(actor)
        with pytest.raises(InsufficientBalanceError):
            await run_world.submit(actor, track_id)
        assert await run_world.count_runs(actor) == 0
        assert await run_world.ledger(actor) == []

    async def test_insufficient_balance_does_not_poison_the_session(
        self, run_world: RunWorld
    ) -> None:
        """Insert + reserve share a savepoint: after the refusal the session still works."""
        actor = await run_world.add_actor(balance=0)
        track_id = await run_world.new_track(actor)
        async with run_world.factory() as session:
            service = run_world.service(session)
            with pytest.raises(InsufficientBalanceError):
                await service.submit(
                    track_id,
                    PlanRunRequest(brief="x"),
                    idempotency_key="k",
                    user_id=actor.user_id,
                    product_id=PRODUCT,
                )
            # no run row survives *inside* the same transaction either
            assert (await session.execute(select(StudioRun))).first() is None

    async def test_a_free_run_writes_no_ledger_rows_at_all(
        self, make_run_world: Callable[..., RunWorld]
    ) -> None:
        world = make_run_world(studio_run_prices={"plan": 0, "lyrics": 0, "revise_lyrics": 0})
        actor = await world.add_actor(balance=0)  # not even an opening credit
        first = await world.new_track(actor, "Done")
        second = await world.new_track(actor, "Cancelled")
        third = await world.new_track(actor, "Failed")
        ok = await world.submit(actor, first)
        cancelled = await world.submit(actor, second)
        failing = await world.submit(actor, third)
        async with world.factory() as session:
            await world.service(session).cancel(
                cancelled.run.id, user_id=actor.user_id, product_id=PRODUCT
            )
            await session.commit()
        await world.drive(FakeLlm(make_draft(), LlmUnavailable("down"), LlmUnavailable("down")))

        assert (await world.run(ok.run.id)).price_tokens == 0
        assert (await world.run(ok.run.id)).billing_account_id is None
        statuses = {(await world.run(w.run.id)).status for w in (ok, cancelled, failing)}
        assert statuses == {"succeeded", "cancelled", "failed"}
        assert await world.ledger(actor) == []

    async def test_a_succeeded_run_keeps_its_charge(self, run_world: RunWorld) -> None:
        actor = await run_world.add_actor()
        track_id = await run_world.new_track(actor)
        await _submit_and_drive(run_world, actor, track_id, FakeLlm(make_draft()))
        assert await run_world.refunds_and_debits(actor) == (1, 0)
        assert await run_world.balance(actor) == 9

    @pytest.mark.parametrize(
        ("script", "code"),
        [
            (LlmUnavailable("x"), "llm_unavailable"),
            (LlmRejected(400), "llm_rejected"),
            (LlmOutputTruncated("x"), "llm_output_truncated"),
            (LlmOutputInvalid("x"), "llm_output_invalid"),
        ],
    )
    async def test_a_failed_run_is_refunded_exactly_once(
        self, run_world: RunWorld, script: Exception, code: str
    ) -> None:
        actor = await run_world.add_actor()
        track_id = await run_world.new_track(actor)
        # ``LlmOutputInvalid`` is re-asked until the call budget is spent
        llm = FakeLlm(script, script, script)
        run = await _submit_and_drive(run_world, actor, track_id, llm)

        assert (run.status, run.error_code) == ("failed", code)
        assert run.finished_at is not None
        assert await run_world.refunds_and_debits(actor) == (1, 1)
        assert await run_world.balance(actor) == 10
        refund = (await run_world.ledger(actor))[-1]
        assert (refund.transaction_type, refund.amount, refund.job_id) == ("refund", 1, run.id)
        assert await run_world.versions(track_id) == []

    @pytest.mark.parametrize(
        ("reason", "code"),
        [
            (DraftRefusal.CONTENT_POLICY, "content_policy_refused"),
            (DraftRefusal.BRIEF_UNUSABLE, "brief_unusable"),
        ],
    )
    async def test_a_refusal_fails_refunded_with_no_version(
        self, run_world: RunWorld, reason: DraftRefusal, code: str
    ) -> None:
        actor = await run_world.add_actor()
        track_id = await run_world.new_track(actor)
        run = await _submit_and_drive(run_world, actor, track_id, FakeLlm(refusal_draft(reason)))

        assert (run.status, run.error_code) == ("failed", code)
        assert run.llm_calls == 1
        assert await run_world.refunds_and_debits(actor) == (1, 1)
        assert await run_world.versions(track_id) == []

    async def test_balance_events_follow_the_ledger(self, run_world: RunWorld) -> None:
        actor = await run_world.add_actor()
        track_id = await run_world.new_track(actor)
        await _submit_and_drive(run_world, actor, track_id, FakeLlm(LlmUnavailable("down")))
        assert [e.delta for e in run_world.bus.balance_events] == [-1, 1]


# ---------------------------------------------------------------------------
# Cancel (invariant 6)
# ---------------------------------------------------------------------------


class TestCancel:
    async def test_a_queued_run_is_cancelled_and_refunded(self, run_world: RunWorld) -> None:
        actor = await run_world.add_actor()
        track_id = await run_world.new_track(actor)
        write = await run_world.submit(actor, track_id)

        async with run_world.factory() as session:
            service = run_world.service(session)
            cancelled = await service.cancel(
                write.run.id, user_id=actor.user_id, product_id=PRODUCT
            )
            await session.commit()
        await service.publish(cancelled)

        run = await run_world.run(write.run.id)
        assert (run.status, run.stage, run.revision) == ("cancelled", None, 2)
        assert run.finished_at is not None
        assert await run_world.refunds_and_debits(actor) == (1, 1)
        assert await run_world.balance(actor) == 10
        assert [e.status for e in run_world.bus.run_events] == ["queued", "cancelled"]
        assert [e.delta for e in run_world.bus.balance_events] == [-1, 1]

    async def test_cancel_frees_the_track(self, run_world: RunWorld) -> None:
        actor = await run_world.add_actor()
        track_id = await run_world.new_track(actor)
        write = await run_world.submit(actor, track_id)
        async with run_world.factory() as session:
            await run_world.service(session).cancel(
                write.run.id, user_id=actor.user_id, product_id=PRODUCT
            )
            await session.commit()
        assert (await run_world.submit(actor, track_id)).changed is True

    async def test_a_cancelled_run_cannot_be_cancelled_again(self, run_world: RunWorld) -> None:
        actor = await run_world.add_actor()
        track_id = await run_world.new_track(actor)
        write = await run_world.submit(actor, track_id)
        async with run_world.factory() as session:
            service = run_world.service(session)
            await service.cancel(write.run.id, user_id=actor.user_id, product_id=PRODUCT)
            await session.commit()
        async with run_world.factory() as session:
            with pytest.raises(RunNotCancellable):
                await run_world.service(session).cancel(
                    write.run.id, user_id=actor.user_id, product_id=PRODUCT
                )
        assert await run_world.refunds_and_debits(actor) == (1, 1)  # still one refund

    async def test_a_running_run_cannot_be_cancelled(self, run_world: RunWorld) -> None:
        actor = await run_world.add_actor()
        track_id = await run_world.new_track(actor)
        write = await run_world.submit(actor, track_id)
        await run_world.executor(FakeLlm()).claim(1)  # now running; never executed
        async with run_world.factory() as session:
            with pytest.raises(RunNotCancellable):
                await run_world.service(session).cancel(
                    write.run.id, user_id=actor.user_id, product_id=PRODUCT
                )
        assert (await run_world.run(write.run.id)).status == "running"
        assert await run_world.refunds_and_debits(actor) == (1, 0)

    @pytest.mark.parametrize("outcome", ["succeeded", "failed"])
    async def test_a_finished_run_cannot_be_cancelled(
        self, run_world: RunWorld, outcome: str
    ) -> None:
        actor = await run_world.add_actor()
        track_id = await run_world.new_track(actor)
        write = await run_world.submit(actor, track_id)
        await run_world.drive(
            FakeLlm(make_draft() if outcome == "succeeded" else LlmUnavailable("x"))
        )
        async with run_world.factory() as session:
            with pytest.raises(RunNotCancellable):
                await run_world.service(session).cancel(
                    write.run.id, user_id=actor.user_id, product_id=PRODUCT
                )

    async def test_another_users_run_is_not_found(self, run_world: RunWorld) -> None:
        owner = await run_world.add_actor()
        intruder = await run_world.add_actor()
        write = await run_world.submit(owner, await run_world.new_track(owner))
        async with run_world.factory() as session:
            with pytest.raises(RunNotFound):
                await run_world.service(session).cancel(
                    write.run.id, user_id=intruder.user_id, product_id=PRODUCT
                )
        assert (await run_world.run(write.run.id)).status == "queued"

    async def test_cancelling_races_the_claim_cleanly(self, run_world: RunWorld) -> None:
        """Whichever wins, the run ends in exactly one state and is refunded iff cancelled."""
        actor = await run_world.add_actor()
        track_id = await run_world.new_track(actor)
        write = await run_world.submit(actor, track_id)

        async def cancel() -> str:
            async with run_world.factory() as session:
                try:
                    await run_world.service(session).cancel(
                        write.run.id, user_id=actor.user_id, product_id=PRODUCT
                    )
                    await session.commit()
                except RunNotCancellable:
                    return "not_cancellable"
                return "cancelled"

        cancel_result, claimed = await asyncio.gather(
            cancel(), run_world.executor(FakeLlm()).claim(1)
        )
        run = await run_world.run(write.run.id)
        if cancel_result == "cancelled":
            assert (run.status, claimed) == ("cancelled", [])
            assert await run_world.refunds_and_debits(actor) == (1, 1)
        else:
            assert (run.status, len(claimed)) == ("running", 1)
            assert await run_world.refunds_and_debits(actor) == (1, 0)


# ---------------------------------------------------------------------------
# Execution (invariants 7, 8, 17)
# ---------------------------------------------------------------------------


class TestExecution:
    async def test_a_successful_run_saves_one_llm_version_and_changes_no_pointer(
        self, run_world: RunWorld
    ) -> None:
        actor = await run_world.add_actor()
        current = make_plan(title="Current", empty=True)
        track_id = await run_world.new_track(actor, plan=current)
        before = await run_world.track_row(track_id)
        (current_version,) = await run_world.versions(track_id)

        run = await _submit_and_drive(run_world, actor, track_id, FakeLlm(make_draft()))

        assert (run.status, run.stage, run.error_code) == ("succeeded", None, None)
        assert run.finished_at is not None
        versions = await run_world.versions(track_id)
        assert len(versions) == 2
        created = versions[1]
        assert run.result_plan_version_id == created.id
        assert created.origin == StudioDocumentOrigin.LLM.value
        assert created.kind == "plan"
        assert created.parent_version_id == current_version.id  # the plan current at claim
        assert created.content["title"] == "Neon Rain"
        assert created.content["idea"] == "a song about rain"
        track = await run_world.track_row(track_id)
        assert track.current_plan_version_id == before.current_plan_version_id == current_version.id
        assert track.current_lyrics_version_id == before.current_lyrics_version_id
        assert await run_world.refunds_and_debits(actor) == (1, 0)

    async def test_a_track_without_a_plan_gets_a_root_version(self, run_world: RunWorld) -> None:
        actor = await run_world.add_actor()
        track_id = await run_world.new_track(actor)
        run = await _submit_and_drive(run_world, actor, track_id, FakeLlm(make_draft()))
        (version,) = await run_world.versions(track_id)
        assert run.result_plan_version_id == version.id
        assert version.parent_version_id is None
        assert (await run_world.track_row(track_id)).current_plan_version_id is None

    async def test_the_parent_is_the_plan_current_at_claim_even_if_it_changes_later(
        self, run_world: RunWorld
    ) -> None:
        actor = await run_world.add_actor()
        track_id = await run_world.new_track(actor, plan=make_plan(title="First", empty=True))
        (first,) = await run_world.versions(track_id)
        await run_world.submit(actor, track_id)

        async def switch_current() -> None:
            async with run_world.factory() as session, session.begin():
                from src.api.schemas.studio import PlanVersionCreate

                await DocumentVersionService(session).create(
                    track_id,
                    PlanVersionCreate(
                        content=make_plan(title="Second", empty=True), make_current=True
                    ),
                    actor.user_id,
                    PRODUCT,
                    origin=StudioDocumentOrigin.USER,
                )

        llm = FakeLlm(make_draft(), before_call=switch_current)
        await run_world.drive(llm)

        versions = await run_world.versions(track_id)
        llm_version = next(v for v in versions if v.origin == "llm")
        assert llm_version.parent_version_id == first.id

    async def test_content_equal_to_the_parent_makes_the_parent_the_result(
        self, run_world: RunWorld
    ) -> None:
        from src.core.uid import new_id  # noqa: F401
        from src.studio.documents import PlanDocument, StyleSpec

        actor = await run_world.add_actor()
        same = PlanDocument(
            title="Neon Rain",
            idea="a song about rain",
            tempo_bpm=120,
            global_style=StyleSpec(tags=("synthwave",), avoid=("autotune",)),
            sections=(),
        )
        track_id = await run_world.new_track(actor, plan=same)
        (parent,) = await run_world.versions(track_id)

        run = await _submit_and_drive(
            run_world,
            actor,
            track_id,
            FakeLlm(_empty_sections_draft()),
        )

        assert run.status == "succeeded"
        assert run.result_plan_version_id == parent.id
        assert len(await run_world.versions(track_id)) == 1

    async def test_the_claim_freezes_the_context_and_stamps_the_run(
        self, run_world: RunWorld
    ) -> None:
        actor = await run_world.add_actor(artist=True)
        track_id = await run_world.new_track(actor, "Night Drive", plan=make_plan(title="Old"))
        write = await run_world.submit(actor, track_id)

        (claimed,) = await run_world.executor(FakeLlm()).claim(4)

        run = await run_world.run(write.run.id)
        assert (run.status, run.stage, run.revision) == ("running", "context", 2)
        assert run.model_id == "model-a"
        assert run.prompt_version == "plan-v1"
        assert run.started_at is not None
        assert run.lease_expires_at is not None
        assert run.lease_expires_at > run.started_at
        snapshot = run.context_snapshot
        assert snapshot is not None
        assert snapshot["track_title"] == "Night Drive"
        assert snapshot["artist"]["name"] == "Nova"
        assert snapshot["concept"] == "SECRET-CONCEPT night drive"
        assert snapshot["content_rating"] == "permissive"  # vex
        assert snapshot["plan_version_id"] is not None
        assert snapshot["current_plan"]["title"] == "Old"
        assert snapshot["memories"] == []
        assert claimed.context.artist is not None
        assert claimed.context.plan_version_id is not None

    async def test_claiming_is_oldest_first_and_bounded(self, run_world: RunWorld) -> None:
        actor = await run_world.add_actor(balance=20)
        runs = []
        for index in range(3):
            track_id = await run_world.new_track(actor, f"T{index}")
            runs.append((await run_world.submit(actor, track_id)).run.id)

        executor = run_world.executor(FakeLlm())
        first = await executor.claim(2)
        assert [c.run_id for c in first] == runs[:2]
        assert [c.run_id for c in await executor.claim(5)] == runs[2:]
        assert await executor.claim(5) == []
        assert await executor.claim(0) == []

    async def test_concurrent_claimers_never_share_a_run(
        self, make_run_world: Callable[..., RunWorld]
    ) -> None:
        run_world = make_run_world(studio_run_max_active_per_user=10)
        actor = await run_world.add_actor(balance=20)
        ids = set()
        for index in range(6):
            track_id = await run_world.new_track(actor, f"T{index}")
            ids.add((await run_world.submit(actor, track_id)).run.id)

        batches = await asyncio.gather(*(run_world.executor(FakeLlm()).claim(2) for _ in range(4)))

        claimed = [c.run_id for batch in batches for c in batch]
        assert len(claimed) == len(set(claimed)) == 6
        assert set(claimed) == ids

    async def test_an_invalid_plan_is_revised_then_saved(self, run_world: RunWorld) -> None:
        actor = await run_world.add_actor()
        track_id = await run_world.new_track(actor)
        llm = FakeLlm(make_draft(tags=["120 BPM"]), make_draft())
        run = await _submit_and_drive(run_world, actor, track_id, llm)

        assert run.status == "succeeded"
        assert (run.llm_calls, run.prompt_tokens, run.completion_tokens) == (2, 200, 100)
        assert llm.call_count == 2
        assert len(await run_world.versions(track_id)) == 1

    async def test_a_plan_that_stays_invalid_fails_with_paths_and_codes(
        self, run_world: RunWorld
    ) -> None:
        """Invariants 8 and 9: the same validation as user input; exactly three model calls."""
        actor = await run_world.add_actor()
        track_id = await run_world.new_track(actor)
        bad = make_draft(tags=["120 BPM"], title="SECRET-TITLE-TEXT")
        llm = FakeLlm(bad, bad, bad)

        run = await _submit_and_drive(run_world, actor, track_id, llm)

        assert llm.call_count == 3
        assert (run.status, run.error_code, run.llm_calls) == ("failed", "plan_invalid", 3)
        assert run.error_detail == {
            "issues": [{"path": "global_style.tags[0]", "code": "reserved_tempo_tag"}]
        }
        assert "SECRET-TITLE-TEXT" not in str(run.error_detail)
        assert await run_world.versions(track_id) == []
        assert await run_world.refunds_and_debits(actor) == (1, 1)

    async def test_a_save_step_refusal_keeps_its_own_code_and_refunds(
        self, run_world: RunWorld, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        actor = await run_world.add_actor()
        track_id = await run_world.new_track(actor)

        async def refuse(*_args: object, **_kwargs: object) -> None:
            raise VersionLimitReached

        monkeypatch.setattr(DocumentVersionService, "create", refuse)
        run = await _submit_and_drive(run_world, actor, track_id, FakeLlm(make_draft()))

        assert (run.status, run.error_code) == ("failed", "version_limit_reached")
        assert await run_world.refunds_and_debits(actor) == (1, 1)

    async def test_an_unexpected_error_is_an_internal_error_and_refunds(
        self, run_world: RunWorld
    ) -> None:
        actor = await run_world.add_actor()
        track_id = await run_world.new_track(actor)
        run = await _submit_and_drive(
            run_world, actor, track_id, FakeLlm(RuntimeError("the brief says SECRET"))
        )
        assert (run.status, run.error_code) == ("failed", "internal_error")
        assert run.error_detail is None
        assert await run_world.refunds_and_debits(actor) == (1, 1)

    async def test_a_deadline_overrun_fails_as_run_timeout_and_refunds(
        self, run_world: RunWorld
    ) -> None:
        from types import SimpleNamespace

        actor = await run_world.add_actor()
        track_id = await run_world.new_track(actor)
        write = await run_world.submit(actor, track_id)

        async def never_returns() -> None:
            await asyncio.sleep(30)

        executor = run_world.executor(
            FakeLlm(make_draft(), before_call=never_returns),
            settings=SimpleNamespace(studio_run_deadline_seconds=0.05),
        )
        for claimed in await executor.claim(1):
            await executor.execute(claimed)

        run = await run_world.run(write.run.id)
        assert (run.status, run.error_code) == ("failed", "run_timeout")
        assert await run_world.refunds_and_debits(actor) == (1, 1)

    async def test_events_carry_a_strictly_increasing_revision_published_after_commit(
        self, run_world: RunWorld
    ) -> None:
        """Invariant 17."""
        actor = await run_world.add_actor()
        track_id = await run_world.new_track(actor)
        seen: list[tuple[int, str, str | None]] = []

        async def check(_user: UUID, event_type: EventType, payload: object) -> None:
            if event_type is not EventType.STUDIO_RUN_UPDATED:
                return
            row = await run_world.run(payload.run_id)  # type: ignore[attr-defined]
            # committed: another connection sees at least this event's state
            assert row.revision >= payload.revision  # type: ignore[attr-defined]
            seen.append((payload.revision, payload.status, payload.stage))  # type: ignore[attr-defined]

        run_world.bus.on_publish = check
        await _submit_and_drive(
            run_world, actor, track_id, FakeLlm(make_draft(tags=["120 BPM"]), make_draft())
        )

        revisions = [revision for revision, _, _ in seen]
        assert revisions == sorted(set(revisions)), revisions
        assert [(status, stage) for _, status, stage in seen] == [
            ("queued", None),
            ("running", "context"),
            ("running", "drafting"),
            ("running", "validating"),
            ("running", "revising"),
            ("running", "validating"),
            ("running", "saving"),
            ("succeeded", None),
        ]
        final = run_world.bus.run_events[-1]
        assert final.error_code is None

    async def test_a_failed_run_publishes_its_error_code(self, run_world: RunWorld) -> None:
        actor = await run_world.add_actor()
        track_id = await run_world.new_track(actor)
        await _submit_and_drive(run_world, actor, track_id, FakeLlm(LlmUnavailable("down")))
        final = run_world.bus.run_events[-1]
        assert (final.status, final.error_code) == ("failed", "llm_unavailable")

    async def test_a_claim_that_cannot_build_its_context_fails_that_run_only(
        self, run_world: RunWorld
    ) -> None:
        actor = await run_world.add_actor(balance=20)
        bad_track = await run_world.new_track(actor, "Bad")
        good_track = await run_world.new_track(actor, "Good")
        bad = await run_world.submit(actor, bad_track)
        good = await run_world.submit(actor, good_track)

        class Flaky:
            async def for_run(self, request, *, session):  # type: ignore[no-untyped-def]
                if request.track_id == bad_track:
                    raise RuntimeError("context exploded")
                from src.studio.services.run_context import ProfileArtistContextProvider

                return await ProfileArtistContextProvider().for_run(request, session=session)

        executor = run_world.executor(FakeLlm(make_draft()), provider=Flaky())
        claimed = await executor.claim(4)

        assert [c.run_id for c in claimed] == [good.run.id]
        failed = await run_world.run(bad.run.id)
        assert (failed.status, failed.error_code) == ("failed", "internal_error")
        assert (await run_world.run(good.run.id)).status == "running"
        assert await run_world.refunds_and_debits(actor) == (2, 1)

    async def test_the_track_cannot_be_deleted_during_an_active_run_but_can_afterwards(
        self, run_world: RunWorld
    ) -> None:
        """Invariant 18."""
        from src.studio.services.track import TrackService

        actor = await run_world.add_actor()
        track_id = await run_world.new_track(actor)
        await run_world.submit(actor, track_id)

        async with run_world.factory() as session:
            with pytest.raises(RunInProgress):
                await TrackService(session).delete(track_id, actor.user_id, PRODUCT)
        assert len(await run_world.runs(track_id)) == 1

        await run_world.drive(FakeLlm(make_draft()))  # finished runs do not block
        async with run_world.factory() as session, session.begin():
            await TrackService(session).delete(track_id, actor.user_id, PRODUCT)
        assert await run_world.runs(track_id) == []
        assert await run_world.versions(track_id) == []
        assert await run_world.refunds_and_debits(actor) == (
            1,
            0,
        )  # a finished run keeps its charge

    async def test_deleting_a_version_keeps_the_run_and_clears_its_result(
        self, run_world: RunWorld
    ) -> None:
        actor = await run_world.add_actor()
        track_id = await run_world.new_track(actor)
        run = await _submit_and_drive(run_world, actor, track_id, FakeLlm(make_draft()))
        assert run.result_plan_version_id is not None
        async with run_world.factory() as session, session.begin():
            await DocumentVersionService(session).delete(
                run.result_plan_version_id, actor.user_id, PRODUCT
            )
        kept = await run_world.run(run.id)
        assert (kept.status, kept.result_plan_version_id) == ("succeeded", None)


def _empty_sections_draft():  # type: ignore[no-untyped-def]
    from src.studio.planner.drafts import PlanDraft

    return PlanDraft(
        refusal=None,
        title="Neon Rain",
        tempo_bpm=120,
        instrumental=False,
        tags=["synthwave"],
        avoid=["autotune"],
        sections=[],
    )


# ---------------------------------------------------------------------------
# Terminal-once (invariant 1) and the sweep (invariant 11)
# ---------------------------------------------------------------------------


class TestTerminalOnce:
    async def _terminal_run(self, world: RunWorld, status: str) -> tuple[UUID, UUID]:
        actor = await world.add_actor()
        track_id = await world.new_track(actor)
        write = await world.submit(actor, track_id)
        if status == "cancelled":
            async with world.factory() as session:
                await world.service(session).cancel(
                    write.run.id, user_id=actor.user_id, product_id=PRODUCT
                )
                await session.commit()
        else:
            await world.drive(
                FakeLlm(make_draft() if status == "succeeded" else LlmUnavailable("x"))
            )
        assert (await world.run(write.run.id)).status == status
        return write.run.id, actor.user_id

    @pytest.mark.parametrize("status", ["succeeded", "failed", "cancelled"])
    async def test_no_transition_leaves_a_terminal_status(
        self, run_world: RunWorld, status: str
    ) -> None:
        run_id, user_id = await self._terminal_run(run_world, status)
        before = await run_world.run(run_id)

        async with run_world.factory() as session, session.begin():
            runs = StudioRunRepository(session)
            assert await runs.set_stage(run_id, StudioRunStage.DRAFTING) is None
            assert await runs.cancel_queued(run_id, user_id=user_id, product_id=PRODUCT) is None
            assert (
                await runs.mark_running(
                    run_id,
                    lease_seconds=60,
                    model_id="m",
                    prompt_version="p",
                    context_snapshot={},
                )
                is None
            )
            assert (
                await runs.finish_succeeded(
                    run_id,
                    result_plan_version_id=None,
                    warnings=[],
                    llm_calls=9,
                    prompt_tokens=9,
                    completion_tokens=9,
                )
                is None
            )
            assert (
                await runs.finish_failed(
                    run_id,
                    error_code="x",
                    error_detail=None,
                    llm_calls=9,
                    prompt_tokens=9,
                    completion_tokens=9,
                )
                is None
            )
            assert await runs.fail_queued(run_id, error_code="x") is None
            assert list(await runs.sweep_expired(error_code="run_timeout")) == []

        after = await run_world.run(run_id)
        assert (after.status, after.revision, after.error_code, after.llm_calls) == (
            before.status,
            before.revision,
            before.error_code,
            before.llm_calls,
        )

    async def test_the_database_refuses_inconsistent_terminal_states(
        self, run_world: RunWorld
    ) -> None:
        from sqlalchemy.exc import IntegrityError

        actor = await run_world.add_actor()
        write = await run_world.submit(actor, await run_world.new_track(actor))
        for statement in (
            "UPDATE studio_runs SET status = 'succeeded' WHERE id = :id",  # no finished_at
            "UPDATE studio_runs SET status = 'failed', finished_at = now() WHERE id = :id",
            "UPDATE studio_runs SET finished_at = now() WHERE id = :id",  # queued + finished
            "UPDATE studio_runs SET status = 'nonsense' WHERE id = :id",
            "UPDATE studio_runs SET stage = 'nonsense' WHERE id = :id",
            "UPDATE studio_runs SET price_tokens = -1 WHERE id = :id",
        ):
            with pytest.raises(IntegrityError):
                async with run_world.factory() as session, session.begin():
                    await session.execute(text(statement), {"id": write.run.id})

    async def test_the_database_allows_one_active_run_per_track_only(
        self, run_world: RunWorld
    ) -> None:
        from sqlalchemy.exc import IntegrityError

        actor = await run_world.add_actor()
        track_id = await run_world.new_track(actor)
        await run_world.submit(actor, track_id)
        with pytest.raises(IntegrityError, match="uq_studio_runs_one_active_per_track"):
            async with run_world.factory() as session, session.begin():
                await StudioRunRepository(session).insert(
                    run_id=new_id(),
                    user_id=actor.user_id,
                    product_id=PRODUCT,
                    track_id=track_id,
                    kind="plan",
                    idempotency_key="other",
                    request_hash="h",
                    request={},
                    model_key="grok",
                    price_tokens=0,
                )


class TestSweep:
    async def test_a_run_past_its_lease_is_failed_and_refunded_once(
        self, run_world: RunWorld
    ) -> None:
        actor = await run_world.add_actor()
        track_id = await run_world.new_track(actor)
        write = await run_world.submit(actor, track_id)
        executor = run_world.executor(FakeLlm())
        await executor.claim(1)  # running; the "worker" then dies
        assert await executor.sweep() == 0  # the lease is still live
        await run_world.expire_lease(write.run.id)

        assert await executor.sweep() == 1
        assert await executor.sweep() == 0  # terminal once

        run = await run_world.run(write.run.id)
        assert (run.status, run.error_code, run.stage) == ("failed", "run_timeout", None)
        assert run.finished_at is not None
        assert await run_world.refunds_and_debits(actor) == (1, 1)
        assert await run_world.balance(actor) == 10
        last = run_world.bus.run_events[-1]
        assert (last.status, last.error_code) == ("failed", "run_timeout")

    async def test_a_queued_run_is_never_swept(self, run_world: RunWorld) -> None:
        actor = await run_world.add_actor()
        write = await run_world.submit(actor, await run_world.new_track(actor))
        assert await run_world.executor(FakeLlm()).sweep() == 0
        assert (await run_world.run(write.run.id)).status == "queued"

    async def test_a_swept_run_is_never_executed_again(self, run_world: RunWorld) -> None:
        actor = await run_world.add_actor()
        track_id = await run_world.new_track(actor)
        write = await run_world.submit(actor, track_id)
        executor = run_world.executor(FakeLlm())
        await executor.claim(1)
        await run_world.expire_lease(write.run.id)
        await executor.sweep()

        assert await executor.claim(4) == []
        assert (await run_world.run(write.run.id)).status == "failed"

    async def test_a_late_finish_after_the_sweep_writes_nothing(self, run_world: RunWorld) -> None:
        """The run was swept while the model was still thinking: no stage write, no version."""
        actor = await run_world.add_actor()
        track_id = await run_world.new_track(actor)
        write = await run_world.submit(actor, track_id)
        executor = run_world.executor(FakeLlm(make_draft()))

        async def sweep_during_the_call() -> None:
            await run_world.expire_lease(write.run.id)
            await executor.sweep()

        llm = FakeLlm(make_draft(), before_call=sweep_during_the_call)
        executor = run_world.executor(llm)
        (claimed,) = await executor.claim(1)
        await executor.execute(claimed)

        run = await run_world.run(write.run.id)
        assert (run.status, run.error_code) == ("failed", "run_timeout")
        assert await run_world.versions(track_id) == []
        assert await run_world.refunds_and_debits(actor) == (1, 1)  # one refund, from the sweep

    async def test_a_sweep_during_the_save_rolls_the_version_back(
        self, run_world: RunWorld
    ) -> None:
        """The finish transaction creates the version, then finds the run swept: all undone."""
        actor = await run_world.add_actor()
        track_id = await run_world.new_track(actor)
        write = await run_world.submit(actor, track_id)
        executor = run_world.executor(FakeLlm(make_draft()))
        (claimed,) = await executor.claim(1)

        real_on_stage = executor._on_stage

        async def on_stage(item, stage):  # type: ignore[no-untyped-def]
            await real_on_stage(item, stage)
            if stage is StudioRunStage.SAVING:
                await run_world.expire_lease(write.run.id)
                await executor.sweep()

        executor._on_stage = on_stage  # type: ignore[method-assign]
        await executor.execute(claimed)

        run = await run_world.run(write.run.id)
        assert (run.status, run.error_code) == ("failed", "run_timeout")
        assert run.result_plan_version_id is None
        assert await run_world.versions(track_id) == []  # the version was rolled back
        assert await run_world.refunds_and_debits(actor) == (1, 1)


# ---------------------------------------------------------------------------
# Connections (invariant 12)
# ---------------------------------------------------------------------------


async def test_no_connection_is_checked_out_while_the_model_is_awaited(
    run_world: RunWorld,
) -> None:
    actor = await run_world.add_actor()
    track_id = await run_world.new_track(actor)
    await run_world.submit(actor, track_id)
    checked_out: list[int] = []

    async def inspect_pool() -> None:
        checked_out.append(run_world.db.engine.sync_engine.pool.checkedout())  # type: ignore[attr-defined]

    llm = FakeLlm(make_draft(tags=["120 BPM"]), make_draft(), before_call=inspect_pool)
    await run_world.drive(llm)

    assert checked_out == [0, 0]  # the first call and the revision call
    assert (await run_world.runs(track_id))[0].status == "succeeded"
