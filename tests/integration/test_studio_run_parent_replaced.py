"""Remediation round 2, RF1 part B: the plan a run started from is deleted while the model works.

Invariant 28: a paid, valid result is never thrown away because its lineage anchor is gone. The
result becomes a child of the track's current plan at save time (a root version when there is none)
and the run records ``parent_version_replaced``. Everything else about a result is unchanged: origin
``llm``, never made current, saved atomically with the run's finish.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.core.enums import StudioDocumentOrigin
from src.db.repositories.studio import StudioTrackRepository
from src.studio.documents import PlanDocument, StyleSpec
from src.studio.errors import ParentVersionNotFound
from src.studio.planner.drafts import PlanDraft
from src.studio.services.mappers import run_out
from src.studio.services.version import DocumentVersionService
from tests.llm_support import FakeLlm
from tests.studio_llm import make_draft
from tests.studio_runs_support import PRODUCT
from tests.studio_support import make_plan

if TYPE_CHECKING:
    from uuid import UUID

    import pytest

    from tests.studio_runs_support import Actor, RunWorld

REPLACED = [{"code": "parent_version_replaced"}]


def _same_as_the_draft() -> PlanDocument:
    """The plan ``_empty_draft`` becomes (the brief of ``RunWorld.submit`` is the idea)."""
    return PlanDocument(
        title="Neon Rain",
        idea="a song about rain",
        tempo_bpm=120,
        global_style=StyleSpec(tags=("synthwave",), avoid=("autotune",)),
        sections=(),
    )


def _empty_draft() -> PlanDraft:
    return PlanDraft(
        refusal=None,
        refusal_memories=[],
        title="Neon Rain",
        tempo_bpm=120,
        instrumental=False,
        tags=["synthwave"],
        avoid=["autotune"],
        sections=[],
    )


async def _claimed_with_a_plan(run_world: RunWorld, llm: FakeLlm):  # type: ignore[no-untyped-def]
    """An actor, a track whose current plan is "Old", and a run claimed against it."""
    actor = await run_world.add_actor()
    track_id = await run_world.new_track(actor, plan=make_plan(title="Old", empty=True))
    (old,) = await run_world.versions(track_id)
    write = await run_world.submit(actor, track_id)
    executor = run_world.executor(llm)
    (claimed,) = await executor.claim(1)
    assert claimed.context.plan_version_id == old.id
    return actor, track_id, old, write.run.id, executor, claimed


async def _clear_current_plan_and_delete(world: RunWorld, actor: Actor, track_id: UUID) -> None:
    """A track that ends with no current plan: the pointer is nulled, then the old plan deleted.

    The services never clear a plan pointer (only the repository can), so this is the one way to
    reach "no current plan at save" for a run that started with one.
    """
    before = await world.track_row(track_id)
    assert before.current_plan_version_id is not None
    async with world.factory() as session, session.begin():
        await StudioTrackRepository(session).set_current(track_id, plan_version_id=None)
        await DocumentVersionService(session).delete(
            before.current_plan_version_id, actor.user_id, PRODUCT
        )


class TestTheParentIsDeletedDuringTheRun:
    async def test_the_result_becomes_a_child_of_the_plan_current_at_save(
        self, run_world: RunWorld
    ) -> None:
        actor, track_id, old, run_id, executor, claimed = await _claimed_with_a_plan(
            run_world, FakeLlm(make_draft())
        )
        replacement = await run_world.replace_current_plan(actor, track_id)

        await executor.execute(claimed)

        run = await run_world.run(run_id)
        assert (run.status, run.error_code) == ("succeeded", None)
        assert run.warnings == REPLACED
        assert [w.code for w in run_out(run).warnings] == ["parent_version_replaced"]
        versions = {v.id: v for v in await run_world.versions(track_id)}
        assert old.id not in versions  # the claim-time parent is really gone
        result = versions[run.result_plan_version_id]  # type: ignore[index]
        assert result.parent_version_id == replacement
        assert result.origin == StudioDocumentOrigin.LLM.value
        assert result.content["title"] == "Neon Rain"
        # A run never changes a pointer, and the model was paid for exactly once.
        assert (await run_world.track_row(track_id)).current_plan_version_id == replacement
        assert run.llm_calls == 1
        assert await run_world.refunds_and_debits(actor) == (1, 0)
        assert await run_world.balance(actor) == 9

    async def test_without_a_current_plan_at_save_the_result_is_a_root_version(
        self, run_world: RunWorld
    ) -> None:
        actor, track_id, old, run_id, executor, claimed = await _claimed_with_a_plan(
            run_world, FakeLlm(make_draft())
        )
        await _clear_current_plan_and_delete(run_world, actor, track_id)

        await executor.execute(claimed)

        run = await run_world.run(run_id)
        assert (run.status, run.error_code) == ("succeeded", None)
        assert run.warnings == REPLACED
        (result,) = await run_world.versions(track_id)
        assert result.id == run.result_plan_version_id
        assert result.id != old.id
        assert result.parent_version_id is None
        assert result.origin == StudioDocumentOrigin.LLM.value
        assert (await run_world.track_row(track_id)).current_plan_version_id is None
        assert await run_world.refunds_and_debits(actor) == (1, 0)

    async def test_content_equal_to_the_fallback_parent_makes_that_parent_the_result(
        self, run_world: RunWorld
    ) -> None:
        actor, track_id, _, run_id, executor, claimed = await _claimed_with_a_plan(
            run_world, FakeLlm(_empty_draft())
        )
        replacement = await run_world.replace_current_plan(
            actor, track_id, plan=_same_as_the_draft()
        )

        await executor.execute(claimed)

        run = await run_world.run(run_id)
        assert (run.status, run.error_code) == ("succeeded", None)
        assert run.result_plan_version_id == replacement  # the result *is* the fallback parent
        assert run.warnings == REPLACED
        assert [v.id for v in await run_world.versions(track_id)] == [replacement]  # no new row
        assert await run_world.refunds_and_debits(actor) == (1, 0)

    async def test_a_second_refusal_is_not_retried_and_fails_the_run_refunded(
        self, run_world: RunWorld, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """One retry, never a loop: a parent refused twice fails the run like any save refusal."""
        attempts: list[UUID | None] = []

        async def refuse(
            _self: object, _track_id: UUID, request: object, *_a: object, **_k: object
        ):  # type: ignore[no-untyped-def]
            attempts.append(request.parent_version_id)  # type: ignore[attr-defined]
            raise ParentVersionNotFound

        actor, track_id, old, run_id, executor, claimed = await _claimed_with_a_plan(
            run_world, FakeLlm(make_draft())
        )
        replacement = await run_world.replace_current_plan(actor, track_id)
        monkeypatch.setattr(DocumentVersionService, "create", refuse)

        await executor.execute(claimed)

        run = await run_world.run(run_id)
        assert (run.status, run.error_code) == ("failed", "parent_version_not_found")
        assert run.warnings == []
        assert attempts == [old.id, replacement]  # the snapshot parent, then the current plan
        assert await run_world.refunds_and_debits(actor) == (1, 1)
        assert await run_world.balance(actor) == 10


class TestTheParentStillExists:
    async def test_no_warning_and_the_snapshot_parent_is_kept(self, run_world: RunWorld) -> None:
        _, track_id, old, run_id, executor, claimed = await _claimed_with_a_plan(
            run_world, FakeLlm(make_draft())
        )

        await executor.execute(claimed)

        run = await run_world.run(run_id)
        assert (run.status, run.warnings) == ("succeeded", [])
        result = next(v for v in await run_world.versions(track_id) if v.origin == "llm")
        assert result.id == run.result_plan_version_id
        assert result.parent_version_id == old.id

    async def test_a_plan_switched_but_not_deleted_is_still_the_parent(
        self, run_world: RunWorld
    ) -> None:
        """Only a *deleted* parent is replaced: the lineage is the plan the run started from."""
        from src.api.schemas.studio import PlanVersionCreate

        actor, track_id, old, run_id, executor, claimed = await _claimed_with_a_plan(
            run_world, FakeLlm(make_draft())
        )
        async with run_world.factory() as session, session.begin():
            await DocumentVersionService(session).create(
                track_id,
                PlanVersionCreate(content=make_plan(title="Other", empty=True), make_current=True),
                actor.user_id,
                PRODUCT,
                origin=StudioDocumentOrigin.USER,
            )

        await executor.execute(claimed)

        run = await run_world.run(run_id)
        assert (run.status, run.warnings) == ("succeeded", [])
        result = next(v for v in await run_world.versions(track_id) if v.origin == "llm")
        assert result.parent_version_id == old.id
