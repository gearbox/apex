"""``studio_runs`` stores SQL ``NULL`` where it means "none", and the database holds two rules by kind.

R2: ``inputs`` (and ``error_detail``) are SQL ``NULL``, not the JSON value ``null`` - checked in SQL,
never through the ORM, which reads both as ``None``. R3: ``inputs IS NULL`` exactly for ``plan`` runs
(``ck_studio_runs_inputs_by_kind``) and only the kind's own result column is ever set
(``ck_studio_runs_result_by_kind``) - invariant 24.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast
from uuid import uuid4

import pytest
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError

from src.core.uid import new_id
from src.db.models.studio import StudioRun
from src.db.repositories.studio import StudioRunRepository
from src.llm.port import LlmUnavailable
from src.studio.run_requests import (
    LyricsRunRequest,
    PlanRunRequest,
    ReviseLyricsRunRequest,
    ReviseTargets,
)
from tests.llm_support import FakeLlm
from tests.studio_llm import (
    make_line_patch,
    make_lyrics_draft,
    make_lyrics_patch,
    section_lines,
)
from tests.studio_lyrics_support import PLAN, S1, S2, S4, doc
from tests.studio_runs_support import PRODUCT

if TYPE_CHECKING:
    from uuid import UUID

    from sqlalchemy import Table

    from tests.studio_runs_support import Actor, RunWorld

FOUR = "la la la la"


async def _row(world: RunWorld, run_id: UUID, expression: str) -> Any:
    """One SQL expression over one run, evaluated by the database."""
    async with world.factory() as session:
        return (
            await session.execute(
                text(f"SELECT {expression} FROM studio_runs WHERE id = :id"),  # noqa: S608
                {"id": run_id},
            )
        ).scalar_one()


async def _cancel(world: RunWorld, actor: Actor, run_id: UUID) -> None:
    async with world.factory() as session:
        await world.service(session).cancel(run_id, user_id=actor.user_id, product_id=PRODUCT)
        await session.commit()


async def _track_with_lyrics(world: RunWorld) -> tuple[Actor, UUID, UUID, UUID]:
    """(actor, track, plan version, lyrics version): a track with both, both current."""
    actor = await world.add_actor()
    track_id = await world.new_track(actor, "Night Drive", PLAN)
    plan_id = (await world.track_row(track_id)).current_plan_version_id
    assert plan_id is not None
    lyrics_id = await world.create_lyrics(
        actor,
        track_id,
        doc((S1, [FOUR, FOUR]), (S2, [FOUR, FOUR]), (S4, [FOUR])),
        plan_version_id=plan_id,
    )
    return actor, track_id, plan_id, lyrics_id


# -- R2: SQL NULL, not JSON null -------------------------------------------------------------------


async def test_inputs_is_sql_null_exactly_for_plan_runs_invariant_24(run_world: RunWorld) -> None:
    actor, track_id, _, _ = await _track_with_lyrics(run_world)
    revise = ReviseLyricsRunRequest(
        instruction="warmer", targets=ReviseTargets(sections=(S2,), lines=())
    )
    seen: dict[str, tuple[bool, str | None]] = {}
    for request in (PlanRunRequest(brief="rain"), LyricsRunRequest(language="en"), revise):
        write = await run_world.submit(actor, track_id, request=request)
        run_id = write.run.id
        kind = (await run_world.run(run_id)).kind
        seen[kind] = (
            await _row(run_world, run_id, "inputs IS NULL"),
            await _row(run_world, run_id, "jsonb_typeof(inputs)"),
        )
        await _cancel(run_world, actor, run_id)  # one active run per track

    # SQL NULL has no JSON type at all; the JSON value ``null`` would say 'null'
    assert seen == {
        "plan": (True, None),
        "lyrics": (False, "object"),
        "revise_lyrics": (False, "object"),
    }


async def test_a_run_never_writes_the_json_value_null_into_any_nullable_json_column(
    run_world: RunWorld,
) -> None:
    """The columns that can hold "nothing": ``inputs``, ``context_snapshot``, ``error_detail``."""
    actor = await run_world.add_actor()
    track_id = await run_world.new_track(actor, "T", PLAN)
    queued = (await run_world.submit(actor, track_id)).run.id
    assert await _row(run_world, queued, "context_snapshot IS NULL AND error_detail IS NULL")
    assert await _row(run_world, queued, "inputs IS NULL")

    await run_world.drive(FakeLlm(LlmUnavailable("down")))  # a failure with no detail
    assert (await run_world.run(queued)).error_code == "llm_unavailable"
    assert await _row(run_world, queued, "error_detail IS NULL")  # not 'null'::jsonb
    assert await _row(run_world, queued, "jsonb_typeof(context_snapshot)") == "object"
    assert await _row(run_world, queued, "inputs IS NULL")


async def test_a_swept_run_has_sql_null_error_detail(run_world: RunWorld) -> None:
    actor = await run_world.add_actor()
    write = await run_world.submit(actor, await run_world.new_track(actor, "T", PLAN))
    executor = run_world.executor(FakeLlm())
    await executor.claim(1)
    await run_world.expire_lease(write.run.id)
    assert await executor.sweep() == 1
    assert await _row(run_world, write.run.id, "error_detail IS NULL")


async def test_error_detail_is_an_object_when_there_is_detail_and_sql_null_otherwise(
    run_world: RunWorld,
) -> None:
    actor, track_id, _, _ = await _track_with_lyrics(run_world)
    hard = make_lyrics_draft(  # a bracket is a hard issue
        section_lines("s1", FOUR), section_lines("s2", "oh [no] no"), section_lines("s4", FOUR)
    )

    # a port error carries no detail
    first = await run_world.submit(actor, track_id, request=LyricsRunRequest(language="en"))
    await run_world.drive(FakeLlm(hard, LlmUnavailable("down")))
    assert (await run_world.run(first.run.id)).error_code == "llm_unavailable"
    assert await _row(run_world, first.run.id, "error_detail IS NULL")

    # a rejection carries issue paths and codes
    second = await run_world.submit(actor, track_id, request=LyricsRunRequest(language="en"))
    keep = make_lyrics_patch(make_line_patch("s2", 1, "still [bad]"))
    await run_world.drive(FakeLlm(hard, keep, keep, keep))
    assert (await run_world.run(second.run.id)).error_code == "lyrics_invalid"
    assert await _row(run_world, second.run.id, "jsonb_typeof(error_detail)") == "object"


# -- R3.3: the two CHECK constraints -----------------------------------------------------------------


async def _insert(world: RunWorld, actor: Actor, track_id: UUID, *, kind: str, inputs: Any) -> None:
    async with world.factory() as session, session.begin():
        await StudioRunRepository(session).insert(
            run_id=new_id(),
            user_id=actor.user_id,
            product_id=PRODUCT,
            track_id=track_id,
            kind=kind,
            idempotency_key=uuid4().hex,
            request_hash="0" * 64,
            request={},
            model_key="grok",
            price_tokens=0,
            inputs=inputs,
        )


@pytest.mark.parametrize("kind", ["lyrics", "revise_lyrics"])
async def test_the_database_refuses_a_lyrics_run_without_inputs_invariant_24(
    run_world: RunWorld, kind: str
) -> None:
    actor = await run_world.add_actor()
    track_id = await run_world.new_track(actor)
    with pytest.raises(IntegrityError, match="ck_studio_runs_inputs_by_kind"):
        await _insert(run_world, actor, track_id, kind=kind, inputs=None)
    assert await run_world.count_runs(actor) == 0


async def test_the_database_refuses_a_plan_run_with_inputs_invariant_24(
    run_world: RunWorld,
) -> None:
    actor = await run_world.add_actor()
    track_id = await run_world.new_track(actor)
    inputs = {"plan_version_id": str(uuid4()), "lyrics_version_id": None, "language": "en"}
    with pytest.raises(IntegrityError, match="ck_studio_runs_inputs_by_kind"):
        await _insert(run_world, actor, track_id, kind="plan", inputs=inputs)
    assert await run_world.count_runs(actor) == 0


async def test_the_database_accepts_each_kind_with_its_own_inputs(run_world: RunWorld) -> None:
    actor = await run_world.add_actor()
    track_id = await run_world.new_track(actor)
    inputs = {"plan_version_id": str(uuid4()), "lyrics_version_id": None, "language": "en"}
    await _insert(run_world, actor, track_id, kind="plan", inputs=None)
    async with run_world.factory() as session, session.begin():
        await session.execute(
            text("UPDATE studio_runs SET status = 'cancelled', finished_at = now()")
        )
    await _insert(run_world, actor, track_id, kind="lyrics", inputs=inputs)
    assert await run_world.count_runs(actor) == 2


async def _running(world: RunWorld, actor: Actor, track_id: UUID, request: Any) -> StudioRun:
    """Submit ``request`` and claim it, so it is ``running`` and may be finished."""
    write = await world.submit(actor, track_id, request=request)
    await world.executor(FakeLlm()).claim(1)
    run = await world.run(write.run.id)
    assert run.status == "running"
    return run


async def _finish(world: RunWorld, run_id: UUID, *, plan: UUID | None, lyrics: UUID | None) -> None:
    async with world.factory() as session, session.begin():
        await StudioRunRepository(session).finish_succeeded(
            run_id,
            result_plan_version_id=plan,
            result_lyrics_version_id=lyrics,
            warnings=[],
            llm_calls=1,
            prompt_tokens=1,
            completion_tokens=1,
        )


async def test_a_plan_run_cannot_finish_with_a_lyrics_result_invariant_24(
    run_world: RunWorld,
) -> None:
    actor, track_id, _, lyrics_id = await _track_with_lyrics(run_world)
    run = await _running(run_world, actor, track_id, PlanRunRequest(brief="rain"))
    with pytest.raises(IntegrityError, match="ck_studio_runs_result_by_kind"):
        await _finish(run_world, run.id, plan=None, lyrics=lyrics_id)
    assert (await run_world.run(run.id)).status == "running"  # nothing was written


@pytest.mark.parametrize("request_kind", ["lyrics", "revise_lyrics"])
async def test_a_lyrics_run_cannot_finish_with_a_plan_result_invariant_24(
    run_world: RunWorld, request_kind: str
) -> None:
    actor, track_id, plan_id, _ = await _track_with_lyrics(run_world)
    request: Any = (
        LyricsRunRequest(language="en")
        if request_kind == "lyrics"
        else ReviseLyricsRunRequest(
            instruction="warmer", targets=ReviseTargets(sections=(S2,), lines=())
        )
    )
    run = await _running(run_world, actor, track_id, request)
    with pytest.raises(IntegrityError, match="ck_studio_runs_result_by_kind"):
        await _finish(run_world, run.id, plan=plan_id, lyrics=None)
    assert (await run_world.run(run.id)).status == "running"


async def test_each_kind_may_finish_with_its_own_result(run_world: RunWorld) -> None:
    actor, track_id, plan_id, lyrics_id = await _track_with_lyrics(run_world)
    plan_run = await _running(run_world, actor, track_id, PlanRunRequest(brief="rain"))
    await _finish(run_world, plan_run.id, plan=plan_id, lyrics=None)
    assert (await run_world.run(plan_run.id)).result_plan_version_id == plan_id

    lyrics_run = await _running(run_world, actor, track_id, LyricsRunRequest(language="en"))
    await _finish(run_world, lyrics_run.id, plan=None, lyrics=lyrics_id)
    assert (await run_world.run(lyrics_run.id)).result_lyrics_version_id == lyrics_id


async def test_the_model_and_the_database_declare_the_same_constraints(
    run_world: RunWorld,
) -> None:
    """``alembic check`` does not compare CHECK constraints, so compare their names here."""
    from sqlalchemy import CheckConstraint

    # ``__table__`` is typed as ``FromClause``; a declarative model's is always a ``Table``.
    table = cast("Table", StudioRun.__table__)
    declared = {
        constraint.name
        for constraint in table.constraints
        if isinstance(constraint, CheckConstraint)
    }
    async with run_world.factory() as session:
        stored = {
            row[0]
            for row in await session.execute(
                text(
                    "SELECT conname FROM pg_constraint "
                    "WHERE conrelid = 'studio_runs'::regclass AND contype = 'c'"
                )
            )
        }
    assert declared == stored
    assert {"ck_studio_runs_inputs_by_kind", "ck_studio_runs_result_by_kind"} <= stored
