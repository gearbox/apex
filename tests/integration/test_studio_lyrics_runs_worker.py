"""One ``StudioRunWorker`` serves every run kind: a tick claims them, ``stop`` drains them.

Driven by ``run_once``; every wait is on something the test controls, never a sleep.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

from src.studio.run_requests import (
    LineTarget,
    LyricsRunRequest,
    ReviseLyricsRunRequest,
    ReviseTargets,
)
from src.workers.studio_run import StudioRunWorker
from tests.studio_llm import (
    make_line_patch,
    make_lyrics_draft,
    make_lyrics_revision,
    section_lines,
)
from tests.studio_lyrics_support import PLAN, S1, S2, doc

if TYPE_CHECKING:
    from collections.abc import Callable

    from tests.studio_runs_support import RunWorld

TIMEOUT = 10
FOUR = "la la la la"


async def test_one_tick_serves_a_lyrics_run_and_a_revision_side_by_side(
    make_run_world: Callable[..., RunWorld],
) -> None:
    world = make_run_world(studio_run_max_active_per_user=5)
    actor = await world.add_actor(balance=20)
    plan_track = await world.new_track(actor, "A", PLAN)
    revise_track = await world.new_track(actor, "B", PLAN)
    plan_id = (await world.track_row(revise_track)).current_plan_version_id
    assert plan_id is not None
    await world.create_lyrics(
        actor, revise_track, doc((S1, [FOUR]), (S2, [FOUR])), plan_version_id=plan_id
    )

    lyrics_run = await world.submit(actor, plan_track, request=LyricsRunRequest(language="en"))
    revise_run = await world.submit(
        actor,
        revise_track,
        request=ReviseLyricsRunRequest(
            instruction="warmer", targets=ReviseTargets(lines=(LineTarget(section_id=S1, line=0),))
        ),
    )

    # the kind decides which schema is asked for: a script keyed by call order of arrival would
    # be racy, so each kind gets its own scripted reply through one routing fake
    replies = {
        "LyricsDraft": make_lyrics_draft(
            section_lines("s1", FOUR), section_lines("s2", FOUR), section_lines("s4", FOUR)
        ),
        "LyricsRevisionDraft": make_lyrics_revision(lines=[make_line_patch("s1", 1, "new opener")]),
    }

    class Routing:
        def __init__(self) -> None:
            self.calls = 0

        async def complete(self, messages, *, schema, schema_name):  # type: ignore[no-untyped-def]  # noqa: ARG002
            from src.llm.port import LlmResult, LlmUsage

            self.calls += 1
            return LlmResult(value=replies[schema.__name__], usage=LlmUsage(3, 4), model_id="m")

    routing = Routing()
    worker = StudioRunWorker(
        executor=world.executor(routing),  # type: ignore[arg-type]
        interval_seconds=1,
        max_concurrent=4,
        redis_client_factory=MagicMock,
    )
    await asyncio.wait_for(worker.run_once(), TIMEOUT)
    await asyncio.wait_for(worker.stop(), TIMEOUT)  # drains both

    first = await world.run(lyrics_run.run.id)
    second = await world.run(revise_run.run.id)
    assert (first.status, first.kind, second.status, second.kind) == (
        "succeeded",
        "lyrics",
        "succeeded",
        "revise_lyrics",
    )
    assert (first.prompt_version, second.prompt_version) == ("lyrics-v1", "revise-lyrics-v1")
    assert routing.calls == 2
    assert first.result_lyrics_version_id is not None
    assert second.result_lyrics_version_id is not None
    assert (first.result_plan_version_id, second.result_plan_version_id) == (None, None)
    assert await world.refunds_and_debits(actor) == (2, 0)
