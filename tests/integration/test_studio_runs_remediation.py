"""Remediation round 1 of the planner runs, on committed rows (invariants 24 and 26).

* A graph that ends in an impossible state fails the run as ``internal_error`` - deliberately named,
  refunded, and logged by its type (not as an assertion that the guard swallows).
* A timeout raised by the save step is not the run's deadline, so it is not ``run_timeout``.
* Tokens of a reply that was paid for but cannot be used are counted on the run.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, ClassVar

import httpx
import msgspec
from pydantic import SecretStr
from structlog.testing import capture_logs

from src.core.llm_models import LlmModelEntry
from src.llm.openai_compatible import OpenAICompatibleLlmAdapter
from src.llm.port import LlmOutputInvalid, LlmOutputTruncated, LlmUsage
from src.studio.planner.graphs import plan as plan_graph
from src.studio.services.version import DocumentVersionService
from tests.llm_support import FakeLlm
from tests.studio_llm import make_draft

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    import pytest

    from src.db.models.studio import StudioRun
    from tests.studio_runs_support import RunWorld

SECRET = "the brief says SECRET-TEXT"
PAID = LlmUsage(prompt_tokens=40, completion_tokens=9)


async def _drive_one(world: RunWorld, llm: Any) -> tuple[Any, StudioRun]:
    actor = await world.add_actor()
    track_id = await world.new_track(actor)
    write = await world.submit(actor, track_id)
    await world.drive(llm)
    return actor, await world.run(write.run.id)


def _internal_error_logs(logs: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    return [log for log in logs if log["event"] == "studio.run.internal_error"]


class TestAnImpossibleGraphState:
    """Invariant 24: a graph that ends without an outcome fails the run as ``internal_error``."""

    async def test_it_fails_internal_error_refunded_and_named_in_the_log(
        self, run_world: RunWorld, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # forced: the router ends the graph on an invalid first draft instead of revising it
        monkeypatch.setattr(plan_graph, "_route", lambda _state: "done")
        llm = FakeLlm(make_draft(tags=["120 BPM"]))

        with capture_logs() as logs:
            actor, run = await _drive_one(run_world, llm)

        assert (run.status, run.error_code) == ("failed", "internal_error")
        assert run.error_detail is None
        assert run.finished_at is not None
        assert await run_world.refunds_and_debits(actor) == (1, 1)
        assert await run_world.balance(actor) == 10
        (record,) = _internal_error_logs(logs)
        assert record["exc_type"] == "PlannerStateError"
        assert record["run_id"] == str(run.id)
        assert await run_world.versions(run.track_id) == []

    async def test_the_failed_run_still_reports_what_the_model_cost(
        self, run_world: RunWorld, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(plan_graph, "_route", lambda _state: "done")
        _, run = await _drive_one(run_world, FakeLlm(make_draft(tags=["120 BPM"])))
        assert (run.llm_calls, run.prompt_tokens, run.completion_tokens) == (1, 100, 50)

    async def test_a_node_reached_without_its_input_fails_the_same_way(
        self, run_world: RunWorld, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A patched node: the failure is the named error, whatever raised it."""
        real: Any = plan_graph._validate

        async def broken(state: Any, runtime: Any) -> dict[str, Any]:
            return await real({**state, "draft": None}, runtime)

        monkeypatch.setattr(plan_graph, "_validate", broken)
        with capture_logs() as logs:
            actor, run = await _drive_one(run_world, FakeLlm(make_draft()))

        assert (run.status, run.error_code) == ("failed", "internal_error")
        assert await run_world.refunds_and_debits(actor) == (1, 1)
        (record,) = _internal_error_logs(logs)
        assert record["exc_type"] == "PlannerStateError"


class TestATimeoutOutsideTheDeadline:
    """R2.4: only the graph is bounded by the run's deadline."""

    async def test_a_timeout_from_the_save_step_is_not_a_run_timeout(
        self, run_world: RunWorld, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        async def database_timeout(*_args: object, **_kwargs: object) -> None:
            raise TimeoutError(SECRET)

        monkeypatch.setattr(DocumentVersionService, "create", database_timeout)
        with capture_logs() as logs:
            actor, run = await _drive_one(run_world, FakeLlm(make_draft()))

        assert (run.status, run.error_code) == ("failed", "internal_error")
        assert run.error_code != "run_timeout"
        assert await run_world.refunds_and_debits(actor) == (1, 1)
        assert await run_world.versions(run.track_id) == []
        (record,) = _internal_error_logs(logs)
        assert record["exc_type"] == "TimeoutError"
        assert SECRET not in json.dumps(logs, default=str)

    async def test_the_deadline_itself_is_still_a_run_timeout(self, run_world: RunWorld) -> None:
        import asyncio
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


class TestUsageOfUnusableReplies:
    """Invariant 26: tokens of a paid-for but unusable reply are counted on the run."""

    async def test_a_truncated_reply_is_counted(self, run_world: RunWorld) -> None:
        _, run = await _drive_one(run_world, FakeLlm(LlmOutputTruncated("cut", usage=PAID)))
        assert (run.status, run.error_code) == ("failed", "llm_output_truncated")
        assert (run.llm_calls, run.prompt_tokens, run.completion_tokens) == (1, 40, 9)

    async def test_every_undecodable_reply_of_a_failed_run_is_counted(
        self, run_world: RunWorld
    ) -> None:
        bad = LlmOutputInvalid("garbled", usage=PAID)
        _, run = await _drive_one(run_world, FakeLlm(bad, bad, bad))
        assert (run.status, run.error_code) == ("failed", "llm_output_invalid")
        assert (run.llm_calls, run.prompt_tokens, run.completion_tokens) == (3, 120, 27)

    async def test_a_re_asked_reply_is_counted_with_the_one_that_succeeded(
        self, run_world: RunWorld
    ) -> None:
        llm = FakeLlm(LlmOutputInvalid("garbled", usage=PAID), make_draft())  # the good one: 100/50
        _, run = await _drive_one(run_world, llm)
        assert run.status == "succeeded"
        assert (run.llm_calls, run.prompt_tokens, run.completion_tokens) == (2, 140, 59)

    async def test_an_error_without_usage_adds_nothing(self, run_world: RunWorld) -> None:
        _, run = await _drive_one(run_world, FakeLlm(LlmOutputTruncated("cut")))
        assert (run.llm_calls, run.prompt_tokens, run.completion_tokens) == (1, 0, 0)


def _adapter_over(*replies: httpx.Response) -> OpenAICompatibleLlmAdapter:
    script = list(replies)

    def handler(_request: httpx.Request) -> httpx.Response:
        return script.pop(0)

    entry = LlmModelEntry(
        base_url="https://llm.example.test/v1", api_key_env="UNUSED", model="test-model"
    )
    return OpenAICompatibleLlmAdapter(
        entry,
        SecretStr("sk-test"),
        httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        model_key="grok",
    )


def _reply(content: str, *, finish: str = "stop", usage: dict[str, int] | None = None) -> Any:
    payload: dict[str, Any] = {
        "model": "served",
        "choices": [{"finish_reason": finish, "message": {"content": content}}],
    }
    if usage is not None:
        payload["usage"] = usage
    return httpx.Response(200, json=payload)


class TestUsageThroughTheRealAdapter:
    """The same invariant end to end: HTTP ``200`` replies the adapter cannot use."""

    USAGE: ClassVar[dict[str, int]] = {"prompt_tokens": 31, "completion_tokens": 13}

    async def test_a_truncated_200_reply_shows_up_in_the_runs_tokens(
        self, run_world: RunWorld
    ) -> None:
        adapter = _adapter_over(_reply('{"title": "cut', finish="length", usage=self.USAGE))
        _, run = await _drive_one(run_world, adapter)
        assert (run.status, run.error_code) == ("failed", "llm_output_truncated")
        assert (run.llm_calls, run.prompt_tokens, run.completion_tokens) == (1, 31, 13)

    async def test_an_undecodable_200_reply_shows_up_in_the_runs_tokens(
        self, run_world: RunWorld
    ) -> None:
        adapter = _adapter_over(
            _reply("not json", usage=self.USAGE),
            _reply(json.dumps({"title": 1}), usage=self.USAGE),
            _reply("", usage=self.USAGE),
        )
        _, run = await _drive_one(run_world, adapter)
        assert (run.status, run.error_code) == ("failed", "llm_output_invalid")
        assert (run.llm_calls, run.prompt_tokens, run.completion_tokens) == (3, 93, 39)

    async def test_a_re_asked_200_reply_is_counted_with_the_good_one(
        self, run_world: RunWorld
    ) -> None:
        good = msgspec.json.encode(make_draft()).decode()
        adapter = _adapter_over(
            _reply("not json", usage=self.USAGE),
            _reply(good, usage={"prompt_tokens": 50, "completion_tokens": 20}),
        )
        _, run = await _drive_one(run_world, adapter)
        assert run.status == "succeeded"
        assert (run.llm_calls, run.prompt_tokens, run.completion_tokens) == (2, 81, 33)
