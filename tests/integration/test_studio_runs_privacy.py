"""Privacy of planner runs (invariant 13).

The brief, the artist's persona, the project concept, every prompt and every model output may reach
the LLM provider and the owner's own rows - and nowhere else: not a log record (success or
failure), not ``error_detail``, not ``RunOut``. LangSmith / LangChain tracing is off and no tracing
client is constructed.
"""

from __future__ import annotations

import json
import logging
import os
from typing import TYPE_CHECKING, Any

import httpx
import msgspec
import pytest

from src.core.enums import StudioRunStage
from src.studio.llm.factory import LlmClientFactory
from src.studio.llm.port import LlmOutputInvalid, LlmUnavailable
from src.studio.services.mappers import run_out
from tests.studio_llm import FakeLlm, make_draft, make_section_draft
from tests.studio_runs_support import PRODUCT

if TYPE_CHECKING:
    from tests.studio_runs_support import RunWorld

BRIEF = "BRIEF-SECRET-rain on neon streets"
PERSONA = "SECRET-PERSONA"
CONCEPT = "SECRET-CONCEPT"
OUTPUT = "MODEL-OUTPUT-SECRET"
SECRETS = (BRIEF, PERSONA, CONCEPT, OUTPUT, "neon noir", "SECRET-KEY-VALUE")


def _records_text(caplog: pytest.LogCaptureFixture) -> str:
    """Everything every log record would show: its message and its formatted exception."""
    parts: list[str] = []
    for record in caplog.records:
        parts.append(record.getMessage())
        if record.exc_info:
            parts.append(logging.Formatter().formatException(record.exc_info))
    return "\n".join(parts)


def _assert_clean(text: str) -> None:
    for secret in SECRETS:
        assert secret not in text, secret


async def _world_with_artist(world: RunWorld) -> tuple[Any, Any]:
    actor = await world.add_actor(artist=True)
    track_id = await world.new_track(actor, "Night Drive")
    return actor, track_id


class TestLogs:
    @pytest.fixture(autouse=True)
    def _capture_everything(self, caplog: pytest.LogCaptureFixture) -> None:
        caplog.set_level(logging.DEBUG)

    async def test_a_successful_run_logs_no_text(
        self, run_world: RunWorld, caplog: pytest.LogCaptureFixture
    ) -> None:
        actor, track_id = await _world_with_artist(run_world)
        await run_world.submit(actor, track_id, brief=BRIEF)
        await run_world.drive(
            FakeLlm(make_draft(tags=["120 BPM"], title=OUTPUT), make_draft(title=OUTPUT))
        )

        text = _records_text(caplog)
        assert "studio.run.finished" in text  # the run was logged ...
        _assert_clean(text)  # ... without any of its text

    @pytest.mark.parametrize(
        "script",
        [
            [LlmUnavailable("provider says: " + BRIEF)],
            [LlmOutputInvalid("garbled: " + OUTPUT)] * 3,
            [RuntimeError(f"unexpected {BRIEF} {PERSONA} {OUTPUT}")],
            [make_draft(tags=["120 BPM"], title=OUTPUT)] * 3,
        ],
        ids=["unavailable", "invalid-output", "internal-error", "plan-invalid"],
    )
    async def test_a_failed_run_logs_no_text(
        self, run_world: RunWorld, caplog: pytest.LogCaptureFixture, script: list[Any]
    ) -> None:
        actor, track_id = await _world_with_artist(run_world)
        write = await run_world.submit(actor, track_id, brief=BRIEF)
        await run_world.drive(FakeLlm(*script))

        run = await run_world.run(write.run.id)
        assert run.status == "failed"
        text = _records_text(caplog)
        assert "studio.run.finished" in text
        # The exception *messages* above are deliberately full of secrets: a port error would
        # never carry any, and an unexpected one must not be logged with its message.
        for secret in (BRIEF, PERSONA, OUTPUT):
            assert secret not in text, secret
        assert BRIEF not in str(run.error_detail)
        assert OUTPUT not in str(run.error_detail)

    async def test_the_sweep_and_the_claim_log_no_text(
        self, run_world: RunWorld, caplog: pytest.LogCaptureFixture
    ) -> None:
        actor, track_id = await _world_with_artist(run_world)
        write = await run_world.submit(actor, track_id, brief=BRIEF)
        executor = run_world.executor(FakeLlm())
        await executor.claim(1)
        await run_world.expire_lease(write.run.id)
        await executor.sweep()
        _assert_clean(_records_text(caplog))

    async def test_a_claim_failure_logs_no_text(
        self, run_world: RunWorld, caplog: pytest.LogCaptureFixture
    ) -> None:
        actor, track_id = await _world_with_artist(run_world)
        await run_world.submit(actor, track_id, brief=BRIEF)

        class Explodes:
            async def for_run(self, request: Any, *, session: Any) -> Any:  # noqa: ARG002
                raise RuntimeError(f"context failed for {PERSONA} {CONCEPT}")

        await run_world.executor(FakeLlm(), provider=Explodes()).claim(1)
        text = _records_text(caplog)
        assert "studio.run.claim_failed" in text
        _assert_clean(text)

    async def test_the_submit_and_cancel_log_no_text(
        self, run_world: RunWorld, caplog: pytest.LogCaptureFixture
    ) -> None:
        actor, track_id = await _world_with_artist(run_world)
        write = await run_world.submit(actor, track_id, brief=BRIEF)
        async with run_world.factory() as session:
            await run_world.service(session).cancel(
                write.run.id, user_id=actor.user_id, product_id=PRODUCT
            )
            await session.commit()
        _assert_clean(_records_text(caplog))


class TestRows:
    async def test_run_out_and_error_detail_hold_no_text(self, run_world: RunWorld) -> None:
        actor, track_id = await _world_with_artist(run_world)
        write = await run_world.submit(actor, track_id, brief=BRIEF)
        bad = make_draft(tags=["120 BPM"], title=OUTPUT, avoid=[OUTPUT])
        await run_world.drive(FakeLlm(bad, bad, bad))

        run = await run_world.run(write.run.id)
        assert run.error_code == "plan_invalid"
        wire = msgspec.json.encode(run_out(run)).decode()
        _assert_clean(wire)
        _assert_clean(json.dumps(run.error_detail))
        for field in ("request", "context_snapshot", "error_detail"):
            assert field not in json.loads(wire) or field == "error_detail"

    async def test_the_owners_own_row_does_keep_what_the_run_was_given(
        self, run_world: RunWorld
    ) -> None:
        """The snapshot is the point of the feature: what the model saw, frozen for the owner."""
        actor, track_id = await _world_with_artist(run_world)
        write = await run_world.submit(actor, track_id, brief=BRIEF)
        await run_world.executor(FakeLlm()).claim(1)
        run = await run_world.run(write.run.id)
        assert run.request["brief"] == BRIEF
        assert run.context_snapshot is not None
        assert PERSONA in run.context_snapshot["artist"]["persona"]
        assert CONCEPT in run.context_snapshot["concept"]


class TestProvider:
    async def test_only_the_provider_receives_the_prompts(
        self, run_world: RunWorld, caplog: pytest.LogCaptureFixture
    ) -> None:
        """End to end through the real adapter: the text goes to the model and nowhere else."""
        caplog.set_level(logging.DEBUG)
        actor, track_id = await _world_with_artist(run_world)
        write = await run_world.submit(actor, track_id, brief=BRIEF)
        sent: list[dict[str, Any]] = []

        def handler(request: httpx.Request) -> httpx.Response:
            sent.append(json.loads(request.content))
            reply = msgspec.json.encode(make_draft(title=OUTPUT)).decode()
            return httpx.Response(
                200,
                json={
                    "model": "served",
                    "choices": [{"finish_reason": "stop", "message": {"content": reply}}],
                    "usage": {"prompt_tokens": 5, "completion_tokens": 6},
                },
            )

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        executor = run_world.executor(FakeLlm())
        executor._llm_factory = LlmClientFactory(run_world.registry, client)
        for claimed in await executor.claim(1):
            await executor.execute(claimed)
        await client.aclose()

        run = await run_world.run(write.run.id)
        assert (run.status, run.llm_calls, run.prompt_tokens, run.completion_tokens) == (
            "succeeded",
            1,
            5,
            6,
        )
        (body,) = sent
        prompt = json.dumps(body["messages"])
        assert BRIEF in prompt  # the provider is the one party that must receive it
        assert PERSONA in prompt
        assert body["response_format"]["json_schema"]["strict"] is True
        _assert_clean(_records_text(caplog))


class TestTracing:
    async def test_tracing_is_off_and_no_tracing_client_is_built(
        self, run_world: RunWorld, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import langsmith

        constructed: list[object] = []
        real_init = langsmith.Client.__init__

        def spy(self: Any, *args: Any, **kwargs: Any) -> None:
            constructed.append(self)
            real_init(self, *args, **kwargs)

        monkeypatch.setattr(langsmith.Client, "__init__", spy)
        actor, track_id = await _world_with_artist(run_world)
        await run_world.submit(actor, track_id, brief=BRIEF)
        await run_world.drive(FakeLlm(make_draft(tags=["120 BPM"]), make_draft()))

        assert constructed == []
        assert os.environ["LANGSMITH_TRACING"] == "false"
        assert os.environ["LANGCHAIN_TRACING_V2"] == "false"
        from langsmith.utils import tracing_is_enabled

        assert tracing_is_enabled() is False

    async def test_the_stage_callback_sees_every_stage_without_leaking_it_anywhere(
        self, run_world: RunWorld
    ) -> None:
        actor, track_id = await _world_with_artist(run_world)
        await run_world.submit(actor, track_id, brief=BRIEF)
        await run_world.drive(FakeLlm(make_draft(make_section_draft("s1"))))
        stages = {e.stage for e in run_world.bus.run_events}
        assert StudioRunStage.SAVING.value in stages
        for event in run_world.bus.run_events:
            _assert_clean(repr(event))
