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
from uuid import UUID

import httpx
import msgspec
import pytest

from src.core.enums import StudioRunStage
from src.llm.factory import LlmClientFactory
from src.llm.port import LlmOutputInvalid, LlmUnavailable
from src.studio.services.mappers import run_out
from tests.llm_support import FakeLlm
from tests.studio_llm import make_draft, make_section_draft
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
            [LlmUnavailable(f"provider says: {BRIEF}")],
            [LlmOutputInvalid(f"garbled: {OUTPUT}")] * 3,
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

        assert not constructed
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


# -- S1-P2a-B: lyrics and revise_lyrics (invariant 18) ------------------------------------------------
#
# The brief, the instruction, the plan's notes, every lyric line, every avoid word, every prompt and
# every model output reach the provider and the owner's own rows - and nothing else: no log record,
# no ``error_detail``, no warning, no ``RunOut``.

LYRICS_BRIEF = "LYRICS-BRIEF-SECRET about leaving"
INSTRUCTION = "INSTRUCTION-SECRET make it warmer"
NOTES = "PLAN-NOTES-SECRET keep it small"
LINE = "LYRIC-LINE-SECRET walking home"
AVOID = "AVOIDWORD-SECRET"
MODEL_LINE = "MODEL-OUTPUT-SECRET [bracket] line"
LYRICS_SECRETS = (
    LYRICS_BRIEF,
    INSTRUCTION,
    NOTES,
    LINE,
    AVOID,
    MODEL_LINE,
    "MODEL-OUTPUT-SECRET",
    PERSONA,
)
PROMPT_FRAGMENTS = ("You write song lyrics", "You revise parts of existing song lyrics", "<plan>")


def _assert_lyrics_clean(text: str) -> None:
    for secret in (*LYRICS_SECRETS, *PROMPT_FRAGMENTS):
        assert secret not in text, secret


async def _lyrics_world(run_world: RunWorld) -> tuple[Any, Any, Any, Any]:
    from src.core.enums import StudioSectionKind as K
    from tests.studio_lyrics_support import doc
    from tests.studio_support import make_plan, make_section

    s1, s2 = UUID(int=0x2001), UUID(int=0x2002)
    plan = make_plan(
        make_section(K.VERSE, section_id=s1, notes=NOTES),
        make_section(K.CHORUS, section_id=s2),
        title="Night Drive",
    )
    actor = await run_world.add_actor(artist=True)
    track_id = await run_world.new_track(actor, "Night Drive", plan)
    plan_id = (await run_world.track_row(track_id)).current_plan_version_id
    assert plan_id is not None
    lyrics_id = await run_world.create_lyrics(
        actor,
        track_id,
        doc((s1, [LINE, "second line"]), (s2, ["third line"]), plan=plan),
        plan_version_id=plan_id,
    )
    return actor, track_id, (s1, s2), lyrics_id


def _lyrics_request(**kwargs: Any) -> Any:
    from src.studio.run_requests import LyricsRunRequest

    return LyricsRunRequest(language="en", brief=LYRICS_BRIEF, avoid_words=(AVOID,), **kwargs)


def _revise_request(sections: tuple[Any, Any], **kwargs: Any) -> Any:
    from src.studio.run_requests import LineTarget, ReviseLyricsRunRequest, ReviseTargets

    return ReviseLyricsRunRequest(
        instruction=INSTRUCTION,
        targets=ReviseTargets(
            sections=(sections[1],), lines=(LineTarget(section_id=sections[0], line=0),)
        ),
        avoid_words=(AVOID,),
        **kwargs,
    )


def _good_lyrics() -> Any:
    from tests.studio_llm import make_lyrics_draft, section_lines

    return make_lyrics_draft(
        section_lines("s1", "fine line one", "fine line two"), section_lines("s2", "fine chorus")
    )


class TestLyricsLogs:
    @pytest.fixture(autouse=True)
    def _capture_everything(self, caplog: pytest.LogCaptureFixture) -> None:
        caplog.set_level(logging.DEBUG)

    async def test_a_successful_lyrics_run_with_a_patch_and_warnings_logs_no_text(
        self, run_world: RunWorld, caplog: pytest.LogCaptureFixture
    ) -> None:
        from tests.studio_llm import (
            make_line_patch,
            make_lyrics_draft,
            make_lyrics_patch,
            section_lines,
        )

        actor, track_id, _, _ = await _lyrics_world(run_world)
        write = await run_world.submit(actor, track_id, request=_lyrics_request())
        flawed = make_lyrics_draft(
            section_lines("s1", MODEL_LINE, f"oh {AVOID.lower()} no"),
            section_lines("s2", "ok chorus"),
        )
        keep = make_lyrics_patch(make_line_patch("s1", 2, f"oh {AVOID.lower()} no"))
        fixed = make_lyrics_patch(make_line_patch("s1", 1, "fine line"))
        await run_world.drive(FakeLlm(flawed, fixed, keep, keep))

        run = await run_world.run(write.run.id)
        assert run.status == "succeeded"
        assert any(w["code"] == "avoid_word_used" for w in run.warnings)
        text = _records_text(caplog)
        assert "studio.run.finished" in text
        _assert_lyrics_clean(text)
        _assert_lyrics_clean(json.dumps(run.warnings))
        _assert_lyrics_clean(msgspec.json.encode(run_out(run)).decode())

    async def test_a_failed_lyrics_run_logs_no_text_and_reports_paths_only(
        self, run_world: RunWorld, caplog: pytest.LogCaptureFixture
    ) -> None:
        from tests.studio_llm import (
            make_line_patch,
            make_lyrics_draft,
            make_lyrics_patch,
            section_lines,
        )

        actor, track_id, _, _ = await _lyrics_world(run_world)
        write = await run_world.submit(actor, track_id, request=_lyrics_request())
        bad = make_lyrics_draft(section_lines("s1", MODEL_LINE, "x y"), section_lines("s2", "z"))
        keep = make_lyrics_patch(make_line_patch("s1", 1, MODEL_LINE))
        await run_world.drive(FakeLlm(bad, keep, keep, keep))

        run = await run_world.run(write.run.id)
        assert (run.status, run.error_code) == ("failed", "lyrics_invalid")
        _assert_lyrics_clean(_records_text(caplog))
        _assert_lyrics_clean(json.dumps(run.error_detail))
        _assert_lyrics_clean(msgspec.json.encode(run_out(run)).decode())

    async def test_a_revision_logs_no_text_either_way(
        self, run_world: RunWorld, caplog: pytest.LogCaptureFixture
    ) -> None:
        from tests.studio_llm import (
            make_line_patch,
            make_lyrics_patch,
            make_lyrics_revision,
            section_lines,
        )

        actor, track_id, sections, _ = await _lyrics_world(run_world)
        ok = await run_world.submit(actor, track_id, request=_revise_request(sections))
        await run_world.drive(
            FakeLlm(
                make_lyrics_revision(
                    sections=[section_lines("s2", "better chorus")],
                    lines=[make_line_patch("s1", 1, "better opener")],
                )
            )
        )
        assert (await run_world.run(ok.run.id)).status == "succeeded"

        bad = await run_world.submit(
            actor, track_id, request=_revise_request(sections), key="second"
        )
        flawed = make_lyrics_revision(
            sections=[section_lines("s2", MODEL_LINE)], lines=[make_line_patch("s1", 1, "fine")]
        )
        keep = make_lyrics_patch(make_line_patch("s2", 1, MODEL_LINE))
        await run_world.drive(FakeLlm(flawed, keep, keep))

        run = await run_world.run(bad.run.id)
        assert (run.status, run.error_code) == ("failed", "lyrics_invalid")
        _assert_lyrics_clean(_records_text(caplog))
        _assert_lyrics_clean(json.dumps(run.error_detail))

    @pytest.mark.parametrize("kind", ["lyrics", "revise_lyrics"])
    async def test_an_unexpected_error_is_logged_by_type_and_frames_only(
        self, run_world: RunWorld, caplog: pytest.LogCaptureFixture, kind: str
    ) -> None:
        actor, track_id, sections, _ = await _lyrics_world(run_world)
        request = _lyrics_request() if kind == "lyrics" else _revise_request(sections)
        write = await run_world.submit(actor, track_id, request=request)
        boom = RuntimeError(f"unexpected {LYRICS_BRIEF} {INSTRUCTION} {LINE} {NOTES} {AVOID}")
        await run_world.drive(FakeLlm(boom))

        run = await run_world.run(write.run.id)
        assert (run.status, run.error_code) == ("failed", "internal_error")
        text = _records_text(caplog)
        assert "studio.run.internal_error" in text
        _assert_lyrics_clean(text)

    @pytest.mark.parametrize(
        "script",
        [[LlmUnavailable(f"provider: {LYRICS_BRIEF}")], [LlmOutputInvalid(f"garbled: {LINE}")] * 5],
        ids=["unavailable", "invalid-output"],
    )
    async def test_port_failures_log_no_text(
        self, run_world: RunWorld, caplog: pytest.LogCaptureFixture, script: list[Any]
    ) -> None:
        actor, track_id, _, _ = await _lyrics_world(run_world)
        write = await run_world.submit(actor, track_id, request=_lyrics_request())
        await run_world.drive(FakeLlm(*script))
        assert (await run_world.run(write.run.id)).status == "failed"
        _assert_lyrics_clean(_records_text(caplog))

    async def test_a_claim_failure_logs_no_text(
        self, run_world: RunWorld, caplog: pytest.LogCaptureFixture
    ) -> None:
        actor, track_id, _, _ = await _lyrics_world(run_world)
        await run_world.submit(actor, track_id, request=_lyrics_request())

        class Explodes:
            async def for_run(self, request: Any, *, session: Any) -> Any:  # noqa: ARG002
                raise RuntimeError(f"context failed for {PERSONA} {NOTES}")

        await run_world.executor(FakeLlm(), provider=Explodes()).claim(1)
        text = _records_text(caplog)
        assert "studio.run.claim_failed" in text
        _assert_lyrics_clean(text)

    async def test_submit_and_cancel_log_no_text(
        self, run_world: RunWorld, caplog: pytest.LogCaptureFixture
    ) -> None:
        actor, track_id, sections, _ = await _lyrics_world(run_world)
        write = await run_world.submit(actor, track_id, request=_revise_request(sections))
        async with run_world.factory() as session:
            await run_world.service(session).cancel(
                write.run.id, user_id=actor.user_id, product_id=PRODUCT
            )
            await session.commit()
        _assert_lyrics_clean(_records_text(caplog))


class TestLyricsRows:
    async def test_the_snapshot_holds_ids_and_hashes_never_a_lyric_or_a_note(
        self, run_world: RunWorld
    ) -> None:
        actor, track_id, sections, _ = await _lyrics_world(run_world)
        write = await run_world.submit(actor, track_id, request=_revise_request(sections))
        await run_world.executor(FakeLlm()).claim(1)
        snapshot = (await run_world.run(write.run.id)).context_snapshot
        assert snapshot is not None
        flat = json.dumps(snapshot)
        for secret in (LINE, NOTES, INSTRUCTION, AVOID):
            assert secret not in flat, secret
        assert PERSONA in flat  # the artist is the one thing the snapshot has always held

    async def test_the_owners_own_request_row_does_keep_what_they_wrote(
        self, run_world: RunWorld
    ) -> None:
        actor, track_id, sections, _ = await _lyrics_world(run_world)
        write = await run_world.submit(actor, track_id, request=_revise_request(sections))
        row = await run_world.run(write.run.id)
        assert row.request["instruction"] == INSTRUCTION
        assert row.request["avoid_words"] == [AVOID]
        assert row.inputs is not None
        assert INSTRUCTION not in json.dumps(row.inputs)  # ids and a language only


class TestLyricsProvider:
    @pytest.mark.parametrize("kind", ["lyrics", "revise_lyrics"])
    async def test_only_the_provider_receives_the_prompts(
        self, run_world: RunWorld, caplog: pytest.LogCaptureFixture, kind: str
    ) -> None:
        """Through the real adapter: the text goes to the model, and nowhere else."""
        from tests.studio_llm import make_line_patch, make_lyrics_revision, section_lines

        caplog.set_level(logging.DEBUG)
        actor, track_id, sections, _ = await _lyrics_world(run_world)
        request = _lyrics_request() if kind == "lyrics" else _revise_request(sections)
        write = await run_world.submit(actor, track_id, request=request)
        sent: list[dict[str, Any]] = []
        reply_value = (
            _good_lyrics()
            if kind == "lyrics"
            else make_lyrics_revision(
                sections=[section_lines("s2", "better chorus")],
                lines=[make_line_patch("s1", 1, "better opener")],
            )
        )

        def handler(http_request: httpx.Request) -> httpx.Response:
            sent.append(json.loads(http_request.content))
            reply = msgspec.json.encode(reply_value).decode()
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
        assert (run.status, run.llm_calls) == ("succeeded", 1), run.error_code
        (body,) = sent
        prompt = json.dumps(body["messages"])
        # the provider is the one party that must receive them
        assert NOTES in prompt
        # the existing lyrics are sent for a revision - and only for a revision
        assert (LINE in prompt) is (kind == "revise_lyrics")
        assert (LYRICS_BRIEF if kind == "lyrics" else INSTRUCTION) in prompt
        assert AVOID in prompt
        assert body["response_format"]["json_schema"]["strict"] is True
        text = _records_text(caplog)
        assert "studio.run.finished" in text
        _assert_lyrics_clean(text)
