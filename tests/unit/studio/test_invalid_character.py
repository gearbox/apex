"""N3: text the **server** builds is checked by the studio's one walker (``invalid_character``).

It never passed through a request, so the API's own refusal of U+0000 never saw it. Entry points:
a plan built in code, lyrics built in code, a model's memory draft, and (``test_timeline.py``) a
timeline built in code, which is the aligner's path.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Annotated, Final
from uuid import UUID

import msgspec
import pytest
from msgspec import Meta, Struct

from src.core.enums import StudioRunKind
from src.core.enums import StudioRunStage as Stage
from src.core.enums import StudioSectionKind as K
from src.studio.chat.draft import MemoryDraftDraft, check_memory_draft
from src.studio.constraints import (
    INVALID_CHARACTER,
    TOO_LONG,
    collect_count_issues,
    collect_range_issues,
    collect_text_issues,
)
from src.studio.documents import LyricLine
from src.studio.planner.context import ArtistSnapshot, RunContextSnapshot
from src.studio.planner.graphs.lyrics import (
    LyricsDeps,
    compile_lyrics_graph,
    run_lyrics_graph,
)
from src.studio.planner.graphs.lyrics import initial_state as initial_lyrics_state
from src.studio.planner.graphs.plan import PlanDeps, compile_plan_graph, run_plan_graph
from src.studio.planner.graphs.plan import initial_state as initial_plan_state
from src.studio.planner.outcome import LyricsReady, PlanReady, RunRejected
from src.studio.run_failures import failure_of_outcome
from src.studio.run_requests import LyricsRunRequest, PlanRunRequest
from src.studio.validation import normalize_lyrics, normalize_plan, validate_built_document
from tests.llm_support import FakeLlm
from tests.studio_llm import (
    make_draft,
    make_line_patch,
    make_lyrics_draft,
    make_lyrics_patch,
    make_section_draft,
    section_lines,
)
from tests.studio_lyrics_support import CONTEXT, PLAN
from tests.studio_support import make_lyrics, make_plan, make_section

if TYPE_CHECKING:
    from src.studio.errors import DocumentIssue
    from src.studio.planner.drafts import LyricsDraft
    from src.studio.planner.graphs.lyrics import LyricsRun
    from src.studio.planner.graphs.plan import PlanRun

pytestmark = pytest.mark.unit

NUL: Final = "\x00"
SECRET: Final = "SECRET-TEXT"
BAD: Final = f"{SECRET}{NUL}tail"


def _pairs(issues: list[DocumentIssue] | tuple[DocumentIssue, ...]) -> list[tuple[str, str]]:
    return [(issue.path, issue.code) for issue in issues]


def _invalid(*paths: str) -> list[tuple[str, str]]:
    return [(path, INVALID_CHARACTER) for path in paths]


# ---------------------------------------------------------------------------
# The walker
# ---------------------------------------------------------------------------


class _Leaf(Struct, frozen=True, kw_only=True):
    free: str
    maybe: str | None = None


class _Shape(Struct, frozen=True, kw_only=True):
    unlimited: str = ""
    limited: Annotated[str, Meta(max_length=3)] = ""
    free_list: list[str] = []
    leaf: _Leaf | None = None
    leaves: tuple[_Leaf, ...] = ()
    patched: str | msgspec.UnsetType = msgspec.UNSET
    number: Annotated[int, Meta(ge=0)] = 0


def _text_issues(value: Struct, path: str = "") -> list[tuple[str, str]]:
    issues: list[DocumentIssue] = []
    collect_text_issues(value, issues, path)
    return _pairs(issues)


class TestTheWalker:
    def test_every_string_is_checked_with_or_without_a_declared_limit(self) -> None:
        shape = _Shape(
            unlimited=NUL,
            limited=NUL,
            free_list=["fine", BAD],
            leaf=_Leaf(free=BAD, maybe=BAD),
            leaves=(_Leaf(free="fine"), _Leaf(free=NUL)),
            patched=NUL,
        )

        assert _text_issues(shape) == _invalid(
            "unlimited",
            "limited",
            "free_list[1]",
            "leaf.free",
            "leaf.maybe",
            "leaves[1].free",
            "patched",
        )

    def test_clean_unset_and_none_values_pass(self) -> None:
        assert _text_issues(_Shape(unlimited="a\n\t\x01", leaf=_Leaf(free=""))) == []
        assert _text_issues(_Shape()) == []

    def test_a_value_that_is_also_too_long_gets_both_codes(self) -> None:
        assert _text_issues(_Shape(limited=f"abc{NUL}")) == [
            ("limited", INVALID_CHARACTER),
            ("limited", TOO_LONG),
        ]

    def test_the_path_is_the_wire_path_and_never_carries_text(self) -> None:
        issues = _text_issues(_Shape(free_list=[BAD]), "content")

        assert issues == _invalid("content.free_list[0]")
        assert SECRET not in repr(issues)

    def test_the_count_and_range_checks_do_not_report_it(self) -> None:
        issues: list[DocumentIssue] = []
        collect_count_issues(_Shape(unlimited=NUL, free_list=[NUL]), issues)
        collect_range_issues(_Shape(unlimited=NUL, free_list=[NUL]), issues)

        assert issues == []


# ---------------------------------------------------------------------------
# The entry points
# ---------------------------------------------------------------------------


def test_a_plan_built_in_code() -> None:
    plan = make_plan(
        make_section(K.VERSE, label=BAD, tags=[BAD], delivery=[BAD], notes=BAD),
        title=BAD,
        idea=BAD,
        references=[BAD],
        tags=["fine", BAD],
        avoid=[f"other{NUL}"],
    )

    normalized, issues = validate_built_document(plan, normalize_plan)

    assert normalized is None
    assert sorted(_pairs(issues)) == sorted(
        _invalid(
            "title",
            "idea",
            "references[0]",
            "global_style.tags[1]",
            "global_style.avoid[0]",
            "sections[0].label",
            "sections[0].style.tags[0]",
            "sections[0].delivery[0]",
            "sections[0].notes",
        )
    )
    assert SECRET not in repr(issues)


def test_lyrics_built_in_code() -> None:
    verse = make_section(K.VERSE)
    plan = make_plan(verse)
    lyrics = make_lyrics((verse.id, ["fine", LyricLine(text=BAD, delivery=(f"soft{NUL}",))]))

    normalized, issues = validate_built_document(
        lyrics, lambda candidate: normalize_lyrics(candidate, plan)
    )

    assert normalized is None
    assert _pairs(issues) == _invalid(
        "sections[0].lines[1].text", "sections[0].lines[1].delivery[0]"
    )
    assert SECRET not in repr(issues)


def test_a_memory_draft() -> None:
    draft = MemoryDraftDraft(refusal=None, title=BAD, body=BAD, themes=["fine", BAD], imagery=[BAD])

    content, issues = check_memory_draft(draft)

    assert content is None
    assert _pairs(issues) == _invalid("title", "body", "themes[1]", "imagery[0]")
    assert SECRET not in repr(issues)


# ---------------------------------------------------------------------------
# What it does to a planner run: asked again within the budget, then rejected
# ---------------------------------------------------------------------------

_PLAN_CONTEXT = RunContextSnapshot(
    track_title="Track",
    artist=ArtistSnapshot(
        id=UUID(int=1),
        name="Nova",
        persona=None,
        voice_notes=None,
        default_language=None,
        style_tags=(),
        global_avoid=(),
        aesthetic_references=(),
        updated_at=datetime(2026, 1, 1, tzinfo=UTC),
    ),
    concept=None,
    content_rating="sfw",
    plan_version_id=None,
    plan_content_hash=None,
    current_plan=None,
    lyrics_version_id=None,
    lyrics_content_hash=None,
)


async def _ignore_stage(_stage: Stage) -> None:
    return None


class TestAPlanRun:
    @staticmethod
    async def _run(llm: FakeLlm) -> PlanRun:
        return await run_plan_graph(
            compile_plan_graph(),
            initial_plan_state(_PLAN_CONTEXT, PlanRunRequest(brief="a brief")),
            PlanDeps(llm=llm, on_stage=_ignore_stage),
        )

    async def test_a_reply_with_a_nul_is_revised(self) -> None:
        llm = FakeLlm(make_draft(make_section_draft("s1", label=BAD)), make_draft())

        run = await self._run(llm)

        assert isinstance(run.outcome, PlanReady)
        assert llm.call_count == 2
        assert "sections[0].label" in llm.calls[1][-1].content
        assert INVALID_CHARACTER in llm.calls[1][-1].content

    async def test_a_reply_that_keeps_its_nul_fails_as_plan_invalid(self) -> None:
        bad = make_draft(title=BAD)
        llm = FakeLlm(bad, bad, bad)

        run = await self._run(llm)

        assert isinstance(run.outcome, RunRejected)
        assert run.outcome.kind is StudioRunKind.PLAN
        failure = failure_of_outcome(run.outcome)
        assert (failure.code, failure.detail) == (
            "plan_invalid",
            {"issues": [{"path": "title", "code": INVALID_CHARACTER}]},
        )
        assert llm.call_count == 3


def _sung(second_line: str) -> LyricsDraft:
    """A full reply for ``PLAN``; ``second_line`` is the chorus's second line."""
    return make_lyrics_draft(
        section_lines("s1", "one line", "two line"),
        section_lines("s2", "one line", second_line),
        section_lines("s4", "one line", "two line"),
    )


class TestALyricsRun:
    @staticmethod
    async def _run(llm: FakeLlm) -> LyricsRun:
        state = initial_lyrics_state(
            CONTEXT, LyricsRunRequest(), PLAN, language="en", line_syllables=None
        )
        return await run_lyrics_graph(
            compile_lyrics_graph(), state, LyricsDeps(llm=llm, on_stage=_ignore_stage)
        )

    async def test_the_one_line_with_a_nul_is_replaced(self) -> None:
        llm = FakeLlm(_sung(BAD), make_lyrics_patch(make_line_patch("s2", 2, "clean again")))

        run = await self._run(llm)

        assert isinstance(run.outcome, LyricsReady)
        assert [line.text for line in run.outcome.lyrics.sections[1].lines] == [
            "one line",
            "clean again",
        ]
        assert llm.call_count == 2
        assert "s2 line 2" in llm.calls[1][-1].content

    async def test_a_line_that_keeps_its_nul_fails_as_lyrics_invalid(self) -> None:
        stubborn = make_lyrics_patch(make_line_patch("s2", 2, BAD))
        llm = FakeLlm(_sung(BAD), stubborn, stubborn, stubborn, stubborn)

        run = await self._run(llm)

        assert isinstance(run.outcome, RunRejected)
        failure = failure_of_outcome(run.outcome)
        assert (failure.code, failure.detail) == (
            "lyrics_invalid",
            {"issues": [{"path": "sections[1].lines[1].text", "code": INVALID_CHARACTER}]},
        )
