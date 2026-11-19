"""One review of a built lyrics document: hard vs soft, line-level vs structural (C7, C9)."""

from __future__ import annotations

from typing import Any

import pytest

from src.studio.documents import LyricLine, LyricsDocument, LyricsSection, SyllableRange
from src.studio.errors import DocumentIssue
from src.studio.planner.lyric_checks import (
    AVOID_WORD_USED,
    SYLLABLE_RANGE_MISSED,
    LineProblem,
    LineRef,
)
from src.studio.planner.lyrics_review import Findings, review
from src.studio.planner.syllables import counter_for
from tests.studio_lyrics_support import PLAN, S1, S2, S4, doc

pytestmark = pytest.mark.unit

RANGE = SyllableRange(min=4, max=6)
FOUR = "la la la la"


def _review(
    document: LyricsDocument, *, structural: tuple[DocumentIssue, ...] = (), **kwargs: Any
) -> Findings:
    return review(
        document,
        PLAN,
        structural=structural,
        counter=counter_for("en"),
        syllables=kwargs.pop("syllables", RANGE),
        avoid=kwargs.pop("avoid", ()),
        **kwargs,
    )


def _built(*sections: tuple[object, list[str]]) -> LyricsDocument:
    """A document as the model's reply builds it: unvalidated."""
    return LyricsDocument(
        language="en",
        sections=tuple(
            LyricsSection(section_id=sid, lines=tuple(LyricLine(text=t) for t in lines))  # type: ignore[arg-type]
            for sid, lines in sections
        ),
    )


def test_a_clean_document_has_nothing_to_fix_and_carries_the_normalized_lyrics() -> None:
    findings = _review(doc((S1, [FOUR]), (S2, [FOUR])))
    assert findings.clean
    assert findings.lyrics is not None
    assert findings.report is not None
    assert (findings.hard, findings.structural, findings.problems) == ((), (), ())


def test_soft_problems_are_listed_as_lines_to_replace_with_plan_indexes() -> None:
    findings = _review(doc((S1, [FOUR]), (S4, ["la", "baby baby baby baby"])), avoid=("baby",))
    assert not findings.clean
    assert findings.hard == ()
    assert findings.lyrics is not None  # a soft miss does not stop the document being valid
    assert findings.problems == (
        LineProblem(plan_index=3, line=0, code=SYLLABLE_RANGE_MISSED, count=1),
        LineProblem(plan_index=3, line=1, code=SYLLABLE_RANGE_MISSED, count=8),
        LineProblem(plan_index=3, line=1, code=AVOID_WORD_USED),
    )


def test_a_hard_line_issue_is_replaced_line_by_line_and_blocks_the_soft_checks() -> None:
    findings = _review(
        _built(
            (S1, [FOUR]),
            (
                S2,
                [FOUR, "oh [no] no"],
            ),
        ),
        syllables=RANGE,
    )
    assert findings.lyrics is None
    assert findings.report is None  # the soft checks wait for a valid document
    assert findings.structural == ()  # one line can fix it
    assert findings.hard == (DocumentIssue("sections[1].lines[1].text", "reserved_characters"),)
    assert findings.problems == (LineProblem(plan_index=1, line=1, code="reserved_characters"),)


def test_a_line_with_two_problems_is_listed_once_per_code() -> None:
    document = LyricsDocument(
        language="en",
        sections=(
            LyricsSection(
                section_id=S1,
                lines=(LyricLine(text="a [b]", delivery=("x", "x", "[y]")),),
            ),
        ),
    )
    findings = _review(document)
    codes = {p.code for p in findings.problems}
    assert codes == {"reserved_characters", "duplicate_tag"}
    assert {(p.plan_index, p.line) for p in findings.problems} == {(0, 0)}
    assert findings.structural == ()


def test_a_section_over_the_line_limit_is_structural_and_not_a_line_problem() -> None:
    findings = _review(_built((S1, [FOUR] * 65)))
    assert findings.problems == ()
    assert DocumentIssue("sections[0].lines", "too_many_items") in findings.structural
    assert findings.structural == findings.hard


def test_an_issue_found_while_building_is_hard_structural_and_hides_the_soft_checks() -> None:
    missing = DocumentIssue("sections[3]", "missing_section_lines")
    findings = _review(doc((S1, ["la"]), (S2, [FOUR])), structural=(missing,))
    assert findings.hard == (missing,)
    assert findings.structural == (missing,)
    assert findings.lyrics is None  # the document is valid, but the reply is not
    assert findings.problems == ()  # and the short line is not mentioned yet
    assert not findings.clean


def test_document_paths_are_readdressed_to_the_plans_sections() -> None:
    # S2 and S4 are document sections 0 and 1, plan sections 1 and 3
    findings = _review(_built((S2, [FOUR]), (S4, [FOUR, "x [y]"])))
    assert findings.hard == (DocumentIssue("sections[3].lines[1].text", "reserved_characters"),)


def test_only_restricts_the_soft_checks_to_the_changed_lines() -> None:
    document = doc((S1, ["la", FOUR]), (S2, ["la"]))
    findings = _review(document, only={LineRef(S2, 0)})
    assert [(p.plan_index, p.line) for p in findings.problems] == [(1, 0)]


def test_a_range_misses_only_when_asked_for() -> None:
    findings = _review(doc((S1, ["la"])), syllables=None)
    assert findings.clean
