"""``validate_plan``: a drafted plan gets the checks a user-submitted plan gets (invariant 8)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from src.studio.planner.drafts import PlanDraft
from src.studio.planner.mapping import plan_from_draft
from src.studio.planner.validate import validate_plan
from tests.studio_llm import make_draft, make_section_draft

if TYPE_CHECKING:
    from src.studio.documents import PlanDocument
    from src.studio.errors import DocumentIssue

pytestmark = pytest.mark.unit


def _validate(draft: PlanDraft) -> tuple[PlanDocument | None, tuple[DocumentIssue, ...]]:
    return validate_plan(plan_from_draft(draft, brief="a brief"))


def _codes(issues: tuple[DocumentIssue, ...]) -> set[tuple[str, str]]:
    return {(i.path, i.code) for i in issues}


def test_a_valid_draft_passes_and_is_normalized() -> None:
    plan, issues = _validate(make_draft(title="  Neon   Rain  "))
    assert issues == ()
    assert plan is not None
    assert plan.title == "Neon Rain"


def test_a_tempo_like_tag_is_reserved() -> None:
    plan, issues = _validate(make_draft(tags=["120 BPM"]))
    assert plan is None
    assert ("global_style.tags[0]", "reserved_tempo_tag") in _codes(issues)


def test_an_instrumental_descriptor_is_reserved() -> None:
    plan, issues = _validate(make_draft(make_section_draft("s1", delivery=["Instrumental"])))
    assert plan is None
    assert ("sections[0].delivery[0]", "reserved_descriptor") in _codes(issues)


def test_a_tag_in_both_lists_conflicts() -> None:
    _, issues = _validate(make_draft(tags=["rock"], avoid=["Rock"]))
    assert ("global_style.avoid[0]", "tag_conflict") in _codes(issues)


def test_a_title_that_is_too_long_after_normalization_is_flagged() -> None:
    # U+0958 expands to two code points under NFC: 120 of them are 240 > the 200 limit.
    _, issues = _validate(make_draft(title="क़" * 120))
    assert ("title", "too_long_after_normalization") in _codes(issues)


def test_too_many_sections_is_flagged_at_the_array() -> None:
    sections = [make_section_draft(f"s{i}") for i in range(65)]
    _, issues = _validate(make_draft(*sections))
    assert ("sections", "too_many_items") in _codes(issues)


def test_too_many_tags_is_flagged() -> None:
    _, issues = _validate(make_draft(tags=[f"tag {i}" for i in range(41)]))
    assert ("global_style.tags", "too_many_items") in _codes(issues)


@pytest.mark.parametrize("tempo", [9000, 10, -5])
def test_a_tempo_outside_the_wire_range_is_flagged(tempo: int) -> None:
    _, issues = _validate(make_draft(tempo_bpm=tempo))
    assert ("tempo_bpm", "out_of_range") in _codes(issues)


@pytest.mark.parametrize("seconds", [0, -1, 601])
def test_a_duration_outside_the_wire_range_is_flagged(seconds: int) -> None:
    _, issues = _validate(make_draft(make_section_draft("s1", duration_seconds=seconds)))
    assert ("sections[0].duration_seconds", "out_of_range") in _codes(issues)


def test_the_total_duration_cap_applies() -> None:
    sections = [make_section_draft(f"s{i}", duration_seconds=600) for i in range(4)]
    _, issues = _validate(make_draft(*sections))
    assert ("sections", "total_duration_exceeded") in _codes(issues)


def test_every_issue_is_collected_not_just_the_first() -> None:
    _, issues = _validate(make_draft(title="  ", tags=["80 BPM"], tempo_bpm=9000))
    assert {i.code for i in issues} >= {
        "empty_after_normalization",
        "reserved_tempo_tag",
        "out_of_range",
    }


def test_issues_never_contain_plan_text() -> None:
    _, issues = _validate(make_draft(title="SECRET-TITLE", tags=["SECRET-TAG 120 BPM"]))
    for issue in issues:
        assert "SECRET" not in issue.path + issue.code


def test_an_empty_plan_is_valid() -> None:
    draft = PlanDraft(
        refusal=None,
        title="Only a title",
        tempo_bpm=None,
        instrumental=True,
        tags=[],
        avoid=[],
        sections=[],
    )
    plan, issues = validate_plan(plan_from_draft(draft, brief="b"))
    assert issues == ()
    assert plan is not None
    assert plan.sections == ()
