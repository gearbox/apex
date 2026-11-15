"""``plan_from_draft`` and ``draft_key_issues``: pure draft -> document mapping."""

from __future__ import annotations

import pytest

from src.core.enums import StudioSectionKind as K
from src.studio.errors import DocumentIssue
from src.studio.planner.mapping import draft_key_issues, plan_from_draft
from tests.studio_llm import make_draft, make_section_draft

pytestmark = pytest.mark.unit


def test_fields_map_across() -> None:
    draft = make_draft(
        make_section_draft(
            "s1",
            K.INTRO,
            label="Hush",
            duration_seconds=20,
            tags=["pads"],
            delivery=["whispered"],
            instrumental=True,
            notes="soft",
        ),
        title="T",
        tempo_bpm=98,
        instrumental=False,
        tags=["synthwave", "dreamy"],
        avoid=["autotune"],
    )
    plan = plan_from_draft(draft, brief="the brief")
    assert plan.title == "T"
    assert plan.tempo_bpm == 98
    assert plan.instrumental is False
    assert plan.global_style.tags == ("synthwave", "dreamy")
    assert plan.global_style.avoid == ("autotune",)
    section = plan.sections[0]
    assert (section.kind, section.label, section.duration_seconds) == (K.INTRO, "Hush", 20)
    assert section.style.tags == ("pads",)
    assert section.delivery == ("whispered",)
    assert section.instrumental is True
    assert section.notes == "soft"


def test_the_brief_becomes_the_idea_and_references_stay_empty() -> None:
    plan = plan_from_draft(make_draft(), brief="  my brief ")
    assert plan.idea == "  my brief "  # normalization is the validator's job, not the mapper's
    assert plan.references == ()
    assert all(section.style.avoid == () for section in plan.sections)


def test_the_server_assigns_distinct_ids() -> None:
    first = plan_from_draft(make_draft(), brief="b")
    second = plan_from_draft(make_draft(), brief="b")
    ids = [s.id for s in first.sections] + [s.id for s in second.sections]
    assert len(set(ids)) == 4


def test_valid_keys_have_no_issues() -> None:
    assert draft_key_issues(make_draft()) == ()


def test_a_duplicate_key_is_flagged_at_the_repeat() -> None:
    draft = make_draft(make_section_draft("s1"), make_section_draft("s2"), make_section_draft("s1"))
    assert draft_key_issues(draft) == (DocumentIssue("sections[2].key", "duplicate_section_key"),)


@pytest.mark.parametrize("key", ["", "   "])
def test_an_empty_key_is_flagged(key: str) -> None:
    draft = make_draft(make_section_draft(key))
    assert draft_key_issues(draft) == (DocumentIssue("sections[0].key", "empty_section_key"),)
