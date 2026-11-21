"""Lyrics drafts <-> documents (C8): keys, structural issues, patches, revisions, and the
properties the design stands on (invariants 8, 9, 11, 20)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from src.studio.documents import LyricLine, LyricsDocument, LyricsSection, SyllableRange
from src.studio.errors import DocumentIssue
from src.studio.planner.drafts import LyricsDraft
from src.studio.planner.errors import UnusableReply
from src.studio.planner.lyric_checks import LineRef
from src.studio.planner.lyrics_mapping import (
    DUPLICATE_SECTION_KEY,
    INSTRUMENTAL_SECTION_HAS_LINES,
    MISSING_SECTION_LINES,
    MISSING_TARGET,
    UNEXPECTED_TARGET,
    UNKNOWN_SECTION_KEY,
    apply_patch,
    apply_revision,
    draft_of,
    issues_in_plan_space,
    key_of,
    line_index,
    line_issue_ref,
    line_number,
    lyrics_from_draft,
    revision_draft_of,
    section_keys,
)
from src.studio.run_requests import LineTarget, ReviseTargets
from tests.studio_llm import (
    make_line_patch,
    make_lyrics_draft,
    make_lyrics_patch,
    make_lyrics_revision,
    section_lines,
)
from tests.studio_lyrics_support import PLAN, S1, S2, S3, S4, doc

if TYPE_CHECKING:
    from uuid import UUID

pytestmark = pytest.mark.unit

RANGE = SyllableRange(min=4, max=6)


def _texts(document: LyricsDocument, section_id: UUID) -> list[str]:
    section = next(s for s in document.sections if s.section_id == section_id)
    return [line.text for line in section.lines]


# -- numbering: the one conversion ------------------------------------------------------------


def test_keys_cover_every_plan_section_instrumental_ones_included() -> None:
    keys = section_keys(PLAN)
    assert list(keys) == ["s1", "s2", "s3", "s4"]
    assert [section.id for section in keys.values()] == [S1, S2, S3, S4]


def test_the_models_numbers_are_one_based_and_ours_are_zero_based() -> None:
    assert (key_of(0), line_number(0)) == ("s1", 1)
    assert line_index(1) == 0
    assert all(line_index(line_number(i)) == i for i in range(64))


def test_a_patch_for_line_one_changes_the_first_line_invariant_20() -> None:
    lyrics = doc((S1, ["a", "b", "c"]))
    patched = apply_patch(
        lyrics,
        make_lyrics_patch(make_line_patch("s1", 1, "first")),
        {LineRef(S1, 0)},
        PLAN,
    )
    assert _texts(patched, S1) == ["first", "b", "c"]


def test_a_patch_for_the_last_numbered_line_changes_the_last_index() -> None:
    lyrics = doc((S1, ["a", "b", "c"]))
    patched = apply_patch(
        lyrics, make_lyrics_patch(make_line_patch("s1", 3, "z")), {LineRef(S1, 2)}, PLAN
    )
    assert _texts(patched, S1) == ["a", "b", "z"]


# -- a whole draft ----------------------------------------------------------------------------


def test_sections_come_out_in_plan_order_whatever_order_the_model_used() -> None:
    draft = make_lyrics_draft(
        section_lines("s4", "four"), section_lines("s2", "two"), section_lines("s1", "one")
    )
    document, issues = lyrics_from_draft(draft, PLAN, language="en", line_syllables=RANGE)
    assert issues == ()
    assert [s.section_id for s in document.sections] == [S1, S2, S4]
    assert (document.language, document.line_syllables) == ("en", RANGE)


def test_keys_are_matched_without_regard_to_case_and_padding() -> None:
    draft = make_lyrics_draft(
        section_lines(" S1 ", "one"), section_lines("s2", "two"), section_lines("s4", "four")
    )
    _, issues = lyrics_from_draft(draft, PLAN, language="en")
    assert issues == ()


def test_the_structural_issues_use_plan_section_paths() -> None:
    draft = make_lyrics_draft(
        section_lines("s1", "one"),
        section_lines("s1", "again"),  # a repeated key
        section_lines("s9", "who"),  # no such section
        section_lines("s3", "sung over an instrumental section"),
        # s2 and s4 are vocal and missing
    )
    document, issues = lyrics_from_draft(draft, PLAN, language="en")
    assert set(issues) == {
        DocumentIssue("sections[0]", DUPLICATE_SECTION_KEY),
        DocumentIssue("sections", UNKNOWN_SECTION_KEY),
        DocumentIssue("sections[2]", INSTRUMENTAL_SECTION_HAS_LINES),
        DocumentIssue("sections[1]", MISSING_SECTION_LINES),
        DocumentIssue("sections[3]", MISSING_SECTION_LINES),
    }
    assert [s.section_id for s in document.sections] == [S1]  # the first s1 wins


def test_a_vocal_section_the_model_left_empty_never_reaches_the_document() -> None:
    draft = make_lyrics_draft(
        section_lines("s1"), section_lines("s2", "two"), section_lines("s4", "four")
    )
    document, issues = lyrics_from_draft(draft, PLAN, language="en")
    assert issues == (DocumentIssue("sections[0]", MISSING_SECTION_LINES),)
    assert S1 not in {s.section_id for s in document.sections}


def test_an_empty_entry_for_an_instrumental_section_is_harmless() -> None:
    draft = make_lyrics_draft(
        section_lines("s1", "one"),
        section_lines("s2", "two"),
        section_lines("s3"),
        section_lines("s4", "four"),
    )
    _, issues = lyrics_from_draft(draft, PLAN, language="en")
    assert issues == ()


def test_the_draft_of_a_document_round_trips_through_the_mapping() -> None:
    lyrics = doc((S1, ["a", LyricLine(text="b", delivery=("soft",))]), (S2, ["c"]), (S4, ["d"]))
    shown = draft_of(lyrics, PLAN)
    assert [s.key for s in shown.sections] == ["s1", "s2", "s4"]
    rebuilt, issues = lyrics_from_draft(shown, PLAN, language="en")
    assert (rebuilt, issues) == (lyrics, ())


# -- patches ----------------------------------------------------------------------------------


def test_a_patch_replaces_exactly_the_requested_lines_and_freezes_the_rest() -> None:
    lyrics = doc((S1, ["a", "b", "c"]), (S2, ["d", "e"]))
    patch = make_lyrics_patch(make_line_patch("s1", 2, "B", ["whispered"]))
    patched = apply_patch(lyrics, patch, {LineRef(S1, 1)}, PLAN)
    assert _texts(patched, S1) == ["a", "B", "c"]
    assert patched.sections[0].lines[1].delivery == ("whispered",)
    assert patched.sections[0].lines[0] is lyrics.sections[0].lines[0]  # the same object
    assert patched.sections[1] is lyrics.sections[1]


def test_a_requested_line_the_reply_leaves_out_stays_as_it_was() -> None:
    lyrics = doc((S1, ["a", "b"]))
    patched = apply_patch(
        lyrics,
        make_lyrics_patch(make_line_patch("s1", 1, "A")),
        {LineRef(S1, 0), LineRef(S1, 1)},
        PLAN,
    )
    assert _texts(patched, S1) == ["A", "b"]


def test_an_empty_patch_changes_nothing() -> None:
    lyrics = doc((S1, ["a"]))
    assert apply_patch(lyrics, make_lyrics_patch(), {LineRef(S1, 0)}, PLAN) is lyrics


@pytest.mark.parametrize(
    "patch",
    [
        make_lyrics_patch(make_line_patch("s1", 2, "not requested")),
        make_lyrics_patch(make_line_patch("s2", 1, "another section")),
        make_lyrics_patch(make_line_patch("s9", 1, "no such key")),
        make_lyrics_patch(make_line_patch("s1", 0, "line zero")),
        make_lyrics_patch(make_line_patch("s1", 7, "past the end")),
        make_lyrics_patch(make_line_patch("s1", 1, "once"), make_line_patch("s1", 1, "twice")),
    ],
)
def test_an_unexpected_patch_is_an_unusable_reply_not_something_trimmed(patch) -> None:  # type: ignore[no-untyped-def]
    lyrics = doc((S1, ["a", "b"]), (S2, ["c"]))
    with pytest.raises(UnusableReply):
        apply_patch(lyrics, patch, {LineRef(S1, 0)}, PLAN)


_WORDS = st.text(alphabet="abcdefg hij", min_size=1, max_size=12).map(str.strip).filter(bool)
_SECTIONS = st.lists(st.lists(_WORDS, min_size=1, max_size=5), min_size=1, max_size=3)
_VOCAL = (S1, S2, S4)
PROPERTY = settings(
    max_examples=200,
    deadline=None,
    derandomize=True,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much],
)


@PROPERTY
@given(data=st.data(), sections=_SECTIONS)
def test_patch_rounds_are_monotonic_invariant_11(
    data: st.DataObject, sections: list[list[str]]
) -> None:
    """A line that was not in the patch request is identical before and after the round."""
    lyrics = doc(*zip(_VOCAL, sections, strict=False))
    every = [
        LineRef(section.section_id, index)
        for section in lyrics.sections
        for index in range(len(section.lines))
    ]
    requested = set(data.draw(st.lists(st.sampled_from(every), unique=True, min_size=1)))
    answered = data.draw(st.lists(st.sampled_from(sorted(requested, key=repr)), unique=True))
    keys = {sid: key for key, sid in ((key_of(i), s.id) for i, s in enumerate(PLAN.sections))}
    patch = make_lyrics_patch(
        *(
            make_line_patch(keys[ref.section_id], line_number(ref.line), f"NEW{n}")
            for n, ref in enumerate(answered)
        )
    )
    patched = apply_patch(lyrics, patch, requested, PLAN)

    assert [len(s.lines) for s in patched.sections] == [len(s.lines) for s in lyrics.sections]
    for section_before, section_after in zip(lyrics.sections, patched.sections, strict=True):
        for index, (before, after) in enumerate(
            zip(section_before.lines, section_after.lines, strict=True)
        ):
            ref = LineRef(section_before.section_id, index)
            if ref in answered:
                assert after.text.startswith("NEW")
            else:
                assert after is before


# -- revisions --------------------------------------------------------------------------------


def _targets(
    sections: tuple[UUID, ...] = (), lines: tuple[tuple[UUID, int], ...] = ()
) -> ReviseTargets:
    return ReviseTargets(
        sections=sections, lines=tuple(LineTarget(section_id=s, line=n) for s, n in lines)
    )


def test_a_section_target_replaces_the_whole_section_and_may_change_its_length() -> None:
    lyrics = doc((S1, ["a", "b"]), (S2, ["c", "d", "e"]))
    reply = make_lyrics_revision(sections=[section_lines("s2", "X", "Y")])
    merged, touched, issues = apply_revision(lyrics, reply, _targets(sections=(S2,)), PLAN)
    assert issues == ()
    assert _texts(merged, S2) == ["X", "Y"]
    assert merged.sections[0] is lyrics.sections[0]
    assert touched == {LineRef(S2, 0), LineRef(S2, 1)}


def test_a_section_target_may_write_a_vocal_section_that_had_no_lines_invariant_9() -> None:
    lyrics = doc((S1, ["a"]), (S4, ["z"]))
    reply = make_lyrics_revision(sections=[section_lines("s2", "chorus one", "chorus two")])
    merged, touched, issues = apply_revision(lyrics, reply, _targets(sections=(S2,)), PLAN)
    assert issues == ()
    assert [s.section_id for s in merged.sections] == [S1, S2, S4]  # inserted in plan order
    assert touched == {LineRef(S2, 0), LineRef(S2, 1)}


def test_a_new_section_goes_first_when_nothing_precedes_it() -> None:
    lyrics = doc((S2, ["b"]))
    reply = make_lyrics_revision(sections=[section_lines("s1", "a")])
    merged, _, _ = apply_revision(lyrics, reply, _targets(sections=(S1,)), PLAN)
    assert [s.section_id for s in merged.sections] == [S1, S2]


def test_a_line_target_replaces_one_line_and_never_changes_the_count_invariant_9() -> None:
    lyrics = doc((S1, ["a", "b", "c"]))
    reply = make_lyrics_revision(lines=[make_line_patch("s1", 2, "B")])
    merged, touched, issues = apply_revision(lyrics, reply, _targets(lines=((S1, 1),)), PLAN)
    assert issues == ()
    assert _texts(merged, S1) == ["a", "B", "c"]
    assert touched == {LineRef(S1, 1)}


def test_sections_and_lines_in_one_request() -> None:
    lyrics = doc((S1, ["a", "b"]), (S2, ["c", "d"]))
    reply = make_lyrics_revision(
        sections=[section_lines("s1", "A")], lines=[make_line_patch("s2", 2, "D")]
    )
    merged, touched, issues = apply_revision(
        lyrics, reply, _targets(sections=(S1,), lines=((S2, 1),)), PLAN
    )
    assert issues == ()
    assert (_texts(merged, S1), _texts(merged, S2)) == (["A"], ["c", "D"])
    assert touched == {LineRef(S1, 0), LineRef(S2, 1)}


def test_whatever_was_not_targeted_is_a_hard_unexpected_target_and_is_ignored() -> None:
    lyrics = doc((S1, ["a", "b"]), (S2, ["c"]))
    reply = make_lyrics_revision(
        sections=[section_lines("s1", "A"), section_lines("s2", "sneaky")],
        lines=[make_line_patch("s1", 2, "sneaky too")],
    )
    merged, _, issues = apply_revision(lyrics, reply, _targets(sections=(S1,)), PLAN)
    assert set(issues) == {
        DocumentIssue("sections[1]", UNEXPECTED_TARGET),
        DocumentIssue("sections[0].lines[1]", UNEXPECTED_TARGET),
    }
    assert _texts(merged, S1) == ["A"]
    assert merged.sections[1] is lyrics.sections[1]


def test_a_missing_target_is_a_hard_issue_and_the_text_stays() -> None:
    lyrics = doc((S1, ["a", "b"]), (S2, ["c"]))
    merged, touched, issues = apply_revision(
        lyrics, make_lyrics_revision(), _targets(sections=(S2,), lines=((S1, 0),)), PLAN
    )
    assert set(issues) == {
        DocumentIssue("sections[1]", MISSING_TARGET),
        DocumentIssue("sections[0].lines[0]", MISSING_TARGET),
    }
    assert merged == lyrics
    assert touched == frozenset()


def test_an_empty_section_in_the_reply_is_a_missing_target() -> None:
    lyrics = doc((S1, ["a"]))
    reply = make_lyrics_revision(sections=[section_lines("s1")])
    merged, _, issues = apply_revision(lyrics, reply, _targets(sections=(S1,)), PLAN)
    assert issues == (DocumentIssue("sections[0]", MISSING_TARGET),)
    assert merged == lyrics


def test_an_unknown_key_and_a_repeated_section_are_reported() -> None:
    lyrics = doc((S1, ["a"]))
    reply = make_lyrics_revision(
        sections=[
            section_lines("s1", "A"),
            section_lines("s1", "A again"),
            section_lines("s9", "x"),
        ],
        lines=[make_line_patch("s9", 1, "x")],
    )
    _, _, issues = apply_revision(lyrics, reply, _targets(sections=(S1,)), PLAN)
    assert set(issues) == {
        DocumentIssue("sections[0]", DUPLICATE_SECTION_KEY),
        DocumentIssue("sections", UNKNOWN_SECTION_KEY),
    }


def test_the_revision_echo_shows_the_current_state_of_what_was_asked_for() -> None:
    lyrics = doc((S1, ["a", "b"]), (S2, ["c", "d"]))
    targets = _targets(sections=(S1,), lines=((S2, 1),))
    echo = revision_draft_of(lyrics, targets, PLAN)
    assert [(s.key, [x.text for x in s.lines]) for s in echo.sections] == [("s1", ["a", "b"])]
    assert [(p.key, p.line, p.text) for p in echo.lines] == [("s2", 2, "d")]
    assert echo.refusal is None


@PROPERTY
@given(data=st.data(), sections=_SECTIONS)
def test_every_line_that_was_not_targeted_is_byte_identical_invariant_8(
    data: st.DataObject, sections: list[list[str]]
) -> None:
    lyrics = doc(*zip(_VOCAL, sections, strict=False))
    present = {s.section_id: len(s.lines) for s in lyrics.sections}

    section_targets = tuple(data.draw(st.lists(st.sampled_from(_VOCAL), unique=True, max_size=3)))
    line_pool = [
        (sid, index)
        for sid, size in present.items()
        if sid not in section_targets
        for index in range(size)
    ]
    line_targets = tuple(
        data.draw(st.lists(st.sampled_from(line_pool), unique=True, max_size=4))
        if line_pool
        else []
    )
    keys = {s.id: key_of(i) for i, s in enumerate(PLAN.sections)}

    # a reply that answers some targets, skips others, and adds things nobody asked for
    answered_sections = [
        section_lines(keys[sid], *data.draw(st.lists(_WORDS, min_size=1, max_size=4)))
        for sid in section_targets
        if data.draw(st.booleans())
    ]
    answered_lines = [
        make_line_patch(keys[sid], line_number(index), data.draw(_WORDS))
        for sid, index in line_targets
        if data.draw(st.booleans())
    ]
    extras = data.draw(st.lists(st.sampled_from(_VOCAL), max_size=2))
    answered_sections += [
        section_lines(keys[sid], "extra")
        for sid in extras
        if sid not in section_targets and data.draw(st.booleans())
    ]
    reply = make_lyrics_revision(sections=answered_sections, lines=answered_lines)

    merged, touched, _ = apply_revision(
        lyrics, reply, _targets(sections=section_targets, lines=line_targets), PLAN
    )

    by_section_after = {s.section_id: s for s in merged.sections}
    for section in lyrics.sections:
        after = by_section_after[section.section_id]
        if section.section_id in section_targets:
            continue  # a targeted section may change completely
        # a line-target-only section keeps its length: only single lines are swapped
        assert len(after.lines) == len(section.lines)
        for index, (before_line, after_line) in enumerate(
            zip(section.lines, after.lines, strict=True)
        ):
            if (section.section_id, index) not in line_targets:
                assert after_line is before_line  # byte-identical: not even re-created
    # nothing outside the targets is reported as touched
    for ref in touched:
        assert ref.section_id in section_targets or (ref.section_id, ref.line) in line_targets
    # sections are still unique and in an order that keeps every untargeted one where it was
    assert len({s.section_id for s in merged.sections}) == len(merged.sections)


# -- issue paths ------------------------------------------------------------------------------


def test_document_issues_are_readdressed_to_the_plan_not_the_document() -> None:
    lyrics = doc((S2, ["a"]), (S4, ["b"]))  # document sections 0, 1 are plan sections 1, 3
    issues = [
        DocumentIssue("sections[1].lines[0].text", "reserved_characters"),
        DocumentIssue("sections[0].lines", "too_many_items"),
        DocumentIssue("sections", "lyrics_too_long"),
        DocumentIssue("", "document_too_large"),
    ]
    assert issues_in_plan_space(issues, lyrics, PLAN) == (
        DocumentIssue("sections[3].lines[0].text", "reserved_characters"),
        DocumentIssue("sections[1].lines", "too_many_items"),
        DocumentIssue("sections", "lyrics_too_long"),
        DocumentIssue("", "document_too_large"),
    )


def test_only_a_path_that_names_a_line_is_a_line_issue() -> None:
    lyrics = doc((S2, ["a"]), (S4, ["b"]))
    ref = line_issue_ref(DocumentIssue("sections[1].lines[0].delivery[2]", "duplicate_tag"), lyrics)
    assert ref == LineRef(S4, 0)
    assert line_issue_ref(DocumentIssue("sections[1].lines", "too_many_items"), lyrics) is None
    assert (
        line_issue_ref(DocumentIssue("sections[1].section_id", "duplicate_section"), lyrics) is None
    )
    assert line_issue_ref(DocumentIssue("sections", "lyrics_too_long"), lyrics) is None
    assert line_issue_ref(DocumentIssue("sections[9].lines[0].text", "x"), lyrics) is None


def test_the_empty_draft_builds_an_empty_document_and_names_every_vocal_section() -> None:
    document, issues = lyrics_from_draft(
        LyricsDraft(refusal=None, refusal_memories=[], sections=[]), PLAN, language="en"
    )
    assert document.sections == ()
    assert [i.path for i in issues] == ["sections[0]", "sections[1]", "sections[3]"]


def test_lyrics_sections_hold_lines_as_the_model_wrote_them_unvalidated() -> None:
    draft = make_lyrics_draft(
        section_lines("s1", "has [brackets]"), section_lines("s2", "ok"), section_lines("s4", "ok")
    )
    document, issues = lyrics_from_draft(draft, PLAN, language="en")
    assert issues == ()
    assert isinstance(document.sections[0], LyricsSection)
    assert document.sections[0].lines[0].text == "has [brackets]"
