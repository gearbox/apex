"""The soft checks of written lyrics: the syllable range, avoid-words, skipped lines (P2a-44/46)."""

from __future__ import annotations

import pytest

from src.studio.documents import SyllableRange
from src.studio.planner.lyric_checks import (
    AVOID_WORD_USED,
    SYLLABLE_CHECK_SKIPPED,
    SYLLABLE_CHECK_SKIPPED_LINES,
    SYLLABLE_RANGE_MISSED,
    LineRef,
    check_lines,
    line_path,
    warnings_of,
)
from src.studio.planner.outcome import MAX_WARNINGS, WARNINGS_TRUNCATED, RunWarning, cap_warnings
from src.studio.planner.syllables import counter_for
from tests.studio_lyrics_support import S1, S2, doc

pytestmark = pytest.mark.unit

RANGE = SyllableRange(min=4, max=6)


def _avoid(line_text: str, *avoid: str) -> bool:
    report = check_lines(doc((S1, [line_text])), counter=None, syllables=None, avoid=avoid)
    return bool(report.issues)


@pytest.mark.parametrize(
    ("line", "avoid", "hit"),
    [
        ("Oh BABY, come", "baby", True),  # whole word, any case, punctuation around it
        ("my babysitter", "baby", False),  # not a substring match
        ("навсегда твой", "навсегда", True),
        ("навсегдашний друг", "навсегда", False),
        ("my   heart!", "my heart", True),  # a phrase: whitespace-insensitive
        ("my broken heart", "my heart", False),  # contiguous tokens only
        (
            "don\N{RIGHT SINGLE QUOTATION MARK}t stop",
            "don't",
            True,
        ),  # U+2019 folds to an apostrophe on both sides
        ("DON'T stop", "don\N{RIGHT SINGLE QUOTATION MARK}t", True),
        ("Café noir", "café", True),  # NFC either way
        ("Cafe\N{COMBINING ACUTE ACCENT} noir", "café", True),
        ("rock'n'roll", "rock", False),  # an apostrophe joins, it does not split
        ("nothing here", "...", False),  # an item with no tokens never matches
    ],
)
def test_avoid_words_match_whole_words_and_phrases(line: str, avoid: str, hit: bool) -> None:
    assert _avoid(line, avoid) is hit


def test_one_avoid_issue_per_line_however_many_words_match() -> None:
    report = check_lines(
        doc((S1, ["baby baby, oh baby"])), counter=None, syllables=None, avoid=["baby", "oh"]
    )
    assert [(i.ref, i.code) for i in report.issues] == [(LineRef(S1, 0), AVOID_WORD_USED)]


def test_a_syllable_miss_carries_the_count_and_a_hit_does_not() -> None:
    lyrics = doc((S1, ["la la la la", "la la la la la la la la la la"]))
    report = check_lines(lyrics, counter=counter_for("ru"), syllables=RANGE, avoid=())
    # "la" is Latin, so Russian cannot count it: both lines are skipped, none missed
    assert report.issues == ()
    assert report.skipped_lines == 2

    russian = doc((S1, ["ла ла ла ла", "ла ла ла ла ла ла ла ла ла ла"]), language="ru")
    report = check_lines(russian, counter=counter_for("ru"), syllables=RANGE, avoid=())
    assert [(i.ref.line, i.code, i.count) for i in report.issues] == [
        (1, SYLLABLE_RANGE_MISSED, 10)
    ]
    assert report.skipped_lines == 0


def test_english_has_a_tolerance_of_one_and_russian_has_none() -> None:
    three = ["la la la"]
    assert (
        check_lines(doc((S1, three)), counter=counter_for("en"), syllables=RANGE, avoid=()).issues
        == ()
    )  # 3 is within 4 - 1
    assert (
        check_lines(
            doc((S1, ["ла ла ла"]), language="ru"),
            counter=counter_for("ru"),
            syllables=RANGE,
            avoid=(),
        ).issues
        != ()
    )  # 3 < 4, exact


def test_uncountable_lines_are_skipped_and_counted_not_reported_as_misses() -> None:
    lyrics = doc((S1, ["room 101", "café au lait", "la la la la"]))
    report = check_lines(lyrics, counter=counter_for("en"), syllables=RANGE, avoid=())
    assert report.issues == ()
    assert report.skipped_lines == 2


def test_no_range_means_no_syllable_check_and_no_skip_notice() -> None:
    report = check_lines(
        doc((S1, ["room 101"])), counter=counter_for("en"), syllables=None, avoid=()
    )
    assert (report.issues, report.skipped_lines, report.syllable_check_skipped) == ((), 0, False)


def test_an_unsupported_language_with_a_range_is_skipped_as_a_whole() -> None:
    report = check_lines(
        doc((S1, ["Guten Tag"]), language="de"), counter=None, syllables=RANGE, avoid=()
    )
    assert report.syllable_check_skipped is True
    assert report.issues == ()


def test_an_unsupported_language_without_a_range_is_not_a_skip() -> None:
    report = check_lines(
        doc((S1, ["Guten Tag"]), language="de"), counter=None, syllables=None, avoid=()
    )
    assert report.syllable_check_skipped is False


def test_avoid_words_are_checked_in_a_language_without_a_counter() -> None:
    report = check_lines(
        doc((S1, ["Hallo Baby"]), language="de"), counter=None, syllables=RANGE, avoid=["baby"]
    )
    assert [i.code for i in report.issues] == [AVOID_WORD_USED]
    assert report.syllable_check_skipped is True


def test_only_restricts_both_checks_and_the_skipped_count() -> None:
    lyrics = doc((S1, ["room 101", "oh baby baby baby baby"]), (S2, ["la", "baby"]))
    only = {LineRef(S1, 1), LineRef(S2, 0)}
    report = check_lines(
        lyrics, counter=counter_for("en"), syllables=RANGE, avoid=["baby"], only=only
    )
    # S1 line 0 ("room 101") is not checked at all: not counted as skipped either; S2 line 1 ("baby") ignored
    assert {(i.ref, i.code) for i in report.issues} == {
        (LineRef(S1, 1), SYLLABLE_RANGE_MISSED),
        (LineRef(S1, 1), AVOID_WORD_USED),
        (LineRef(S2, 0), SYLLABLE_RANGE_MISSED),
    }
    assert report.skipped_lines == 0


def test_issues_come_in_document_order_syllables_before_avoid_words() -> None:
    lyrics = doc((S1, ["la", "baby baby baby baby baby baby baby"]), (S2, ["baby"]))
    report = check_lines(lyrics, counter=counter_for("en"), syllables=RANGE, avoid=["baby"])
    assert [(i.ref.section_id, i.ref.line, i.code) for i in report.issues] == [
        (S1, 0, SYLLABLE_RANGE_MISSED),
        (S1, 1, SYLLABLE_RANGE_MISSED),
        (S1, 1, AVOID_WORD_USED),
        (S2, 0, SYLLABLE_RANGE_MISSED),
        (S2, 0, AVOID_WORD_USED),
    ]


def test_warnings_address_the_result_document_not_the_plan() -> None:
    # Only the second plan section has lines: in the DOCUMENT it is section 0.
    lyrics = doc((S2, ["la", "baby"]))
    report = check_lines(lyrics, counter=counter_for("en"), syllables=RANGE, avoid=["baby"])
    warnings = warnings_of(report, lyrics)
    paths = {(w.code, w.path) for w in warnings}
    assert (SYLLABLE_RANGE_MISSED, "sections[0].lines[0]") in paths
    assert (AVOID_WORD_USED, "sections[0].lines[1]") in paths
    assert line_path(0, 1) == "sections[0].lines[1]"


def test_warnings_for_skips_carry_a_count_or_nothing() -> None:
    lyrics = doc((S1, ["room 101", "oh no 2"]))
    report = check_lines(lyrics, counter=counter_for("en"), syllables=RANGE, avoid=())
    assert warnings_of(report, lyrics) == (RunWarning(SYLLABLE_CHECK_SKIPPED_LINES, count=2),)
    german = doc((S1, ["Hallo"]), language="de")
    report = check_lines(german, counter=None, syllables=RANGE, avoid=())
    assert warnings_of(report, german) == (RunWarning(SYLLABLE_CHECK_SKIPPED),)


def test_a_warning_never_carries_the_matched_word_or_any_text() -> None:
    lyrics = doc((S1, ["the SECRETWORD is here"]))
    report = check_lines(lyrics, counter=None, syllables=None, avoid=["secretword"])
    (warning,) = warnings_of(report, lyrics)
    assert "secretword" not in repr(warning).casefold()
    assert warning.to_json() == {"code": AVOID_WORD_USED, "path": "sections[0].lines[0]"}


def test_the_stored_form_has_only_the_fields_that_are_set() -> None:
    assert RunWarning("x").to_json() == {"code": "x"}
    assert RunWarning("x", "sections[0].lines[1]", 7).to_json() == {
        "code": "x",
        "path": "sections[0].lines[1]",
        "count": 7,
    }


def test_at_most_a_hundred_warnings_are_kept_and_the_marker_is_included() -> None:
    few = [RunWarning("w", f"p{i}") for i in range(MAX_WARNINGS)]
    assert cap_warnings(few) == few
    many = [RunWarning("w", f"p{i}") for i in range(MAX_WARNINGS + 50)]
    capped = cap_warnings(many)
    assert len(capped) == MAX_WARNINGS
    assert capped[-1] == RunWarning(WARNINGS_TRUNCATED)
    assert capped[:-1] == many[: MAX_WARNINGS - 1]
