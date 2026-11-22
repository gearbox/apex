"""``lead_with``: the executor's ``pinned_memory_unavailable`` goes first and always survives."""

from __future__ import annotations

import pytest

from src.studio.planner.outcome import (
    MAX_WARNINGS,
    PINNED_MEMORY_UNAVAILABLE,
    REVISION_INCOMPLETE,
    WARNINGS_TRUNCATED,
    RunWarning,
    cap_warnings,
    lead_with,
)

pytestmark = pytest.mark.unit

LEAD = RunWarning(PINNED_MEMORY_UNAVAILABLE, count=2)


def _plain(count: int) -> list[RunWarning]:
    return [
        RunWarning("syllable_range_missed", path=f"sections[0].lines[{i}]") for i in range(count)
    ]


def test_the_warning_is_first_and_the_rest_keep_their_order() -> None:
    rest = _plain(3)
    assert lead_with(LEAD, rest) == [LEAD, *rest]
    assert lead_with(LEAD, []) == [LEAD]


def test_one_short_of_full_still_fits_without_dropping_anything() -> None:
    rest = _plain(MAX_WARNINGS - 1)
    result = lead_with(LEAD, rest)
    assert len(result) == MAX_WARNINGS
    assert result == [LEAD, *rest]


def test_a_full_list_gives_up_one_entry_but_keeps_the_lead_and_stays_within_the_cap() -> None:
    result = lead_with(LEAD, _plain(MAX_WARNINGS))
    assert len(result) == MAX_WARNINGS
    assert result[0] == LEAD
    assert result[-1].code == WARNINGS_TRUNCATED  # nothing marked the cut, so this does


def test_a_closing_marker_stays_last_in_a_full_list() -> None:
    for marker in (WARNINGS_TRUNCATED, REVISION_INCOMPLETE):
        full = [*_plain(MAX_WARNINGS - 1), RunWarning(marker)]
        result = lead_with(LEAD, full)
        assert len(result) == MAX_WARNINGS
        assert (result[0], result[-1]) == (LEAD, RunWarning(marker))


def test_a_list_cut_by_cap_warnings_keeps_revision_incomplete_through_lead_with() -> None:
    capped = cap_warnings([*_plain(150)])
    assert len(capped) == MAX_WARNINGS
    result = lead_with(LEAD, capped)
    assert len(result) == MAX_WARNINGS
    assert (result[0], result[-1].code) == (LEAD, WARNINGS_TRUNCATED)
