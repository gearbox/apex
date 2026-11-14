"""Taking the artist's tags back out of a pasted Style / Exclude box (the renderer's inverse)."""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from src.studio.importers.subtraction import subtract_artist_tags
from src.studio.renderers.base import dedupe_ci

pytestmark = pytest.mark.unit


def render(plan: list[str], artist: list[str]) -> list[str]:
    """What the renderer writes for the tags: plan first, then the artist's, repeats dropped."""
    return dedupe_ci([*plan, *artist])


class TestSmallestK:
    def test_the_artist_suffix_is_removed(self) -> None:
        assert subtract_artist_tags(["lofi", "warm", "female vocals"], ["female vocals"]) == [
            "lofi",
            "warm",
        ]

    def test_no_artist_subtracts_nothing(self) -> None:
        assert subtract_artist_tags(["a", "b"], []) == ["a", "b"]

    def test_nothing_pasted_is_nothing(self) -> None:
        assert subtract_artist_tags([], ["a"]) == []
        assert subtract_artist_tags([], []) == []

    def test_a_tag_shared_with_the_artist_stays_in_the_plan(self) -> None:
        # P=[x, a], A=[b, a] renders [x, a, b]; the smallest k that fits keeps `a` in the plan.
        assert subtract_artist_tags(["x", "a", "b"], ["b", "a"]) == ["x", "a"]

    def test_mixed_case_overlap_keeps_the_plan_tag(self) -> None:
        # P=[a], A=[A, d, B] renders [a, d, B]; only the membership test is case-insensitive.
        assert render(["a"], ["A", "d", "B"]) == ["a", "d", "B"]
        assert subtract_artist_tags(["a", "d", "B"], ["A", "d", "B"]) == ["a"]

    def test_the_suffix_is_compared_exactly(self) -> None:
        # The renderer writes the artist's own spelling, so a different case is not the artist's.
        assert subtract_artist_tags(["x", "FEMALE"], ["female"]) == ["x", "FEMALE"]

    def test_an_edited_text_subtracts_nothing(self) -> None:
        # The artist's `b` is nowhere in the text: no suffix fits, so everything stays in the plan.
        assert subtract_artist_tags(["a", "z"], ["b"]) == ["a", "z"]
        # Present, but not as the suffix the renderer would have written.
        assert subtract_artist_tags(["b", "a", "z"], ["b", "q"]) == ["b", "a", "z"]

    def test_the_artists_own_duplicates_are_collapsed_first(self) -> None:
        assert subtract_artist_tags(["x", "v"], ["v", "V"]) == ["x"]

    def test_a_plan_that_is_all_artist_tags_is_empty(self) -> None:
        assert subtract_artist_tags(["a", "b"], ["a", "b"]) == []

    def test_works_for_the_exclude_box_too(self) -> None:
        assert subtract_artist_tags(["noise", "autotune"], ["autotune"]) == ["noise"]


_word = st.text(alphabet="abcABC", min_size=1, max_size=2)


@st.composite
def _lists(draw: st.DrawFn) -> tuple[list[str], list[str]]:
    plan = dedupe_ci(draw(st.lists(_word, max_size=6)))
    artist = dedupe_ci(draw(st.lists(_word, max_size=6)))
    return plan, artist


@settings(max_examples=500, deadline=None, derandomize=True)
@given(_lists())
def test_re_rendering_the_subtraction_reproduces_the_text(
    pair: tuple[list[str], list[str]],
) -> None:
    """``dedupe_ci([*subtract(render, A), *A]) == render`` — for every plan and artist."""
    plan, artist = pair
    rendered = render(plan, artist)
    assert dedupe_ci([*subtract_artist_tags(rendered, artist), *artist]) == rendered


@settings(max_examples=500, deadline=None, derandomize=True)
@given(_lists())
def test_the_subtraction_is_never_larger_than_the_plan_it_came_from(
    pair: tuple[list[str], list[str]],
) -> None:
    plan, artist = pair
    assert len(subtract_artist_tags(render(plan, artist), artist)) <= len(plan)
