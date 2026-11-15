"""``policy_block``: what the content rating adds to the system prompt."""

from __future__ import annotations

import pytest

from src.core.product import ContentRating
from src.studio.planner.policy import policy_block

pytestmark = pytest.mark.unit

ALWAYS = (
    "Never produce sexual content involving minors, sexual or defamatory content about real "
    "people, or content that promotes violence against a group of people."
)


def test_sfw_adds_the_general_audience_rule() -> None:
    block = policy_block(ContentRating.SFW)
    assert block.splitlines()[0] == ALWAYS
    assert "Keep all themes suitable for a general audience" in block
    assert "Adult themes are allowed." not in block


def test_permissive_allows_adult_themes_but_keeps_the_hard_rule() -> None:
    block = policy_block(ContentRating.PERMISSIVE)
    assert block.splitlines()[0] == ALWAYS
    assert block.splitlines()[1] == "Adult themes are allowed."
    assert "general audience" not in block


@pytest.mark.parametrize("rating", list(ContentRating))
def test_every_rating_has_a_block(rating: ContentRating) -> None:
    assert ALWAYS in policy_block(rating)
