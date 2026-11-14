"""Taking an artist's tags back out of a pasted Style / Exclude box.

The renderer writes ``dedupe_ci([*plan_tags, *artist_tags])`` — the plan's tags first, then the
artist's, repeated ones dropped. Importing must undo exactly that, or the artist's tags would be
copied into the plan and every export would carry them twice (once as the plan's, once as the
artist's) the day the artist changes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.studio.renderers.base import dedupe_ci

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = ["subtract_artist_tags"]


def subtract_artist_tags(tokens: Sequence[str], artist_tags: Sequence[str]) -> list[str]:
    """The plan's tags: ``tokens`` without the suffix the renderer appended for the artist.

    The exact inverse of the renderer's merge. It finds the **smallest** ``k`` such that::

        tokens[k:] == [a for a in dedupe_ci(artist_tags) if a.casefold() not in folded(tokens[:k])]

    i.e. everything from ``k`` on is precisely the artist's tags that the plan's tags did not
    already contain; ``tokens[:k]`` is then the plan's tags and re-rendering them reproduces
    ``tokens``. The suffix is compared **exactly** (the renderer writes the artist's own
    spelling); only the "already in the plan" test is case-insensitive. A casefold comparison of
    the suffix as well would mis-split about 4 % of random cases (plan ``[a]``, artist
    ``[A, d, B]``).

    The smallest ``k`` keeps a tag the plan and the artist share in the plan
    (``plan [x, a]``, ``artist [b, a]`` -> ``[x, a, b]`` -> plan ``[x, a]``).

    Args:
        tokens: The pasted tags (already trimmed and deduplicated), tempo removed.
        artist_tags: The project artist's exportable tags (``style_tags`` or ``global_avoid``).

    Returns:
        The plan's tags. When no ``k`` fits (the text was edited by hand), nothing is subtracted
        and every token stays in the plan.
    """
    artist = dedupe_ci(artist_tags)
    items = list(tokens)
    for k in range(len(items) + 1):
        in_plan = {token.casefold() for token in items[:k]}
        expected = [tag for tag in artist if tag.casefold() not in in_plan]
        if items[k:] == expected:
            return items[:k]
    return items
