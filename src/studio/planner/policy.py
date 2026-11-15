"""The content-policy block of a planner prompt, by the product's content rating."""

from __future__ import annotations

from typing import Final

from src.core.product import ContentRating

__all__ = ["policy_block"]

_ALWAYS: Final = (
    "Never produce sexual content involving minors, sexual or defamatory content about real "
    "people, or content that promotes violence against a group of people."
)
_SFW: Final = (
    "Keep all themes suitable for a general audience: no explicit sexual content and no graphic "
    "violence."
)
_PERMISSIVE: Final = "Adult themes are allowed."


def policy_block(rating: ContentRating) -> str:
    """What the model may and may not write for a product with ``rating``.

    Args:
        rating: The resolved product's content rating.

    Returns:
        The block to place in the system prompt.
    """
    return _ALWAYS + "\n" + (_SFW if rating is ContentRating.SFW else _PERMISSIVE)
