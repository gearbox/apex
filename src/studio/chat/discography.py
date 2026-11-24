"""The artist's own tracks, as the chat prompt shows them (P2c-21). Pure.

The artist is interviewed about what it has made, so a reply is given its discography: a
``Tracks:`` line (up to 30 titles, most recently touched first) and, for the most recent tracks
that have lyrics, the title and the lyrics as plain text - the section name on its own line, then
the lines. Nothing here is a plan, a note or a reference: only what a listener would hear.

Packing follows the memory rule (P2b-3): items are taken in order **while each next one fits the
budget; the first that does not ends the selection** (never skipping ahead to a smaller, older
track). The ``Tracks:`` line is packed the same way. The block is ``None`` when the artist has no
tracks. User text is cleaned like every prompt block (``<`` and ``>`` removed).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

from src.studio.constraints import DISCOGRAPHY_BUDGET, DISCOGRAPHY_TITLES_MAX
from src.studio.planner.prompts.blocks import clean

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = ["DiscographySection", "DiscographyTrack", "render_discography"]

_SEPARATOR: Final = "\n\n"
_INDENT: Final = "  "
_TITLES_PREFIX: Final = "Tracks: "
_TITLES_JOIN: Final = "; "


@dataclass(frozen=True, slots=True)
class DiscographySection:
    """One lyrics section as a listener reads it: its name and its lines."""

    name: str
    lines: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class DiscographyTrack:
    """One track of the artist; ``sections`` is empty when it has no lyrics."""

    title: str
    sections: tuple[DiscographySection, ...] = ()


def _titles_line(tracks: Sequence[DiscographyTrack], titles_max: int, budget: int) -> str:
    """``Tracks: a; b; c`` - up to ``titles_max`` titles, as many as fit ``budget``."""
    line = _TITLES_PREFIX.rstrip()
    chosen: list[str] = []
    for track in tracks[:titles_max]:
        candidate = _TITLES_PREFIX + _TITLES_JOIN.join([*chosen, clean(track.title)])
        if len(candidate) > budget:
            break
        chosen.append(clean(track.title))
        line = candidate
    return line if chosen else ""


def _render_track(track: DiscographyTrack) -> str:
    """``- <title>`` and, indented, every section name followed by its lines."""
    lines = [f"- {clean(track.title)}"]
    for section in track.sections:
        lines.append(f"{_INDENT}{clean(section.name)}")
        lines.extend(f"{_INDENT}{_INDENT}{clean(line)}" for line in section.lines)
    return "\n".join(lines)


def render_discography(
    tracks: Sequence[DiscographyTrack],
    *,
    budget: int = DISCOGRAPHY_BUDGET,
    titles_max: int = DISCOGRAPHY_TITLES_MAX,
) -> str | None:
    """The body of the ``<discography>`` block, at most ``budget`` characters; ``None`` if empty.

    Args:
        tracks: The artist's tracks, most recently touched first.
        budget: Most characters of the body.
        titles_max: Most titles in the ``Tracks:`` line.
    """
    if not tracks:
        return None
    parts: list[str] = []
    size = 0
    if titles := _titles_line(tracks, titles_max, budget):
        parts.append(titles)
        size = len(titles)
    for track in tracks:
        if not track.sections:
            continue
        rendered = _render_track(track)
        cost = len(rendered) + (len(_SEPARATOR) if parts else 0)
        if size + cost > budget:
            break
        parts.append(rendered)
        size += cost
    return _SEPARATOR.join(parts) if parts else None
