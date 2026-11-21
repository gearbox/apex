"""Rendering and selecting the artist memories a run is given (P2-M2, P2b-3). Pure.

``render_memory`` is the **one** place a memory becomes prompt text. The same function measures the
budget when a run is submitted (the pins must fit), picks the memories when a worker claims the run,
and writes the ``<artist_memories>`` block of the prompt - so the three can never disagree about how
many characters a memory costs.

``select_memories`` is deterministic and has one rule: take the pinned memories, then the
auto-fill candidates, **in order, while the next one fits the budget; stop at the first that does
not.** It never skips ahead to a smaller, less salient memory: that would invert the salience order.

No database, API or LLM imports. Memory text is user data: nothing here logs it.
"""

from __future__ import annotations

from itertools import chain
from typing import TYPE_CHECKING, Final

from src.studio.constraints import MEMORY_CONTEXT_BUDGET
from src.studio.planner.prompts.blocks import clean

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from src.studio.planner.context import MemorySnapshot

__all__ = ["render_memories", "render_memory", "rendered_size", "select_memories"]

_SEPARATOR: Final = "\n\n"
"""Between two memories: a blank line. Every line *inside* a memory is non-empty."""
_INDENT: Final = "  "


def render_memory(memory: MemorySnapshot) -> str:
    """One memory exactly as it appears inside ``<artist_memories>``::

        - <title> (<occurred_on>; salience <n>)
          themes: a, b, c
          imagery: x; y; z
          <body>

    The date is left out when there is none, and a line without content is omitted. Every user
    string is cleaned like the other prompt blocks (``<`` / ``>`` removed). Each line of the body
    is indented, a blank one included, so only the separator between memories is an empty line.
    """
    facts = f"{memory.occurred_on.isoformat()}; " if memory.occurred_on is not None else ""
    lines = [f"- {clean(memory.title)} ({facts}salience {memory.salience})"]
    if themes := [clean(theme) for theme in memory.themes]:
        lines.append(f"{_INDENT}themes: " + ", ".join(themes))
    if imagery := [clean(image) for image in memory.imagery]:
        lines.append(f"{_INDENT}imagery: " + "; ".join(imagery))
    if body := clean(memory.body).strip():
        lines.extend(f"{_INDENT}{line}" for line in body.split("\n"))
    return "\n".join(lines)


def render_memories(memories: Sequence[MemorySnapshot]) -> str:
    """The body of the ``<artist_memories>`` block: the memories, a blank line between each."""
    return _SEPARATOR.join(render_memory(memory) for memory in memories)


def rendered_size(memories: Sequence[MemorySnapshot]) -> int:
    """Characters of the block body, separators included - the number the budget limits."""
    return len(render_memories(memories))


def select_memories(
    pinned: Sequence[MemorySnapshot],
    candidates: Iterable[MemorySnapshot],
    *,
    auto_fill: bool,
    budget: int = MEMORY_CONTEXT_BUDGET,
) -> tuple[MemorySnapshot, ...]:
    """The memories a run is given, in prompt order.

    Args:
        pinned: The user's pins, in request order, already reduced to the ones still available.
        candidates: Active memories in the P2-M2 order (``salience DESC, occurred_on DESC NULLS
            LAST, id``) with the pins' ids already left out. Read lazily, only as far as needed.
        auto_fill: Whether to add candidates after the pins.
        budget: Most characters the rendered block body may have.

    Returns:
        The pins first, then candidates, as long as each next memory fits ``budget``; the first
        one that does not fit ends the selection (nothing after it is considered). Pins that were
        validated at submit fit; one that no longer does (it was edited since) ends the selection
        too, so the rendered size never exceeds ``budget``.
    """
    chosen: list[MemorySnapshot] = []
    size = 0
    for memory in chain(pinned, candidates) if auto_fill else pinned:
        cost = len(render_memory(memory)) + (len(_SEPARATOR) if chosen else 0)
        if size + cost > budget:
            break
        chosen.append(memory)
        size += cost
    return tuple(chosen)
