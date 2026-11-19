"""The worker-side handlers, one per run kind.

Importing this module loads every graph (``langgraph``); only the executor's wiring does.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.studio.services.run_kinds.lyrics import LyricsHandler
from src.studio.services.run_kinds.plan import PlanHandler
from src.studio.services.run_kinds.revise_lyrics import ReviseLyricsHandler

if TYPE_CHECKING:
    from src.core.enums import StudioRunKind
    from src.studio.services.run_kinds.base import RunHandler

__all__ = ["default_handlers"]


def default_handlers() -> dict[StudioRunKind, RunHandler]:
    """A handler for every run kind (each compiles its graph once; reuse the result)."""
    handlers: tuple[RunHandler, ...] = (PlanHandler(), LyricsHandler(), ReviseLyricsHandler())
    return {handler.kind: handler for handler in handlers}
