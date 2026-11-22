"""Builders for artist-memory unit tests: a memory snapshot with a stable id."""

from __future__ import annotations

from datetime import UTC, date, datetime
from uuid import UUID

from src.studio.planner.context import MemorySnapshot

__all__ = ["make_memory"]


def make_memory(
    number: int = 1,
    *,
    title: str | None = None,
    body: str | None = None,
    themes: tuple[str, ...] = (),
    imagery: tuple[str, ...] = (),
    occurred_on: date | None = None,
    salience: int = 3,
) -> MemorySnapshot:
    """A memory whose id is ``UUID(int=number)``; the title and body default to the number."""
    return MemorySnapshot(
        id=UUID(int=number),
        title=title if title is not None else f"Memory {number}",
        body=body if body is not None else f"Body of memory {number}.",
        themes=themes,
        imagery=imagery,
        occurred_on=occurred_on,
        salience=salience,
        updated_at=datetime(2026, 1, 1, tzinfo=UTC),
    )
