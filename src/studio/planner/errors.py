"""Errors of the planner itself (not of the model it calls)."""

from __future__ import annotations

__all__ = ["PlannerStateError"]


class PlannerStateError(RuntimeError):
    """The graph reached a state its wiring should make impossible.

    Raised deliberately - and only with a fixed message that names no data - where an ``assert``
    used to sit: it must survive ``python -O``, and a broken invariant must be named in the log by
    its type. The run it ends is failed as ``internal_error`` and refunded.
    """
