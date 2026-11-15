"""Keep LangSmith / LangChain tracing off: studio prompts must reach the LLM provider only (P2-M9).

``langgraph`` (and the ``langsmith`` client it pulls in) read these variables to decide whether to
ship run inputs and outputs - the brief, the artist persona, the model's drafts - to a third
party. The studio therefore refuses to start with tracing switched on, and pins both switches to
``false`` before any ``langgraph`` import can read them.

Only variable *names* appear in errors; a value (an API key) is never echoed.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from collections.abc import MutableMapping

__all__ = [
    "TRACING_API_KEY_VARS",
    "TRACING_SWITCH_VARS",
    "TracingEnabledError",
    "force_tracing_off",
]

TRACING_SWITCH_VARS: Final = ("LANGSMITH_TRACING", "LANGCHAIN_TRACING_V2")
"""Switches pinned to ``false`` (and refused when truthy)."""

TRACING_API_KEY_VARS: Final = ("LANGSMITH_API_KEY", "LANGCHAIN_API_KEY")
"""Credentials whose mere presence means somebody expects traces to be shipped."""

_FALSY: Final = frozenset({"", "0", "false", "no", "off"})


class TracingEnabledError(RuntimeError):
    """The environment would let prompts be traced to a third party."""


def force_tracing_off(environ: MutableMapping[str, str] | None = None) -> None:
    """Pin both tracing switches to ``false``, or refuse when the environment asks for tracing.

    Args:
        environ: The environment to check and update (``os.environ`` by default).

    Raises:
        TracingEnabledError: A switch is truthy, or a tracing API key is set. The message names
            the variable only.
    """
    env = os.environ if environ is None else environ
    for name in TRACING_SWITCH_VARS:
        if env.get(name, "").strip().lower() not in _FALSY:
            raise TracingEnabledError(f"{name} must not be enabled while studio runs are on")
    for name in TRACING_API_KEY_VARS:
        if env.get(name, "").strip():
            raise TracingEnabledError(f"{name} must not be set while studio runs are on")
    for name in TRACING_SWITCH_VARS:
        env[name] = "false"
