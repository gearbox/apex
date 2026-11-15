"""The studio's port to a language model: one structured completion, typed errors.

The planner depends on this module only; which provider answers is an adapter detail
(``openai_compatible.py``) chosen by configuration. The errors are plain exceptions, not
``StudioError``: the run service maps them to a run's ``error_code``. **They never carry response
text** - a model may echo the brief or the artist's persona, and an exception message ends up in
logs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Protocol

if TYPE_CHECKING:
    from collections.abc import Sequence

    import msgspec

__all__ = [
    "LlmError",
    "LlmMessage",
    "LlmOutputInvalid",
    "LlmOutputTruncated",
    "LlmRejected",
    "LlmResult",
    "LlmUnavailable",
    "LlmUsage",
    "StructuredLlmPort",
]


@dataclass(frozen=True, slots=True)
class LlmMessage:
    """One chat message."""

    role: Literal["system", "user", "assistant"]
    content: str


@dataclass(frozen=True, slots=True)
class LlmUsage:
    """Tokens one call used, as the provider reports them (0 when it does not)."""

    prompt_tokens: int
    completion_tokens: int


@dataclass(frozen=True, slots=True)
class LlmResult[T]:
    """A decoded structured reply."""

    value: T
    usage: LlmUsage
    model_id: str


class LlmError(Exception):
    """Base of the port's errors. Messages are static and carry no response text."""


class LlmUnavailable(LlmError):
    """Transport failure, timeout, ``429`` or ``5xx`` - after the adapter's one retry."""


class LlmRejected(LlmError):
    """The provider refused the request (a ``4xx`` other than ``429``)."""

    def __init__(self, status_code: int) -> None:
        super().__init__(f"LLM request rejected (HTTP {status_code})")
        self.status_code = status_code


class LlmOutputTruncated(LlmError):
    """The reply hit the output-token limit (``finish_reason == "length"``)."""


class LlmOutputInvalid(LlmError):
    """Empty reply, not JSON, or JSON that does not decode into the requested schema."""


class StructuredLlmPort(Protocol):
    """One structured completion."""

    async def complete[T: msgspec.Struct](
        self, messages: Sequence[LlmMessage], *, schema: type[T], schema_name: str
    ) -> LlmResult[T]:
        """Ask the model for a JSON object that decodes into ``schema``.

        Args:
            messages: The conversation so far.
            schema: A ``msgspec.Struct`` with no field defaults (see ``to_strict_schema``).
            schema_name: A short identifier the provider shows in its own logs.

        Raises:
            LlmUnavailable: The provider could not be reached or kept failing.
            LlmRejected: The provider refused the request.
            LlmOutputTruncated: The reply was cut off.
            LlmOutputInvalid: The reply is not a valid ``schema`` object.
        """
        ...
