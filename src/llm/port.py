"""The port to a language model: one structured completion, typed errors.

Callers depend on this module only; which provider answers is an adapter detail
(``openai_compatible.py``) chosen by configuration. The errors are plain exceptions, not API
errors: each caller maps them to its own outcome. **They never carry response text** - a model may
echo the prompt it was given, and an exception message ends up in logs. The one thing a failed reply
may carry is its token usage, which is numbers only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Protocol

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Sequence

    import msgspec

__all__ = [
    "ChatChunk",
    "ChatLlmPort",
    "LlmError",
    "LlmMessage",
    "LlmOutputInvalid",
    "LlmOutputTruncated",
    "LlmRejected",
    "LlmReplyError",
    "LlmResult",
    "LlmUnavailable",
    "LlmUsage",
    "StreamEnd",
    "StructuredLlmPort",
    "TextDelta",
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


class LlmReplyError(LlmError):
    """The provider answered (and billed) but the reply is unusable.

    ``usage`` is what the provider reported for that reply (numbers only, never text), or ``None``
    when the failure came before any usage was known. A caller that meters tokens adds it before
    giving up, so the recorded usage does not undercount the runs that spend the most.
    """

    def __init__(self, message: str, *, usage: LlmUsage | None = None) -> None:
        super().__init__(message)
        self.usage = usage


class LlmOutputTruncated(LlmReplyError):
    """The reply hit the output-token limit (``finish_reason == "length"``)."""


class LlmOutputInvalid(LlmReplyError):
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
            LlmOutputTruncated: The reply was cut off (``usage`` holds its tokens when known).
            LlmOutputInvalid: The reply is not a valid ``schema`` object (``usage`` as above).
        """
        ...


@dataclass(frozen=True, slots=True)
class TextDelta:
    """A piece of a reply that is being written. User-visible content: never logged."""

    text: str


@dataclass(frozen=True, slots=True)
class StreamEnd:
    """The last chunk of a stream: why it stopped, what it cost and who answered."""

    finish_reason: str | None
    """``"stop"`` | ``"length"`` | ``"content_filter"`` | another provider value; ``None`` when the
    stream ended without saying."""
    usage: LlmUsage
    """Zeros when the provider reports none."""
    model_id: str | None


type ChatChunk = TextDelta | StreamEnd


class ChatLlmPort(Protocol):
    """A streamed chat completion: text as it is written, then how it ended.

    Separate from :class:`StructuredLlmPort` on purpose: a stream cannot be retried once the first
    text has been delivered, and its failure modes (a reply cut off mid-way) are not those of a
    structured call.
    """

    def stream(
        self, messages: Sequence[LlmMessage], *, max_output_tokens: int
    ) -> AsyncIterator[ChatChunk]:
        """Text deltas, then exactly one :class:`StreamEnd`.

        Args:
            messages: The conversation so far (a leading system message, then the history).
            max_output_tokens: The output-token limit of this reply.

        Raises:
            LlmUnavailable: The provider could not be reached or kept failing before any text was
                delivered, or the connection broke mid-stream (the reply cannot be spliced).
            LlmRejected: The provider refused the request.
            LlmOutputInvalid: The stream is not a well-formed completion stream.
        """
        ...
