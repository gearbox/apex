"""The "ask, and re-ask when the reply cannot be used" loop every planner graph shares.

A reply that does not decode (``LlmOutputInvalid``) - or that decodes but ``accept`` refuses
(:class:`UnusableReply`, e.g. a patch for a line nobody asked about) - is asked for again within
the **same** call budget: it consumes a call but never a revision. The invalid reply itself is not
echoed back. On the last call the error propagates (the run fails ``llm_output_invalid``). Every
reply that was paid for is counted, usable or not. No ``langgraph`` here, so it is unit-testable
without the graph library.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from src.llm.port import LlmMessage, LlmOutputInvalid
from src.studio.planner.errors import UnusableReply
from src.studio.planner.prompts.blocks import REASK_MESSAGE

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    import msgspec

    from src.llm.port import StructuredLlmPort

__all__ = ["Asked", "ask_structured"]


@dataclass(frozen=True, slots=True)
class Asked[R]:
    """An accepted reply and the running totals after it."""

    value: R
    llm_calls: int
    prompt_tokens: int
    completion_tokens: int


async def ask_structured[T: msgspec.Struct, R](
    llm: StructuredLlmPort,
    messages: Sequence[LlmMessage],
    *,
    schema: type[T],
    schema_name: str,
    max_calls: int,
    llm_calls: int,
    prompt_tokens: int,
    completion_tokens: int,
    accept: Callable[[T], R],
) -> Asked[R]:
    """Ask for a ``schema`` reply that ``accept`` takes, re-asking within ``max_calls``.

    Args:
        llm: The port.
        messages: The conversation to send.
        schema: The draft Struct to decode into.
        schema_name: A short identifier the provider shows in its own logs.
        max_calls: The run's call cap (``llm_calls`` already counts the earlier calls).
        llm_calls: Calls made so far.
        prompt_tokens: Prompt tokens so far.
        completion_tokens: Completion tokens so far.
        accept: Turns a decoded reply into what the caller needs; raises :class:`UnusableReply`
            when the reply must be asked for again.

    Raises:
        LlmError: Any port error. An undecodable reply on the last call re-raises as it came; an
            unusable one becomes ``LlmOutputInvalid``.
    """
    conversation = list(messages)
    calls = llm_calls
    while True:
        calls += 1
        try:
            result = await llm.complete(conversation, schema=schema, schema_name=schema_name)
        except LlmOutputInvalid as exc:
            # The reply was paid for even though it is unusable: keep the counters in step.
            if exc.usage is not None:
                prompt_tokens += exc.usage.prompt_tokens
                completion_tokens += exc.usage.completion_tokens
            if calls >= max_calls:
                raise
            conversation.append(LlmMessage("user", REASK_MESSAGE))
            continue
        prompt_tokens += result.usage.prompt_tokens
        completion_tokens += result.usage.completion_tokens
        try:
            value = accept(result.value)
        except UnusableReply:
            if calls >= max_calls:
                raise LlmOutputInvalid("the reply was not usable", usage=result.usage) from None
            conversation.append(LlmMessage("user", REASK_MESSAGE))
            continue
        return Asked(
            value=value,
            llm_calls=calls,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )
