"""A scripted ``StructuredLlmPort`` for tests of anything that calls a model.

CI never calls a model. ``FakeLlm`` replies from a queue (one entry per call: a value to return, or
an exception to raise) and records the messages it was sent, so a test can assert on exactly what
was asked. It knows nothing about any feature; plan-draft builders live in ``tests/studio_llm.py``.
"""

from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING

from src.llm.port import LlmMessage, LlmResult, LlmUsage

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Sequence

    import msgspec

    from src.llm.port import StructuredLlmPort

__all__ = ["FakeLlm", "as_port"]


class FakeLlm:
    """A scripted ``StructuredLlmPort``.

    Args:
        *script: One entry per expected call - a Struct to return or an exception to raise.
            A call beyond the script raises ``AssertionError`` (an unexpected extra model call).
        usage: Tokens each successful reply reports.
        before_call: Awaited at the start of every call (e.g. to inspect the connection pool).
    """

    def __init__(
        self,
        *script: msgspec.Struct | BaseException,
        usage: LlmUsage = LlmUsage(prompt_tokens=100, completion_tokens=50),  # noqa: B008
        before_call: Callable[[], Awaitable[None]] | None = None,
    ) -> None:
        self._script = deque(script)
        self._usage = usage
        self._before_call = before_call
        self.calls: list[list[LlmMessage]] = []
        """The messages of every call so far."""

    @property
    def call_count(self) -> int:
        return len(self.calls)

    async def complete[T: msgspec.Struct](
        self,
        messages: Sequence[LlmMessage],
        *,
        schema: type[T],
        schema_name: str,  # noqa: ARG002
    ) -> LlmResult[T]:
        self.calls.append([LlmMessage(m.role, m.content) for m in messages])
        if self._before_call is not None:
            await self._before_call()
        if not self._script:
            raise AssertionError("FakeLlm was called more often than its script allows")
        item = self._script.popleft()
        if isinstance(item, BaseException):
            raise item
        assert isinstance(item, schema)
        return LlmResult(value=item, usage=self._usage, model_id="fake-model")


def as_port(fake: FakeLlm) -> StructuredLlmPort:
    """``FakeLlm`` typed as the port (it already satisfies the protocol)."""
    return fake
