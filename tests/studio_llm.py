"""Test doubles and builders for the studio planner: a scripted ``FakeLlm`` and draft builders.

CI never calls a model. ``FakeLlm`` is a ``StructuredLlmPort`` that replies from a queue (one entry
per call: a draft, or an exception to raise) and records the messages it was sent, so a test can
assert on exactly what the planner asked.
"""

from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING

from src.core.enums import StudioSectionKind
from src.studio.llm.port import LlmMessage, LlmResult, LlmUsage
from src.studio.planner.drafts import DraftRefusal, PlanDraft, PlanSectionDraft

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Sequence

    import msgspec

    from src.studio.llm.port import StructuredLlmPort

__all__ = [
    "FakeLlm",
    "make_draft",
    "make_section_draft",
    "refusal_draft",
]

K = StudioSectionKind


def make_section_draft(
    key: str = "s1",
    kind: StudioSectionKind = K.VERSE,
    *,
    label: str | None = None,
    duration_seconds: int | None = None,
    tags: Sequence[str] = (),
    delivery: Sequence[str] = (),
    instrumental: bool = False,
    notes: str | None = None,
) -> PlanSectionDraft:
    return PlanSectionDraft(
        key=key,
        kind=kind,
        label=label,
        duration_seconds=duration_seconds,
        tags=list(tags),
        delivery=list(delivery),
        instrumental=instrumental,
        notes=notes,
    )


def make_draft(
    *sections: PlanSectionDraft,
    title: str = "Neon Rain",
    tempo_bpm: int | None = 120,
    instrumental: bool = False,
    tags: Sequence[str] = ("synthwave",),
    avoid: Sequence[str] = ("autotune",),
    refusal: DraftRefusal | None = None,
) -> PlanDraft:
    """A valid draft (two sections unless given some)."""
    return PlanDraft(
        refusal=refusal,
        title=title,
        tempo_bpm=tempo_bpm,
        instrumental=instrumental,
        tags=list(tags),
        avoid=list(avoid),
        sections=list(
            sections
            or (make_section_draft("s1", K.VERSE), make_section_draft("s2", K.CHORUS, tags=["big"]))
        ),
    )


def refusal_draft(reason: DraftRefusal) -> PlanDraft:
    """What a model returns to decline: the reason and empty values."""
    return PlanDraft(
        refusal=reason,
        title="",
        tempo_bpm=None,
        instrumental=False,
        tags=[],
        avoid=[],
        sections=[],
    )


class FakeLlm:
    """A scripted ``StructuredLlmPort``.

    Args:
        *script: One entry per expected call - a ``PlanDraft`` to return or an exception to raise.
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
