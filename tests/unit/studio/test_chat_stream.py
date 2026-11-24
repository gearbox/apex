"""``ReplyStream``: the sentinel buffer and the coalescing (invariants 5 and 6). A fake clock."""

from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from src.studio.chat.stream import Delta, ReplyStream
from src.studio.constraints import CHAT_REFUSAL_SENTINEL, CHAT_SENTINEL_BUFFER

pytestmark = pytest.mark.unit


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def _play(
    chunks: list[str], clock: Clock | None = None, step: float = 0.0
) -> tuple[ReplyStream, list[Delta]]:
    clock = clock or Clock()
    stream = ReplyStream(clock=clock)
    deltas: list[Delta] = []
    for chunk in chunks:
        if delta := stream.feed(chunk):
            deltas.append(delta)
        if stream.refused:
            break
        clock.advance(step)
    if last := stream.finish():
        deltas.append(last)
    return stream, deltas


class TestTheSentinelBuffer:
    def test_a_refusal_publishes_nothing_at_all(self) -> None:
        stream, deltas = _play(["⟦decl", "ined⟧"])
        assert stream.refused
        assert deltas == []

    def test_a_refusal_is_noticed_as_soon_as_the_sentinel_is_complete(self) -> None:
        stream = ReplyStream(clock=Clock())
        assert stream.feed("⟦declined⟧") is None
        assert stream.refused
        assert stream.feed(" and more") is None  # reading has stopped; nothing more is taken
        assert stream.text == CHAT_REFUSAL_SENTINEL

    def test_leading_whitespace_does_not_hide_the_sentinel(self) -> None:
        stream, deltas = _play(["  \n", "⟦declined⟧", " I cannot"])
        assert stream.refused
        assert deltas == []

    def test_a_refusal_with_prose_after_it_is_still_a_refusal(self) -> None:
        stream, deltas = _play(["⟦declined⟧ I'm sorry, but that would break the rules."])
        assert stream.refused
        assert deltas == []

    def test_a_short_reply_that_is_not_the_sentinel_is_delivered(self) -> None:
        stream, deltas = _play(["Hello", " there!"])
        assert not stream.refused
        assert len(stream.text) < CHAT_SENTINEL_BUFFER
        assert [d.text for d in deltas] == ["Hello there!"]

    def test_nothing_leaves_the_buffer_before_it_holds_enough_text(self) -> None:
        stream = ReplyStream(clock=Clock())
        assert stream.feed("a" * (CHAT_SENTINEL_BUFFER - 1)) is None
        first = stream.feed("b")
        assert first == Delta(1, "a" * (CHAT_SENTINEL_BUFFER - 1) + "b")

    def test_the_held_text_goes_out_as_the_first_delta_in_one_piece(self) -> None:
        stream, deltas = _play(
            ["I remember ", "the night ", "we played in the rain, ", "barefoot."]
        )
        assert deltas[0].text.startswith("I remember the night")
        assert "".join(d.text for d in deltas) == stream.text

    def test_the_sentinel_in_the_middle_of_a_reply_is_just_text(self) -> None:
        stream, _ = _play(["I could say ⟦declined⟧ but I will not, because it is a nice day."])
        assert not stream.refused

    def test_an_empty_stream_has_nothing_to_publish(self) -> None:
        stream, deltas = _play([])
        assert not stream.refused
        assert deltas == []
        assert stream.text == ""


class TestCoalescing:
    def test_chunks_within_the_interval_are_merged(self) -> None:
        clock = Clock()
        stream = ReplyStream(clock=clock)
        first = stream.feed("x" * 40)  # opens the gate: published at once
        assert first == Delta(1, "x" * 40)
        clock.advance(0.01)
        assert stream.feed("a") is None
        clock.advance(0.01)
        assert stream.feed("b") is None
        clock.advance(0.1)
        assert stream.feed("c") == Delta(2, "abc")

    def test_enough_waiting_characters_force_a_publish(self) -> None:
        clock = Clock()
        stream = ReplyStream(clock=clock)
        stream.feed("x" * 40)
        clock.advance(0.001)
        assert stream.feed("y" * 150) is None
        forced = stream.feed("z" * 60)  # 210 characters wait: no need to wait for the clock
        assert forced == Delta(2, "y" * 150 + "z" * 60)

    def test_the_end_flushes_what_waits(self) -> None:
        clock = Clock()
        stream = ReplyStream(clock=clock)
        stream.feed("x" * 40)
        clock.advance(0.01)
        stream.feed("tail")
        assert stream.finish() == Delta(2, "tail")
        assert stream.finish() is None  # nothing left

    def test_seq_starts_at_one_and_has_no_gaps(self) -> None:
        clock = Clock()
        _, deltas = _play(["w" * 45, *["abc"] * 20, "end"], clock, step=0.2)
        assert [d.seq for d in deltas] == list(range(1, len(deltas) + 1))
        assert len(deltas) > 3

    def test_concatenated_deltas_are_the_stored_text(self) -> None:
        chunks = [" lead", *[f"chunk-{i} " for i in range(30)], "last."]
        stream, deltas = _play(chunks, step=0.03)
        assert "".join(d.text for d in deltas) == "".join(chunks)
        assert stream.text == "".join(chunks)


@given(
    chunks=st.lists(st.text(min_size=1, max_size=40), max_size=40),
    step=st.floats(min_value=0.0, max_value=0.3),
)
def test_for_any_stream_the_deltas_rebuild_the_text_or_there_are_none(
    chunks: list[str], step: float
) -> None:
    """Invariant 6 as a property: seq is 1..n and the deltas concatenate to the stored text; a
    refusal publishes nothing."""
    stream, deltas = _play(chunks, Clock(), step)
    assert [d.seq for d in deltas] == list(range(1, len(deltas) + 1))
    if stream.refused:
        assert deltas == []
    else:
        assert "".join(d.text for d in deltas) == "".join(chunks)
        assert stream.text == "".join(chunks)
