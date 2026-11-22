"""What a reply stream publishes, and when: the sentinel buffer and the coalescing. Pure.

Two rules shape the deltas a client sees (P2c-15, P2c-19):

* **The sentinel is checked before anything is published.** The first
  :data:`~src.studio.constraints.CHAT_SENTINEL_BUFFER` characters are held back. If the stripped
  text starts with the refusal sentinel the stream is a refusal and *nothing* was ever sent; there
  is no "publish, then retract". Otherwise the held text goes out as the first delta.
* **Deltas are coalesced**: a delta is published when at least ``min_interval`` seconds have passed
  since the last publish, or ``max_chars`` characters are waiting; and once more at the end. One
  loop, no timers, no tasks: the check runs when a chunk arrives.

``seq`` starts at 1 and grows by 1 per publish, so a client can see a gap; the concatenation of
every published delta is exactly :attr:`ReplyStream.text`, the text that is stored.

The clock is injected so the rules are tested without sleeping. No logging: this is reply text.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

from src.studio.constraints import CHAT_REFUSAL_SENTINEL, CHAT_SENTINEL_BUFFER

if TYPE_CHECKING:
    from collections.abc import Callable

__all__ = ["DEFAULT_MAX_CHARS", "DEFAULT_MIN_INTERVAL", "Delta", "ReplyStream"]

DEFAULT_MIN_INTERVAL: Final = 0.1
"""Seconds between two published deltas (a delta is also sent as soon as enough text waits)."""
DEFAULT_MAX_CHARS: Final = 200
"""Waiting characters that force a publish."""


@dataclass(frozen=True, slots=True)
class Delta:
    """One piece of the reply to publish: its place in the sequence and its text."""

    seq: int
    text: str


class ReplyStream:
    """Folds the text chunks of one reply into the deltas to publish."""

    def __init__(
        self,
        *,
        clock: Callable[[], float] = time.monotonic,
        min_interval: float = DEFAULT_MIN_INTERVAL,
        max_chars: int = DEFAULT_MAX_CHARS,
        sentinel_buffer: int = CHAT_SENTINEL_BUFFER,
    ) -> None:
        self._clock = clock
        self._min_interval = min_interval
        self._max_chars = max_chars
        self._sentinel_buffer = sentinel_buffer
        self._parts: list[str] = []
        self._waiting: list[str] = []
        self._waiting_chars = 0
        self._open = False
        """The sentinel decision is made and the reply is not a refusal: deltas may go out."""
        self._refused = False
        self._last_publish: float | None = None
        self._seq = 0

    @property
    def refused(self) -> bool:
        """The reply starts with the refusal sentinel: stop reading, publish nothing."""
        return self._refused

    @property
    def text(self) -> str:
        """Everything received so far - the text that is stored when the reply completes."""
        return "".join(self._parts)

    @property
    def published(self) -> int:
        """How many deltas have been handed out (the last ``seq``)."""
        return self._seq

    def feed(self, chunk: str) -> Delta | None:
        """Take one text chunk of the stream.

        Returns:
            The delta to publish now, or ``None`` (held back by the sentinel buffer, or coalesced).
            After a refusal is detected nothing is ever returned.
        """
        if self._refused or not chunk:
            return None
        self._parts.append(chunk)
        self._waiting.append(chunk)
        self._waiting_chars += len(chunk)
        if not self._open:
            held = self.text.strip()
            if held.startswith(CHAT_REFUSAL_SENTINEL):
                self._refused = True
                return None
            if len(held) < self._sentinel_buffer:
                return None  # still deciding: nothing leaves the buffer yet
            self._open = True
            return self._publish()  # the held text is the first delta
        if self._waiting_chars >= self._max_chars or self._interval_passed():
            return self._publish()
        return None

    def finish(self) -> Delta | None:
        """The stream ended: decide a still-undecided reply and flush what waits.

        Returns:
            The last delta to publish, or ``None`` (a refusal, an empty reply, or nothing waiting).
        """
        if self._refused:
            return None
        if not self._open:
            if self.text.strip().startswith(CHAT_REFUSAL_SENTINEL):
                self._refused = True
                return None
            self._open = True  # a reply shorter than the buffer that is not the sentinel
        return self._publish() if self._waiting_chars else None

    # -- internals ---------------------------------------------------------------------------

    def _interval_passed(self) -> bool:
        return (
            self._last_publish is None or self._clock() - self._last_publish >= self._min_interval
        )

    def _publish(self) -> Delta:
        text = "".join(self._waiting)
        self._waiting = []
        self._waiting_chars = 0
        self._seq += 1
        self._last_publish = self._clock()
        return Delta(seq=self._seq, text=text)
