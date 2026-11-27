"""Deterministic LRC and SRT exports of a realized timeline (S3a).

The renderers see a :class:`TimedLyricsProjection` and nothing else: it has fields for times, line
text and word stamps only, so a plan's notes, idea, references or labels cannot reach an export
because there is no way to carry them (P1-12, as for the Suno renderer).

**Where the text comes from.** A line's text comes only from the **lyrics version**; every time
comes only from the **timeline**. The one thing a lyrics version does not have is a division into
words, so enhanced LRC writes each word as the timeline stored it (``WordSpan.text``, display data)
next to its stamp; SRT and plain LRC never use it.

**Lines are in time order**, by their start in the timeline: a subtitle file's cues must be
chronological, and the lyrics document may list its sections in another order than the song has them
(S3-4a). The lines of a valid timeline are non-empty and do not overlap, so their starts are
distinct and the order is total.

**No time is ever invented.** A lyric line the timeline does not time is left out and counted
(``untimed_lines``). Words whose span an aligner inferred are exported as stored and counted
(``inferred_word_timings``), so the user knows to check them.

Rendering is pure (the same input yields byte-identical output). Warnings are codes and counts,
never text.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, Protocol

from src.core.enums import StudioTimedLyricsFormat

if TYPE_CHECKING:
    from collections.abc import Mapping

    from src.studio.documents import LyricsDocument
    from src.studio.timeline import TimelineDocument

__all__ = [
    "TIMED_LYRICS_RENDERERS",
    "LrcRenderer",
    "RenderedTimedLyrics",
    "SrtRenderer",
    "TimedLine",
    "TimedLyricsProjection",
    "TimedLyricsRenderer",
    "TimedLyricsWarning",
    "TimedWord",
    "project_timed_lyrics",
]

UNTIMED_LINES: Final = "untimed_lines"
NO_LYRICS: Final = "no_lyrics"
INFERRED_WORD_TIMINGS: Final = "inferred_word_timings"


@dataclass(frozen=True, slots=True)
class TimedWord:
    """One word stamp of a timed line."""

    start_ms: int
    text: str
    inferred: bool


@dataclass(frozen=True, slots=True)
class TimedLine:
    """One lyric line with its place in the audio."""

    start_ms: int
    end_ms: int
    text: str
    words: tuple[TimedWord, ...]


@dataclass(frozen=True, slots=True)
class TimedLyricsProjection:
    """Everything a lyric-file renderer may see."""

    lines: tuple[TimedLine, ...]
    """The timed lines, in time order."""
    untimed: int
    """Lyric lines the timeline does not time (left out of ``lines``)."""


def project_timed_lyrics(
    timeline: TimelineDocument, lyrics: LyricsDocument
) -> TimedLyricsProjection:
    """Join a timeline's times to its lyrics' text.

    Args:
        timeline: A stored timeline (validated against ``lyrics``).
        lyrics: The lyrics version its audio version links.

    Returns:
        The timed lines in time order and the number of lyric lines without a span.
    """
    spans = {(span.section_id, span.line_index): span for span in timeline.lines}
    lines: list[TimedLine] = []
    untimed = 0
    for section in lyrics.sections:
        for line_index, lyric_line in enumerate(section.lines):
            span = spans.get((section.section_id, line_index))
            if span is None:
                untimed += 1
                continue
            lines.append(
                TimedLine(
                    start_ms=span.start_ms,
                    end_ms=span.end_ms,
                    text=lyric_line.text,
                    words=tuple(
                        TimedWord(start_ms=word.start_ms, text=word.text, inferred=word.inferred)
                        for word in span.words
                    ),
                )
            )
    lines.sort(key=lambda line: line.start_ms)
    return TimedLyricsProjection(lines=tuple(lines), untimed=untimed)


@dataclass(frozen=True, slots=True)
class TimedLyricsWarning:
    """Something the export could not carry, or that the user should check."""

    code: str
    count: int | None = None


@dataclass(frozen=True, slots=True)
class RenderedTimedLyrics:
    text: str
    """The whole file; ``""`` when there is nothing to time."""
    warnings: tuple[TimedLyricsWarning, ...]


class TimedLyricsRenderer(Protocol):
    """A deterministic renderer of one lyric-file format."""

    format: StudioTimedLyricsFormat

    def render(self, projection: TimedLyricsProjection | None) -> RenderedTimedLyrics:
        """Render ``projection``; ``None`` means the audio links no lyrics."""
        ...


def _no_lyrics() -> RenderedTimedLyrics:
    return RenderedTimedLyrics(text="", warnings=(TimedLyricsWarning(NO_LYRICS),))


def _untimed_warning(projection: TimedLyricsProjection) -> tuple[TimedLyricsWarning, ...]:
    if not projection.untimed:
        return ()
    return (TimedLyricsWarning(UNTIMED_LINES, projection.untimed),)


def _lrc_stamp(ms: int) -> str:
    """``mm:ss.xx``; centiseconds are truncated (``ms // 10``), never rounded up."""
    minutes, rest = divmod(ms // 10, 6000)
    seconds, centiseconds = divmod(rest, 100)
    return f"{minutes:02d}:{seconds:02d}.{centiseconds:02d}"


def _lrc_word(text: str) -> str:
    """A word as it can sit in a line-based file: any whitespace run (a line break included)
    becomes one space, so a word can never end its line or forge the next one."""
    return " ".join(text.split())


def _lrc_line(line: TimedLine) -> str:
    stamp = f"[{_lrc_stamp(line.start_ms)}]"
    if not line.words:
        return stamp + line.text
    return stamp + " ".join(
        f"<{_lrc_stamp(word.start_ms)}>{_lrc_word(word.text)}" for word in line.words
    )


class LrcRenderer:
    """LRC: one ``[mm:ss.xx]`` line per timed lyric line, enhanced ``<mm:ss.xx>`` word stamps
    when the line has words. No header tags (no title, artist or length)."""

    format: StudioTimedLyricsFormat = StudioTimedLyricsFormat.LRC

    def render(self, projection: TimedLyricsProjection | None) -> RenderedTimedLyrics:
        if projection is None:
            return _no_lyrics()
        warnings = list(_untimed_warning(projection))
        inferred = sum(word.inferred for line in projection.lines for word in line.words)
        if inferred:
            warnings.append(TimedLyricsWarning(INFERRED_WORD_TIMINGS, inferred))
        return RenderedTimedLyrics(
            text="".join(f"{_lrc_line(line)}\n" for line in projection.lines),
            warnings=tuple(warnings),
        )


def _srt_stamp(ms: int) -> str:
    """``HH:MM:SS,mmm``."""
    hours, rest = divmod(ms, 3_600_000)
    minutes, rest = divmod(rest, 60_000)
    seconds, milliseconds = divmod(rest, 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"


class SrtRenderer:
    """SRT: one numbered cue per timed lyric line, each followed by a blank line."""

    format: StudioTimedLyricsFormat = StudioTimedLyricsFormat.SRT

    def render(self, projection: TimedLyricsProjection | None) -> RenderedTimedLyrics:
        if projection is None:
            return _no_lyrics()
        return RenderedTimedLyrics(
            text="".join(
                f"{number}\n{_srt_stamp(line.start_ms)} --> {_srt_stamp(line.end_ms)}\n"
                f"{line.text}\n\n"
                for number, line in enumerate(projection.lines, start=1)
            ),
            warnings=_untimed_warning(projection),
        )


TIMED_LYRICS_RENDERERS: Final[Mapping[StudioTimedLyricsFormat, TimedLyricsRenderer]] = {
    renderer.format: renderer for renderer in (LrcRenderer(), SrtRenderer())
}
