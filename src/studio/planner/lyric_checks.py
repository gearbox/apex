"""The soft checks of written lyrics: the syllable range and the avoid-words (P2a-44, P2a-46).

Pure. A miss here is never a failure: after the model's call budget is spent the run still succeeds
and the misses become warnings (:func:`warnings_of`). Everything is read from the **normalized**
document, so a path (``sections[i].lines[j]``) addresses the very document that is saved.

``avoid_words`` match whole words or phrases, case-insensitively, after NFC, with no stemming: the
phrase's tokens must occur contiguously in the line's tokens (``baby`` matches "Oh BABY, come" and
not "babysitter"; ``my heart`` matches "my   heart!" and not "my broken heart"). The word that
matched is never recorded, only the line.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final
from uuid import UUID  # noqa: TC003 - runtime: dataclass annotations

from src.studio.planner.outcome import RunWarning
from src.studio.planner.syllables import fold_line

if TYPE_CHECKING:
    from collections.abc import Collection, Sequence

    from src.studio.documents import LyricsDocument, SyllableRange
    from src.studio.planner.syllables import SyllableCounter

__all__ = [
    "AVOID_WORD_USED",
    "SYLLABLE_CHECK_SKIPPED",
    "SYLLABLE_CHECK_SKIPPED_LINES",
    "SYLLABLE_RANGE_MISSED",
    "LineCheckReport",
    "LineIssue",
    "LineProblem",
    "LineRef",
    "check_lines",
    "line_path",
    "warnings_of",
]

SYLLABLE_RANGE_MISSED: Final = "syllable_range_missed"
AVOID_WORD_USED: Final = "avoid_word_used"
SYLLABLE_CHECK_SKIPPED_LINES: Final = "syllable_check_skipped_lines"
SYLLABLE_CHECK_SKIPPED: Final = "syllable_check_skipped"

_WORD: Final = re.compile(r"[^\W_]+(?:'[^\W_]+)*")


@dataclass(frozen=True, slots=True)
class LineRef:
    """One line of a lyrics document: its plan section and its **0-based** position in it."""

    section_id: UUID
    line: int


@dataclass(frozen=True, slots=True)
class LineIssue:
    """A soft problem with one line. ``count`` is the syllables counted (range misses only)."""

    ref: LineRef
    code: str
    count: int | None = None


@dataclass(frozen=True, slots=True)
class LineProblem:
    """One thing wrong with one line, addressed in the **plan's** section space (both 0-based).

    What a follow-up request asks the model to fix; the line's text is never part of it.
    """

    plan_index: int
    line: int
    code: str
    count: int | None = None
    """Syllables counted (for ``syllable_range_missed``)."""


@dataclass(frozen=True, slots=True)
class LineCheckReport:
    """What :func:`check_lines` found."""

    issues: tuple[LineIssue, ...]
    skipped_lines: int
    """Checked lines that could not be counted (a digit, or a word outside the alphabet)."""
    syllable_check_skipped: bool
    """A syllable range was asked for but the language has no counter."""


def line_path(section_index: int, line: int) -> str:
    """The document path of a line: ``sections[i].lines[j]`` (both 0-based)."""
    return f"sections[{section_index}].lines[{line}]"


def _tokens(text: str) -> list[str]:
    return _WORD.findall(fold_line(text))


def _phrases(avoid: Sequence[str]) -> list[list[str]]:
    """The token lists of the avoid words; one with no tokens (only punctuation) never matches."""
    return [tokens for word in avoid if (tokens := _tokens(word))]


def _contains(line: list[str], phrase: list[str]) -> bool:
    width = len(phrase)
    return any(line[start : start + width] == phrase for start in range(len(line) - width + 1))


def check_lines(
    lyrics: LyricsDocument,
    *,
    counter: SyllableCounter | None,
    syllables: SyllableRange | None,
    avoid: Sequence[str],
    only: Collection[LineRef] | None = None,
) -> LineCheckReport:
    """The soft issues of ``lyrics``.

    Args:
        lyrics: A **normalized** document.
        counter: The language's syllable counter, or ``None`` when it has none.
        syllables: The range every line should have, or ``None`` for no syllable check.
        avoid: Words or phrases that must not appear.
        only: Check just these lines (a revision checks what it changed, never the user's own
            lines); ``None`` checks all of them.

    Returns:
        The issues in document order (the syllable miss of a line before its avoid-word hit), how
        many checked lines could not be counted, and whether the language has no counter.
    """
    phrases = _phrases(avoid)
    issues: list[LineIssue] = []
    skipped = 0
    for section in lyrics.sections:
        for index, line in enumerate(section.lines):
            ref = LineRef(section.section_id, index)
            if only is not None and ref not in only:
                continue
            if counter is not None and syllables is not None:
                count = counter.count(line.text)
                if count is None:
                    skipped += 1
                elif not counter.within(count, syllables.min, syllables.max):
                    issues.append(LineIssue(ref, SYLLABLE_RANGE_MISSED, count))
            if phrases:
                tokens = _tokens(line.text)
                if any(_contains(tokens, phrase) for phrase in phrases):
                    issues.append(LineIssue(ref, AVOID_WORD_USED))
    return LineCheckReport(
        issues=tuple(issues),
        skipped_lines=skipped,
        syllable_check_skipped=syllables is not None and counter is None,
    )


def warnings_of(report: LineCheckReport, lyrics: LyricsDocument) -> tuple[RunWarning, ...]:
    """The run warnings a report stands for; paths index ``lyrics`` (the document that is saved).

    Args:
        report: Built from ``lyrics`` (so every issue's section is in it).
        lyrics: The result document.
    """
    position = {section.section_id: index for index, section in enumerate(lyrics.sections)}
    warnings = [
        RunWarning(
            issue.code,
            path=line_path(position[issue.ref.section_id], issue.ref.line),
            count=issue.count,
        )
        for issue in report.issues
    ]
    if report.skipped_lines:
        warnings.append(RunWarning(SYLLABLE_CHECK_SKIPPED_LINES, count=report.skipped_lines))
    if report.syllable_check_skipped:
        warnings.append(RunWarning(SYLLABLE_CHECK_SKIPPED))
    return tuple(warnings)
