"""One review of a built lyrics document: what is wrong, and how it can be fixed. Pure.

The same step serves ``lyrics`` and ``revise_lyrics``. It validates the document exactly as a
user's document is validated (P2a-7), sorts the **hard** problems into those a replacement of one
line can fix and those that need the model's whole reply again, and - only for a document that
validated - runs the **soft** checks (the syllable range, the avoid-words; P2a-44).

Issue paths in a review are in the **plan's** section space (``sections[p]…``): a failed run saved
no document, so those are the only paths a client has (P2a-49).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from src.studio.planner.lyric_checks import LineProblem, check_lines
from src.studio.planner.lyrics_mapping import issues_in_plan_space, line_issue_ref, plan_positions
from src.studio.validation import normalize_lyrics, validate_built_document

if TYPE_CHECKING:
    from collections.abc import Collection, Sequence

    from src.studio.documents import LyricsDocument, PlanDocument, SyllableRange
    from src.studio.errors import DocumentIssue
    from src.studio.planner.lyric_checks import LineCheckReport, LineRef
    from src.studio.planner.syllables import SyllableCounter

__all__ = ["Findings", "review"]


@dataclass(frozen=True, slots=True)
class Findings:
    """What a review found."""

    hard: tuple[DocumentIssue, ...]
    """Every hard issue; a run that ends with one fails as ``lyrics_invalid``."""
    structural: tuple[DocumentIssue, ...]
    """The hard issues no replacement of single lines can fix: the model must answer in full."""
    problems: tuple[LineProblem, ...]
    """Lines to replace: the hard line-level issues, else the soft ones."""
    lyrics: LyricsDocument | None
    """The normalized document - set exactly when there is no hard issue."""
    report: LineCheckReport | None
    """The soft checks - set exactly when the document validated."""

    @property
    def clean(self) -> bool:
        """Nothing to fix: no hard issue and no soft problem."""
        return not self.hard and not self.problems


def review(
    document: LyricsDocument,
    plan: PlanDocument,
    *,
    structural: Sequence[DocumentIssue],
    counter: SyllableCounter | None,
    syllables: SyllableRange | None,
    avoid: Sequence[str],
    only: Collection[LineRef] | None = None,
) -> Findings:
    """Review ``document`` (built from the model's reply, in plan order).

    Args:
        document: The built, not yet validated, document.
        plan: The plan it is for.
        structural: Plan-space issues found while building it (a missing section, an unexpected
            target ...). They are hard and need the model's whole reply again.
        counter: The language's syllable counter, or ``None``.
        syllables: The range every checked line should have, or ``None``.
        avoid: Words or phrases no checked line may contain.
        only: Check just these lines softly (a revision checks what it changed).
    """
    validated, document_issues = validate_built_document(
        document, lambda candidate: normalize_lyrics(candidate, plan)
    )
    positions = plan_positions(plan)
    line_problems: list[LineProblem] = []
    whole: list[DocumentIssue] = []
    for issue in document_issues:
        ref = line_issue_ref(issue, document)
        if ref is None:
            whole.append(issue)
        else:
            line_problems.append(LineProblem(positions[ref.section_id], ref.line, issue.code))
    structural_issues = (*structural, *issues_in_plan_space(whole, document, plan))
    if validated is None or structural:
        return Findings(
            hard=(*structural, *issues_in_plan_space(document_issues, document, plan)),
            structural=structural_issues,
            problems=tuple(dict.fromkeys(line_problems)),
            lyrics=None,
            report=None,
        )
    report = check_lines(validated, counter=counter, syllables=syllables, avoid=avoid, only=only)
    return Findings(
        hard=(),
        structural=(),
        problems=tuple(
            LineProblem(positions[issue.ref.section_id], issue.ref.line, issue.code, issue.count)
            for issue in report.issues
        ),
        lyrics=validated,
        report=report,
    )
