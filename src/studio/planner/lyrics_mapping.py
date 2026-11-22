"""Lyrics drafts <-> documents, purely. The server assigns every id; nothing is trusted from the model.

Two index spaces meet here and must never be mixed up:

* The **model** sees sections as ``s1 … sN`` (the plan's own order, instrumental sections counted)
  and lines **1-based**.
* The **API** and every issue path are **0-based**. A failed run's issue paths index the *input
  plan's sections* (``sections[p]…``, P2a-49), because no document was saved; the document paths of
  a succeeded run's warnings index the *result* document.

:func:`key_of`, :func:`line_number` and :func:`line_index` are the one place the model's numbering
is converted, in either direction.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Final
from uuid import UUID  # noqa: TC003 - runtime: dict key annotations are evaluated by tooling

from src.studio.documents import LyricLine, LyricsDocument, LyricsSection
from src.studio.errors import DocumentIssue
from src.studio.planner.drafts import (
    LinePatchDraft,
    LyricLineDraft,
    LyricsDraft,
    LyricsRevisionDraft,
    LyricsSectionDraft,
)
from src.studio.planner.errors import UnusableReply
from src.studio.planner.lyric_checks import LineRef
from src.studio.validation import is_instrumental

if TYPE_CHECKING:
    from collections.abc import Collection, Iterable, Sequence

    from src.studio.documents import PlanDocument, PlanSection, SyllableRange
    from src.studio.planner.drafts import LyricsPatchDraft
    from src.studio.run_requests import ReviseTargets

__all__ = [
    "DUPLICATE_SECTION_KEY",
    "INSTRUMENTAL_SECTION_HAS_LINES",
    "MISSING_SECTION_LINES",
    "MISSING_TARGET",
    "UNEXPECTED_TARGET",
    "UNKNOWN_SECTION_KEY",
    "apply_patch",
    "apply_revision",
    "draft_of",
    "issues_in_plan_space",
    "key_of",
    "line_index",
    "line_issue_ref",
    "line_number",
    "lyrics_from_draft",
    "plan_positions",
    "revision_draft_of",
    "section_keys",
]

UNKNOWN_SECTION_KEY: Final = "unknown_section_key"
DUPLICATE_SECTION_KEY: Final = "duplicate_section_key"
MISSING_SECTION_LINES: Final = "missing_section_lines"
INSTRUMENTAL_SECTION_HAS_LINES: Final = "instrumental_section_has_lines"
UNEXPECTED_TARGET: Final = "unexpected_target"
MISSING_TARGET: Final = "missing_target"

_SECTION_PATH: Final = re.compile(r"^sections\[(\d+)\]")
_LINE_PATH: Final = re.compile(r"^sections\[(\d+)\]\.lines\[(\d+)\](?:\.|\[|$)")


# -- the one conversion between the model's numbering and ours --------------------------------


def key_of(plan_index: int) -> str:
    """The key the model uses for the plan section at ``plan_index`` (0-based): ``s<index + 1>``."""
    return f"s{plan_index + 1}"


def line_number(index: int) -> int:
    """A 0-based line index as the model is shown it (1-based)."""
    return index + 1


def line_index(number: int) -> int:
    """A 1-based line number as the model wrote it, as a 0-based index."""
    return number - 1


def _normal_key(key: str) -> str:
    return key.strip().casefold()


def section_keys(plan: PlanDocument) -> dict[str, PlanSection]:
    """``s1 … sN`` over **all** plan sections, in order."""
    return {key_of(index): section for index, section in enumerate(plan.sections)}


def plan_positions(plan: PlanDocument) -> dict[UUID, int]:
    """The 0-based position of every plan section."""
    return {section.id: index for index, section in enumerate(plan.sections)}


def _line(text: str, delivery: Iterable[str]) -> LyricLine:
    """A (not yet validated) line: the document's own validation judges it, never this function."""
    return LyricLine(text=text, delivery=tuple(delivery))


def _lines_of(entry: LyricsSectionDraft) -> tuple[LyricLine, ...]:
    return tuple(_line(line.text, line.delivery) for line in entry.lines)


# -- a whole draft -----------------------------------------------------------------------------


def lyrics_from_draft(
    draft: LyricsDraft,
    plan: PlanDocument,
    *,
    language: str,
    line_syllables: SyllableRange | None = None,
) -> tuple[LyricsDocument, tuple[DocumentIssue, ...]]:
    """Build the (not yet validated) lyrics a draft stands for, with the structural problems.

    Sections come out in **plan order**, whatever order the model used. A vocal section the model
    left out (or left empty) is *not* put into the document - ``LyricsSection.lines`` must have a
    line - and is reported as ``missing_section_lines`` instead.

    Args:
        draft: The model's reply.
        plan: The plan the lyrics are for.
        language: The lyrics' language tag.
        line_syllables: The syllable range to store on the document (or ``None``).

    Returns:
        ``(document, issues)``. ``issues`` use plan-section paths (``sections[p]``); an unknown key
        has no plan section and is reported at ``sections``.
    """
    keys = section_keys(plan)
    positions = plan_positions(plan)
    issues: list[DocumentIssue] = []
    entries: dict[UUID, LyricsSectionDraft] = {}
    for entry in draft.sections:
        section = keys.get(_normal_key(entry.key))
        if section is None:
            issues.append(DocumentIssue("sections", UNKNOWN_SECTION_KEY))
            continue
        path = f"sections[{positions[section.id]}]"
        if section.id in entries:
            issues.append(DocumentIssue(path, DUPLICATE_SECTION_KEY))
        elif is_instrumental(section):
            if entry.lines:
                issues.append(DocumentIssue(path, INSTRUMENTAL_SECTION_HAS_LINES))
        else:
            entries[section.id] = entry

    sections: list[LyricsSection] = []
    for index, section in enumerate(plan.sections):
        if is_instrumental(section):
            continue
        written = entries.get(section.id)
        if written is None or not written.lines:
            issues.append(DocumentIssue(f"sections[{index}]", MISSING_SECTION_LINES))
            continue
        sections.append(LyricsSection(section_id=section.id, lines=_lines_of(written)))
    document = LyricsDocument(
        language=language,
        line_syllables=line_syllables,
        sections=tuple(sections),
    )
    return document, _unique(issues)


def draft_of(lyrics: LyricsDocument, plan: PlanDocument) -> LyricsDraft:
    """A document back in the shape the model writes (to show it its own current work).

    Sections the document has are kept in its own order; keys come from the plan.
    """
    positions = plan_positions(plan)
    return LyricsDraft(
        refusal=None,
        refusal_memories=[],
        sections=[
            LyricsSectionDraft(
                key=key_of(positions[section.section_id]),
                lines=[
                    LyricLineDraft(text=line.text, delivery=list(line.delivery))
                    for line in section.lines
                ],
            )
            for section in lyrics.sections
            if section.section_id in positions
        ],
    )


def revision_draft_of(
    lyrics: LyricsDocument, targets: ReviseTargets, plan: PlanDocument
) -> LyricsRevisionDraft:
    """The parts of ``lyrics`` a revision was asked to write, in the shape the model writes them.

    Shows the model its own work as it stands (replacements already merged) when it is asked to
    fix a few of its lines.
    """
    positions = plan_positions(plan)
    by_id = {section.section_id: section for section in lyrics.sections}
    sections = [
        LyricsSectionDraft(
            key=key_of(positions[section_id]),
            lines=[
                LyricLineDraft(text=line.text, delivery=list(line.delivery))
                for line in by_id[section_id].lines
            ],
        )
        for section_id in targets.sections
        if section_id in by_id
    ]
    lines = [
        LinePatchDraft(
            key=key_of(positions[target.section_id]),
            line=line_number(target.line),
            text=by_id[target.section_id].lines[target.line].text,
            delivery=list(by_id[target.section_id].lines[target.line].delivery),
        )
        for target in targets.lines
        if target.section_id in by_id and target.line < len(by_id[target.section_id].lines)
    ]
    return LyricsRevisionDraft(refusal=None, refusal_memories=[], sections=sections, lines=lines)


def _unique(issues: Iterable[DocumentIssue]) -> tuple[DocumentIssue, ...]:
    return tuple(dict.fromkeys(issues))


# -- replacing single lines --------------------------------------------------------------------


def _replace_lines(
    lyrics: LyricsDocument, replacements: dict[LineRef, LyricLine]
) -> LyricsDocument:
    """``lyrics`` with exactly the lines in ``replacements`` swapped; every other line is the same
    object, so it is byte-identical."""
    if not replacements:
        return lyrics
    touched_sections = {ref.section_id for ref in replacements}
    sections = tuple(
        section
        if section.section_id not in touched_sections
        else LyricsSection(
            section_id=section.section_id,
            lines=tuple(
                replacements.get(LineRef(section.section_id, index), line)
                for index, line in enumerate(section.lines)
            ),
        )
        for section in lyrics.sections
    )
    return LyricsDocument(
        language=lyrics.language, line_syllables=lyrics.line_syllables, sections=sections
    )


def _resolve_patch(
    patch: LinePatchDraft, keys: dict[str, PlanSection]
) -> tuple[LineRef, PlanSection] | None:
    section = keys.get(_normal_key(patch.key))
    if section is None:
        return None
    return LineRef(section.id, line_index(patch.line)), section


def apply_patch(
    lyrics: LyricsDocument,
    patch: LyricsPatchDraft,
    requested: Collection[LineRef],
    plan: PlanDocument,
) -> LyricsDocument:
    """Replace exactly the requested lines the patch answers; every other line stays as it is.

    A line that was asked for and is missing from the reply simply stays as it was (it is still
    rejected, and the next round asks again).

    Raises:
        UnusableReply: A replacement names an unknown key or line, a line nobody asked about, or
            the same line twice. The reply is *invalid*, never silently trimmed.
    """
    keys = section_keys(plan)
    sizes = {section.section_id: len(section.lines) for section in lyrics.sections}
    replacements: dict[LineRef, LyricLine] = {}
    for entry in patch.lines:
        resolved = _resolve_patch(entry, keys)
        if resolved is None:
            raise UnusableReply
        ref, _ = resolved
        if ref not in requested or ref in replacements:
            raise UnusableReply
        if not 0 <= ref.line < sizes.get(ref.section_id, 0):
            raise UnusableReply
        replacements[ref] = _line(entry.text, entry.delivery)
    return _replace_lines(lyrics, replacements)


# -- a revision --------------------------------------------------------------------------------


def _insert_in_plan_order(
    sections: list[LyricsSection], new: LyricsSection, positions: dict[UUID, int]
) -> None:
    """Add a section the lyrics did not have after the last section that precedes it in the plan."""
    target = positions[new.section_id]
    at = 0
    for index, existing in enumerate(sections):
        if positions.get(existing.section_id, target) < target:
            at = index + 1
    sections.insert(at, new)


def apply_revision(
    lyrics: LyricsDocument,
    draft: LyricsRevisionDraft,
    targets: ReviseTargets,
    plan: PlanDocument,
) -> tuple[LyricsDocument, frozenset[LineRef], tuple[DocumentIssue, ...]]:
    """Merge a revision reply into ``lyrics``: only what was targeted changes.

    Whole-section targets replace (or create) the section's lines; line targets replace one line
    each and never change a section's length. Everything else in ``lyrics`` is carried over as the
    same objects.

    Returns:
        ``(merged, touched, issues)``: the merged (not yet validated) document, the lines the
        model wrote, and ``unexpected_target`` / ``missing_target`` / ``unknown_section_key`` /
        ``duplicate_section_key`` issues with plan-section paths. Whatever was unexpected is
        ignored in ``merged``; whatever is missing stays as it was.
    """
    keys = section_keys(plan)
    positions = plan_positions(plan)
    section_targets = list(dict.fromkeys(targets.sections))
    line_targets = {LineRef(target.section_id, target.line) for target in targets.lines}
    issues: list[DocumentIssue] = []

    rewritten: dict[UUID, tuple[LyricLine, ...]] = {}
    for entry in draft.sections:
        section = keys.get(_normal_key(entry.key))
        if section is None:
            issues.append(DocumentIssue("sections", UNKNOWN_SECTION_KEY))
            continue
        path = f"sections[{positions[section.id]}]"
        if section.id not in section_targets:
            issues.append(DocumentIssue(path, UNEXPECTED_TARGET))
        elif section.id in rewritten:
            issues.append(DocumentIssue(path, DUPLICATE_SECTION_KEY))
        elif entry.lines:
            rewritten[section.id] = _lines_of(entry)
    issues.extend(
        DocumentIssue(f"sections[{positions[section_id]}]", MISSING_TARGET)
        for section_id in section_targets
        if section_id not in rewritten
    )

    replacements: dict[LineRef, LyricLine] = {}
    for patch in draft.lines:
        resolved = _resolve_patch(patch, keys)
        if resolved is None:
            issues.append(DocumentIssue("sections", UNKNOWN_SECTION_KEY))
            continue
        ref, section = resolved
        where = f"sections[{positions[section.id]}]"
        if ref not in line_targets or ref in replacements:
            path = f"{where}.lines[{ref.line}]" if ref.line >= 0 else where
            issues.append(DocumentIssue(path, UNEXPECTED_TARGET))
            continue
        replacements[ref] = _line(patch.text, patch.delivery)
    issues.extend(
        DocumentIssue(f"sections[{positions[ref.section_id]}].lines[{ref.line}]", MISSING_TARGET)
        for ref in sorted(
            line_targets - replacements.keys(), key=lambda r: (positions[r.section_id], r.line)
        )
    )

    touched: set[LineRef] = set()
    sections: list[LyricsSection] = []
    present: set[UUID] = set()
    for existing in lyrics.sections:
        present.add(existing.section_id)
        lines = rewritten.get(existing.section_id)
        if lines is not None:
            sections.append(LyricsSection(section_id=existing.section_id, lines=lines))
            touched.update(LineRef(existing.section_id, i) for i in range(len(lines)))
        else:
            sections.append(existing)
    for section_id, lines in rewritten.items():
        if (
            section_id not in present
        ):  # a vocal section that had no lines yet: the revision writes it
            _insert_in_plan_order(
                sections, LyricsSection(section_id=section_id, lines=lines), positions
            )
            touched.update(LineRef(section_id, i) for i in range(len(lines)))
    merged = _replace_lines(
        LyricsDocument(
            language=lyrics.language, line_syllables=lyrics.line_syllables, sections=tuple(sections)
        ),
        replacements,
    )
    touched.update(replacements)
    return merged, frozenset(touched), _unique(issues)


# -- issue paths -------------------------------------------------------------------------------


def line_issue_ref(issue: DocumentIssue, document: LyricsDocument) -> LineRef | None:
    """The line a *document-space* issue is about, or ``None`` when it is not about one line.

    A line-level problem (a bracket, a cue that reads as a header, an over-long line) can be fixed
    by replacing that line; anything else (a section's size, the document's size) cannot.
    """
    match = _LINE_PATH.match(issue.path)
    if match is None:
        return None
    section_index, line = int(match[1]), int(match[2])
    if section_index >= len(document.sections):
        return None
    return LineRef(document.sections[section_index].section_id, line)


def issues_in_plan_space(
    issues: Sequence[DocumentIssue], document: LyricsDocument, plan: PlanDocument
) -> tuple[DocumentIssue, ...]:
    """Re-address document-space issues (``sections[i]`` = the i-th section of ``document``) to the
    plan's sections (``sections[p]``), the space a failed run reports in (P2a-49)."""
    positions = plan_positions(plan)

    def readdress(issue: DocumentIssue) -> DocumentIssue:
        match = _SECTION_PATH.match(issue.path)
        if match is None or int(match[1]) >= len(document.sections):
            return issue
        plan_index = positions.get(document.sections[int(match[1])].section_id)
        if plan_index is None:
            return issue
        return DocumentIssue(f"sections[{plan_index}]{issue.path[match.end() :]}", issue.code)

    return _unique(readdress(issue) for issue in issues)
