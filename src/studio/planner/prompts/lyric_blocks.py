"""Blocks and follow-up requests shared by ``lyrics-v1`` and ``revise-lyrics-v1``.

A line problem is explained to the model by **fixed text per issue code**; the explanation may
carry numbers (a syllable count and the allowed range) and never the line's own text, so a prompt
cannot be made to echo user content back at the model beyond what the model itself wrote.

Everything the model sees uses its own numbering (``s<n>``, lines from 1): the conversions live in
``lyrics_mapping`` (``key_of``, ``line_number``) and are used here, never re-spelled.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Final

from src.llm.port import LlmMessage
from src.studio.planner.lyric_checks import AVOID_WORD_USED, SYLLABLE_RANGE_MISSED
from src.studio.planner.lyrics_mapping import key_of, line_number
from src.studio.planner.prompts.blocks import block, clean
from src.studio.validation import is_instrumental

if TYPE_CHECKING:
    from collections.abc import Sequence

    from src.studio.documents import PlanDocument, SyllableRange
    from src.studio.errors import DocumentIssue
    from src.studio.planner.lyric_checks import LineProblem

__all__ = [
    "AVOID_BLOCK",
    "LINE_EXPLANATIONS",
    "STRUCTURAL_EXPLANATIONS",
    "avoid_block",
    "build_patch_request",
    "build_rejection_request",
    "plan_block",
    "syllable_block",
]

_EN_DASH: Final = "\N{EN DASH}"
AVOID_BLOCK: Final = "avoid_words"
"""The block's own name; the explanations refer to it."""

LINE_EXPLANATIONS: Final[dict[str, str]] = {
    SYLLABLE_RANGE_MISSED: "{count} syllables counted, allowed {min}" + _EN_DASH + "{max}",
    AVOID_WORD_USED: f"uses a phrase from <{AVOID_BLOCK}>",
    "reserved_characters": "contains a square bracket",
    "empty_after_normalization": "has an empty value; write real text and drop empty cues",
    "reserved_delivery_tag": "has cues that read like a section name; use other cues or none",
    "duplicate_tag": "repeats a cue; remove the repeat",
    "too_long_after_normalization": "has a value that is too long; shorten it",
    "too_many_items": "has too many cues; use fewer",
}
"""Why a single line was rejected, by issue code."""
_GENERIC_LINE: Final = "fix this line"

STRUCTURAL_EXPLANATIONS: Final[dict[str, str]] = {
    "unknown_section_key": 'a key is not in <plan>; use only the keys "s1", "s2", … it lists',
    "duplicate_section_key": "this key appears more than once; return each key once",
    "missing_section_lines": "this section has no lines; write lines for it",
    "instrumental_section_has_lines": "this section is instrumental; return no lines for it",
    "unexpected_target": "this was not listed in <targets>; return nothing for it",
    "missing_target": "this was listed in <targets> but is missing from the reply; return it",
    "too_many_items": "there are too many lines in a section; use at most 64",
    "lyrics_too_long": "the lyrics are too long in total; shorten them",
    "document_too_large": "the lyrics are too large; make them shorter",
    "too_long_after_normalization": "a value is too long; shorten it",
    "empty_after_normalization": "a value is empty; write real text",
}
"""Why a whole reply was rejected (it is asked for again in full), by issue code."""
_GENERIC_STRUCTURE: Final = "fix this part"

_PATCH_SHAPE: Final = '{"refusal": null, "lines": [{"key", "line", "text", "delivery"}]}'
_SECTION_PATH: Final = re.compile(r"^sections\[(\d+)\](?:\.lines\[(\d+)\])?")


def _one_line(text: str) -> str:
    return clean(" ".join(text.split()))


def plan_block(plan: PlanDocument) -> str:
    """The plan as the model reads it: its facts, then one line per section under its key.

    Instrumental sections are marked so the model writes nothing for them.
    """
    lines = [f"title: {_one_line(plan.title)}"]
    if plan.idea:
        lines.append(f"idea: {_one_line(plan.idea)}")
    if plan.tempo_bpm is not None:
        lines.append(f"tempo: {plan.tempo_bpm}")
    if plan.global_style.tags:
        lines.append("tags: " + ", ".join(clean(tag) for tag in plan.global_style.tags))
    for index, section in enumerate(plan.sections):
        parts = [f"{key_of(index)}: {section.kind.value}"]
        if section.label:
            parts[0] += f" ({_one_line(section.label)})"
        if is_instrumental(section):
            parts[0] += " [instrumental]"
        if section.delivery:
            parts.append("delivery: " + ", ".join(clean(item) for item in section.delivery))
        if section.style.tags:
            parts.append("tags: " + ", ".join(clean(tag) for tag in section.style.tags))
        if section.duration_seconds is not None:
            parts.append(f"duration: {section.duration_seconds} s")
        if section.notes:
            parts.append("notes: " + _one_line(section.notes))
        lines.append("; ".join(parts))
    return block("plan", "\n".join(lines))


def syllable_block(line_syllables: SyllableRange | None) -> str | None:
    """``<line_syllables>`` (min, an en dash, max), or ``None`` when no range is asked for."""
    if line_syllables is None:
        return None
    return block("line_syllables", f"{line_syllables.min}{_EN_DASH}{line_syllables.max}")


def avoid_block(words: Sequence[str]) -> str | None:
    """``<avoid_words>``, one word or phrase per line, or ``None`` when there are none."""
    if not words:
        return None
    return block(AVOID_BLOCK, "\n".join(clean(word) for word in words))


def _explain_line(problem: LineProblem, line_syllables: SyllableRange | None) -> str:
    template = LINE_EXPLANATIONS.get(problem.code, _GENERIC_LINE)
    if problem.code == SYLLABLE_RANGE_MISSED:
        if problem.count is None or line_syllables is None:
            return _GENERIC_LINE
        return template.format(count=problem.count, min=line_syllables.min, max=line_syllables.max)
    return template


def build_patch_request(
    problems: Sequence[LineProblem], line_syllables: SyllableRange | None
) -> LlmMessage:
    """The user turn that has exactly the rejected lines replaced, one for one.

    Lines are addressed as the model numbered them (``s2 line 3``); each carries a fixed
    explanation and never the line's text.
    """
    lines = [
        f"- {key_of(problem.plan_index)} line {line_number(problem.line)}: "
        f"{_explain_line(problem, line_syllables)}"
        for problem in problems
    ]
    head = (
        "Some lines were rejected. Return replacements for exactly these lines and no others, as "
        + _PATCH_SHAPE
        + ":"
    )
    return LlmMessage("user", head + "\n" + "\n".join(lines))


def _where(path: str) -> str:
    """A plan-space issue path in the model's own words."""
    match = _SECTION_PATH.match(path)
    if match is None:
        return "(the whole reply)"
    if match[2] is None:
        return key_of(int(match[1]))
    return f"{key_of(int(match[1]))} line {line_number(int(match[2]))}"


def _explain_issue(code: str) -> str:
    return STRUCTURAL_EXPLANATIONS.get(code) or LINE_EXPLANATIONS.get(code) or _GENERIC_STRUCTURE


def build_rejection_request(heading: str, issues: Sequence[DocumentIssue]) -> LlmMessage:
    """The user turn that has a whole rejected reply fixed and returned in full.

    Args:
        heading: The sentence that introduces the list (what was rejected).
        issues: Plan-space issues; only paths and codes are used, never text.
    """
    lines = [
        f"- {_where(issue.path)}: {issue.code} — {_explain_issue(issue.code)}" for issue in issues
    ]
    return LlmMessage("user", heading + "\n" + "\n".join(lines))
