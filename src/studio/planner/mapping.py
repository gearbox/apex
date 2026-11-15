"""Draft -> document, purely. The server assigns every id; nothing is trusted from the model."""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.core.uid import new_id
from src.studio.documents import PlanDocument, PlanSection, StyleSpec
from src.studio.errors import DocumentIssue

if TYPE_CHECKING:
    from src.studio.planner.drafts import PlanDraft

__all__ = ["DUPLICATE_SECTION_KEY", "EMPTY_SECTION_KEY", "draft_key_issues", "plan_from_draft"]

DUPLICATE_SECTION_KEY = "duplicate_section_key"
EMPTY_SECTION_KEY = "empty_section_key"


def draft_key_issues(draft: PlanDraft) -> tuple[DocumentIssue, ...]:
    """Issues with the section keys the model chose: an empty key, or one used twice.

    Keys are how a draft's sections are told apart before the server gives them ids; the model is
    asked to number them ``s1``, ``s2``, ... and a repeat means it lost count.
    """
    issues: list[DocumentIssue] = []
    seen: set[str] = set()
    for index, section in enumerate(draft.sections):
        key = section.key.strip()
        path = f"sections[{index}].key"
        if not key:
            issues.append(DocumentIssue(path, EMPTY_SECTION_KEY))
        elif key in seen:
            issues.append(DocumentIssue(path, DUPLICATE_SECTION_KEY))
        seen.add(key)
    return tuple(issues)


def plan_from_draft(draft: PlanDraft, *, brief: str) -> PlanDocument:
    """Build the (not yet validated) plan a draft stands for.

    ``idea`` is the user's brief; ``references`` and every section's ``style.avoid`` stay empty
    (the model's ``avoid`` list is the plan-level one). Section ids are minted here.
    """
    return PlanDocument(
        title=draft.title,
        idea=brief,
        tempo_bpm=draft.tempo_bpm,
        instrumental=draft.instrumental,
        global_style=StyleSpec(tags=tuple(draft.tags), avoid=tuple(draft.avoid)),
        references=(),
        sections=tuple(
            PlanSection(
                id=new_id(),
                kind=section.kind,
                label=section.label,
                duration_seconds=section.duration_seconds,
                style=StyleSpec(tags=tuple(section.tags), avoid=()),
                delivery=tuple(section.delivery),
                instrumental=section.instrumental,
                notes=section.notes,
            )
            for section in draft.sections
        ),
    )
