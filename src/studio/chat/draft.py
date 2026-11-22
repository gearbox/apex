"""A memory draft: what the model returns, and the check that makes it a memory (P2c-7, P2c-28).

``MemoryDraftDraft`` is shaped for a model (no defaults: a defaulted field would be optional in the
strict JSON schema; a required ``refusal`` so the model can decline in-band) and declares **no
length limits** - the strict schema drops them anyway and a decode error could not say which field
was wrong. The limits are the memory's own (RD6): the draft is normalized with the memory
normalisers and re-checked against the limits declared on :class:`MemoryContent`, which mirror
``MemoryCreate``'s (a test holds the two together). A draft that does not pass is asked for once
more with the issues (paths and codes, never text); then the job fails ``memory_draft_invalid``.

Nothing here logs: a draft is model output derived from the user's chat.
"""

from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING, Annotated

from msgspec import Meta, Struct

from src.studio.constraints import (
    MEMORY_BODY_MAX,
    MEMORY_IMAGE_MAX,
    MEMORY_IMAGERY_MAX,
    MEMORY_THEME_MAX,
    MEMORY_THEMES_MAX,
    MEMORY_TITLE_MAX,
    collect_count_issues,
    collect_length_issues,
)
from src.studio.errors import DocumentIssue
from src.studio.validation import normalize_block, normalize_text, normalize_tokens

if TYPE_CHECKING:
    from collections.abc import Mapping

__all__ = [
    "MemoryContent",
    "MemoryDraftDraft",
    "MemoryDraftRefusal",
    "check_memory_draft",
]

_EMPTY = "empty_after_normalization"


class MemoryDraftRefusal(StrEnum):
    """Why the model declined to write a memory."""

    CONTENT_POLICY = "content_policy"
    NOTHING_TO_REMEMBER = "nothing_to_remember"


class MemoryDraftDraft(Struct, frozen=True, kw_only=True, forbid_unknown_fields=True):
    """The model's memory."""

    refusal: MemoryDraftRefusal | None
    title: str
    body: str
    themes: list[str]
    imagery: list[str]


class MemoryContent(Struct, frozen=True, kw_only=True):
    """A normalized memory draft: the fields of a memory, with the limits of ``MemoryCreate``.

    Built without validation (``Struct(...)`` skips the declared ``Meta``); the limits are applied
    by ``collect_length_issues`` / ``collect_count_issues``, exactly as the memory service does.
    """

    title: Annotated[str, Meta(min_length=1, max_length=MEMORY_TITLE_MAX)]
    body: Annotated[str, Meta(min_length=1, max_length=MEMORY_BODY_MAX)]
    themes: Annotated[
        list[Annotated[str, Meta(max_length=MEMORY_THEME_MAX)]],
        Meta(max_length=MEMORY_THEMES_MAX),
    ]
    imagery: Annotated[
        list[Annotated[str, Meta(max_length=MEMORY_IMAGE_MAX)]],
        Meta(max_length=MEMORY_IMAGERY_MAX),
    ]

    def to_json(self) -> Mapping[str, object]:
        """The stored form (``studio_memory_drafts.draft``)."""
        return {
            "title": self.title,
            "body": self.body,
            "themes": list(self.themes),
            "imagery": list(self.imagery),
        }


def check_memory_draft(
    draft: MemoryDraftDraft,
) -> tuple[MemoryContent | None, tuple[DocumentIssue, ...]]:
    """Normalize a draft like a memory and check it against the memory's limits.

    Returns:
        ``(content, ())`` when it is a valid memory, else ``(None, issues)`` with *every* issue
        found (paths and codes, never text).
    """
    issues: list[DocumentIssue] = []
    title = normalize_text(draft.title)
    if not title:
        issues.append(DocumentIssue("title", _EMPTY))
    body = normalize_block(draft.body)
    if body is None:
        issues.append(DocumentIssue("body", _EMPTY))
    themes = normalize_tokens(draft.themes, "themes", issues, check_reserved=True)
    imagery = normalize_tokens(draft.imagery, "imagery", issues, check_reserved=True)
    content = MemoryContent(
        title=title, body=body or "", themes=list(themes), imagery=list(imagery)
    )
    # RD6: NFC can expand a value past its limit, and the model does not count characters.
    collect_length_issues(content, issues)
    collect_count_issues(content, issues)
    return (None, tuple(issues)) if issues else (content, ())
