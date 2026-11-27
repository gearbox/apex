"""Request bodies of ``POST /v1/studio/tracks/{id}/runs`` (a tagged union on ``kind``).

They live in the studio package, not in ``src/api/schemas``, because the planner reads them and
may not import the API layer. ``src/api/schemas/studio_runs.py`` re-exports them for the routes.

A request is stored and hashed **as sent** (after normalization); what a run *works on* - the plan,
the lyrics, the language - is resolved once, when the run is accepted, and kept apart as
:class:`RunInputs` (P2a-40), so an idempotent replay never depends on where the track's current
pointers are now.
"""

from __future__ import annotations

from typing import Annotated, cast
from uuid import UUID

import msgspec
from msgspec import Meta, Struct

from src.studio.constraints import collect_text_issues
from src.studio.documents import LanguageTag, SyllableRange
from src.studio.errors import DocumentIssue, StudioDocumentInvalid
from src.studio.validation import normalize_block, normalize_tokens

__all__ = [
    "AvoidWord",
    "LineTarget",
    "LyricsRunRequest",
    "PlanRunRequest",
    "ReviseLyricsRunRequest",
    "ReviseTargets",
    "RunInputs",
    "RunRequest",
    "normalize_run_request",
]

AvoidWord = Annotated[str, Meta(min_length=1, max_length=40)]
_MEMORY_IDS = Annotated[tuple[UUID, ...], Meta(max_length=50)]
_AVOID_WORDS = Annotated[tuple[AvoidWord, ...], Meta(max_length=30)]


class PlanRunRequest(
    Struct, forbid_unknown_fields=True, kw_only=True, tag="plan", tag_field="kind"
):
    """Plan a track from a brief."""

    brief: Annotated[str, Meta(min_length=1, max_length=4000)]
    """The user's own words about the song. Sent to the configured LLM provider."""
    language: LanguageTag | None = None
    memory_ids: _MEMORY_IDS = ()
    """Artist memories to pin, in the order the model should see them (P2-M3). Each must be an
    active memory of the project's artist, named once; together they must fit the memory budget.
    The same two fields exist on every run kind."""
    auto_fill: bool = True
    """After the pins, fill what is left of the budget with the artist's most salient active
    memories. ``false``: use only the pins."""


class LyricsRunRequest(
    Struct, forbid_unknown_fields=True, kw_only=True, tag="lyrics", tag_field="kind"
):
    """Write lyrics for a plan version."""

    brief: Annotated[str, Meta(max_length=4000)] | None = None
    """Optional extra direction. Sent to the configured LLM provider."""
    plan_version_id: UUID | None = None
    """The plan to write for; default: the track's current plan."""
    language: LanguageTag | None = None
    """Default: the track's current lyrics' language, then the artist's, else the request fails."""
    line_syllables: SyllableRange | None = None
    avoid_words: _AVOID_WORDS = ()
    memory_ids: _MEMORY_IDS = ()
    auto_fill: bool = True


class LineTarget(Struct, frozen=True, forbid_unknown_fields=True, kw_only=True):
    """One line of the lyrics to replace."""

    section_id: UUID
    line: Annotated[int, Meta(ge=0, le=63)]
    """0-based, as in issue paths."""


class ReviseTargets(Struct, frozen=True, forbid_unknown_fields=True, kw_only=True):
    """What a revision may change; everything else in the lyrics stays byte-identical."""

    sections: Annotated[tuple[UUID, ...], Meta(max_length=64)] = ()
    """Sections to rewrite as a whole (their number of lines may change)."""
    lines: Annotated[tuple[LineTarget, ...], Meta(max_length=64)] = ()
    """Single lines to replace, one for one."""


class ReviseLyricsRunRequest(
    Struct, forbid_unknown_fields=True, kw_only=True, tag="revise_lyrics", tag_field="kind"
):
    """Revise parts of a lyrics version."""

    instruction: Annotated[str, Meta(min_length=1, max_length=1000)]
    """What to change. Sent to the configured LLM provider."""
    targets: ReviseTargets
    lyrics_version_id: UUID | None = None
    """The lyrics to revise; default: the track's current lyrics."""
    line_syllables: SyllableRange | None = None
    """Default: the input lyrics' own range."""
    avoid_words: _AVOID_WORDS = ()
    memory_ids: _MEMORY_IDS = ()
    auto_fill: bool = True


RunRequest = PlanRunRequest | LyricsRunRequest | ReviseLyricsRunRequest
"""Tagged union decoded from the ``kind`` discriminant."""


class RunInputs(Struct, frozen=True, kw_only=True):
    """What a run works on, resolved when it was accepted (P2a-40). Stored in ``studio_runs.inputs``.

    For ``lyrics``: the plan to write for. For ``revise_lyrics``: the lyrics to revise and **their**
    plan (which need not be the track's current one). ``plan`` runs have no inputs.
    """

    plan_version_id: UUID
    language: str
    lyrics_version_id: UUID | None = None


# -- normalization -----------------------------------------------------------------------------


def _normalized_brief(request: PlanRunRequest, issues: list[DocumentIssue]) -> PlanRunRequest:
    brief = normalize_block(request.brief)
    if brief is None:
        issues.append(DocumentIssue("brief", "empty_after_normalization"))
        return request
    return msgspec.structs.replace(request, brief=brief)


def _check_syllables(range_: SyllableRange | None, issues: list[DocumentIssue]) -> None:
    if range_ is not None and range_.min > range_.max:
        issues.append(DocumentIssue("line_syllables", "syllable_range_inverted"))


def _check_targets(targets: ReviseTargets, issues: list[DocumentIssue]) -> None:
    if not targets.sections and not targets.lines:
        issues.append(DocumentIssue("targets", "targets_required"))
    seen_sections: set[UUID] = set()
    for index, section_id in enumerate(targets.sections):
        if section_id in seen_sections:
            issues.append(DocumentIssue(f"targets.sections[{index}]", "duplicate_target"))
        seen_sections.add(section_id)
    seen_lines: set[tuple[UUID, int]] = set()
    for index, target in enumerate(targets.lines):
        path = f"targets.lines[{index}]"
        if (target.section_id, target.line) in seen_lines:
            issues.append(DocumentIssue(path, "duplicate_target"))
        seen_lines.add((target.section_id, target.line))
        if target.section_id in seen_sections:
            issues.append(DocumentIssue(path, "overlapping_targets"))


def _normalized_lyrics(request: LyricsRunRequest, issues: list[DocumentIssue]) -> LyricsRunRequest:
    _check_syllables(request.line_syllables, issues)
    return msgspec.structs.replace(
        request,
        brief=normalize_block(request.brief),
        avoid_words=normalize_tokens(
            request.avoid_words, "avoid_words", issues, check_reserved=True
        ),
    )


def _normalized_revision(
    request: ReviseLyricsRunRequest, issues: list[DocumentIssue]
) -> ReviseLyricsRunRequest:
    instruction = normalize_block(request.instruction)
    if instruction is None:
        issues.append(DocumentIssue("instruction", "empty_after_normalization"))
        instruction = request.instruction
    _check_syllables(request.line_syllables, issues)
    _check_targets(request.targets, issues)
    return msgspec.structs.replace(
        request,
        instruction=instruction,
        avoid_words=normalize_tokens(
            request.avoid_words, "avoid_words", issues, check_reserved=True
        ),
    )


def normalize_run_request[R: RunRequest](request: R) -> R:
    """The request as it is stored and hashed: text NFC-normalized, and re-checked (RD6).

    One entry point for every kind. A plan request's behaviour is unchanged (its brief is
    normalized; nothing else is touched). Every issue found is collected into one error.

    Raises:
        StudioDocumentInvalid: The brief / instruction is empty after normalization, an avoid word
            is empty, repeated or has reserved characters (``avoid_words[i]``), the syllable range
            is inverted (``line_syllables``), the targets are empty (``targets_required``), repeat
            (``duplicate_target``) or overlap (``overlapping_targets``), or NFC pushed a value past
            its declared limit (``too_long_after_normalization``).
    """
    issues: list[DocumentIssue] = []
    normalized: RunRequest
    if isinstance(request, PlanRunRequest):
        normalized = _normalized_brief(request, issues)
    elif isinstance(request, LyricsRunRequest):
        normalized = _normalized_lyrics(request, issues)
    else:
        normalized = _normalized_revision(request, issues)
    collect_text_issues(normalized, issues)
    if issues:
        raise StudioDocumentInvalid(tuple(issues))
    return cast("R", normalized)
