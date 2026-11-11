"""Pure validation, normalization and hashing of studio documents.

Every function here is free of I/O. Validation **collects all issues** and raises one
:class:`StudioDocumentInvalid` (P1-6); a document that passes is returned as a *normalized copy*
— the form that is stored and hashed.

Normalization (P1-15): Unicode NFC, trim, collapse internal whitespace — for every tag, label and
line. Case-insensitive comparisons use ``str.casefold()`` (covers non-ASCII, e.g. Cyrillic).
Issue paths address the offending field (``sections[2].style.tags[1]``) and never contain text.
"""

from __future__ import annotations

import hashlib
import unicodedata
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final
from uuid import UUID  # noqa: TC003 — runtime: used in type hints evaluated by dataclass tools

import msgspec

from src.core.enums import StudioDocumentKind, StudioSectionKind
from src.studio.documents import (
    LyricLine,
    LyricsDocument,
    LyricsSection,
    PlanDocument,
    PlanSection,
    StyleSpec,
)
from src.studio.errors import DocumentIssue, StudioDocumentInvalid

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

__all__ = [
    "MAX_TOTAL_DURATION_SECONDS",
    "NormalizedArtistStyle",
    "check_pair_compatible",
    "content_hash",
    "is_instrumental",
    "normalize_artist_style",
    "normalize_block",
    "normalize_lyrics",
    "normalize_plan",
    "normalize_text",
    "to_content",
]

MAX_TOTAL_DURATION_SECONDS: Final = 1800
"""Upper bound on the sum of a plan's planned section durations (30 minutes)."""

_RESERVED: Final = frozenset("[]")
"""Characters that would break Suno meta-tags ``[Verse]`` if they appeared inside one."""

_EMPTY: Final = "empty_after_normalization"


def normalize_text(value: str) -> str:
    """NFC, trim and collapse internal whitespace (including newlines) to single spaces."""
    return " ".join(unicodedata.normalize("NFC", value).split())


def normalize_block(value: str | None) -> str | None:
    """Normalize multi-line free text: NFC, ``\\n`` line endings, trimmed; ``None`` when blank.

    Inner whitespace and line breaks are preserved — this is for prose (ideas, notes, persona),
    not for tokens.
    """
    if value is None:
        return None
    text = unicodedata.normalize("NFC", value).replace("\r\n", "\n").replace("\r", "\n").strip()
    return text or None


def _has_reserved(value: str) -> bool:
    return any(ch in _RESERVED for ch in value)


def _normalize_tokens(
    values: Iterable[str],
    path: str,
    issues: list[DocumentIssue],
    *,
    check_reserved: bool,
    check_duplicates: bool = True,
) -> tuple[str, ...]:
    """Normalize a list of short tokens, recording empty / reserved / duplicate issues."""
    result: list[str] = []
    seen: set[str] = set()
    for index, raw in enumerate(values):
        item_path = f"{path}[{index}]"
        value = normalize_text(raw)
        if not value:
            issues.append(DocumentIssue(item_path, _EMPTY))
        else:
            if check_reserved and _has_reserved(value):
                issues.append(DocumentIssue(item_path, "reserved_characters"))
            if check_duplicates:
                key = value.casefold()
                if key in seen:
                    issues.append(DocumentIssue(item_path, "duplicate_tag"))
                seen.add(key)
        result.append(value)
    return tuple(result)


def _normalize_tag_pair(
    tags: Sequence[str],
    avoid: Sequence[str],
    tags_path: str,
    avoid_path: str,
    issues: list[DocumentIssue],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Normalize a (tags, avoid) pair and flag tags that appear in both."""
    norm_tags = _normalize_tokens(tags, tags_path, issues, check_reserved=True)
    norm_avoid = _normalize_tokens(avoid, avoid_path, issues, check_reserved=True)
    wanted = {tag.casefold() for tag in norm_tags if tag}
    issues.extend(
        DocumentIssue(f"{avoid_path}[{index}]", "tag_conflict")
        for index, tag in enumerate(norm_avoid)
        if tag and tag.casefold() in wanted
    )
    return norm_tags, norm_avoid


def _normalize_style(style: StyleSpec, path: str, issues: list[DocumentIssue]) -> StyleSpec:
    tags, avoid = _normalize_tag_pair(
        style.tags, style.avoid, f"{path}.tags", f"{path}.avoid", issues
    )
    return StyleSpec(tags=tags, avoid=avoid)


def is_instrumental(section: PlanSection) -> bool:
    """A section that must not carry lyrics."""
    return section.instrumental or section.kind is StudioSectionKind.INSTRUMENTAL


def normalize_plan(plan: PlanDocument) -> PlanDocument:
    """Validate and normalize a plan.

    Args:
        plan: A plan that already passed msgspec's wire constraints.

    Returns:
        The normalized copy (what gets stored and hashed).

    Raises:
        StudioDocumentInvalid: With every issue found (empty text, duplicate section ids or
            tags, tag conflicts, reserved characters, total duration over the cap).
    """
    issues: list[DocumentIssue] = []

    title = normalize_text(plan.title)
    if not title:
        issues.append(DocumentIssue("title", _EMPTY))

    references = _normalize_tokens(
        plan.references, "references", issues, check_reserved=False, check_duplicates=False
    )
    global_style = _normalize_style(plan.global_style, "global_style", issues)

    sections: list[PlanSection] = []
    seen_ids: set[UUID] = set()
    total_duration = 0
    for index, section in enumerate(plan.sections):
        base = f"sections[{index}]"
        if section.id in seen_ids:
            issues.append(DocumentIssue(f"{base}.id", "duplicate_section_id"))
        seen_ids.add(section.id)

        label: str | None = None
        if section.label is not None:
            label = normalize_text(section.label)
            if not label:
                issues.append(DocumentIssue(f"{base}.label", _EMPTY))
            elif _has_reserved(label):
                issues.append(DocumentIssue(f"{base}.label", "reserved_characters"))

        sections.append(
            PlanSection(
                id=section.id,
                kind=section.kind,
                label=label,
                duration_seconds=section.duration_seconds,
                style=_normalize_style(section.style, f"{base}.style", issues),
                delivery=_normalize_tokens(
                    section.delivery, f"{base}.delivery", issues, check_reserved=True
                ),
                instrumental=section.instrumental,
                notes=normalize_block(section.notes),
            )
        )
        total_duration += section.duration_seconds or 0

    if total_duration > MAX_TOTAL_DURATION_SECONDS:
        issues.append(DocumentIssue("sections", "total_duration_exceeded"))

    if issues:
        raise StudioDocumentInvalid(tuple(issues))
    return PlanDocument(
        title=title,
        idea=normalize_block(plan.idea),
        tempo_bpm=plan.tempo_bpm,
        global_style=global_style,
        references=references,
        sections=tuple(sections),
    )


def check_pair_compatible(plan: PlanDocument, lyrics: LyricsDocument) -> tuple[DocumentIssue, ...]:
    """The plan-dependent part of lyrics validation.

    Used on create, by make-current / set-current and by export: the track's current
    (plan, lyrics) pair must never violate it.

    Returns:
        ``unknown_section`` for lyrics sections the plan does not have, and
        ``instrumental_section_has_lines`` for lyrics on an instrumental plan section. Empty
        when the pair is compatible.
    """
    by_id = {section.id: section for section in plan.sections}
    issues: list[DocumentIssue] = []
    for index, lyrics_section in enumerate(lyrics.sections):
        plan_section = by_id.get(lyrics_section.section_id)
        if plan_section is None:
            issues.append(DocumentIssue(f"sections[{index}].section_id", "unknown_section"))
        elif is_instrumental(plan_section):
            issues.append(DocumentIssue(f"sections[{index}]", "instrumental_section_has_lines"))
    return tuple(issues)


def normalize_lyrics(lyrics: LyricsDocument, plan: PlanDocument) -> LyricsDocument:
    """Validate lyrics against the plan they are based on and normalize them.

    Args:
        lyrics: Lyrics that already passed msgspec's wire constraints.
        plan: The (already normalized) plan the lyrics are based on.

    Returns:
        The normalized copy.

    Raises:
        StudioDocumentInvalid: With every issue found.
    """
    issues: list[DocumentIssue] = []

    if lyrics.line_syllables is not None and lyrics.line_syllables.min > lyrics.line_syllables.max:
        issues.append(DocumentIssue("line_syllables", "syllable_range_inverted"))

    sections: list[LyricsSection] = []
    seen: set[UUID] = set()
    for index, section in enumerate(lyrics.sections):
        base = f"sections[{index}]"
        if section.section_id in seen:
            issues.append(DocumentIssue(f"{base}.section_id", "duplicate_section"))
        seen.add(section.section_id)

        lines: list[LyricLine] = []
        for line_index, line in enumerate(section.lines):
            line_base = f"{base}.lines[{line_index}]"
            text = normalize_text(line.text)
            if not text:
                issues.append(DocumentIssue(f"{line_base}.text", _EMPTY))
            elif _has_reserved(text):
                issues.append(DocumentIssue(f"{line_base}.text", "reserved_characters"))
            lines.append(
                LyricLine(
                    text=text,
                    delivery=_normalize_tokens(
                        line.delivery, f"{line_base}.delivery", issues, check_reserved=True
                    ),
                )
            )
        sections.append(LyricsSection(section_id=section.section_id, lines=tuple(lines)))

    normalized = LyricsDocument(
        language=lyrics.language, line_syllables=lyrics.line_syllables, sections=tuple(sections)
    )
    issues.extend(check_pair_compatible(plan, normalized))
    if issues:
        raise StudioDocumentInvalid(tuple(issues))
    return normalized


@dataclass(frozen=True, slots=True)
class NormalizedArtistStyle:
    """The validated, exportable parts of an artist (plus the planner-only references)."""

    style_tags: tuple[str, ...]
    global_avoid: tuple[str, ...]
    aesthetic_references: tuple[str, ...]


def normalize_artist_style(
    *,
    style_tags: Sequence[str],
    global_avoid: Sequence[str],
    aesthetic_references: Sequence[str] = (),
) -> NormalizedArtistStyle:
    """Validate an artist's tag lists like a ``StyleSpec`` (``style_tags`` vs ``global_avoid``).

    Raises:
        StudioDocumentInvalid: With every issue found (paths ``style_tags[i]`` etc.).
    """
    issues: list[DocumentIssue] = []
    tags, avoid = _normalize_tag_pair(
        style_tags, global_avoid, "style_tags", "global_avoid", issues
    )
    references = _normalize_tokens(
        aesthetic_references,
        "aesthetic_references",
        issues,
        check_reserved=False,
        check_duplicates=False,
    )
    if issues:
        raise StudioDocumentInvalid(tuple(issues))
    return NormalizedArtistStyle(
        style_tags=tags, global_avoid=avoid, aesthetic_references=references
    )


def to_content(document: PlanDocument | LyricsDocument) -> dict[str, object]:
    """The JSONB form of a normalized document.

    Goes through the JSON encoder so the result is exactly what PostgreSQL will hand back
    (lists, not tuples; UUIDs and enums as strings) — what is written equals what is read.
    """
    content = msgspec.json.decode(msgspec.json.encode(document))
    assert isinstance(content, dict)  # noqa: S101 — a Struct always encodes to an object
    return content


def content_hash(
    kind: StudioDocumentKind,
    based_on_plan_version_id: UUID | None,
    document: PlanDocument | LyricsDocument,
) -> str:
    """Stable identity of a version's content: ``sha256(kind \\n based_on \\n canonical-json)``.

    ``based_on_plan_version_id`` is part of the hash because the same lyrics text over a
    different plan is a different version. ``document`` must already be normalized.
    """
    head = f"{kind.value}\n{based_on_plan_version_id or ''}\n".encode()
    return hashlib.sha256(head + msgspec.json.encode(document)).hexdigest()
