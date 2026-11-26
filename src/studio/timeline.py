"""Realized timelines (S3a): *when* an audio version's plan sections and lyric lines happen.

Pure module: the shape of a timeline document, its normalization, its validation against the
audio's duration, plan and lyrics, completeness and the content hash. No I/O, and nothing imported
from ``src.api`` or ``src.db``.

Times are integer milliseconds and every span is half-open, ``[start_ms, end_ms)``: two spans that
touch do not overlap. A timeline names no plan or lyrics version: those are its audio version's
links, which freeze once the audio version has a timeline (S3-5). Together with immutable versions
and an immutable audio duration, that is why a stored timeline stays valid for as long as it exists.

**All four Structs set ``omit_defaults=True`` (S3-14).** The content hash is computed over
``msgspec.json.encode(document)`` and the stored JSONB is that same encoding, so a field added
later with a default changes neither the encoding nor the hash of a row already stored: replay
detection and ``timeline_unchanged`` keep working with no data migration. It also means a word an
aligner placed with certainty stores no ``inferred`` key. A change that old rows cannot decode
needs a new ``TIMELINE_SCHEMA_VERSION`` and an Alembic data migration (D5: no read-time shims).

Every limit is declared on the Struct ``Meta`` and nowhere else.
"""

from __future__ import annotations

import hashlib
import unicodedata
from dataclasses import dataclass
from typing import TYPE_CHECKING, Annotated, Final, Protocol
from uuid import UUID

import msgspec
from msgspec import Meta, Struct

from src.core.media_limits import AUDIO_MAX_MS
from src.studio.constraints import collect_length_issues
from src.studio.errors import DocumentIssue, StudioDocumentSchemaError
from src.studio.validation import MAX_DOCUMENT_BYTES

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from src.studio.documents import LyricsDocument, PlanDocument

__all__ = [
    "TIMELINE_SCHEMA_VERSION",
    "LineSpan",
    "Ms",
    "SectionSpan",
    "TimelineDocument",
    "WordSpan",
    "check_timeline_limits",
    "decode_stored_timeline",
    "is_complete",
    "normalize_timeline",
    "timeline_content",
    "timeline_hash",
    "validate_timeline",
]

TIMELINE_SCHEMA_VERSION: Final = 1

_HASH_PREFIX: Final = b"timeline\n"

# -- issue codes (S3-4) ----------------------------------------------------------------------------

SPAN_OUT_OF_RANGE: Final = "span_out_of_range"
UNKNOWN_SECTION: Final = "unknown_section"
DUPLICATE_SECTION: Final = "duplicate_section"
SECTION_ORDER: Final = "section_order"
SECTION_OVERLAP: Final = "section_overlap"
UNKNOWN_LINE: Final = "unknown_line"
DUPLICATE_LINE: Final = "duplicate_line"
LINE_SECTION_UNTIMED: Final = "line_section_untimed"
LINE_OUTSIDE_SECTION: Final = "line_outside_section"
LINE_ORDER: Final = "line_order"
LINE_OVERLAP: Final = "line_overlap"
WORD_OUTSIDE_LINE: Final = "word_outside_line"
WORD_ORDER: Final = "word_order"
WORD_OVERLAP: Final = "word_overlap"

# -- the document ----------------------------------------------------------------------------------

Ms = Annotated[int, Meta(ge=0, le=AUDIO_MAX_MS)]
"""A point in the audio, in milliseconds."""


class SectionSpan(
    Struct, frozen=True, kw_only=True, forbid_unknown_fields=True, omit_defaults=True
):
    """When one plan section (``PlanSection.id``) is heard."""

    section_id: UUID
    start_ms: Ms
    end_ms: Ms


class WordSpan(Struct, frozen=True, kw_only=True, forbid_unknown_fields=True, omit_defaults=True):
    """When one word of a line is sung. ``text`` is display data: it is not compared with the
    line's text (an aligner tokenizes its own way)."""

    # No NUL: PostgreSQL JSONB cannot store U+0000, and the resulting 500 would put the whole
    # statement - every word of the timeline - into the error log.
    text: Annotated[str, Meta(min_length=1, max_length=100, pattern=r"\A[^\x00]*\Z")]
    start_ms: Ms
    end_ms: Ms
    inferred: bool = False
    """The aligner could not place this word; its span was inferred (S3-14). Display and quality
    data only: validation treats the word like any other, and a user who corrects it sends it
    without the flag, which makes it a measured word."""


class LineSpan(Struct, frozen=True, kw_only=True, forbid_unknown_fields=True, omit_defaults=True):
    """When one lyric line is sung. The line is ``(section_id, line_index)`` in the audio's lyrics
    version; lyric versions are immutable, so the index is stable."""

    section_id: UUID
    line_index: Annotated[int, Meta(ge=0, le=63)]
    start_ms: Ms
    end_ms: Ms
    words: Annotated[tuple[WordSpan, ...], Meta(max_length=64)] = ()


class TimelineDocument(
    Struct, frozen=True, kw_only=True, forbid_unknown_fields=True, omit_defaults=True
):
    """A realized timeline (schema_version 1): section spans and, optionally, line spans.

    A timeline may be partial: not every section or line has to be timed (see
    :func:`is_complete`). The server never invents a time.
    """

    sections: Annotated[tuple[SectionSpan, ...], Meta(max_length=64)]
    lines: Annotated[tuple[LineSpan, ...], Meta(max_length=4096)] = ()


# -- normalization, limits, identity ---------------------------------------------------------------


def normalize_timeline(document: TimelineDocument) -> TimelineDocument:
    """The stored form: word text in Unicode NFC (RD6). Nothing else is changed (times are
    integers, and word text is otherwise kept as sent).

    NFC can make a value longer than the wire check saw; call :func:`check_timeline_limits` on the
    result.
    """
    return msgspec.structs.replace(
        document,
        lines=tuple(
            msgspec.structs.replace(
                line,
                words=tuple(
                    msgspec.structs.replace(word, text=unicodedata.normalize("NFC", word.text))
                    for word in line.words
                ),
            )
            for line in document.lines
        ),
    )


def check_timeline_limits(document: TimelineDocument) -> list[DocumentIssue]:
    """The limits that only the **normalized** document can be checked against.

    Args:
        document: The result of :func:`normalize_timeline`.

    Returns:
        ``too_long_after_normalization`` at each value past its declared length (RD6), and one
        ``document_too_large`` (path ``""``) when the compact JSON exceeds
        :data:`~src.studio.validation.MAX_DOCUMENT_BYTES`.
    """
    issues: list[DocumentIssue] = []
    collect_length_issues(document, issues)
    if len(msgspec.json.encode(document)) > MAX_DOCUMENT_BYTES:
        issues.append(DocumentIssue("", "document_too_large"))
    return issues


def timeline_hash(document: TimelineDocument) -> str:
    """Stable identity of a timeline's content: ``sha256("timeline\\n" + canonical-json)``."""
    return hashlib.sha256(_HASH_PREFIX + msgspec.json.encode(document)).hexdigest()


def timeline_content(document: TimelineDocument) -> dict[str, object]:
    """The JSONB form of a normalized timeline: exactly what PostgreSQL hands back (lists, not
    tuples; UUIDs as strings; defaults omitted)."""
    content: dict[str, object] = msgspec.json.decode(msgspec.json.encode(document), type=dict)
    return content


def decode_stored_timeline(schema_version: int, content: Mapping[str, object]) -> TimelineDocument:
    """Decode a stored ``content`` JSONB.

    Raises:
        StudioDocumentSchemaError: ``schema_version`` is not one this code understands. Never
            falls back to a best-effort decode.
        msgspec.ValidationError: The stored content does not match the schema (a data bug).
    """
    if schema_version != TIMELINE_SCHEMA_VERSION:
        raise StudioDocumentSchemaError(
            f"unsupported timeline schema_version {schema_version} "
            f"(this build understands {TIMELINE_SCHEMA_VERSION})"
        )
    return msgspec.convert(content, TimelineDocument, strict=True)


# -- validation ------------------------------------------------------------------------------------


class _Span(Protocol):
    @property
    def start_ms(self) -> int: ...

    @property
    def end_ms(self) -> int: ...


@dataclass(frozen=True, slots=True)
class _Placed:
    """A span together with where it should come in its sequence (``rank``)."""

    path: str
    rank: int
    start_ms: int
    end_ms: int


def validate_timeline(
    document: TimelineDocument,
    *,
    duration_ms: int,
    plan: PlanDocument,
    lyrics: LyricsDocument | None,
) -> list[DocumentIssue]:
    """Every problem of ``document`` against the audio's duration, plan and lyrics (S3-4).

    Args:
        document: A normalized timeline.
        duration_ms: The audio version's duration.
        plan: The plan the audio version links.
        lyrics: The lyrics it links, or ``None`` (then no line is known).

    Returns:
        All issues (empty when the timeline is valid). Paths are ``sections[i]``, ``lines[j]`` and
        ``lines[j].words[k]``; they never contain text. Gaps are fine, and a partial timeline is
        valid. "Plan order" is the order of the plan's sections; "lyrics order" is the order of
        the lyrics document's sections, then ``line_index``. A sequence is in order when it is
        listed in that order **and** no span starts before the one listed ahead of it.
    """
    issues: list[DocumentIssue] = []
    timed_sections = _check_sections(document.sections, plan, duration_ms, issues)
    _check_lines(document.lines, lyrics, timed_sections, duration_ms, issues)
    return issues


def _check_range(span: _Span, path: str, duration_ms: int, issues: list[DocumentIssue]) -> None:
    if not 0 <= span.start_ms < span.end_ms <= duration_ms:
        issues.append(DocumentIssue(path, SPAN_OUT_OF_RANGE))


def _check_sequence(
    placed: Sequence[_Placed],
    *,
    order_code: str,
    overlap_code: str,
    issues: list[DocumentIssue],
) -> None:
    """Order and overlap of ``placed``, given in the order the document lists them."""
    previous: _Placed | None = None
    for item in placed:
        if previous is not None and (
            item.rank < previous.rank or item.start_ms < previous.start_ms
        ):
            issues.append(DocumentIssue(item.path, order_code))
        previous = item
    # Half-open intervals, swept by start time (a stable sort: a tie keeps document order). A
    # span with no width or inverted is empty, so it overlaps nothing; it is out of range already.
    latest_end = 0
    for item in sorted(
        (item for item in placed if item.start_ms < item.end_ms), key=lambda item: item.start_ms
    ):
        if item.start_ms < latest_end:
            issues.append(DocumentIssue(item.path, overlap_code))
        latest_end = max(latest_end, item.end_ms)


def _check_sections(
    spans: Sequence[SectionSpan],
    plan: PlanDocument,
    duration_ms: int,
    issues: list[DocumentIssue],
) -> dict[UUID, SectionSpan]:
    """Check the section spans; returns the first span of each section id (what lines sit in)."""
    plan_rank = {section.id: index for index, section in enumerate(plan.sections)}
    timed: dict[UUID, SectionSpan] = {}
    placed: list[_Placed] = []
    for index, span in enumerate(spans):
        path = f"sections[{index}]"
        _check_range(span, path, duration_ms, issues)
        rank = plan_rank.get(span.section_id)
        if rank is None:
            issues.append(DocumentIssue(path, UNKNOWN_SECTION))
        if span.section_id in timed:
            issues.append(DocumentIssue(path, DUPLICATE_SECTION))
            continue
        timed[span.section_id] = span
        if rank is not None:
            placed.append(_Placed(path, rank, span.start_ms, span.end_ms))
    _check_sequence(placed, order_code=SECTION_ORDER, overlap_code=SECTION_OVERLAP, issues=issues)
    return timed


def _line_keys(lyrics: LyricsDocument | None) -> list[tuple[UUID, int]]:
    """Every ``(section_id, line_index)`` of the lyrics, in lyrics order."""
    if lyrics is None:
        return []
    return [
        (section.section_id, line_index)
        for section in lyrics.sections
        for line_index in range(len(section.lines))
    ]


def _check_lines(
    spans: Sequence[LineSpan],
    lyrics: LyricsDocument | None,
    timed_sections: Mapping[UUID, SectionSpan],
    duration_ms: int,
    issues: list[DocumentIssue],
) -> None:
    line_rank = {key: rank for rank, key in enumerate(_line_keys(lyrics))}
    seen: set[tuple[UUID, int]] = set()
    placed: list[_Placed] = []
    for index, line in enumerate(spans):
        path = f"lines[{index}]"
        _check_range(line, path, duration_ms, issues)
        _check_words(line, path, duration_ms, issues)
        key = (line.section_id, line.line_index)
        duplicate = key in seen
        seen.add(key)
        if duplicate:
            issues.append(DocumentIssue(path, DUPLICATE_LINE))
        rank = line_rank.get(key)
        if rank is None:
            # Nothing to place it by: a line the lyrics do not have belongs to no section.
            issues.append(DocumentIssue(path, UNKNOWN_LINE))
            continue
        section = timed_sections.get(line.section_id)
        if section is None:
            issues.append(DocumentIssue(path, LINE_SECTION_UNTIMED))
        elif not (section.start_ms <= line.start_ms and line.end_ms <= section.end_ms):
            issues.append(DocumentIssue(path, LINE_OUTSIDE_SECTION))
        if not duplicate:
            placed.append(_Placed(path, rank, line.start_ms, line.end_ms))
    _check_sequence(placed, order_code=LINE_ORDER, overlap_code=LINE_OVERLAP, issues=issues)


def _check_words(
    line: LineSpan, line_path: str, duration_ms: int, issues: list[DocumentIssue]
) -> None:
    placed: list[_Placed] = []
    for index, word in enumerate(line.words):
        path = f"{line_path}.words[{index}]"
        _check_range(word, path, duration_ms, issues)
        if not (line.start_ms <= word.start_ms and word.end_ms <= line.end_ms):
            issues.append(DocumentIssue(path, WORD_OUTSIDE_LINE))
        # Words have no identity of their own: the order they are listed in is their order.
        placed.append(_Placed(path, index, word.start_ms, word.end_ms))
    _check_sequence(placed, order_code=WORD_ORDER, overlap_code=WORD_OVERLAP, issues=issues)


# -- completeness ----------------------------------------------------------------------------------


def is_complete(
    document: TimelineDocument, *, plan: PlanDocument, lyrics: LyricsDocument | None
) -> bool:
    """Whether everything is timed: every plan section has a span **and**, when the audio has
    lyrics, every lyric line of every lyrics section has one.

    An instrumental section has no lines in valid lyrics, so its span is all it needs. Meant for a
    document that passed :func:`validate_timeline`.
    """
    timed_sections = {span.section_id for span in document.sections}
    if any(section.id not in timed_sections for section in plan.sections):
        return False
    timed_lines = {(span.section_id, span.line_index) for span in document.lines}
    return all(key in timed_lines for key in _line_keys(lyrics))
