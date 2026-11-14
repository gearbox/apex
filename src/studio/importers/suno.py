"""Parser for the text boxes of Suno's create page: Style, Exclude and Lyrics.

This is the inverse of ``src/studio/renderers/suno.py`` and shares its vocabulary through
``src/studio/section_names.py``. It is **pure** (no I/O, no clock, no randomness): the same three
strings always give the same :class:`ParsedSuno`. Turning that into plan and lyrics documents —
section ids, carry-over from the current versions, artist tags — is the build step in
``suno_build.py``; persisting them is ``services/importer.py``.

The parser is lenient and never silent: whatever it has to guess or drop becomes an
:class:`ImportWarning` carrying the 1-based line of the Lyrics it concerns. The one thing it
refuses is a bracket inside a lyric line (``unsupported_inline_tag``), because the studio forbids
brackets in text and there is no faithful reading of one.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Final

from src.core.enums import StudioSectionKind
from src.studio.errors import DocumentIssue, ImportInvalid
from src.studio.renderers.base import dedupe_ci
from src.studio.section_names import is_header_text, resolve_kind, split_header
from src.studio.tempo import tempo_of
from src.studio.validation import normalize_text

__all__ = [
    "ImportWarning",
    "ParsedLine",
    "ParsedSection",
    "ParsedSuno",
    "parse_suno",
    "split_items",
]

_BRACKET: Final = re.compile(r"^\[(.+)\]$")
_INSTRUMENTAL: Final = "instrumental"


@dataclass(frozen=True, slots=True)
class ImportWarning:
    """Something the parser guessed, dropped or did not understand."""

    code: str
    line: int | None = None
    """1-based line in the Lyrics input; ``None`` for Style / Exclude / global warnings."""


@dataclass(frozen=True, slots=True)
class ParsedLine:
    text: str
    delivery: tuple[str, ...]
    source_line: int


@dataclass(frozen=True, slots=True)
class ParsedSection:
    header_name: str
    """Exactly as written (after NFC + trim); ``""`` for an implicit section."""
    header_line: str | None
    """The whole bracket line, canonical (``[Verse 1: whispered]``); ``None`` if implicit."""
    kind: StudioSectionKind
    descriptors: tuple[str, ...]
    """After removing the ``instrumental`` cue."""
    instrumental: bool
    lines: tuple[ParsedLine, ...]
    source_line: int | None


@dataclass(frozen=True, slots=True)
class ParsedSuno:
    style_tokens: tuple[str, ...]
    """BPM removed; the artist's tags are **not** subtracted yet (the service does that)."""
    tempo_bpm: int | None
    exclude_tokens: tuple[str, ...]
    sections: tuple[ParsedSection, ...]
    warnings: tuple[ImportWarning, ...]


def split_items(text: str) -> list[str]:
    """Comma-separated items of a Style / Exclude box: NFC, trimmed, empty ones skipped."""
    items = (normalize_text(item) for item in text.split(","))
    return [item for item in items if item]


def parse_suno(*, style: str, exclude: str, lyrics: str) -> ParsedSuno:
    """Parse the three boxes of a Suno song.

    Args:
        style: The Style box.
        exclude: The Exclude box.
        lyrics: The Lyrics box (``\\r\\n`` and ``\\r`` are read as ``\\n``).

    Returns:
        The parse, with every warning. Deterministic.

    Raises:
        ImportInvalid: A lyric line contains ``[`` or ``]`` without being a whole bracket line
            (``unsupported_inline_tag``, one issue per such line).
    """
    warnings: list[ImportWarning] = []
    style_tokens, tempo = _parse_style(style, warnings)
    exclude_tokens = _dedupe(split_items(exclude), warnings)
    sections = _parse_lyrics(lyrics, warnings)
    return ParsedSuno(
        style_tokens=tuple(style_tokens),
        tempo_bpm=tempo,
        exclude_tokens=tuple(exclude_tokens),
        sections=tuple(sections),
        warnings=tuple(warnings),
    )


# -- Style / Exclude --------------------------------------------------------------------------


def _dedupe(tokens: list[str], warnings: list[ImportWarning]) -> list[str]:
    """Drop case-insensitive repeats (first wins), one ``duplicate_style_item`` per drop."""
    unique = dedupe_ci(tokens)
    warnings.extend(ImportWarning("duplicate_style_item") for _ in range(len(tokens) - len(unique)))
    return unique


def _parse_style(style: str, warnings: list[ImportWarning]) -> tuple[list[str], int | None]:
    """Style tokens without the tempo, and the tempo (the first item :func:`tempo_of` reads).

    Later tempo items are **dropped** (``duplicate_tempo``), never kept as tags: a tag that reads
    as a tempo is reserved (``reserved_tempo_tag``), so the plan could not hold it.
    """
    tempo: int | None = None
    rest: list[str] = []
    for token in split_items(style):
        bpm = tempo_of(token)
        if bpm is None:
            rest.append(token)
        elif tempo is None:
            tempo = bpm
        else:
            warnings.append(ImportWarning("duplicate_tempo"))
    return _dedupe(rest, warnings), tempo


# -- Lyrics -----------------------------------------------------------------------------------


class _Kind(Enum):
    BLANK = auto()
    HEADER = auto()
    DELIVERY = auto()
    TEXT = auto()


@dataclass(frozen=True, slots=True)
class _Line:
    number: int
    kind: _Kind
    text: str
    """The canonical line: NFC, trimmed, whitespace collapsed."""
    content: str = ""
    """Inside the brackets (trimmed) of a header or delivery line."""


def _classify(lyrics: str) -> list[_Line]:
    """Tag every line as blank / header / delivery / text.

    A bracket line is a header when it opens a blank-line-separated block, or when what it holds
    reads as a section header; any other bracket line is the delivery of the next text line.
    """
    raw_lines = lyrics.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    result: list[_Line] = []
    bad: list[int] = []
    block_start = True
    for number, raw in enumerate(raw_lines, start=1):
        text = normalize_text(raw)
        if not text:
            result.append(_Line(number, _Kind.BLANK, ""))
            block_start = True
            continue
        match = _BRACKET.match(text)
        content = match.group(1).strip() if match else ""
        if match and content:
            kind = _Kind.HEADER if block_start or is_header_text(content) else _Kind.DELIVERY
            result.append(_Line(number, kind, text, content))
        elif "[" in text or "]" in text:
            bad.append(number)
        else:
            result.append(_Line(number, _Kind.TEXT, text))
        block_start = False
    if bad:
        raise ImportInvalid(
            tuple(DocumentIssue(f"lyrics[{n}]", "unsupported_inline_tag") for n in bad),
            {f"lyrics[{n}]": n for n in bad},
        )
    return result


@dataclass(slots=True)
class _Draft:
    """A section being assembled."""

    header_name: str
    header_line: str | None
    kind: StudioSectionKind
    descriptors: tuple[str, ...]
    instrumental: bool
    source_line: int | None
    lines: list[ParsedLine] = field(default_factory=list)

    def freeze(self) -> ParsedSection:
        return ParsedSection(
            header_name=self.header_name,
            header_line=self.header_line,
            kind=self.kind,
            descriptors=self.descriptors,
            instrumental=self.instrumental,
            lines=tuple(self.lines),
            source_line=self.source_line,
        )


def _header_draft(line: _Line, warnings: list[ImportWarning]) -> _Draft:
    name, raw_descriptors = split_header(line.content)
    if not name:
        # "[: x]": nothing before the colon to read as a name, so the whole text is the name.
        name, raw_descriptors = line.content, []
    instrumental = any(item.casefold() == _INSTRUMENTAL for item in raw_descriptors)
    kept = [item for item in raw_descriptors if item.casefold() != _INSTRUMENTAL]
    descriptors = dedupe_ci(kept)
    if len(descriptors) != len(kept):
        warnings.append(ImportWarning("duplicate_descriptor", line.number))
    kind = resolve_kind(name)
    if kind is None:
        kind = StudioSectionKind.INTERLUDE
        warnings.append(ImportWarning("unknown_section_name", line.number))
    return _Draft(
        header_name=name,
        header_line=f"[{line.content}]",
        kind=kind,
        descriptors=tuple(descriptors),
        instrumental=instrumental,
        source_line=line.number,
    )


def _delivery_of(line: _Line, warnings: list[ImportWarning]) -> tuple[str, ...]:
    """The tokens of a delivery line, deduplicated (``duplicate_descriptor`` when one was)."""
    tokens = [item.strip() for item in line.content.split(",")]
    kept = [token for token in tokens if token]
    unique = dedupe_ci(kept)
    if len(unique) != len(kept):
        warnings.append(ImportWarning("duplicate_descriptor", line.number))
    return tuple(unique)


def _parse_lyrics(lyrics: str, warnings: list[ImportWarning]) -> list[ParsedSection]:
    lines = _classify(lyrics)
    has_headers = any(line.kind is _Kind.HEADER for line in lines)

    drafts: list[_Draft] = []
    current: _Draft | None = None
    pending: tuple[tuple[str, ...], int] | None = None  # (delivery, its line)

    def drop_pending() -> None:
        nonlocal pending
        if pending is not None:
            warnings.append(ImportWarning("dangling_delivery", pending[1]))
            pending = None

    for line in lines:
        if line.kind is _Kind.BLANK:
            drop_pending()
        elif line.kind is _Kind.HEADER:
            drop_pending()
            current = _header_draft(line, warnings)
            drafts.append(current)
        elif line.kind is _Kind.DELIVERY:
            drop_pending()
            tokens = _delivery_of(line, warnings)
            if tokens:
                pending = (tokens, line.number)
            else:  # "[,]" — nothing to deliver
                warnings.append(ImportWarning("dangling_delivery", line.number))
        else:  # TEXT
            if current is None:
                code = "text_before_first_header" if has_headers else "no_section_headers"
                warnings.append(ImportWarning(code, line.number))
                current = _Draft(
                    header_name="",
                    header_line=None,
                    kind=StudioSectionKind.VERSE,
                    descriptors=(),
                    instrumental=False,
                    source_line=None,
                )
                drafts.append(current)
            delivery = pending[0] if pending is not None else ()
            pending = None
            current.lines.append(ParsedLine(line.text, delivery, line.number))
    drop_pending()
    return [draft.freeze() for draft in drafts]
