"""From a parse to documents: the pure half of the Suno import (no I/O).

``suno.py`` reads text; this module decides what the plan and the lyrics *are*, given the track's
current versions:

* **Section ids** (I3). A section keeps the id of the first still-unused section of the current
  plan that it continues — first by the **whole header line** (a label may itself contain ``:``,
  and an unchanged header must reclaim its own section even after a reorder), then by the
  rendered label equal to the imported header name — so the lyrics, and everything that will
  refer to a section later, survive an edit. A section with no counterpart gets a deterministic
  ``uuid5`` — importing the same text twice gives the same ids.
* **Carry-over** (I3a). Export -> Suno -> import must be a no-op for an unchanged song, so what a
  header cannot say is taken from the current plan: a section whose *whole header line* is
  byte-identical to the current plan's is copied verbatim; any other matched section keeps the
  current plan's duration, notes and local avoid list; the plan keeps its idea and references; the
  lyrics keep their syllable range.
* **Title and flag** (I9). What the request leaves out is the current plan's, not a default: the
  title falls back to the current plan's, then the track's; ``instrumental`` to the current
  plan's, then ``false``.
* **Artist tags** are taken back out of Style / Exclude (:func:`subtract_artist_tags`).

The documents returned are *not normalized*: the service runs ``normalize_plan`` /
``normalize_lyrics`` on them and uses :class:`Provenance` to point any issue at the pasted line.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, Literal
from uuid import UUID, uuid5

import msgspec

from src.studio.documents import (
    LyricLine,
    LyricsDocument,
    LyricsSection,
    PlanDocument,
    PlanSection,
    StyleSpec,
)
from src.studio.errors import DocumentIssue, ImportInvalid
from src.studio.importers.subtraction import subtract_artist_tags
from src.studio.importers.suno import ImportWarning
from src.studio.renderers.base import dedupe_ci
from src.studio.renderers.suno import render_header
from src.studio.section_names import default_labels, rendered_labels
from src.studio.tempo import tempo_of

if TYPE_CHECKING:
    from collections.abc import Sequence

    from src.core.enums import StudioSectionKind
    from src.studio.importers.suno import ParsedSection, ParsedSuno
    from src.studio.renderers.base import ArtistRenderInput

__all__ = ["BuiltImport", "Provenance", "build_documents", "to_import_invalid"]

_SECTION_PATH: Final = re.compile(r"^sections\[(\d+)\](?:\.lines\[(\d+)\])?")
_ID_NAMESPACE: Final = "suno-import"


@dataclass(frozen=True, slots=True)
class Provenance:
    """Where each part of the built documents came from in the pasted Lyrics (1-based lines)."""

    plan_lines: tuple[int | None, ...]
    """Per plan section: its header line (or its first lyric line when it has no header)."""
    lyrics_plan_index: tuple[int, ...]
    """Per lyrics section: the index of the plan section it belongs to."""
    lyrics_lines: tuple[tuple[int, ...], ...]
    """Per lyrics section, per line: the line it was read from."""

    def source_line(self, document: Literal["plan", "lyrics"], path: str) -> int | None:
        """The pasted line behind a document path (``sections[i]...``), if there is one."""
        match = _SECTION_PATH.match(path)
        if match is None:
            return None
        index = int(match.group(1))
        if document == "plan":
            return self.plan_lines[index] if index < len(self.plan_lines) else None
        if index >= len(self.lyrics_plan_index):
            return None
        if match.group(2) is not None:
            line = int(match.group(2))
            if line < len(self.lyrics_lines[index]):
                return self.lyrics_lines[index][line]
        return self.plan_lines[self.lyrics_plan_index[index]]


@dataclass(frozen=True, slots=True)
class BuiltImport:
    plan: PlanDocument
    lyrics: LyricsDocument | None
    """``None`` when the pasted Lyrics have no lyric line (headers only, or nothing)."""
    provenance: Provenance
    warnings: tuple[ImportWarning, ...]
    """The parser's warnings, then the builder's."""


def to_import_invalid(
    issues: Sequence[DocumentIssue], document: Literal["plan", "lyrics"], provenance: Provenance
) -> ImportInvalid:
    """An :class:`ImportInvalid` for document issues, with paths prefixed and mapped to lines.

    ``sections[2].label`` of the plan becomes ``plan.sections[2].label``; the line of the header
    it was read from goes to ``source_lines`` under the same path.
    """
    mapped: list[DocumentIssue] = []
    lines: dict[str, int] = {}
    for issue in issues:
        path = f"{document}.{issue.path}" if issue.path else document
        mapped.append(DocumentIssue(path, issue.code))
        line = provenance.source_line(document, issue.path)
        if line is not None:
            lines[path] = line
    return ImportInvalid(tuple(mapped), lines)


def build_documents(
    parsed: ParsedSuno,
    *,
    track_id: UUID,
    track_title: str,
    title: str | None,
    instrumental: bool | None,
    language: str | None,
    parent_plan: PlanDocument | None,
    parent_lyrics: LyricsDocument | None,
    artist: ArtistRenderInput | None,
) -> BuiltImport:
    """Build the plan (and the lyrics, if there are lyric lines) a parse stands for.

    Args:
        parsed: The parse of the pasted boxes.
        track_id: The track; seeds the ids of sections that have no counterpart.
        track_title: The track's title, the last fallback for the plan title.
        title: The request's plan title; ``None`` keeps the current plan's, else the track's.
        instrumental: The request's switch; ``None`` keeps the current plan's, else ``False``.
            Never inferred from the text (I7).
        language: Lyrics language, already resolved (request -> current lyrics -> artist).
        parent_plan: The track's current plan, if any.
        parent_lyrics: The track's current lyrics, if any.
        artist: The project artist's exportable style, if any.

    Raises:
        ImportInvalid: ``reserved_tempo_tag`` (the current plan holds a global tag the importer
            reads as a tempo, from before the tag was reserved: fix the plan first),
            ``instrumental_with_lyric_lines`` (the plan is, or was asked to be, instrumental and
            the Lyrics have lines) or ``language_required`` (lines, and no language to give them).
    """
    if parent_plan is not None and (legacy := tuple(
        DocumentIssue(
            f"plan.global_style.tags[{index}]", "reserved_tempo_tag"
        )
        for index, tag in enumerate(parent_plan.global_style.tags)
        if tempo_of(tag) is not None
    )):
        raise ImportInvalid(legacy)

    if title is None:
        title = parent_plan.title if parent_plan is not None else track_title
    if instrumental is None:
        instrumental = parent_plan.instrumental if parent_plan is not None else False

    sections = parsed.sections
    line_numbers = tuple(line.source_line for section in sections for line in section.lines)
    if line_numbers and instrumental:
        raise ImportInvalid(
            (DocumentIssue("lyrics", "instrumental_with_lyric_lines"),), {"lyrics": line_numbers}
        )
    if line_numbers and language is None:
        raise ImportInvalid((DocumentIssue("language", "language_required"),))

    parent_sections = parent_plan.sections if parent_plan is not None else ()
    parent_labels = rendered_labels((s.kind, s.label) for s in parent_sections)
    parent_headers = [
        render_header(parent, label)
        for parent, label in zip(parent_sections, parent_labels, strict=True)
    ]
    matches = _match_parent_sections(sections, parent_labels, parent_headers)
    plan_sections = _build_plan_sections(
        track_id, sections, parent_sections, parent_headers, matches
    )

    parent_style = parent_plan.global_style if parent_plan is not None else None
    tags = _plan_tokens(
        parsed.style_tokens,
        artist.style_tags if artist else (),
        parent_style.tags if parent_style else None,
    )
    avoid = _plan_tokens(
        parsed.exclude_tokens,
        artist.global_avoid if artist else (),
        parent_style.avoid if parent_style else None,
    )
    plan = PlanDocument(
        title=title,
        idea=parent_plan.idea if parent_plan is not None else None,
        tempo_bpm=parsed.tempo_bpm,
        instrumental=instrumental,
        global_style=StyleSpec(tags=tuple(tags), avoid=tuple(avoid)),
        references=parent_plan.references if parent_plan is not None else (),
        sections=tuple(plan_sections),
    )

    warnings = list(parsed.warnings)
    lyrics: LyricsDocument | None = None
    lyrics_index: list[int] = []
    if line_numbers:
        assert language is not None  # noqa: S101 — checked above
        lyrics_index = [i for i, section in enumerate(sections) if section.lines]
        lyrics = LyricsDocument(
            language=language,
            line_syllables=parent_lyrics.line_syllables if parent_lyrics is not None else None,
            sections=tuple(
                LyricsSection(
                    section_id=plan_sections[i].id,
                    lines=tuple(
                        LyricLine(text=ln.text, delivery=ln.delivery) for ln in sections[i].lines
                    ),
                )
                for i in lyrics_index
            ),
        )
    elif sections and not instrumental:
        warnings.append(ImportWarning("no_lyric_lines"))

    provenance = Provenance(
        plan_lines=tuple(
            section.source_line
            if section.source_line is not None
            else (section.lines[0].source_line if section.lines else None)
            for section in sections
        ),
        lyrics_plan_index=tuple(lyrics_index),
        lyrics_lines=tuple(tuple(ln.source_line for ln in sections[i].lines) for i in lyrics_index),
    )
    return BuiltImport(plan=plan, lyrics=lyrics, provenance=provenance, warnings=tuple(warnings))


def _plan_tokens(
    tokens: Sequence[str], artist_tags: Sequence[str], parent_tags: Sequence[str] | None
) -> list[str]:
    """The plan's tags (or avoid list) behind the pasted ``tokens``.

    Rendering is not injective: plan ``[a]`` with artist ``[a, b]`` and plan ``[]`` with the same
    artist both write ``a, b``. The pure inverse (:func:`subtract_artist_tags`) can only pick the
    smallest plan, which would rewrite a plan that has not changed. So when the **current plan's
    own tags already render to exactly what was pasted**, they are kept (I3a's principle: an
    unchanged song imports as a no-op).
    """
    if parent_tags is not None and dedupe_ci([*parent_tags, *artist_tags]) == list(tokens):
        return list(parent_tags)
    return subtract_artist_tags(tokens, artist_tags)


def _match_parent_sections(
    sections: Sequence[ParsedSection],
    parent_labels: Sequence[str],
    parent_headers: Sequence[str],
) -> list[int | None]:
    """For each imported section, the index of the parent section it continues (I3).

    Two passes, each taking the first parent section that is not yet taken:

    1. **The whole header line.** The parent whose rendered header (``render_header``) equals the
       imported header line exactly. A label may itself contain ``:`` (``Verse 1: The Return``),
       which the parser reads as name + descriptor, so the name alone would never find it again;
       and an unchanged header reclaims its own section even when sections were reordered.
    2. **The rendered label.** For what is still unmatched, the parent whose rendered label
       (explicit, else the default for its position) equals the imported header name exactly.

    A section without a header (implicit) never matches.
    """
    taken: set[int] = set()
    matches: list[int | None] = [None] * len(sections)

    def claim(index: int, candidates: Sequence[str], wanted: str) -> None:
        found = next(
            (i for i, value in enumerate(candidates) if i not in taken and value == wanted), None
        )
        if found is not None:
            taken.add(found)
            matches[index] = found

    for index, section in enumerate(sections):
        if section.header_line is not None:
            claim(index, parent_headers, section.header_line)
    for index, section in enumerate(sections):
        if matches[index] is None and section.header_line is not None:
            claim(index, parent_labels, section.header_name)
    return matches


def _build_plan_sections(
    track_id: UUID,
    sections: Sequence[ParsedSection],
    parent_sections: Sequence[PlanSection],
    parent_headers: Sequence[str],
    matches: Sequence[int | None],
) -> list[PlanSection]:
    # I3a(a): a matched section whose whole header line is unchanged is the parent's, verbatim.
    verbatim = [
        match is not None and section.header_line == parent_headers[match]
        for section, match in zip(sections, matches, strict=True)
    ]
    kinds: list[StudioSectionKind] = [
        parent_sections[match].kind if is_verbatim and match is not None else section.kind
        for section, match, is_verbatim in zip(sections, matches, verbatim, strict=True)
    ]
    # Numbering depends on every kind, so labels are decided once the kinds are final.
    defaults = default_labels(kinds)
    ids = _section_ids(track_id, sections, parent_sections, matches)

    built: list[PlanSection] = []
    for index, section in enumerate(sections):
        match = matches[index]
        parent = parent_sections[match] if match is not None else None
        if parent is not None and verbatim[index]:
            label = parent.label
            if label is None and defaults[index] != section.header_name:
                # The parent relied on a number that the new neighbours changed: pin the name.
                label = section.header_name
            built.append(msgspec.structs.replace(parent, label=label))
            continue
        explicit = section.header_line is not None and section.header_name != defaults[index]
        built.append(
            PlanSection(
                id=ids[index],
                kind=section.kind,
                label=section.header_name if explicit else None,
                # I3a(b): what a header cannot carry stays with the section.
                duration_seconds=parent.duration_seconds if parent is not None else None,
                style=StyleSpec(
                    tags=section.descriptors, avoid=parent.style.avoid if parent is not None else ()
                ),
                delivery=(),
                instrumental=section.instrumental,
                notes=parent.notes if parent is not None else None,
            )
        )
    return built


def _section_ids(
    track_id: UUID,
    sections: Sequence[ParsedSection],
    parent_sections: Sequence[PlanSection],
    matches: Sequence[int | None],
) -> list[UUID]:
    """A matched section keeps its parent's id; the rest get a deterministic ``uuid5``."""
    ids: list[UUID | None] = [
        parent_sections[match].id if match is not None else None for match in matches
    ]
    used = {id_ for id_ in ids if id_ is not None}
    for ordinal, (section, id_) in enumerate(zip(sections, ids, strict=True)):
        if id_ is not None:
            continue
        attempt = 0
        while True:
            name = f"{_ID_NAMESPACE}:{ordinal}:{section.header_name}"
            candidate = uuid5(track_id, name if attempt == 0 else f"{name}:{attempt}")
            if candidate not in used:
                break
            attempt += 1  # a parent that is not matched here already owns this id
        used.add(candidate)
        ids[ordinal] = candidate
    return [id_ for id_ in ids if id_ is not None]
