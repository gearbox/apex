"""Section names: one definition shared by the renderer, the validators and the Suno importer.

A Suno lyrics box marks sections with bracketed headers (``[Verse 1]``, ``[Chorus: whispered]``).
Three places must agree on what such a header *means*:

* the renderer writes ``[<label>: <descriptors>]`` headers (:func:`default_labels`,
  :func:`rendered_labels`);
* the validators refuse lyric-line delivery tags that the importer would read back as a section
  header (:func:`is_header_text`);
* the importer maps a header name onto a :class:`~src.core.enums.StudioSectionKind`
  (:func:`resolve_kind`, :func:`split_header`).

Keeping them in this one pure module is what makes ``render -> import -> render`` stable.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Final

from src.core.enums import StudioSectionKind as K

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

__all__ = [
    "ALIASES",
    "UNNUMBERED_KINDS",
    "default_labels",
    "display_name",
    "is_header_text",
    "rendered_labels",
    "resolve_kind",
    "split_header",
]

ALIASES: Final[Mapping[str, K]] = {
    "intro": K.INTRO,
    "verse": K.VERSE,
    "pre chorus": K.PRE_CHORUS,
    "prechorus": K.PRE_CHORUS,
    "chorus": K.CHORUS,
    "refrain": K.CHORUS,
    "post chorus": K.POST_CHORUS,
    "postchorus": K.POST_CHORUS,
    "hook": K.HOOK,
    "bridge": K.BRIDGE,
    "build": K.BUILD,
    "build up": K.BUILD,
    "buildup": K.BUILD,
    "drop": K.DROP,
    "breakdown": K.BREAKDOWN,
    "break": K.BREAKDOWN,
    "instrumental": K.INSTRUMENTAL,
    "solo": K.INSTRUMENTAL,
    "interlude": K.INTERLUDE,
    "outro": K.OUTRO,
    "end": K.OUTRO,
    "fade out": K.OUTRO,
}
"""Header name -> section kind. Keys are already normalized by :func:`_key`."""

UNNUMBERED_KINDS: Final = frozenset({K.CHORUS, K.PRE_CHORUS, K.POST_CHORUS, K.HOOK})
"""Kinds that repeat by design: an unlabeled second chorus is still just "Chorus"."""

_DISPLAY_NAMES: Final[Mapping[K, str]] = {
    K.PRE_CHORUS: "Pre-Chorus",
    K.POST_CHORUS: "Post-Chorus",
}

_HEADER: Final = re.compile(r"^(?P<name>.+?)(?:\s+(?P<n>\d{1,2}))?$")
"""A header name with an optional trailing number: ``Verse``, ``Verse 2``, ``Pre-Chorus 12``."""


def _key(name: str) -> str:
    """Alias-table key of a header name: casefolded, ``-`` as a space, whitespace collapsed."""
    return " ".join(name.casefold().replace("-", " ").split())


def display_name(kind: K) -> str:
    """The name the renderer writes for a section of ``kind`` (``Pre-Chorus``, ``Breakdown``)."""
    return _DISPLAY_NAMES.get(kind) or kind.value.replace("_", " ").title()


def default_labels(kinds: Iterable[K]) -> list[str]:
    """Header label of each section *without* an explicit label, given only the kinds in order.

    The occurrence number ``n`` counts every section of the same kind, whether or not the others
    carry a custom label; kinds in :data:`UNNUMBERED_KINDS` never get one.
    """
    kind_list = list(kinds)
    totals: dict[K, int] = {}
    for kind in kind_list:
        totals[kind] = totals.get(kind, 0) + 1

    seen: dict[K, int] = {}
    labels: list[str] = []
    for kind in kind_list:
        seen[kind] = seen.get(kind, 0) + 1
        name = display_name(kind)
        if totals[kind] > 1 and kind not in UNNUMBERED_KINDS:
            name = f"{name} {seen[kind]}"
        labels.append(name)
    return labels


def rendered_labels(sections: Iterable[tuple[K, str | None]]) -> list[str]:
    """Header label per ``(kind, explicit label)``: the explicit label, else the default one."""
    items: Sequence[tuple[K, str | None]] = list(sections)
    defaults = default_labels(kind for kind, _ in items)
    return [label or default for (_, label), default in zip(items, defaults, strict=True)]


def resolve_kind(name: str) -> K | None:
    """The kind a header name stands for (``Verse 2`` -> verse, ``Refrain`` -> chorus), or ``None``.

    A trailing number of one or two digits is ignored. Matching is casefolded and treats ``-``
    like a space, so ``Pre-Chorus``, ``pre chorus`` and ``PRE-CHORUS 2`` are the same name.
    """
    match = _HEADER.match(name.strip())
    if match is None:
        return None
    return ALIASES.get(_key(match.group("name")))


def split_header(content: str) -> tuple[str, list[str]]:
    """Split the inside of a header bracket into its name and its descriptors.

    ``'Verse 1: whispered, cello'`` -> ``('Verse 1', ['whispered', 'cello'])``. The split is at the
    **first** ``:`` only (a descriptor may contain one); descriptors are split on ``,``, trimmed,
    and empty ones are dropped.
    """
    name, _, rest = content.partition(":")
    descriptors = [item.strip() for item in rest.split(",")]
    return name.strip(), [item for item in descriptors if item]


def is_header_text(content: str) -> bool:
    """Whether the text inside a bracket reads as a section header.

    True iff the part before the first ``:`` resolves to a kind (optionally followed by a number).
    """
    return resolve_kind(split_header(content)[0]) is not None
