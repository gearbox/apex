"""The round-trip report: does rendering the imported documents give back what was pasted?

Information only (it never fails an import). ``false`` means the import changed the text in some
way worth a look — a tempo spelt ``120bpm`` is written back as ``120 BPM``, a bracket that was
dropped as a dangling delivery is gone, two blank lines became one.

Both sides are canonicalised first so that differences nobody would call a change (line endings,
trailing blanks, padding) do not count.
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass

__all__ = ["RoundTrip", "canonical_items", "canonical_lyrics", "compare"]


@dataclass(frozen=True, slots=True)
class RoundTrip:
    style: bool
    exclude: bool
    lyrics: bool


def _nfc(value: str) -> str:
    return unicodedata.normalize("NFC", value).replace("\r\n", "\n").replace("\r", "\n")


def canonical_items(value: str) -> str:
    """A Style / Exclude box: NFC, split on ``,``, each item trimmed, rejoined with ``", "``."""
    return ", ".join(item.strip() for item in _nfc(value).split(","))


def canonical_lyrics(value: str) -> str:
    """The Lyrics box: NFC, ``\\n`` line ends, lines trimmed, blank runs collapsed and trimmed."""
    lines: list[str] = []
    for raw in _nfc(value).split("\n"):
        line = raw.strip()
        if line or (lines and lines[-1]):
            lines.append(line)
    while lines and not lines[-1]:
        lines.pop()
    return "\n".join(lines)


def compare(*, pasted: tuple[str, str, str], rendered: tuple[str, str, str]) -> RoundTrip:
    """Compare ``(style, exclude, lyrics)`` pasted against rendered, canonicalised."""
    return RoundTrip(
        style=canonical_items(pasted[0]) == canonical_items(rendered[0]),
        exclude=canonical_items(pasted[1]) == canonical_items(rendered[1]),
        lyrics=canonical_lyrics(pasted[2]) == canonical_lyrics(rendered[2]),
    )
