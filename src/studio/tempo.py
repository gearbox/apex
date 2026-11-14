"""The one definition of what a tempo looks like in a Suno Style box.

The renderer writes ``tempo_bpm`` as the **last** Style item (``120 BPM``) and the importer reads
the first item that looks like a tempo back into ``tempo_bpm``. Both go through this module, so
they cannot drift apart.

A tag that :func:`tempo_of` would read as a tempo is **reserved** (``reserved_tempo_tag``): the
tempo lives in ``tempo_bpm`` only. That is what makes the importer's first-tempo rule an exact
inverse of the renderer — the renderer then writes at most one tempo token, always last. Without
the reservation ``120 BPM, rock`` (a typical hand-written Style, read as tempo 120) and the render
of tags ``[120 BPM, rock]`` would be the same text for two different documents, and no parsing
rule could satisfy both.

Pure: no I/O.
"""

from __future__ import annotations

import re
from typing import Final

__all__ = ["TEMPO_MAX", "TEMPO_MIN", "tempo_of", "tempo_token"]

TEMPO_MIN: Final = 40
TEMPO_MAX: Final = 240

_TEMPO: Final = re.compile(r"^(\d{2,3})\s*bpm$")


def tempo_of(token: str) -> int | None:
    """The tempo a Style item stands for.

    ``120 BPM`` / ``120bpm`` / ``120 bpm`` (any case, optional space) give ``120`` when the value
    is between :data:`TEMPO_MIN` and :data:`TEMPO_MAX` (inclusive); anything else — out of range
    (``300 BPM``), another word order (``BPM 120``), a longer number, extra text — gives ``None``
    and is an ordinary tag.

    Args:
        token: One item of a Style box (surrounding whitespace is ignored).
    """
    match = _TEMPO.match(token.strip().casefold())
    if match is None:
        return None
    value = int(match.group(1))
    return value if TEMPO_MIN <= value <= TEMPO_MAX else None


def tempo_token(bpm: int) -> str:
    """What the renderer writes for a tempo: ``120 BPM``."""
    return f"{bpm} BPM"
