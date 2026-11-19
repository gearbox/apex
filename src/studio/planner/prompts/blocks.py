"""Building blocks shared by every planner prompt.

User-supplied text (the artist description, the project concept, a brief, a plan, lyrics) is data,
not instruction: it is wrapped in labelled blocks, and ``<`` / ``>`` are removed from it first so a
value cannot close its own block early. Pure; a prompt that uses these is changed by adding a new
prompt version, never by editing a published one in place.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from src.studio.planner.context import ArtistSnapshot

__all__ = ["REASK_MESSAGE", "artist_lines", "block", "clean"]

REASK_MESSAGE: Final = (
    "Your reply was not valid for the schema. "
    "Return only one JSON object that matches the schema exactly."
)
"""Sent after a reply that did not decode; the invalid reply itself is never echoed back."""


def clean(text: str) -> str:
    """User text with the characters that could close a block removed."""
    return text.replace("<", "").replace(">", "")


def block(name: str, body: str) -> str:
    """``body`` as a labelled block."""
    return f"<{name}>\n{body}\n</{name}>"


def artist_lines(artist: ArtistSnapshot) -> str:
    """The artist's planner-relevant fields, one per line."""
    lines = [f"name: {clean(artist.name)}"]
    if artist.persona:
        lines.append(f"persona: {clean(artist.persona)}")
    if artist.voice_notes:
        lines.append(f"voice notes: {clean(artist.voice_notes)}")
    if artist.style_tags:
        lines.append("style tags: " + ", ".join(clean(tag) for tag in artist.style_tags))
    if artist.global_avoid:
        lines.append("things to avoid: " + ", ".join(clean(tag) for tag in artist.global_avoid))
    if artist.aesthetic_references:
        lines.append(
            "aesthetic references: " + ", ".join(clean(ref) for ref in artist.aesthetic_references)
        )
    return "\n".join(lines)
