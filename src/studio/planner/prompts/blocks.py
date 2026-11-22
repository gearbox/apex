"""Building blocks shared by every planner prompt.

User-supplied text (the artist description, the project concept, a brief, a plan, lyrics) is data,
not instruction: it is wrapped in labelled blocks, and ``<`` / ``>`` are removed from it first so a
value cannot close its own block early. Pure; a prompt that uses these is changed by adding a new
prompt version, never by editing a published one in place.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from src.studio.planner.context import ArtistSnapshot, PlanSummary

__all__ = ["REASK_MESSAGE", "artist_lines", "block", "clean", "plan_lines"]

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


def plan_lines(plan: PlanSummary) -> str:
    """The track's current plan, compactly: enough to plan around, not to copy."""
    lines = [f"title: {clean(plan.title)}"]
    if plan.tempo_bpm is not None:
        lines.append(f"tempo: {plan.tempo_bpm}")
    if plan.instrumental:
        lines.append("instrumental: yes")
    if plan.tags:
        lines.append("tags: " + ", ".join(clean(tag) for tag in plan.tags))
    if plan.avoid:
        lines.append("avoid: " + ", ".join(clean(tag) for tag in plan.avoid))
    for index, section in enumerate(plan.sections, start=1):
        label = f" ({clean(section.label)})" if section.label else ""
        flag = " [instrumental]" if section.instrumental else ""
        lines.append(f"section {index}: {section.kind.value}{label}{flag}")
    return "\n".join(lines)
