"""Request bodies of ``POST /v1/studio/tracks/{id}/runs`` (a tagged union on ``kind``).

They live in the studio package, not in ``src/api/schemas``, because the planner reads them and
may not import the API layer. ``src/api/schemas/studio_runs.py`` re-exports them for the routes.
"""

from __future__ import annotations

from typing import Annotated
from uuid import UUID

import msgspec
from msgspec import Meta, Struct

from src.studio.constraints import collect_length_issues
from src.studio.documents import LanguageTag
from src.studio.errors import DocumentIssue, StudioDocumentInvalid
from src.studio.validation import normalize_block

__all__ = [
    "LyricsRunRequest",
    "PlanRunRequest",
    "ReviseLyricsRunRequest",
    "RunRequest",
    "normalize_plan_run_request",
]


class PlanRunRequest(
    Struct, forbid_unknown_fields=True, kw_only=True, tag="plan", tag_field="kind"
):
    """Plan a track from a brief."""

    brief: Annotated[str, Meta(min_length=1, max_length=4000)]
    """The user's own words about the song. Sent to the configured LLM provider."""
    language: LanguageTag | None = None
    memory_ids: Annotated[tuple[UUID, ...], Meta(max_length=50)] = ()
    """Artist memories to ground the plan in. Not available yet: non-empty is refused."""
    auto_fill: bool = True


class LyricsRunRequest(Struct, kw_only=True, tag="lyrics", tag_field="kind"):
    """Write lyrics for the current plan. Declared so the wire union is final; not available yet.

    Deliberately **without** ``forbid_unknown_fields``: any body of this kind decodes and is then
    refused with ``422 run_kind_not_available`` instead of a field error for a feature that has
    no fields yet.
    """


class ReviseLyricsRunRequest(Struct, kw_only=True, tag="revise_lyrics", tag_field="kind"):
    """Revise the current lyrics. Declared so the wire union is final; not available yet."""


RunRequest = PlanRunRequest | LyricsRunRequest | ReviseLyricsRunRequest
"""Tagged union decoded from the ``kind`` discriminant."""


def normalize_plan_run_request(request: PlanRunRequest) -> PlanRunRequest:
    """The request as it is stored and hashed: the brief NFC-normalized, and re-checked (RD6).

    Raises:
        StudioDocumentInvalid: The brief is empty after normalization, or NFC pushed it past the
            declared limit (``too_long_after_normalization``).
    """
    brief = normalize_block(request.brief)
    if brief is None:
        raise StudioDocumentInvalid((DocumentIssue("brief", "empty_after_normalization"),))
    normalized = msgspec.structs.replace(request, brief=brief)
    issues: list[DocumentIssue] = []
    collect_length_issues(normalized, issues)
    if issues:
        raise StudioDocumentInvalid(tuple(issues))
    return normalized
