"""Studio domain errors.

Every client-facing studio failure is a :class:`StudioError`: a stable machine ``code``, an HTTP
``status_code`` and an optional structured ``detail``. One exception handler in ``src/api/app.py``
turns any of them into an ``ErrorEnvelope`` — no per-route try/except ladders (P1-8).

Messages are static strings: they never echo user-supplied text (names, tags, lyrics, ideas), so
nothing here can leak content into logs or error bodies.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

__all__ = [
    "ArtistInUse",
    "ArtistLimitReached",
    "ArtistNameConflict",
    "ArtistNotFound",
    "DocumentIssue",
    "InvalidCursor",
    "LyricsIncompatibleWithPlan",
    "ParentVersionNotFound",
    "PlanRequired",
    "PlanVersionNotFound",
    "ProjectHasTracks",
    "StudioDocumentInvalid",
    "StudioDocumentSchemaError",
    "StudioError",
    "StudioNotFound",
    "TrackLimitReached",
    "TrackOrderMismatch",
    "VersionHasChildren",
    "VersionIsCurrent",
    "VersionIsLyricsBase",
    "VersionLimitReached",
    "VersionUnchanged",
]


@dataclass(frozen=True, slots=True)
class DocumentIssue:
    """One validation problem: where (``path``) and what (``code``).

    ``path`` looks like ``sections[2].style.tags[1]``. It never contains document text.
    """

    path: str
    code: str


def _issues_detail(issues: tuple[DocumentIssue, ...]) -> dict[str, object]:
    return {"issues": [{"path": issue.path, "code": issue.code} for issue in issues]}


class StudioError(Exception):
    """Base of all client-facing studio errors."""

    code: ClassVar[str] = "studio_error"
    status_code: ClassVar[int] = 400
    message: ClassVar[str] = "Studio request failed."

    def __init__(self, detail: dict[str, object] | None = None) -> None:
        super().__init__(self.code)
        self.detail = detail


# --- 404 -------------------------------------------------------------------------------------


class StudioNotFound(StudioError):
    """The resource does not exist, or belongs to another user/product."""

    code = "not_found"
    status_code = 404
    message = "Resource not found."


class ArtistNotFound(StudioError):
    code = "artist_not_found"
    status_code = 404
    message = "Artist not found."


class ParentVersionNotFound(StudioError):
    """The parent version is missing, or belongs to another track or kind."""

    code = "parent_version_not_found"
    status_code = 404
    message = "Parent version not found on this track."


class PlanVersionNotFound(StudioError):
    """The plan version a lyrics version is based on is missing, or is not a plan of this track."""

    code = "plan_version_not_found"
    status_code = 404
    message = "Plan version not found on this track."


# --- 400 / 422 -------------------------------------------------------------------------------


class InvalidCursor(StudioError):
    code = "invalid_cursor"
    status_code = 400
    message = "Invalid pagination cursor."


class StudioDocumentInvalid(StudioError):
    """A document (or artist style) failed validation; carries *all* issues (P1-6)."""

    code = "document_invalid"
    status_code = 422
    message = "The document is invalid."

    def __init__(self, issues: tuple[DocumentIssue, ...]) -> None:
        super().__init__(_issues_detail(issues))
        self.issues = issues


class TrackOrderMismatch(StudioError):
    """The submitted track id list is not exactly the project's track set."""

    code = "track_order_mismatch"
    status_code = 422
    message = "The track order must list every track of the project exactly once."


# --- 409 -------------------------------------------------------------------------------------


class ArtistNameConflict(StudioError):
    code = "artist_name_conflict"
    status_code = 409
    message = "An artist with this name already exists."


class ArtistLimitReached(StudioError):
    code = "artist_limit_reached"
    status_code = 409
    message = "The maximum number of artists has been reached."


class ArtistInUse(StudioError):
    code = "artist_in_use"
    status_code = 409
    message = "The artist is used by one or more projects."

    def __init__(self, project_count: int | None = None) -> None:
        super().__init__(None if project_count is None else {"project_count": project_count})


class ProjectHasTracks(StudioError):
    code = "project_has_tracks"
    status_code = 409
    message = "The project still has tracks."


class TrackLimitReached(StudioError):
    code = "track_limit_reached"
    status_code = 409
    message = "The maximum number of tracks per project has been reached."


class VersionLimitReached(StudioError):
    code = "version_limit_reached"
    status_code = 409
    message = "The maximum number of versions for this document has been reached."


class VersionUnchanged(StudioError):
    """The content equals its parent's — there is nothing new to version."""

    code = "version_unchanged"
    status_code = 409
    message = "The content is identical to the parent version."


class PlanRequired(StudioError):
    code = "plan_required"
    status_code = 409
    message = "The track needs a plan first."


class LyricsIncompatibleWithPlan(StudioError):
    """The (plan, lyrics) pair is incompatible; ``detail.issues`` says why."""

    code = "lyrics_incompatible_with_plan"
    status_code = 409
    message = "The lyrics do not fit the plan."

    def __init__(self, issues: tuple[DocumentIssue, ...]) -> None:
        super().__init__(_issues_detail(issues))
        self.issues = issues


class VersionIsCurrent(StudioError):
    code = "version_is_current"
    status_code = 409
    message = "The version is the track's current version."


class VersionHasChildren(StudioError):
    code = "version_has_children"
    status_code = 409
    message = "Other versions were derived from this version."

    def __init__(self, count: int) -> None:
        super().__init__({"count": count})


class VersionIsLyricsBase(StudioError):
    code = "version_is_lyrics_base"
    status_code = 409
    message = "Lyrics versions are based on this plan version."

    def __init__(self, count: int) -> None:
        super().__init__({"count": count})


# --- 500-class (never a StudioError: must reach the global handler and be logged loudly) ------


class StudioDocumentSchemaError(Exception):
    """A stored document has a ``schema_version`` this code cannot decode — a deployment bug.

    Deliberately *not* a :class:`StudioError`: it must surface as a 500 with a traceback, never as
    a silent fallback decode.
    """
