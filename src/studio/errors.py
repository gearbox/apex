"""Studio domain errors.

Every client-facing studio failure is a :class:`StudioError`: a stable machine ``code``, an HTTP
``status_code`` and an optional structured ``detail``. One exception handler in ``src/api/app.py``
turns any of them into an ``ErrorEnvelope`` — no per-route try/except ladders (P1-8).

Messages are static strings: they never echo user-supplied text (names, tags, lyrics, ideas), so
nothing here can leak content into logs or error bodies.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

if TYPE_CHECKING:
    from collections.abc import Mapping

__all__ = [
    "ArtistBusy",
    "ArtistInUse",
    "ArtistLimitReached",
    "ArtistNameConflict",
    "ArtistNotFound",
    "AssetNotEligible",
    "AssetNotFound",
    "AudioVersionIsCurrent",
    "AudioVersionLimitReached",
    "AudioVersionNotFound",
    "CastLabelTaken",
    "CastLimitReached",
    "CastLookNameTaken",
    "CastLookNotFound",
    "CastMemberInUse",
    "CastMemberNotFound",
    "CastNameTaken",
    "CastReferenceNotFound",
    "ChatBusy",
    "ChatFull",
    "ChatLimitReached",
    "ChatNotFound",
    "ChatRateLimited",
    "ChatReplyInProgress",
    "DocumentIssue",
    "IdempotencyConflict",
    "IdempotencyKeyRequired",
    "ImportInvalid",
    "InvalidAssetRef",
    "InvalidCursor",
    "InvalidReferenceOrder",
    "LyricsIncompatibleWithPlan",
    "LyricsRequired",
    "LyricsVersionNotFound",
    "MemoryDraftInProgress",
    "MemoryDraftNotFound",
    "MemoryLimitReached",
    "MemoryNotFound",
    "MessageNotComplete",
    "MessageNotFound",
    "ParentVersionNotFound",
    "PlanRequired",
    "PlanVersionNotFound",
    "ProjectHasTracks",
    "RunInProgress",
    "RunLimitReached",
    "RunNotCancellable",
    "RunNotFound",
    "RunRateLimited",
    "StudioDocumentInvalid",
    "StudioDocumentSchemaError",
    "StudioError",
    "StudioNotFound",
    "TrackLimitReached",
    "TrackOrderMismatch",
    "VersionHasAudio",
    "VersionHasChildren",
    "VersionInUseByRun",
    "VersionIsCurrent",
    "VersionIsLyricsBase",
    "VersionKindMismatch",
    "VersionLimitReached",
    "VersionNotFound",
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


class LyricsVersionNotFound(StudioError):
    """The lyrics version a revision works on is missing, or is not a lyrics version of this track."""

    code = "lyrics_version_not_found"
    status_code = 404
    message = "Lyrics version not found on this track."


class MemoryNotFound(StudioError):
    """No such memory for the caller (unknown, or another artist's / user's / product's)."""

    code = "memory_not_found"
    status_code = 404
    message = "Memory not found."


class RunNotFound(StudioError):
    """No such run for the caller."""

    code = "run_not_found"
    status_code = 404
    message = "Run not found."


# --- cast (S1-P3) -----------------------------------------------------------------------------


class CastMemberNotFound(StudioError):
    """No such cast member for the caller (unknown, another user's, another product's)."""

    code = "cast_member_not_found"
    status_code = 404
    message = "Cast member not found."


class CastReferenceNotFound(StudioError):
    """No such reference on this cast member."""

    code = "cast_reference_not_found"
    status_code = 404
    message = "Cast reference not found."


class CastLookNotFound(StudioError):
    """No such look on this cast member."""

    code = "cast_look_not_found"
    status_code = 404
    message = "Cast look not found."


class VersionNotFound(StudioError):
    """A plan / lyrics version a request links to is not a version of this track."""

    code = "version_not_found"
    status_code = 404
    message = "Version not found on this track."


class AudioVersionNotFound(StudioError):
    """No such audio version for the caller (unknown, another track's, user's or product's)."""

    code = "audio_version_not_found"
    status_code = 404
    message = "Audio version not found."


class AssetNotFound(StudioError):
    """The library asset is missing, or owned by another user or another product (all the same)."""

    code = "asset_not_found"
    status_code = 404
    message = "Library asset not found."


class AssetNotEligible(StudioError):
    """The library asset cannot be used as a reference; ``detail.reason`` says why.

    ``reason`` is one of ``derivative``, ``media_kind_not_accepted``, ``job_not_available``.
    """

    code = "asset_not_eligible"
    status_code = 422
    message = "The library asset cannot be used as a reference."

    def __init__(self, reason: str) -> None:
        super().__init__({"reason": reason})


# --- 400 / 422 -------------------------------------------------------------------------------


class VersionKindMismatch(StudioError):
    """A realized-version link names a version of the other kind (a lyrics id as the plan link).

    ``detail`` is ``{"field": "plan_version_id" | "lyrics_version_id", "expected": "plan" |
    "lyrics"}`` - names only, never document text.
    """

    code = "version_kind_mismatch"
    status_code = 422
    message = "The version is not of the kind this link takes."

    def __init__(self, field: str, expected: str) -> None:
        super().__init__({"field": field, "expected": expected})


class IdempotencyKeyRequired(StudioError):
    """``POST .../runs`` needs a usable ``Idempotency-Key`` header (1-128 characters)."""

    code = "idempotency_key_required"
    status_code = 400
    message = "An Idempotency-Key header of 1 to 128 characters is required."


class InvalidCursor(StudioError):
    code = "invalid_cursor"
    status_code = 400
    message = "Invalid pagination cursor."


class InvalidAssetRef(StudioError):
    """A query parameter that must be an asset ref (``upload:<uuid>`` / ``output:<uuid>``) is not."""

    code = "invalid_asset_ref"
    status_code = 400
    message = "Invalid asset reference."


class StudioDocumentInvalid(StudioError):
    """A document (or artist style) failed validation; carries *all* issues (P1-6)."""

    code = "document_invalid"
    status_code = 422
    message = "The document is invalid."

    def __init__(self, issues: tuple[DocumentIssue, ...]) -> None:
        super().__init__(_issues_detail(issues))
        self.issues = issues


class ImportInvalid(StudioError):
    """Pasted Suno text cannot become a plan / lyrics; carries *all* issues and where they are.

    ``issues`` addresses the offending value (``lyrics[7]``, ``plan.sections[2].label``,
    ``plan.global_style.tags[1]``, ``lyrics.sections[0].lines[3].text``, ``language``). Where the
    value came from a line of the pasted Lyrics, ``source_lines`` maps the same path to that
    1-based line: an ``int``, or a list of them when one issue stands for several lines
    (``instrumental_with_lyric_lines``). Neither ever contains pasted text.
    """

    code = "import_invalid"
    status_code = 422
    message = "The pasted text cannot be imported."

    def __init__(
        self,
        issues: tuple[DocumentIssue, ...],
        source_lines: Mapping[str, int | tuple[int, ...]] | None = None,
    ) -> None:
        lines = {
            path: list(line) if isinstance(line, tuple) else line
            for path, line in (source_lines or {}).items()
        }
        super().__init__({**_issues_detail(issues), "source_lines": lines})
        self.issues = issues
        self.source_lines = dict(source_lines or {})


class TrackOrderMismatch(StudioError):
    """The submitted track id list is not exactly the project's track set."""

    code = "track_order_mismatch"
    status_code = 422
    message = "The track order must list every track of the project exactly once."


class InvalidReferenceOrder(StudioError):
    """The submitted reference id list is not exactly the member's reference set."""

    code = "invalid_order"
    status_code = 422
    message = "The reference order must list every reference of the cast member exactly once."


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


class MemoryLimitReached(StudioError):
    """The artist already holds the maximum number of active memories."""

    code = "memory_limit_reached"
    status_code = 409
    message = "The maximum number of active memories for this artist has been reached."


class CastNameTaken(StudioError):
    code = "cast_name_taken"
    status_code = 409
    message = "A cast member with this name already exists."


class CastLabelTaken(StudioError):
    code = "cast_label_taken"
    status_code = 409
    message = "This cast member already has a reference with this label."


class CastLookNameTaken(StudioError):
    code = "cast_look_name_taken"
    status_code = 409
    message = "This cast member already has a look with this name."


class CastLimitReached(StudioError):
    """A cap was reached; ``detail`` is ``{"kind": "members" | "references" | "looks", "limit"}``."""

    code = "cast_limit_reached"
    status_code = 409
    message = "The maximum number of cast items has been reached."

    def __init__(self, kind: str, limit: int) -> None:
        super().__init__({"kind": kind, "limit": limit})


class CastMemberInUse(StudioError):
    """An artist uses the member as its performer; ``detail.blockers`` counts them (D12 shape)."""

    code = "cast_member_in_use"
    status_code = 409
    message = "The cast member is the performer of one or more artists."

    def __init__(self, artist_count: int) -> None:
        super().__init__({"blockers": [{"code": "artists", "count": artist_count}]})


class AudioVersionLimitReached(StudioError):
    """A track holds at most ``limit`` audio versions; ``detail`` is ``{"limit": n}``."""

    code = "audio_version_limit_reached"
    status_code = 409
    message = "The maximum number of audio versions for this track has been reached."

    def __init__(self, limit: int) -> None:
        super().__init__({"limit": limit})


class AudioVersionIsCurrent(StudioError):
    code = "audio_version_is_current"
    status_code = 409
    message = "The audio version is the track's current audio."


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

    def __init__(self, *, status_code: int | None = None) -> None:
        """Allow attach validation to use its specified 422 while existing callers stay 409."""
        super().__init__()
        if status_code is not None:
            self.status_code = status_code  # type: ignore[misc]


class LyricsRequired(StudioError):
    """A revision needs lyrics, and the track has none (and none was named)."""

    code = "lyrics_required"
    status_code = 409
    message = "The track needs lyrics first."


class LyricsIncompatibleWithPlan(StudioError):
    """The (plan, lyrics) pair is incompatible; ``detail.issues`` says why."""

    code = "lyrics_incompatible_with_plan"
    status_code = 409
    message = "The lyrics do not fit the plan."

    def __init__(self, issues: tuple[DocumentIssue, ...]) -> None:
        super().__init__(_issues_detail(issues))
        self.issues = issues


class RunInProgress(StudioError):
    """The track already has a queued or running run (also: the track cannot be deleted)."""

    code = "run_in_progress"
    status_code = 409
    message = "The track already has a run in progress."


class RunLimitReached(StudioError):
    """The user already has the maximum number of queued or running runs."""

    code = "run_limit_reached"
    status_code = 409
    message = "The maximum number of active runs has been reached."


class RunNotCancellable(StudioError):
    """Only a ``queued`` run can be cancelled."""

    code = "run_not_cancellable"
    status_code = 409
    message = "Only a queued run can be cancelled."


class IdempotencyConflict(StudioError):
    """The Idempotency-Key was already used for a different request."""

    code = "idempotency_conflict"
    status_code = 409
    message = "This Idempotency-Key was already used with a different request."


class RunRateLimited(StudioError):
    """Too many runs created in the last hour."""

    code = "run_rate_limited"
    status_code = 429
    message = "Too many runs were started recently; try again later."


# --- artist chat (S1-P2c) ---------------------------------------------------------------------


class ChatNotFound(StudioError):
    """No such chat for the caller (unknown, another user's, another product's)."""

    code = "chat_not_found"
    status_code = 404
    message = "Chat not found."


class MessageNotFound(StudioError):
    """No such chat message for the caller."""

    code = "message_not_found"
    status_code = 404
    message = "Message not found."


class MemoryDraftNotFound(StudioError):
    """No such memory draft for the caller."""

    code = "memory_draft_not_found"
    status_code = 404
    message = "Memory draft not found."


class ChatLimitReached(StudioError):
    """The artist already has the maximum number of chats."""

    code = "chat_limit_reached"
    status_code = 409
    message = "The artist has the maximum number of chats."


class ChatFull(StudioError):
    """The chat holds the maximum number of messages: start a new chat."""

    code = "chat_full"
    status_code = 409
    message = "This chat is full; start a new chat."


class ChatReplyInProgress(StudioError):
    """One reply at a time per chat: the previous message has not been answered yet."""

    code = "chat_reply_in_progress"
    status_code = 409
    message = "The artist is still answering the previous message."


class ChatBusy(StudioError):
    """A chat with a pending reply or memory draft cannot be deleted."""

    code = "chat_busy"
    status_code = 409
    message = "The chat has a reply or a memory draft in progress."


class ArtistBusy(StudioError):
    """An artist with a pending chat reply or memory draft cannot be deleted (S1-P2c R1).

    The cascade ``artist → chats → messages / drafts`` would delete a paid, pending row whose
    reservation then never gets refunded; the user retries in a few seconds, as for ``ChatBusy``.
    """

    code = "artist_busy"
    status_code = 409
    message = "The artist has a chat reply or memory draft in progress."


class MemoryDraftInProgress(StudioError):
    """A draft of this message is already being written."""

    code = "memory_draft_in_progress"
    status_code = 409
    message = "A memory draft of this message is already in progress."


class MessageNotComplete(StudioError):
    """Only a ``complete`` message can be the source of a memory draft."""

    code = "message_not_complete"
    status_code = 422
    message = "Only a completed message can be turned into a memory draft."


class ChatRateLimited(StudioError):
    """Too many replies and drafts pending, or created in the last hour."""

    code = "chat_rate_limited"
    status_code = 429
    message = "Too many chat requests were made recently; try again later."


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


class VersionInUseByRun(StudioError):
    """An active run takes this version as its input: deleting it would lose the paid result."""

    code = "version_in_use_by_run"
    status_code = 409
    message = "A run in progress is working on this version."


class VersionHasAudio(StudioError):
    """Audio versions realize this plan / lyrics version; ``detail.count`` says how many."""

    code = "version_has_audio"
    status_code = 409
    message = "Audio versions are linked to this version."

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
