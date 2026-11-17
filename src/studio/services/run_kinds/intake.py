"""The request side of a run kind: normalize the body, and resolve what the run works on.

``StudioRunService.submit`` calls these under the **track lock**. Resolution reads and writes
nothing and costs nothing: a wrong or missing input is a ``4xx`` here, before any run row or
ledger entry exists (P2a-40). What it returns is stored on the run as ``inputs`` and is all the
worker ever uses - the track's current pointers are never consulted again, so an idempotent replay
can never conflict because they moved.

No ``langgraph`` here: the API process imports this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol
from uuid import UUID  # noqa: TC003 - runtime: dataclass annotations

from src.core.enums import StudioDocumentKind, StudioRunKind
from src.db.repositories.studio import (
    StudioArtistRepository,
    StudioDocumentVersionRepository,
    StudioProjectRepository,
    StudioTrackRepository,
)
from src.studio.errors import (
    DocumentIssue,
    LyricsRequired,
    LyricsVersionNotFound,
    PlanRequired,
    PlanVersionNotFound,
    StudioDocumentInvalid,
    StudioNotFound,
)
from src.studio.run_requests import (
    LyricsRunRequest,
    PlanRunRequest,
    ReviseLyricsRunRequest,
    RunInputs,
    RunRequest,
    normalize_run_request,
)
from src.studio.services.mappers import decode_lyrics, decode_plan
from src.studio.validation import is_instrumental, normalize_lyrics

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from src.db.models.studio import StudioDocumentVersion, StudioTrack
    from src.studio.documents import LyricsDocument, PlanDocument

__all__ = ["IntakeContext", "RunIntake", "intake_for"]


@dataclass(frozen=True, slots=True)
class IntakeContext:
    """Whose track the run is for, and the request transaction to read through."""

    session: AsyncSession
    track_id: UUID
    user_id: UUID
    product_id: str


class RunIntake(Protocol):
    """The request side of one run kind."""

    @property
    def kind(self) -> StudioRunKind:
        """The kind this intake serves."""
        ...

    def normalize(self, request: RunRequest) -> RunRequest:
        """The request as it is stored and hashed.

        Raises:
            StudioDocumentInvalid: With every issue found.
        """
        ...

    async def resolve_inputs(self, context: IntakeContext, request: RunRequest) -> RunInputs | None:
        """What the run works on (``None`` when it needs nothing resolved).

        Raises:
            StudioNotFound: The track is not the caller's.
            StudioError: An input is wrong or missing (``plan_version_not_found``, ``plan_required``,
                ``lyrics_version_not_found``, ``lyrics_required``, ``document_invalid`` ...).
        """
        ...


async def _read_track(
    context: IntakeContext,
) -> tuple[StudioTrack, StudioDocumentVersion | None, StudioDocumentVersion | None]:
    """The track and the rows its current pointers name, in one consistent read."""
    found = await StudioTrackRepository(context.session).get_with_current_versions(
        context.track_id, user_id=context.user_id, product_id=context.product_id
    )
    if found is None:
        raise StudioNotFound
    return found[0], found[1], found[2]


async def _version_of(
    context: IntakeContext, version_id: UUID, kind: StudioDocumentKind
) -> StudioDocumentVersion | None:
    """A version of this track and kind, or ``None``."""
    row = await StudioDocumentVersionRepository(context.session).get_on_track(
        version_id, context.track_id, user_id=context.user_id, product_id=context.product_id
    )
    return row if row is not None and row.kind == kind.value else None


async def _artist_default_language(context: IntakeContext, track: StudioTrack) -> str | None:
    """The default language of the project's artist, if it has one."""
    profile = await StudioProjectRepository(context.session).get(
        track.library_project_id, user_id=context.user_id, product_id=context.product_id
    )
    if profile is None or profile.artist_id is None:
        return None
    artist = await StudioArtistRepository(context.session).get(
        profile.artist_id, user_id=context.user_id, product_id=context.product_id
    )
    return artist.default_language if artist is not None else None


class PlanIntake:
    """``plan``: nothing to resolve."""

    kind = StudioRunKind.PLAN

    def normalize(self, request: RunRequest) -> RunRequest:
        return normalize_run_request(request)

    async def resolve_inputs(self, context: IntakeContext, request: RunRequest) -> RunInputs | None:  # noqa: ARG002
        return None


class LyricsIntake:
    """``lyrics``: the plan to write for, and the language."""

    kind = StudioRunKind.LYRICS

    def normalize(self, request: RunRequest) -> RunRequest:
        return normalize_run_request(request)

    async def resolve_inputs(self, context: IntakeContext, request: RunRequest) -> RunInputs | None:
        if not isinstance(request, LyricsRunRequest):
            raise TypeError("lyrics intake given another kind's request")
        track, plan_row, lyrics_row = await _read_track(context)
        if request.plan_version_id is not None:
            plan_row = await _version_of(context, request.plan_version_id, StudioDocumentKind.PLAN)
            if plan_row is None:
                raise PlanVersionNotFound
        elif plan_row is None:
            raise PlanRequired
        plan = decode_plan(plan_row)

        issues: list[DocumentIssue] = []
        if plan.instrumental or all(is_instrumental(section) for section in plan.sections):
            issues.append(DocumentIssue("plan_version_id", "plan_has_no_vocal_sections"))
        language = request.language
        if language is None and lyrics_row is not None:
            language = decode_lyrics(lyrics_row).language
        if language is None:
            language = await _artist_default_language(context, track)
        if language is None:
            issues.append(DocumentIssue("language", "language_required"))
        if issues or language is None:
            raise StudioDocumentInvalid(tuple(issues))
        return RunInputs(plan_version_id=plan_row.id, language=language)


def _prefixed(issues: tuple[DocumentIssue, ...], prefix: str) -> tuple[DocumentIssue, ...]:
    return tuple(
        DocumentIssue(f"{prefix}.{issue.path}" if issue.path else prefix, issue.code)
        for issue in issues
    )


def _target_issues(
    request: ReviseLyricsRunRequest, plan: PlanDocument, lyrics: LyricsDocument
) -> list[DocumentIssue]:
    """Every target that cannot be revised, addressed in the request (``targets.sections[i]``)."""
    sections = {section.id: section for section in plan.sections}
    sizes = {section.section_id: len(section.lines) for section in lyrics.sections}
    issues: list[DocumentIssue] = []
    for index, section_id in enumerate(request.targets.sections):
        path = f"targets.sections[{index}]"
        section = sections.get(section_id)
        if section is None:
            issues.append(DocumentIssue(path, "unknown_section"))
        elif plan.instrumental or is_instrumental(section):
            issues.append(DocumentIssue(path, "instrumental_section"))
    for index, target in enumerate(request.targets.lines):
        if target.line >= sizes.get(target.section_id, 0):
            issues.append(DocumentIssue(f"targets.lines[{index}]", "line_out_of_range"))
    return issues


class ReviseLyricsIntake:
    """``revise_lyrics``: the lyrics to revise, their plan, and whether the targets fit them."""

    kind = StudioRunKind.REVISE_LYRICS

    def normalize(self, request: RunRequest) -> RunRequest:
        return normalize_run_request(request)

    async def resolve_inputs(self, context: IntakeContext, request: RunRequest) -> RunInputs | None:
        if not isinstance(request, ReviseLyricsRunRequest):
            raise TypeError("revise intake given another kind's request")
        _, _, lyrics_row = await _read_track(context)
        if request.lyrics_version_id is not None:
            lyrics_row = await _version_of(
                context, request.lyrics_version_id, StudioDocumentKind.LYRICS
            )
            if lyrics_row is None:
                raise LyricsVersionNotFound
        elif lyrics_row is None:
            raise LyricsRequired
        if lyrics_row.based_on_plan_version_id is None:  # pragma: no cover - a database CHECK
            raise RuntimeError("lyrics version without a plan")
        plan_row = await _version_of(
            context, lyrics_row.based_on_plan_version_id, StudioDocumentKind.PLAN
        )
        if plan_row is None:  # pragma: no cover - a composite foreign key
            raise RuntimeError("lyrics version whose plan is missing")
        plan = decode_plan(plan_row)
        lyrics = decode_lyrics(lyrics_row)

        # A revision starts from lyrics that are valid today (P2a-47): finding out after the
        # model was paid would be too late.
        try:
            normalize_lyrics(lyrics, plan)
        except StudioDocumentInvalid as exc:
            raise StudioDocumentInvalid(_prefixed(exc.issues, "lyrics")) from None
        issues = _target_issues(request, plan, lyrics)
        if issues:
            raise StudioDocumentInvalid(tuple(issues))
        return RunInputs(
            plan_version_id=plan_row.id,
            lyrics_version_id=lyrics_row.id,
            language=lyrics.language,
        )


_INTAKES: dict[type, RunIntake] = {
    PlanRunRequest: PlanIntake(),
    LyricsRunRequest: LyricsIntake(),
    ReviseLyricsRunRequest: ReviseLyricsIntake(),
}


def intake_for(request: RunRequest) -> RunIntake:
    """The intake of ``request``'s kind."""
    return _INTAKES[type(request)]
