"""What every run kind's worker-side handler is, and the pieces they share.

The executor (``run_executor.py``) owns the guarded state machine - claim, stages, deadline, the
atomic finish, failure, refund, sweep - and knows no kind. A :class:`RunHandler` supplies the rest:
how its request is decoded, which documents it loads when the run is claimed, its graph, and which
version it saves.

Logs carry ids, kinds and counts only - never document text.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Final, Protocol
from uuid import UUID  # noqa: TC003 - runtime: dataclass annotations

import msgspec
import structlog

from src.core.enums import StudioDocumentKind, StudioDocumentOrigin, StudioRunKind
from src.db.repositories.studio import StudioDocumentVersionRepository, StudioTrackRepository
from src.studio.errors import ParentVersionNotFound, StudioNotFound, VersionUnchanged
from src.studio.planner.errors import PlannerStateError
from src.studio.planner.outcome import RunWarning
from src.studio.run_requests import RunInputs, RunRequest
from src.studio.services.version import DocumentVersionService

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Mapping

    from sqlalchemy.ext.asyncio import AsyncSession

    from src.api.schemas.studio import VersionCreate
    from src.core.enums import StudioRunStage
    from src.db.models.studio import StudioDocumentVersion, StudioRun
    from src.llm.port import StructuredLlmPort
    from src.studio.documents import LyricsDocument, PlanDocument
    from src.studio.planner.context import RunContextSnapshot
    from src.studio.planner.outcome import RunOutcome

__all__ = [
    "WARNING_PARENT_VERSION_REPLACED",
    "WARNING_REVISION_UNCHANGED",
    "ClaimedRun",
    "LoadedRun",
    "RunHandler",
    "SavedRun",
    "SavedVersion",
    "StageCallback",
    "WorkingDocuments",
    "create_llm_version",
    "decode_request_as",
    "load_input_version",
    "require_inputs",
    "save_under_current",
]

logger = structlog.get_logger(__name__)

WARNING_PARENT_VERSION_REPLACED: Final = "parent_version_replaced"
"""Run warning: the version the run started from was deleted before the result was saved."""
WARNING_REVISION_UNCHANGED: Final = "revision_unchanged"
"""Run warning: the revision changed nothing, so the input version is the result."""

type StageCallback = Callable[[StudioRunStage], Awaitable[None]]
"""Awaited when a stage starts; the executor records it (and may stop the run)."""


@dataclass(frozen=True, slots=True)
class WorkingDocuments:
    """The documents a run works on, loaded when it was claimed and carried in memory.

    Versions are immutable and a version an active run uses cannot be deleted (P2a-41), so the
    snapshot stores ids and hashes only.
    """

    plan: PlanDocument | None = None
    lyrics: LyricsDocument | None = None


@dataclass(frozen=True, slots=True)
class ClaimedRun:
    """A run this worker moved to ``running`` and now executes."""

    run_id: UUID
    user_id: UUID
    product_id: str
    track_id: UUID
    kind: StudioRunKind
    request: RunRequest
    context: RunContextSnapshot
    model_key: str
    price_tokens: int
    inputs: RunInputs | None = None
    documents: WorkingDocuments = field(default_factory=WorkingDocuments)


@dataclass(frozen=True, slots=True)
class LoadedRun:
    """What a handler's ``load`` hands the claim: the snapshot to freeze, and the documents."""

    context: RunContextSnapshot
    inputs: RunInputs | None = None
    documents: WorkingDocuments = field(default_factory=WorkingDocuments)


@dataclass(frozen=True, slots=True)
class SavedRun:
    """What a handler's ``save`` wrote: which result pointer to set, and the run's warnings."""

    result_plan_version_id: UUID | None = None
    result_lyrics_version_id: UUID | None = None
    warnings: tuple[RunWarning, ...] = ()


class RunHandler(Protocol):
    """The worker side of one run kind."""

    @property
    def kind(self) -> StudioRunKind:
        """The kind this handler serves."""
        ...

    @property
    def prompt_version(self) -> str:
        """The prompt version recorded on the run when it starts."""
        ...

    def decode_request(self, stored: Mapping[str, object]) -> RunRequest:
        """The run's stored request, decoded.

        Raises:
            PlannerStateError: It is not this kind's request.
        """
        ...

    async def load(
        self,
        session: AsyncSession,
        run: StudioRun,
        request: RunRequest,
        context: RunContextSnapshot,
    ) -> LoadedRun:
        """Load the working documents inside the claim transaction and complete the snapshot.

        Raises:
            RuntimeError: An input version is gone (a broken invariant: the delete guard keeps
                them alive); the claim fails the run ``internal_error`` and refunds it.
        """
        ...

    async def run(
        self, claimed: ClaimedRun, *, llm: StructuredLlmPort, on_stage: StageCallback
    ) -> RunOutcome:
        """Run the kind's graph. Touches no database; port errors propagate."""
        ...

    async def save(
        self, session: AsyncSession, claimed: ClaimedRun, outcome: RunOutcome
    ) -> SavedRun:
        """Write the result as an ordinary version (``origin = llm``, never made current).

        Runs inside the executor's atomic "save + mark succeeded" transaction.
        """
        ...


# -- helpers the handlers share ----------------------------------------------------------------


def decode_request_as[R: RunRequest](stored: Mapping[str, object], expected: type[R]) -> R:
    """Decode a stored request and check it is the ``expected`` kind's.

    Raises:
        PlannerStateError: It decodes to another kind.
    """
    request = msgspec.convert(stored, RunRequest)
    if not isinstance(request, expected):
        raise PlannerStateError("the stored request does not match the run kind")
    return request


def require_inputs(run: StudioRun) -> RunInputs:
    """The inputs a lyrics run was accepted with.

    Raises:
        RuntimeError: There are none - a lyrics run is never inserted without them.
    """
    if run.inputs is None:
        raise RuntimeError("studio run without inputs")
    return msgspec.convert(run.inputs, RunInputs)


async def load_input_version(
    session: AsyncSession,
    run: StudioRun,
    version_id: UUID,
    kind: StudioDocumentKind,
) -> StudioDocumentVersion:
    """An input version of the run's track and ``kind``.

    Raises:
        RuntimeError: It is gone (see :meth:`RunHandler.load`).
    """
    row = await StudioDocumentVersionRepository(session).get_on_track(
        version_id, run.track_id, user_id=run.user_id, product_id=run.product_id
    )
    if row is None or row.kind != kind.value:
        raise RuntimeError("studio run input version is missing")
    return row


@dataclass(frozen=True, slots=True)
class SavedVersion:
    """The version a save produced (or found)."""

    version_id: UUID | None
    unchanged: bool
    """``True`` when the result equals the parent, so the parent *is* the result."""
    warnings: tuple[RunWarning, ...] = ()


async def create_llm_version(
    versions: DocumentVersionService,
    claimed: ClaimedRun,
    request: VersionCreate,
) -> SavedVersion:
    """Create the run's result as an ordinary version: ``origin = llm``, never made current.

    The one place a run writes a version. An unchanged result *is* the parent used.
    """
    try:
        created, _ = await versions.create(
            claimed.track_id,
            request,
            claimed.user_id,
            claimed.product_id,
            origin=StudioDocumentOrigin.LLM,
        )
    except VersionUnchanged:
        return SavedVersion(request.parent_version_id, unchanged=True)
    return SavedVersion(created.version.id, unchanged=False)


async def save_under_current(
    session: AsyncSession,
    claimed: ClaimedRun,
    *,
    kind: StudioDocumentKind,
    parent_id: UUID | None,
    build: Callable[[UUID | None], VersionCreate],
) -> SavedVersion:
    """Save a result as a child of ``parent_id``; if that parent is gone, of the current one.

    A paid, valid result is never thrown away because its lineage anchor was deleted while the
    model worked: it becomes a child of the track's **current version of this kind** (a root when
    there is none) and the run records ``parent_version_replaced`` (P2a-29). The refused ``create``
    has locked the track and written nothing, so the pointer read here cannot move before the
    retry. One retry, never a loop: a second refusal propagates.

    Args:
        session: The executor's transaction.
        claimed: The run.
        kind: Which current pointer the fallback reads.
        parent_id: The claim-time parent (the version that was current then), or ``None``.
        build: The version request for a given parent.

    Raises:
        StudioNotFound: The track is gone (cannot happen: the refused create locked this row).
    """
    versions = DocumentVersionService(session)
    try:
        return await create_llm_version(versions, claimed, build(parent_id))
    except ParentVersionNotFound:
        pass
    track = await StudioTrackRepository(session).lock(
        claimed.track_id, user_id=claimed.user_id, product_id=claimed.product_id
    )
    if track is None:  # pragma: no cover - the refused create locked this very row
        raise StudioNotFound
    logger.info("studio.run.parent_replaced", run_id=str(claimed.run_id), kind=kind.value)
    current = (
        track.current_plan_version_id
        if kind is StudioDocumentKind.PLAN
        else track.current_lyrics_version_id
    )
    saved = await create_llm_version(versions, claimed, build(current))
    return SavedVersion(
        saved.version_id,
        saved.unchanged,
        (RunWarning(WARNING_PARENT_VERSION_REPLACED), *saved.warnings),
    )
