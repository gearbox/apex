"""Planner runs, request side: submit, read, list and cancel.

A run is a database row driven by guarded transitions
(``queued → running → succeeded | failed``, ``queued → cancelled``), terminal once. Lock order is
always **track, then run**. This service is request-scoped and commit-free (the route commits, then
publishes); the worker side lives in ``run_executor.py``, which alone imports the planner graph.

Logs carry ids, kinds, model keys and error codes only - never the brief, the artist, a prompt or
model output.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final
from uuid import UUID  # noqa: TC003 - runtime: used in dataclass annotations

import msgspec
import structlog
from sqlalchemy.exc import IntegrityError

from src.api.schemas.pagination import CursorPage, decode_cursor, encode_cursor
from src.core.uid import new_id
from src.db.integrity import violated_constraint
from src.db.models.studio import RUN_ONE_ACTIVE_PER_TRACK, RUN_USER_IDEMPOTENCY
from src.db.repositories.studio import StudioRunRepository, StudioTrackRepository
from src.studio.errors import (
    IdempotencyConflict,
    IdempotencyKeyRequired,
    InvalidCursor,
    RunInProgress,
    RunLimitReached,
    RunNotCancellable,
    RunNotFound,
    RunRateLimited,
    StudioNotFound,
)
from src.studio.services.mappers import run_out
from src.studio.services.run_kinds.intake import IntakeContext, check_memory_pins, intake_for

if TYPE_CHECKING:
    from collections.abc import Mapping

    from sqlalchemy.ext.asyncio import AsyncSession

    from src.api.schemas.studio_runs import RunOut
    from src.api.services.billing import BalanceEvent, BillingService
    from src.core.config import Settings
    from src.core.enums import StudioRunKind
    from src.db.models.studio import StudioRun
    from src.studio.run_requests import RunInputs, RunRequest
    from src.studio.services.run_events import StudioRunEvents

__all__ = ["RunWrite", "StudioRunService", "model_key_for"]

logger = structlog.get_logger(__name__)

_MAX_KEY_LENGTH: Final = 128
_CONSTRAINTS: Final = frozenset({RUN_ONE_ACTIVE_PER_TRACK, RUN_USER_IDEMPOTENCY})


@dataclass(frozen=True, slots=True)
class RunWrite:
    """The result of a request-side operation, to be published once the transaction committed."""

    run: StudioRun
    changed: bool
    """``False`` for an idempotent replay: nothing was written, nothing is published."""
    balance_event: BalanceEvent | None = None


def model_key_for(
    kind: StudioRunKind, *, by_kind: Mapping[StudioRunKind, str], default_key: str
) -> str:
    """The ``LLM_MODELS`` key that serves a run of ``kind``: its own routing entry, else the default.

    Pure. The studio owns this routing (``STUDIO_LLM_MODEL_BY_KIND``); the shared registry only
    knows keys. Startup has already checked that every value is a registry key.
    """
    return by_kind.get(kind, default_key)


def _request_hash(track_id: UUID, request: RunRequest) -> str:
    """Identity of a submit: the track plus the normalized body (same key + same hash = replay).

    The request **as sent** (normalized): never the resolved inputs, so a replay is the same run
    even after the track's current pointers moved (P2a-40).
    """
    return hashlib.sha256(f"{track_id}\n".encode() + msgspec.json.encode(request)).hexdigest()


class StudioRunService:
    """Submit, read, list and cancel runs (one request transaction; never commits)."""

    def __init__(
        self,
        session: AsyncSession,
        *,
        billing: BillingService,
        settings: Settings,
        events: StudioRunEvents,
    ) -> None:
        self._session = session
        self._billing = billing
        self._settings = settings
        self._events = events
        self._runs = StudioRunRepository(session)
        self._tracks = StudioTrackRepository(session)

    # -- submit ------------------------------------------------------------------------------

    async def submit(
        self,
        track_id: UUID,
        request: RunRequest,
        *,
        idempotency_key: str | None,
        user_id: UUID,
        product_id: str,
    ) -> RunWrite:
        """Accept a run (or replay it): validate, resolve its inputs, check the limits, insert it
        ``queued``, reserve.

        What the run works on (the plan, the lyrics, the language) is resolved here, under the
        track lock, and stored on the run: a wrong or missing input is a ``4xx`` with no charge,
        and the worker never looks at the track's current pointers again (P2a-40). Pinned
        memories are checked here too; which memories the model is actually given is decided when
        a worker claims the run (P2b-2). The run row and the token reservation are written
        together; a refusal at any step leaves neither behind.

        Args:
            track_id: Track the run is for.
            request: The decoded body.
            idempotency_key: The ``Idempotency-Key`` header value.
            user_id: Owner scope.
            product_id: Product scope.

        Returns:
            The run; ``changed`` is ``False`` when the key and body match an earlier submit.

        Raises:
            IdempotencyKeyRequired: No usable key.
            StudioDocumentInvalid: The body is invalid after normalization (an empty brief or
                instruction, a bad avoid word, empty or overlapping targets ...), an input
                cannot be used (no vocal section in the plan, no language, lyrics that are not
                valid today, a target that does not fit), or a pinned memory cannot be used
                (``memory_ids``: no artist, unknown, archived, repeated, over the budget).
            IdempotencyConflict: The key belongs to a different request.
            StudioNotFound: The track is not the caller's.
            PlanRequired: ``lyrics`` without a plan to write for.
            PlanVersionNotFound: The named plan is not a plan version of this track.
            LyricsRequired: ``revise_lyrics`` without lyrics to revise.
            LyricsVersionNotFound: The named lyrics are not a lyrics version of this track.
            RunRateLimited: Too many runs were created in the last hour.
            RunLimitReached: The user has too many active runs.
            RunInProgress: The track already has an active run.
            InsufficientBalanceError: The reservation was refused (``402``).
        """
        key = self._valid_key(idempotency_key)
        intake = intake_for(request)
        normalized = intake.normalize(request)
        request_hash = _request_hash(track_id, normalized)

        replay = await self._replay(key, request_hash, user_id, product_id)
        if replay is not None:
            return replay

        track = await self._tracks.lock(track_id, user_id=user_id, product_id=product_id)
        if track is None:
            raise StudioNotFound
        # An identical submit may have committed while this one waited for the track lock.
        replay = await self._replay(key, request_hash, user_id, product_id)
        if replay is not None:
            return replay

        intake_context = IntakeContext(self._session, track_id, user_id, product_id)
        inputs = await intake.resolve_inputs(intake_context, normalized)
        # Pins are validated like every other input: now, with no charge (P2b-1).
        await check_memory_pins(intake_context, track.library_project_id, normalized.memory_ids)

        await self._runs.lock_user_quota(user_id)
        if (
            await self._runs.count_created_in_last_hour(user_id=user_id, product_id=product_id)
            >= self._settings.studio_runs_per_user_per_hour
        ):
            raise RunRateLimited
        if (
            await self._runs.count_active(user_id=user_id, product_id=product_id)
            >= self._settings.studio_run_max_active_per_user
        ):
            raise RunLimitReached

        kind = intake.kind
        model_key = model_key_for(
            kind,
            by_kind=self._settings.studio_llm_model_by_kind,
            default_key=self._settings.llm_default_model,
        )
        price = self._settings.studio_run_prices[kind]
        try:
            return await self._insert_and_reserve(
                track_id=track_id,
                kind=kind,
                key=key,
                request_hash=request_hash,
                request=normalized,
                inputs=inputs,
                model_key=model_key,
                price=price,
                user_id=user_id,
                product_id=product_id,
            )
        except IntegrityError as exc:
            constraint = violated_constraint(exc, _CONSTRAINTS)
            if constraint == RUN_ONE_ACTIVE_PER_TRACK:
                raise RunInProgress from None
            if constraint == RUN_USER_IDEMPOTENCY:
                # The same key raced in on another track (a different request hash).
                replay = await self._replay(key, request_hash, user_id, product_id)
                if replay is not None:
                    return replay
            raise

    async def _insert_and_reserve(
        self,
        *,
        track_id: UUID,
        kind: StudioRunKind,
        key: str,
        request_hash: str,
        request: RunRequest,
        inputs: RunInputs | None,
        model_key: str,
        price: int,
        user_id: UUID,
        product_id: str,
    ) -> RunWrite:
        """Insert the run and reserve its price in one savepoint: both, or neither."""
        async with self._session.begin_nested():
            run = await self._runs.insert(
                run_id=new_id(),
                user_id=user_id,
                product_id=product_id,
                track_id=track_id,
                kind=kind.value,
                idempotency_key=key,
                request_hash=request_hash,
                request=msgspec.to_builtins(request),
                inputs=msgspec.to_builtins(inputs) if inputs is not None else None,
                model_key=model_key,
                price_tokens=price,
            )
            balance_event: BalanceEvent | None = None
            if price > 0:
                account = await self._billing.resolve_account_for_user(
                    user_id, session=self._session
                )
                reserved = await self._billing.check_and_reserve(
                    account.id,
                    price,
                    run.id,
                    metadata={"studio_run_id": str(run.id), "kind": kind.value},
                    session=self._session,
                    product_id=product_id,
                    user_id=user_id,
                    description="Studio run",
                )
                await self._runs.set_billing_account(run.id, account.id)
                balance_event = reserved.event
        logger.info(
            "studio.run.submitted",
            run_id=str(run.id),
            track_id=str(track_id),
            kind=kind.value,
            model_key=model_key,
            price_tokens=price,
        )
        return RunWrite(run=run, changed=True, balance_event=balance_event)

    async def _replay(
        self, key: str, request_hash: str, user_id: UUID, product_id: str
    ) -> RunWrite | None:
        """The run an earlier submit with this key created, or ``None``; a different body is 409."""
        existing = await self._runs.get_by_key(
            user_id=user_id, product_id=product_id, idempotency_key=key
        )
        if existing is None:
            return None
        if existing.request_hash != request_hash:
            raise IdempotencyConflict
        return RunWrite(run=existing, changed=False)

    @staticmethod
    def _valid_key(raw: str | None) -> str:
        key = (raw or "").strip()
        if not key or len(key) > _MAX_KEY_LENGTH:
            raise IdempotencyKeyRequired
        return key

    # -- read --------------------------------------------------------------------------------

    async def get(self, run_id: UUID, *, user_id: UUID, product_id: str) -> RunOut:
        """One run.

        Raises:
            RunNotFound: No such run for the caller.
        """
        run = await self._runs.get(run_id, user_id=user_id, product_id=product_id)
        if run is None:
            raise RunNotFound
        return run_out(run)

    async def list_for_track(
        self,
        track_id: UUID,
        *,
        user_id: UUID,
        product_id: str,
        limit: int,
        cursor: str | None,
    ) -> CursorPage[RunOut]:
        """A track's runs, newest first.

        Raises:
            InvalidCursor: ``cursor`` is not one this endpoint issued.
            StudioNotFound: The track is not the caller's.
        """
        try:
            cursor_ts, cursor_id = decode_cursor(cursor) if cursor is not None else (None, None)
        except ValueError as exc:
            raise InvalidCursor from exc
        if await self._tracks.get(track_id, user_id=user_id, product_id=product_id) is None:
            raise StudioNotFound
        rows = await self._runs.list_page(
            track_id,
            user_id=user_id,
            product_id=product_id,
            limit=limit,
            cursor_ts=cursor_ts,
            cursor_id=cursor_id,
        )
        has_more = len(rows) > limit
        page = list(rows[:limit])
        return CursorPage(
            items=[run_out(r) for r in page],
            limit=limit,
            has_more=has_more,
            next_cursor=encode_cursor(page[-1].created_at, page[-1].id)
            if has_more and page
            else None,
        )

    # -- cancel ------------------------------------------------------------------------------

    async def cancel(self, run_id: UUID, *, user_id: UUID, product_id: str) -> RunWrite:
        """``queued → cancelled``, refunding the reservation in the same transaction.

        Raises:
            RunNotFound: No such run for the caller.
            RunNotCancellable: The run is already running or finished.
        """
        run = await self._runs.cancel_queued(run_id, user_id=user_id, product_id=product_id)
        if run is None:
            if await self._runs.get(run_id, user_id=user_id, product_id=product_id) is None:
                raise RunNotFound
            raise RunNotCancellable
        balance_event = None
        if run.price_tokens > 0:
            refunded = await self._billing.refund(
                run.id,
                description="Studio run cancelled",
                metadata={"studio_run_id": str(run.id), "reason": "cancelled"},
                session=self._session,
                product_id=product_id,
                user_id=user_id,
            )
            balance_event = refunded.event
        logger.info("studio.run.cancelled", run_id=str(run.id), track_id=str(run.track_id))
        return RunWrite(run=run, changed=True, balance_event=balance_event)

    # -- after commit ------------------------------------------------------------------------

    async def publish(self, write: RunWrite) -> None:
        """Publish what ``write`` changed. Call strictly after the transaction committed."""
        if not write.changed:
            return
        await self._events.balance(write.balance_event)
        await self._events.run_updated(write.run)
