"""The asyncio tasks that write artist replies and memory drafts (P2c-18).

``ChatTaskSupervisor`` is a process singleton created in the app lifespan when the chat is enabled.
A request commits its rows, then hands the supervisor a job; the supervisor starts **one task per
job** and keeps the task set. **No database connection is held while a model call is awaited**: the
request transaction is over, the task opens a short transaction of its own only to finish or fail
the row.

Reply task, in order: stream the model's text through :class:`~src.studio.chat.stream.ReplyStream`
(the sentinel buffer and the coalescing), publish deltas, then **finish** or **fail**:

* *finish*: one guarded ``UPDATE … WHERE status = 'pending'`` to ``complete`` with the text, usage
  and ``truncated``; then ``studio.chat.message.updated``.
* *fail*: the same guarded update to ``failed`` **and** the refund of the reservation, in one
  transaction - the refund only if the update hit the row. If it hit none, the sweep (or the task)
  got there first and owns the money: log and stop. Then the event.

**The HTTP connection plays no part.** Nothing in a task observes the client: a disconnect changes
nothing and a finished reply is charged (P2c-12). Money comes back only when there is *no reply*:
provider error, refusal, deadline, shutdown or crash.

Failure modes: ``LlmError`` → ``llm_unavailable``; the deadline → ``reply_timeout``; a cancelled
task (shutdown) → ``server_shutdown``, then the cancellation propagates; any other exception is
logged (``studio.chat_task_crashed``: type and frames, never the message) and the row is left to
the sweep, which fails and refunds it after the deadline. ``shutdown()`` cancels every task and
waits up to a grace period for their fail paths.

Memory text and chat text never reach a log: logs carry ids, counts, codes and model keys.
"""

from __future__ import annotations

import asyncio
import time
import traceback
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

import structlog

from src.db.repositories.studio_chat import (
    StudioChatMessageRepository,
    StudioMemoryDraftRepository,
)
from src.llm.port import (
    LlmError,
    LlmMessage,
    LlmOutputInvalid,
    LlmReplyError,
    LlmUsage,
    StreamEnd,
    TextDelta,
)
from src.studio.chat.draft import MemoryDraftDraft, MemoryDraftRefusal, check_memory_draft
from src.studio.chat.failures import (
    ERR_CONTENT_POLICY,
    ERR_DRAFT_INVALID,
    ERR_LLM_UNAVAILABLE,
    ERR_REPLY_EMPTY,
    ERR_REPLY_TIMEOUT,
    ERR_SERVER_SHUTDOWN,
)
from src.studio.chat.prompts.memory_draft_v1 import REASK_MESSAGE, build_fix_messages
from src.studio.chat.refunds import refund_reservation
from src.studio.chat.stream import ReplyStream

if TYPE_CHECKING:
    from collections.abc import Callable, Coroutine
    from typing import Any

    from src.api.services.billing import BalanceEvent, BillingService
    from src.db.models.studio_chat import StudioArtistChatMessage, StudioMemoryDraft
    from src.db.session import DatabaseManager
    from src.llm.port import ChatLlmPort, StructuredLlmPort
    from src.studio.chat.draft import MemoryContent
    from src.studio.chat.jobs import ChatReplyJob, MemoryDraftJob
    from src.studio.chat.ports import ChatEvents
    from src.studio.chat.stream import Delta

__all__ = ["ChatTaskSupervisor"]

logger = structlog.get_logger(__name__)

_FINISH_LENGTH: Final = "length"
_FINISH_CONTENT_FILTER: Final = "content_filter"
_DRAFT_MAX_CALLS: Final = 2
"""One call and one fix call (P2c-7)."""
_DRAFT_SCHEMA_NAME: Final = "memory_draft"


def _frames(exc: BaseException) -> str:
    """The traceback of ``exc`` without its message (a message can quote user data)."""
    return "".join(traceback.format_tb(exc.__traceback__))


@dataclass(slots=True)
class _Tokens:
    """What a job's model calls cost, so a failed job still records what was spent."""

    prompt: int = 0
    completion: int = 0

    def add(self, usage: LlmUsage | None) -> None:
        if usage is not None:
            self.prompt += usage.prompt_tokens
            self.completion += usage.completion_tokens


@dataclass(frozen=True, slots=True)
class _Failed:
    """A job that ends without a result: why (an ``ERR_*`` code)."""

    code: str


@dataclass(frozen=True, slots=True)
class _DraftWritten:
    content: MemoryContent
    model_id: str


class ChatTaskSupervisor:
    """Owns the asyncio tasks that answer chat messages and write memory drafts."""

    def __init__(
        self,
        *,
        db_manager: DatabaseManager,
        billing: BillingService,
        chat_port_for: Callable[[str], ChatLlmPort],
        structured_port_for: Callable[[str], StructuredLlmPort],
        events: ChatEvents,
        shutdown_grace_seconds: float,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """Create the supervisor.

        Args:
            db_manager: Source of the short finish / fail transactions.
            billing: Refunds a failed job's reservation.
            chat_port_for: Registry key -> the streaming port (the factory's ``chat``).
            structured_port_for: Registry key -> the structured port (a draft's model).
            events: Post-commit notifications; never raises.
            shutdown_grace_seconds: How long ``shutdown`` waits for the fail paths of its tasks.
            clock: Monotonic seconds (injectable for the coalescing tests).
        """
        self._db = db_manager
        self._billing = billing
        self._chat_port_for = chat_port_for
        self._structured_port_for = structured_port_for
        self._events = events
        self._grace = shutdown_grace_seconds
        self._clock = clock
        self._tasks: set[asyncio.Task[None]] = set()
        self._closing = False

    @property
    def in_flight(self) -> int:
        """Replies and drafts this process is writing right now."""
        return len(self._tasks)

    @property
    def tasks(self) -> frozenset[asyncio.Task[None]]:
        """The tasks in flight (a snapshot): what a test awaits instead of sleeping."""
        return frozenset(self._tasks)

    # -- starting ----------------------------------------------------------------------------

    def start_reply(self, job: ChatReplyJob) -> asyncio.Task[None] | None:
        """Start writing a reply. The row is committed and ``pending``.

        Returns the task (tests await it), or ``None`` when the supervisor is shutting down - the
        row is then left to the sweep of the next process, which fails and refunds it.
        """
        return self._start(self._run_reply(job), f"studio-chat-reply-{job.message_id}")

    def start_memory_draft(self, job: MemoryDraftJob) -> asyncio.Task[None] | None:
        """Start writing a memory draft. See :meth:`start_reply`."""
        return self._start(self._run_draft(job), f"studio-memory-draft-{job.draft_id}")

    def _start(self, coro: Coroutine[Any, Any, None], name: str) -> asyncio.Task[None] | None:
        if self._closing:
            coro.close()
            logger.warning("studio.chat_task_refused_shutting_down", task=name)
            return None
        task = asyncio.create_task(coro, name=name)
        self._tasks.add(task)
        task.add_done_callback(self._task_done)
        return task

    def _task_done(self, task: asyncio.Task[None]) -> None:
        self._tasks.discard(task)
        if task.cancelled():
            return
        if (exc := task.exception()) is not None:  # the bodies guard themselves; this is a bug
            logger.error(
                "studio.chat_task_crashed",
                task=task.get_name(),
                exc_type=type(exc).__name__,
                frames=_frames(exc),
            )

    async def shutdown(self) -> None:
        """Fail and refund every in-flight job (``server_shutdown``), waiting up to the grace.

        Cancels each task; a task's cancellation handler runs its fail path (the guarded update and
        the refund) before it lets the cancellation go. A task that cannot finish in time is
        abandoned, its row left to the next process's sweep.
        """
        self._closing = True
        tasks = set(self._tasks)
        for task in tasks:
            task.cancel()
        if not tasks:
            return
        _, pending = await asyncio.wait(tasks, timeout=self._grace)
        if pending:
            logger.warning("studio.chat_shutdown_abandoned", abandoned=len(pending))

    # -- the reply task ----------------------------------------------------------------------

    async def _run_reply(self, job: ChatReplyJob) -> None:
        started = time.monotonic()
        tokens = _Tokens()
        try:
            await self._write_reply(job, tokens, started)
        except asyncio.CancelledError:
            await self._fail_reply(job, ERR_SERVER_SHUTDOWN, tokens, started)
            raise
        except Exception as exc:  # the task guard: log it, leave the row to the sweep
            # Not ``logger.exception``: an exception message can quote user data.
            logger.error(  # noqa: TRY400
                "studio.chat_task_crashed",
                message_id=str(job.message_id),
                exc_type=type(exc).__name__,
                frames=_frames(exc),
            )

    async def _write_reply(self, job: ChatReplyJob, tokens: _Tokens, started: float) -> None:
        stream = ReplyStream(clock=self._clock)
        end: StreamEnd | None = None
        try:
            async with asyncio.timeout(job.deadline_seconds):
                chunks = self._chat_port_for(job.model_key).stream(
                    job.messages, max_output_tokens=job.max_output_tokens
                )
                try:
                    async for chunk in chunks:
                        if isinstance(chunk, TextDelta):
                            if delta := stream.feed(chunk.text):
                                await self._publish_delta(job, delta)
                            if stream.refused:
                                break  # a refusal: stop reading, nothing was ever published
                        elif isinstance(chunk, StreamEnd):
                            end = chunk
                finally:
                    aclose = getattr(chunks, "aclose", None)
                    if aclose is not None:
                        await aclose()
        except TimeoutError:
            await self._fail_reply(job, ERR_REPLY_TIMEOUT, tokens, started)
            return
        except LlmError as exc:
            if isinstance(exc, LlmReplyError):
                tokens.add(exc.usage)
            await self._fail_reply(job, ERR_LLM_UNAVAILABLE, tokens, started)
            return

        if end is not None:
            tokens.add(end.usage)
        if stream.refused:
            await self._fail_reply(job, ERR_CONTENT_POLICY, tokens, started)
            return
        if end is None:  # the port promises a StreamEnd; a stream without one is a broken one
            await self._fail_reply(job, ERR_LLM_UNAVAILABLE, tokens, started)
            return
        if end.finish_reason == _FINISH_CONTENT_FILTER:
            await self._fail_reply(job, ERR_CONTENT_POLICY, tokens, started)
            return
        if not stream.text.strip():
            await self._fail_reply(job, ERR_REPLY_EMPTY, tokens, started)
            return
        if delta := stream.finish():
            await self._publish_delta(job, delta)
        await self._finish_reply(
            job,
            text=stream.text,
            end=end,
            truncated=end.finish_reason == _FINISH_LENGTH,
            started=started,
        )

    async def _publish_delta(self, job: ChatReplyJob, delta: Delta) -> None:
        """Best-effort: a publish failure is logged and never stops the reply."""
        try:
            await self._events.delta(
                user_id=job.user_id,
                chat_id=job.chat_id,
                message_id=job.message_id,
                seq=delta.seq,
                text=delta.text,
            )
        except Exception:
            logger.warning("studio.chat_delta_publish_failed", message_id=str(job.message_id))

    async def _finish_reply(
        self, job: ChatReplyJob, *, text: str, end: StreamEnd, truncated: bool, started: float
    ) -> None:
        """``pending → complete`` (guarded), then the event. The reply is charged."""
        async with self._db.session() as session:
            message = await StudioChatMessageRepository(session).finish_complete(
                job.message_id,
                user_id=job.user_id,
                product_id=job.product_id,
                content=text,
                model_id=end.model_id,
                prompt_tokens=end.usage.prompt_tokens,
                completion_tokens=end.usage.completion_tokens,
                truncated=truncated,
            )
        if message is None:  # the sweep failed (and refunded) it while the reply was written
            logger.warning("studio.chat_finish_lost", message_id=str(job.message_id))
            return
        await self._notify_message(message)
        self._log_reply(job, message.status, None, end.usage, truncated, started)

    async def _fail_reply(
        self, job: ChatReplyJob, code: str, tokens: _Tokens, started: float
    ) -> None:
        """``pending → failed`` and the refund, in one transaction; then the events."""
        balance: BalanceEvent | None = None
        async with self._db.session() as session:
            message = await StudioChatMessageRepository(session).finish_failed(
                job.message_id,
                user_id=job.user_id,
                product_id=job.product_id,
                error_code=code,
                prompt_tokens=tokens.prompt,
                completion_tokens=tokens.completion,
            )
            if message is not None:
                balance = await refund_reservation(
                    self._billing,
                    session,
                    job_id=job.message_id,
                    price=job.price_tokens,
                    product_id=job.product_id,
                    user_id=job.user_id,
                    reason=code,
                    description="Studio chat reply failed",
                )
        if message is None:  # already finished by the other side, which owns the money
            logger.warning("studio.chat_finish_lost", message_id=str(job.message_id))
            return
        await self._events.balance(balance)
        await self._notify_message(message)
        self._log_reply(
            job, message.status, code, LlmUsage(tokens.prompt, tokens.completion), False, started
        )

    async def _notify_message(self, message: StudioArtistChatMessage) -> None:
        try:
            await self._events.message_updated(message)
        except Exception:
            logger.warning("studio.chat_event_publish_failed", message_id=str(message.id))

    @staticmethod
    def _log_reply(
        job: ChatReplyJob,
        status: str,
        error_code: str | None,
        usage: LlmUsage,
        truncated: bool,
        started: float,
    ) -> None:
        logger.info(
            "studio.chat_reply_finished",
            message_id=str(job.message_id),
            chat_id=str(job.chat_id),
            status=status,
            error_code=error_code,
            model_key=job.model_key,
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            truncated=truncated,
            memory_count=len(job.memory_ids_used),
            duration_ms=int((time.monotonic() - started) * 1000),
        )

    # -- the memory draft task ---------------------------------------------------------------

    async def _run_draft(self, job: MemoryDraftJob) -> None:
        started = time.monotonic()
        tokens = _Tokens()
        try:
            await self._write_draft(job, tokens, started)
        except asyncio.CancelledError:
            await self._fail_draft(job, ERR_SERVER_SHUTDOWN, tokens, started)
            raise
        except Exception as exc:  # the task guard: log it, leave the row to the sweep
            logger.error(  # noqa: TRY400 - an exception message can quote user data
                "studio.chat_task_crashed",
                draft_id=str(job.draft_id),
                exc_type=type(exc).__name__,
                frames=_frames(exc),
            )

    async def _write_draft(self, job: MemoryDraftJob, tokens: _Tokens, started: float) -> None:
        try:
            async with asyncio.timeout(job.deadline_seconds):
                outcome = await self._ask_for_draft(job, tokens)
        except TimeoutError:
            await self._fail_draft(job, ERR_REPLY_TIMEOUT, tokens, started)
            return
        except LlmError as exc:
            if isinstance(exc, LlmReplyError):
                tokens.add(exc.usage)
            await self._fail_draft(job, ERR_LLM_UNAVAILABLE, tokens, started)
            return
        if isinstance(outcome, _Failed):
            await self._fail_draft(job, outcome.code, tokens, started)
            return
        await self._finish_draft(job, outcome, tokens, started)

    async def _ask_for_draft(self, job: MemoryDraftJob, tokens: _Tokens) -> _DraftWritten | _Failed:
        """Ask for a draft, and once more with the issues when it is not a valid memory."""
        port = self._structured_port_for(job.model_key)
        conversation: list[LlmMessage] = list(job.messages)
        for call in range(1, _DRAFT_MAX_CALLS + 1):
            last_call = call == _DRAFT_MAX_CALLS
            try:
                result = await port.complete(
                    conversation, schema=MemoryDraftDraft, schema_name=_DRAFT_SCHEMA_NAME
                )
            except LlmOutputInvalid as exc:
                tokens.add(exc.usage)  # paid for even though unusable
                if last_call:
                    return _Failed(ERR_DRAFT_INVALID)
                # The undecodable reply is never echoed back.
                conversation = [*job.messages, LlmMessage("user", REASK_MESSAGE)]
                continue
            tokens.add(result.usage)
            draft = result.value
            if draft.refusal is MemoryDraftRefusal.CONTENT_POLICY:
                return _Failed(ERR_CONTENT_POLICY)
            if draft.refusal is MemoryDraftRefusal.NOTHING_TO_REMEMBER:
                return _Failed(ERR_DRAFT_INVALID)
            content, issues = check_memory_draft(draft)
            if content is not None:
                return _DraftWritten(content=content, model_id=result.model_id)
            if last_call:
                return _Failed(ERR_DRAFT_INVALID)
            conversation = [*job.messages, *build_fix_messages(draft, issues)]
        return _Failed(ERR_DRAFT_INVALID)  # unreachable: the last call returns above

    async def _finish_draft(
        self, job: MemoryDraftJob, written: _DraftWritten, tokens: _Tokens, started: float
    ) -> None:
        """``pending → complete`` with the draft (guarded), then the event. The draft is charged."""
        async with self._db.session() as session:
            draft = await StudioMemoryDraftRepository(session).finish_complete(
                job.draft_id,
                user_id=job.user_id,
                product_id=job.product_id,
                draft=written.content.to_json(),
                model_id=written.model_id,
                prompt_tokens=tokens.prompt,
                completion_tokens=tokens.completion,
            )
        if draft is None:
            logger.warning("studio.chat_finish_lost", draft_id=str(job.draft_id))
            return
        await self._notify_draft(draft)
        self._log_draft(job, draft.status, None, tokens, started)

    async def _fail_draft(
        self, job: MemoryDraftJob, code: str, tokens: _Tokens, started: float
    ) -> None:
        """``pending → failed`` and the refund, in one transaction; then the events."""
        balance: BalanceEvent | None = None
        async with self._db.session() as session:
            draft = await StudioMemoryDraftRepository(session).finish_failed(
                job.draft_id,
                user_id=job.user_id,
                product_id=job.product_id,
                error_code=code,
                prompt_tokens=tokens.prompt,
                completion_tokens=tokens.completion,
            )
            if draft is not None:
                balance = await refund_reservation(
                    self._billing,
                    session,
                    job_id=job.draft_id,
                    price=job.price_tokens,
                    product_id=job.product_id,
                    user_id=job.user_id,
                    reason=code,
                    description="Studio memory draft failed",
                )
        if draft is None:
            logger.warning("studio.chat_finish_lost", draft_id=str(job.draft_id))
            return
        await self._events.balance(balance)
        await self._notify_draft(draft)
        self._log_draft(job, draft.status, code, tokens, started)

    async def _notify_draft(self, draft: StudioMemoryDraft) -> None:
        try:
            await self._events.draft_updated(draft)
        except Exception:
            logger.warning("studio.chat_event_publish_failed", draft_id=str(draft.id))

    @staticmethod
    def _log_draft(
        job: MemoryDraftJob, status: str, error_code: str | None, tokens: _Tokens, started: float
    ) -> None:
        logger.info(
            "studio.memory_draft_finished",
            draft_id=str(job.draft_id),
            chat_id=str(job.chat_id),
            status=status,
            error_code=error_code,
            model_key=job.model_key,
            prompt_tokens=tokens.prompt,
            completion_tokens=tokens.completion,
            duration_ms=int((time.monotonic() - started) * 1000),
        )
