"""A committed world for artist-chat integration tests: a real supervisor over a scripted model.

Builds on ``RunWorld`` (committed users with a funded account, an artist, memories, tracks, plans
and lyrics, the recording bus, the ledger readers). The streaming model is scripted - CI never calls
one - and the supervisor, the services, the sweep and the repositories are the real ones, over the
real PostgreSQL: the reply task opens its own short transactions exactly as in production.
"""

from __future__ import annotations

import asyncio
import dataclasses
from collections import deque
from typing import TYPE_CHECKING, Any
from uuid import UUID

from sqlalchemy import func, select, text

from src.api.schemas.studio_chat import ChatCreate, MessageCreate
from src.api.services.billing import BillingService
from src.db.models.studio_chat import (
    StudioArtistChat,
    StudioArtistChatMessage,
    StudioMemoryDraft,
)
from src.llm.port import LlmUsage, StreamEnd, TextDelta
from src.studio.chat.supervisor import ChatTaskSupervisor
from src.studio.chat.sweep import ChatSweeper
from src.studio.services.chat import ChatService, ChatWrite
from src.studio.services.chat_context import ArtistChatContextAssembler
from src.studio.services.chat_events import StudioChatEvents
from src.studio.services.memory_draft import DraftWrite, MemoryDraftService
from tests.llm_support import FakeLlm
from tests.studio_runs_support import PRODUCT, Actor, RunWorld, make_settings

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Sequence

    from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession

    from src.core.config import Settings
    from src.llm.port import ChatChunk, LlmMessage

__all__ = [
    "CHAT_MODELS",
    "ChatWorld",
    "Gate",
    "ScriptedChat",
    "make_chat_settings",
    "reply",
]

CHAT_MODELS: dict[str, dict[str, Any]] = {
    "grok": {
        "base_url": "https://llm-a.example.test/v1",
        "api_key_env": "KEY_A",
        "model": "model-a",
        "supports_streaming": True,
    },
    "qwen": {
        "base_url": "https://llm-b.example.test/v1",
        "api_key_env": "KEY_B",
        "model": "model-b",
        "structured_output": "json_object",
        "supports_streaming": True,
    },
}


def make_chat_settings(**overrides: Any) -> Settings:
    """Settings with the studio, planner runs and the chat enabled and two streaming models."""
    return make_settings(**{"studio_chat_enabled": True, "llm_models": CHAT_MODELS, **overrides})


class Gate:
    """A point in a scripted stream where the model "thinks" until the test lets it go on."""

    def __init__(self) -> None:
        self.reached = asyncio.Event()
        self._open = asyncio.Event()

    def release(self) -> None:
        self._open.set()

    async def wait(self) -> None:
        self.reached.set()
        await self._open.wait()


def reply(
    *texts: str,
    finish: str | None = "stop",
    usage: tuple[int, int] = (120, 40),
    model_id: str = "served-model",
) -> list[object]:
    """A scripted reply: the text chunks, then the end of the stream."""
    items: list[object] = [*texts]
    if finish is not None or usage:
        items.append(
            StreamEnd(
                finish_reason=finish,
                usage=LlmUsage(prompt_tokens=usage[0], completion_tokens=usage[1]),
                model_id=model_id,
            )
        )
    return items


class ScriptedChat:
    """A scripted ``ChatLlmPort``: one script per call.

    A script is a list of items played in order: a ``str`` is a text delta, a ``StreamEnd`` is the
    end, a ``Gate`` waits for the test, an exception is raised. A call beyond the script raises
    ``AssertionError`` (an unexpected extra model call).
    """

    def __init__(self, *scripts: Sequence[object]) -> None:
        self._scripts = deque(scripts)
        self.calls: list[list[LlmMessage]] = []
        """The messages of every call so far."""
        self.max_output_tokens: list[int] = []

    def add(self, *scripts: Sequence[object]) -> None:
        self._scripts.extend(scripts)

    async def stream(
        self, messages: Sequence[LlmMessage], *, max_output_tokens: int
    ) -> AsyncIterator[ChatChunk]:
        self.calls.append(list(messages))
        self.max_output_tokens.append(max_output_tokens)
        if not self._scripts:
            raise AssertionError("ScriptedChat was called more often than its script allows")
        for item in self._scripts.popleft():
            if isinstance(item, Gate):
                await item.wait()
            elif isinstance(item, BaseException):
                raise item
            elif isinstance(item, StreamEnd):
                yield item
            else:
                yield TextDelta(str(item))

    @property
    def system_messages(self) -> list[str]:
        return [call[0].content for call in self.calls]


class _Clock:
    """A clock the test owns: the coalescing then depends on characters, not on wall time."""

    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class ChatWorld(RunWorld):
    """``RunWorld`` plus a real chat supervisor over a scripted streaming model."""

    def __init__(
        self, engine: AsyncEngine, database_url: str, settings: Settings | None = None
    ) -> None:
        super().__init__(engine, database_url, settings or make_chat_settings())
        self.chat_llm = ScriptedChat()
        self.draft_llm = FakeLlm()
        self.clock = _Clock()
        self.supervisor = self.new_supervisor()

    def new_supervisor(self) -> ChatTaskSupervisor:
        return ChatTaskSupervisor(
            db_manager=self.db,
            billing=BillingService(),
            chat_port_for=lambda _key: self.chat_llm,
            structured_port_for=lambda _key: self.draft_llm,
            events=StudioChatEvents(self.bus),  # type: ignore[arg-type]
            shutdown_grace_seconds=5,
            clock=self.clock,
        )

    def sweeper(self, *, deadline_seconds: int | None = None) -> ChatSweeper:
        return ChatSweeper(
            db_manager=self.db,
            billing=BillingService(),
            events=StudioChatEvents(self.bus),  # type: ignore[arg-type]
            deadline_seconds=deadline_seconds or self.settings.studio_chat_reply_deadline_seconds,
        )

    async def close(self) -> None:
        await self.settle()
        await super().close()

    # -- services ----------------------------------------------------------------------------

    def chat_service(
        self, session: AsyncSession, *, supervisor: ChatTaskSupervisor | None = None
    ) -> ChatService:
        return ChatService(
            session,
            billing=BillingService(),
            settings=self.settings,
            events=StudioChatEvents(self.bus),  # type: ignore[arg-type]
            assembler=ArtistChatContextAssembler(),
            supervisor=supervisor or self.supervisor,
        )

    def draft_service(self, session: AsyncSession) -> MemoryDraftService:
        return MemoryDraftService(
            session,
            billing=BillingService(),
            settings=self.settings,
            events=StudioChatEvents(self.bus),  # type: ignore[arg-type]
            supervisor=self.supervisor,
        )

    # -- doing things ------------------------------------------------------------------------

    async def new_chat(
        self, actor: Actor, title: str = "Interview", *, artist: UUID | None = None
    ) -> UUID:
        async with self.factory() as session, session.begin():
            chat = await self.chat_service(session).create(
                artist or _artist_of(actor), actor.user_id, PRODUCT, ChatCreate(title=title)
            )
        return chat.id

    async def send(
        self,
        actor: Actor,
        chat_id: UUID,
        text_: str = "Tell me about the rooftop.",
        *,
        client_message_id: UUID | None = None,
        deadline_seconds: float | None = None,
        dispatch: bool = True,
    ) -> ChatWrite:
        """What the route does: send, commit, then dispatch (events, then the task)."""
        async with self.factory() as session:
            service = self.chat_service(session)
            write = await service.send(
                chat_id,
                actor.user_id,
                PRODUCT,
                MessageCreate(
                    client_message_id=client_message_id or UUID(int=_next()), content=text_
                ),
            )
            await session.commit()
        if deadline_seconds is not None and write.job is not None:
            write = dataclasses.replace(
                write,
                job=dataclasses.replace(write.job, deadline_seconds=deadline_seconds),  # type: ignore[arg-type]
            )
        if dispatch:
            async with self.factory() as session:
                await self.chat_service(session).dispatch(write)
        return write

    async def ask(
        self, actor: Actor, chat_id: UUID, text_: str = "Tell me about the rooftop.", **kw: Any
    ) -> tuple[StudioArtistChatMessage, StudioArtistChatMessage]:
        """Send, let the reply finish, and return the stored pair."""
        write = await self.send(actor, chat_id, text_, **kw)
        await self.settle()
        return await self.message(write.user_message.id), await self.message(
            write.artist_message.id
        )

    async def settle(self) -> None:
        """Wait for every task in flight (no sleeping: the tasks themselves are awaited)."""
        while tasks := self.supervisor.tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def request_draft(
        self, actor: Actor, message_id: UUID, key: str | None = None, *, dispatch: bool = True
    ) -> DraftWrite:
        async with self.factory() as session:
            write = await self.draft_service(session).create(
                message_id,
                idempotency_key=key if key is not None else f"key-{_next()}",
                user_id=actor.user_id,
                product_id=PRODUCT,
            )
            await session.commit()
        if dispatch:
            async with self.factory() as session:
                await self.draft_service(session).dispatch(write)
        return write

    async def draft_of(
        self, actor: Actor, message_id: UUID, key: str | None = None
    ) -> StudioMemoryDraft:
        """Request a draft, let it finish, return the stored row."""
        write = await self.request_draft(actor, message_id, key)
        await self.settle()
        return await self.draft(write.draft.id)

    # -- reads -------------------------------------------------------------------------------

    async def message(self, message_id: UUID) -> StudioArtistChatMessage:
        async with self.factory() as session:
            return (
                await session.execute(
                    select(StudioArtistChatMessage)
                    .where(StudioArtistChatMessage.id == message_id)
                    .execution_options(populate_existing=True)
                )
            ).scalar_one()

    async def messages(self, chat_id: UUID) -> list[StudioArtistChatMessage]:
        async with self.factory() as session:
            result = await session.execute(
                select(StudioArtistChatMessage)
                .where(StudioArtistChatMessage.chat_id == chat_id)
                .order_by(StudioArtistChatMessage.position)
            )
            return list(result.scalars())

    async def chat(self, chat_id: UUID) -> StudioArtistChat:
        async with self.factory() as session:
            return (
                await session.execute(
                    select(StudioArtistChat)
                    .where(StudioArtistChat.id == chat_id)
                    .execution_options(populate_existing=True)
                )
            ).scalar_one()

    async def draft(self, draft_id: UUID) -> StudioMemoryDraft:
        async with self.factory() as session:
            return (
                await session.execute(
                    select(StudioMemoryDraft)
                    .where(StudioMemoryDraft.id == draft_id)
                    .execution_options(populate_existing=True)
                )
            ).scalar_one()

    async def count(self, model: Any, actor: Actor) -> int:
        """Rows of ``model`` (any table with a ``user_id``) that belong to ``actor``: scoped, so a
        test never depends on what other tests left behind."""
        async with self.factory() as session:
            return (
                await session.execute(
                    select(func.count()).select_from(model).where(model.user_id == actor.user_id)
                )
            ).scalar_one()

    async def age(self, *, message_id: UUID | None = None, draft_id: UUID | None = None) -> None:
        """Make a pending row look orphaned: created an hour ago."""
        async with self.factory() as session, session.begin():
            if message_id is not None:
                await session.execute(
                    text(
                        "UPDATE studio_artist_chat_messages "
                        "SET created_at = now() - interval '1 hour' WHERE id = :id"
                    ),
                    {"id": message_id},
                )
            if draft_id is not None:
                await session.execute(
                    text(
                        "UPDATE studio_memory_drafts "
                        "SET created_at = now() - interval '1 hour' WHERE id = :id"
                    ),
                    {"id": draft_id},
                )

    # -- bus ---------------------------------------------------------------------------------

    def deltas(self, message_id: UUID) -> list[Any]:
        return [
            p
            for _, t, p in self.bus.events
            if t.value == "studio.chat.delta" and p.message_id == message_id
        ]

    def updates(self, message_id: UUID) -> list[Any]:
        return [
            p
            for _, t, p in self.bus.events
            if t.value == "studio.chat.message.updated" and p.message_id == message_id
        ]

    def draft_updates(self, draft_id: UUID) -> list[Any]:
        return [
            p
            for _, t, p in self.bus.events
            if t.value == "studio.memory_draft.updated" and p.draft_id == draft_id
        ]


_counter = 0


def _next() -> int:
    global _counter  # noqa: PLW0603 - a test id source
    _counter += 1
    return _counter + 10_000


def _artist_of(actor: Actor) -> UUID:
    if actor.artist_id is None:
        raise ValueError("the actor has no artist (add_actor(artist=True))")
    return actor.artist_id
