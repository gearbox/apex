"""A committed studio world for planner-run integration tests.

The executor opens its own short sessions (that is the point of the design), so these tests cannot
run inside the rolled-back savepoint of the ``db_session`` fixture: rows are committed and removed
again afterwards. ``token_transactions`` is trigger-immutable, hence the ``replica`` role when
cleaning up.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any
from uuid import UUID, uuid4

import msgspec
from pydantic import SecretStr
from sqlalchemy import delete, func, select, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from src.api.schemas.studio import ProjectProfilePut
from src.api.security.jwt import JWTConfig, JWTService
from src.api.services.billing import BillingService
from src.api.services.legal.errors import LegalAcceptanceRequiredError  # noqa: F401
from src.core.config import Settings
from src.core.enums import StudioReleaseKind, StudioRunStatus
from src.core.llm_models import parse_model_entries
from src.core.uid import new_id
from src.db.models.billing import TokenAccount, TokenTransaction
from src.db.models.library import LibraryProject
from src.db.models.studio import StudioDocumentVersion, StudioRun, StudioTrack
from src.db.models.user import User
from src.db.session import DatabaseManager
from src.llm.registry import LlmModelRegistry
from src.studio.run_requests import PlanRunRequest
from src.studio.services.project import StudioProjectService
from src.studio.services.run import RunWrite, StudioRunService
from src.studio.services.run_context import MemoryArtistContextProvider
from src.studio.services.run_events import StudioRunEvents
from src.studio.services.run_executor import ClaimedRun, StudioRunExecutor
from src.studio.services.run_kinds.registry import default_handlers
from src.studio.services.track import TrackService
from src.studio.services.version import DocumentVersionService
from tests.legal_support import TEST_LEGAL_DIGEST

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncEngine

    from src.api.schemas.events import EventType
    from src.api.services.billing import BalanceEvent
    from src.llm.port import StructuredLlmPort
    from src.studio.documents import LyricsDocument, PlanDocument

__all__ = [
    "PRODUCT",
    "SECRET",
    "Actor",
    "RecordingBus",
    "RunWorld",
    "make_registry",
    "make_settings",
]

PRODUCT = "vex"
SECRET = "test_secret_key_for_testing_only_256bits_long"

MODELS: dict[str, dict[str, Any]] = {
    "grok": {
        "base_url": "https://llm-a.example.test/v1",
        "api_key_env": "KEY_A",
        "model": "model-a",
    },
    "qwen": {
        "base_url": "https://llm-b.example.test/v1",
        "api_key_env": "KEY_B",
        "model": "model-b",
        "structured_output": "json_object",
    },
}


def make_settings(**overrides: Any) -> Settings:
    """Settings with studio runs enabled and a two-model registry (prices 1 / 1 / 1)."""
    base: dict[str, Any] = {
        "jwt_secret_key": "x" * 40,
        "studio_enabled": True,
        "studio_runs_enabled": True,
        "llm_models": MODELS,
        "llm_default_model": "grok",
    }
    return Settings(_env_file=None, **base | overrides)  # type: ignore[call-arg]


def make_registry(settings: Settings | None = None) -> LlmModelRegistry:
    settings = settings or make_settings()
    return LlmModelRegistry(
        parse_model_entries(settings.llm_models),
        settings.llm_default_model,
        {key: SecretStr(f"secret-for-{key}") for key in settings.llm_models},
    )


class RecordingBus:
    """Stands in for ``EventBus``: records what would have been published."""

    def __init__(self) -> None:
        self.events: list[tuple[UUID, EventType, Any]] = []
        self.balance_events: list[BalanceEvent] = []
        self.on_publish: Any = None
        """Optional async callable awaited with each ``(user_id, event_type, payload)``."""

    async def publish(self, *, user_id: UUID, event_type: EventType, payload: object) -> None:
        self.events.append((user_id, event_type, payload))
        if self.on_publish is not None:
            await self.on_publish(user_id, event_type, payload)

    async def publish_balance(self, event: BalanceEvent | None) -> None:
        if event is not None:
            self.balance_events.append(event)

    @property
    def run_events(self) -> list[Any]:
        """The ``studio.run.updated`` payloads, in publish order."""
        return [p for _, t, p in self.events if t.value == "studio.run.updated"]


@dataclass
class Actor:
    """A committed user with a funded personal account and one studio-enabled project."""

    user_id: UUID
    account_id: UUID
    project_id: UUID
    tracks: list[UUID] = field(default_factory=list)
    artist_id: UUID | None = None
    """The project's artist (``add_actor(artist=True)``), else ``None``."""

    @property
    def headers(self) -> dict[str, str]:
        token, _ = JWTService(JWTConfig(secret_key=SECRET)).create_access_token(
            self.user_id, product_id=PRODUCT, legal_digest=TEST_LEGAL_DIGEST
        )
        return {"Authorization": f"Bearer {token}"}


class RunWorld:
    """Committed rows, services and cleanup for planner-run tests."""

    def __init__(
        self, engine: AsyncEngine, database_url: str, settings: Settings | None = None
    ) -> None:
        self.engine = engine
        self.factory = async_sessionmaker(bind=engine, expire_on_commit=False)
        self.db = DatabaseManager(database_url, pool_size=4, max_overflow=0)
        self.settings = settings or make_settings()
        self.registry = make_registry(self.settings)
        self.bus = RecordingBus()
        self.actors: list[Actor] = []

    # -- setup / teardown --------------------------------------------------------------------

    async def add_actor(self, *, balance: int = 10, artist: bool = False) -> Actor:
        user = User(
            id=new_id(),
            email=f"runs-{uuid4().hex}@example.com",
            password_hash="x" * 64,
            product_id=PRODUCT,
            is_active=True,
        )
        account = TokenAccount(
            id=new_id(), user_id=user.id, account_type="personal", product_id=PRODUCT
        )
        project = LibraryProject(
            id=new_id(), product_id=PRODUCT, user_id=user.id, name=f"Album {uuid4().hex[:6]}"
        )
        async with self.factory() as session, session.begin():
            session.add(user)
            await session.flush()
            session.add_all([account, project])
            await session.flush()
            if balance:
                session.add(
                    TokenTransaction(
                        id=new_id(),
                        account_id=account.id,
                        transaction_type="credit",
                        amount=balance,
                        balance_after=balance,
                        product_id=PRODUCT,
                    )
                )
            artist_id = None
            if artist:
                from src.api.schemas.studio import ArtistCreate
                from src.studio.services.artist import ArtistService

                created = await ArtistService(session).create(
                    user.id,
                    PRODUCT,
                    ArtistCreate(
                        name="Nova",
                        persona="A SECRET-PERSONA synth-pop singer",
                        voice_notes="breathy",
                        style_tags=["dreamy"],
                        global_avoid=["autotune"],
                        aesthetic_references=["neon noir"],
                        default_language="en",
                    ),
                )
                artist_id = created.id
            await StudioProjectService(session).upsert_profile(
                project.id,
                user.id,
                PRODUCT,
                ProjectProfilePut(
                    release_kind=StudioReleaseKind.ALBUM,
                    artist_id=artist_id,
                    concept="SECRET-CONCEPT night drive" if artist else None,
                ),
            )
        actor = Actor(
            user_id=user.id, account_id=account.id, project_id=project.id, artist_id=artist_id
        )
        self.actors.append(actor)
        return actor

    async def close(self) -> None:
        """Remove everything this world committed.

        ``token_transactions`` is trigger-immutable, so the ledger is cleared under the
        ``replica`` role - which also switches off foreign-key triggers, so the cascades of
        ``DELETE users`` must run in a *separate* transaction.
        """
        async with self.factory() as session, session.begin():
            await session.execute(text("SET LOCAL session_replication_role = 'replica'"))
            for actor in self.actors:
                await session.execute(
                    delete(TokenTransaction).where(TokenTransaction.account_id == actor.account_id)
                )
        async with self.factory() as session, session.begin():
            for actor in self.actors:
                await session.execute(delete(User).where(User.id == actor.user_id))
        await self.db.close()

    # -- services ----------------------------------------------------------------------------

    def service(self, session: AsyncSession, **overrides: Any) -> StudioRunService:
        return StudioRunService(
            session,
            billing=BillingService(),
            settings=overrides.pop("settings", self.settings),
            events=StudioRunEvents(self.bus),  # type: ignore[arg-type]
        )

    def executor(
        self,
        llm: StructuredLlmPort,
        *,
        settings: Any = None,
        provider: Any = None,
    ) -> StudioRunExecutor:
        return StudioRunExecutor(
            db_manager=self.db,
            billing=BillingService(),
            registry=self.registry,
            llm_factory=lambda _key: llm,
            context_provider=provider or MemoryArtistContextProvider(),
            events=StudioRunEvents(self.bus),  # type: ignore[arg-type]
            settings=settings or self.settings,
            handlers=default_handlers(),
        )

    # -- data --------------------------------------------------------------------------------

    async def add_memory(
        self, actor: Actor, *, artist_id: UUID | None = None, **fields: Any
    ) -> UUID:
        """A committed memory of the actor's artist (or of ``artist_id``); returns its id.

        ``fields`` are ``MemoryCreate`` fields; the title defaults to a unique one.
        """
        from src.api.schemas.studio import MemoryCreate
        from src.studio.services.memory import MemoryService

        target = artist_id or actor.artist_id
        if target is None:
            raise ValueError("the actor has no artist")
        fields.setdefault("title", f"Memory {uuid4().hex[:6]}")
        fields.setdefault("body", "a quiet evening")
        async with self.factory() as session, session.begin():
            created = await MemoryService(session).create(
                target, actor.user_id, PRODUCT, MemoryCreate(**fields)
            )
        return created.id

    async def set_memory(self, actor: Actor, memory_id: UUID, **fields: Any) -> None:
        """Edit, archive or restore a memory (``MemoryPatch`` fields), committed.

        ``status`` may be given as its wire string (``"archived"``): the route decodes it to the
        enum, so this does the same.
        """
        from src.api.schemas.studio import MemoryPatch
        from src.core.enums import StudioMemoryStatus
        from src.studio.services.memory import MemoryService

        if isinstance(fields.get("status"), str):
            fields["status"] = StudioMemoryStatus(fields["status"])

        async with self.factory() as session, session.begin():
            await MemoryService(session).patch(
                memory_id, actor.user_id, PRODUCT, MemoryPatch(**fields)
            )

    async def delete_memory(self, actor: Actor, memory_id: UUID) -> None:
        from src.studio.services.memory import MemoryService

        async with self.factory() as session, session.begin():
            await MemoryService(session).delete(memory_id, actor.user_id, PRODUCT)

    async def new_track(
        self, actor: Actor, title: str = "Track", plan: PlanDocument | None = None
    ) -> UUID:
        from src.api.schemas.studio import PlanVersionCreate
        from src.core.enums import StudioDocumentOrigin

        async with self.factory() as session, session.begin():
            track = await TrackService(session).create(
                actor.project_id, actor.user_id, PRODUCT, title
            )
            if plan is not None:
                await DocumentVersionService(session).create(
                    track.id,
                    PlanVersionCreate(content=plan, make_current=True),
                    actor.user_id,
                    PRODUCT,
                    origin=StudioDocumentOrigin.USER,
                )
        actor.tracks.append(track.id)
        return track.id

    async def replace_current_plan(
        self, actor: Actor, track_id: UUID, plan: PlanDocument | None = None
    ) -> UUID:
        """What a user does in the UI, in one transaction: make another plan current, then delete
        the plan (and the lyrics) that were current before.

        The new plan detaches any lyrics it does not fit; the old current lyrics, which the old
        plan was the base of, go first so the old plan can be deleted. Returns the new plan's id.
        """
        from src.api.schemas.studio import PlanVersionCreate
        from src.core.enums import StudioDocumentOrigin, StudioIncompatibleLyricsPolicy
        from tests.studio_support import make_plan

        before = await self.track_row(track_id)
        async with self.factory() as session, session.begin():
            versions = DocumentVersionService(session)
            created, _ = await versions.create(
                track_id,
                PlanVersionCreate(
                    content=plan or make_plan(title="Replacement", empty=True),
                    make_current=True,
                    incompatible_lyrics=StudioIncompatibleLyricsPolicy.DETACH,
                ),
                actor.user_id,
                PRODUCT,
                origin=StudioDocumentOrigin.USER,
            )
            for old in (before.current_lyrics_version_id, before.current_plan_version_id):
                if old is not None:
                    await versions.delete(old, actor.user_id, PRODUCT)
        return created.version.id

    async def submit(
        self,
        actor: Actor,
        track_id: UUID,
        *,
        brief: str = "a song about rain",
        key: str | None = None,
        request: Any = None,
    ) -> RunWrite:
        """Submit, commit and publish - what the route does."""
        async with self.factory() as session:
            service = self.service(session)
            write = await service.submit(
                track_id,
                request or PlanRunRequest(brief=brief),
                idempotency_key=key if key is not None else f"key-{uuid4().hex}",
                user_id=actor.user_id,
                product_id=PRODUCT,
            )
            await session.commit()
        await service.publish(write)
        return write

    async def drive(self, llm: StructuredLlmPort, *, slots: int = 4) -> list[ClaimedRun]:
        """One worker tick, run to completion: claim, then execute each claimed run."""
        executor = self.executor(llm)
        claimed = await executor.claim(slots)
        for item in claimed:
            await executor.execute(item)
        return claimed

    async def create_plan(
        self, actor: Actor, track_id: UUID, plan: PlanDocument, *, make_current: bool = True
    ) -> UUID:
        """A user-made plan version (committed). Returns its id."""
        from src.api.schemas.studio import PlanVersionCreate
        from src.core.enums import StudioDocumentOrigin

        async with self.factory() as session, session.begin():
            created, _ = await DocumentVersionService(session).create(
                track_id,
                PlanVersionCreate(content=plan, make_current=make_current),
                actor.user_id,
                PRODUCT,
                origin=StudioDocumentOrigin.USER,
            )
        return created.version.id

    async def create_lyrics(
        self,
        actor: Actor,
        track_id: UUID,
        lyrics: LyricsDocument,
        *,
        plan_version_id: UUID,
        make_current: bool = True,
        parent_version_id: UUID | None = None,
    ) -> UUID:
        """A user-made lyrics version (committed). Returns its id."""
        from src.api.schemas.studio import LyricsVersionCreate
        from src.core.enums import StudioDocumentOrigin

        async with self.factory() as session, session.begin():
            created, _ = await DocumentVersionService(session).create(
                track_id,
                LyricsVersionCreate(
                    content=lyrics,
                    based_on_plan_version_id=plan_version_id,
                    parent_version_id=parent_version_id,
                    make_current=make_current,
                ),
                actor.user_id,
                PRODUCT,
                origin=StudioDocumentOrigin.USER,
            )
        return created.version.id

    async def insert_raw_lyrics(
        self,
        actor: Actor,
        track_id: UUID,
        *,
        plan_version_id: UUID,
        content: dict[str, Any],
        make_current: bool = False,
    ) -> UUID:
        """A lyrics row written **without validation**: what an older, laxer release may have
        stored (a bracket in a line, a delivery that reads as a header). It still decodes."""
        from src.core.enums import StudioDocumentKind
        from src.db.repositories.studio import (
            StudioDocumentVersionRepository,
            StudioTrackRepository,
        )
        from src.studio.documents import CURRENT_SCHEMA_VERSION

        async with self.factory() as session, session.begin():
            tracks = StudioTrackRepository(session)
            await tracks.lock(track_id, user_id=actor.user_id, product_id=PRODUCT)
            row, _ = await StudioDocumentVersionRepository(session).insert_or_get_sibling(
                user_id=actor.user_id,
                product_id=PRODUCT,
                track_id=track_id,
                kind=StudioDocumentKind.LYRICS,
                version_number=await tracks.allocate_version_number(
                    track_id, StudioDocumentKind.LYRICS
                ),
                parent_version_id=None,
                based_on_plan_version_id=plan_version_id,
                schema_version=CURRENT_SCHEMA_VERSION[StudioDocumentKind.LYRICS],
                origin="user",
                content=content,
                content_hash=uuid4().hex + uuid4().hex,
            )
            if make_current:
                await tracks.set_current(track_id, lyrics_version_id=row.id)
        return row.id

    async def set_current(
        self,
        actor: Actor,
        track_id: UUID,
        *,
        plan_version_id: Any = msgspec.UNSET,
        lyrics_version_id: Any = msgspec.UNSET,
    ) -> None:
        async with self.factory() as session, session.begin():
            await DocumentVersionService(session).set_current(
                track_id,
                actor.user_id,
                PRODUCT,
                plan_version_id=plan_version_id,
                lyrics_version_id=lyrics_version_id,
            )

    async def delete_version(self, actor: Actor, version_id: UUID) -> None:
        async with self.factory() as session, session.begin():
            await DocumentVersionService(session).delete(version_id, actor.user_id, PRODUCT)

    # -- reads -------------------------------------------------------------------------------

    async def run(self, run_id: UUID) -> StudioRun:
        async with self.factory() as session:
            return (
                await session.execute(
                    select(StudioRun)
                    .where(StudioRun.id == run_id)
                    .execution_options(populate_existing=True)
                )
            ).scalar_one()

    async def runs(self, track_id: UUID) -> list[StudioRun]:
        async with self.factory() as session:
            result = await session.execute(
                select(StudioRun)
                .where(StudioRun.track_id == track_id)
                .order_by(StudioRun.created_at)
            )
            return list(result.scalars())

    async def count_runs(self, actor: Actor) -> int:
        async with self.factory() as session:
            return (
                await session.execute(
                    select(func.count())
                    .select_from(StudioRun)
                    .where(StudioRun.user_id == actor.user_id)
                )
            ).scalar_one()

    async def ledger(self, actor: Actor) -> list[TokenTransaction]:
        async with self.factory() as session:
            result = await session.execute(
                select(TokenTransaction)
                .where(TokenTransaction.account_id == actor.account_id)
                .order_by(TokenTransaction.created_at, TokenTransaction.id)
            )
            return list(result.scalars())

    async def balance(self, actor: Actor) -> int:
        async with self.factory() as session:
            return await BillingService().get_balance(actor.account_id, session=session)

    async def versions(self, track_id: UUID) -> list[StudioDocumentVersion]:
        async with self.factory() as session:
            result = await session.execute(
                select(StudioDocumentVersion)
                .where(StudioDocumentVersion.track_id == track_id)
                .order_by(StudioDocumentVersion.version_number)
            )
            return list(result.scalars())

    async def track_row(self, track_id: UUID) -> StudioTrack:
        async with self.factory() as session:
            return (
                await session.execute(
                    select(StudioTrack)
                    .where(StudioTrack.id == track_id)
                    .execution_options(populate_existing=True)
                )
            ).scalar_one()

    async def expire_lease(self, run_id: UUID) -> None:
        """Make a running run look abandoned: its lease ended a second ago."""
        async with self.factory() as session, session.begin():
            await session.execute(
                text(
                    "UPDATE studio_runs SET lease_expires_at = now() - interval '1 second' "
                    "WHERE id = :id"
                ),
                {"id": run_id},
            )

    async def refunds_and_debits(self, actor: Actor) -> tuple[int, int]:
        """(debits, refunds) counted in the ledger."""
        rows = await self.ledger(actor)
        return (
            sum(r.transaction_type == "debit" for r in rows),
            sum(r.transaction_type == "refund" for r in rows),
        )

    @staticmethod
    def terminal(run: StudioRun) -> bool:
        return StudioRunStatus(run.status) in StudioRunStatus.terminal()
