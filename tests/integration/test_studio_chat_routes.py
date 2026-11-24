"""The artist chat over HTTP: real controllers, guards, DI, handlers and PostgreSQL.

``AsyncTestClient`` runs handlers on another event loop, so the reply task is **not** started from
inside a request here: the app is given a supervisor that only records the jobs it is handed (the
route's contract is "commit, publish, hand over the job"), and the test then starts each job on its
own loop with the real supervisor and reads the result back over HTTP.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, Any
from uuid import UUID, uuid4

import pytest
import pytest_asyncio
from litestar import Litestar
from litestar.di import Provide
from litestar.exceptions import HTTPException
from litestar.testing import AsyncTestClient
from sqlalchemy.ext.asyncio import AsyncSession  # noqa: TC002 - runtime: Litestar DI annotation

from src.api.app import (
    global_exception_handler,
    http_exception_handler,
    insufficient_balance_handler,
    legal_acceptance_required_handler,
    studio_error_handler,
)
from src.api.dependencies.common import (
    get_product_id,
    get_studio_artist_service,
    get_studio_memory_service,
)
from src.api.routes.studio_artists import StudioArtistController
from src.api.routes.studio_chat import StudioChatController
from src.api.routes.studio_memories import StudioMemoryController
from src.api.security.jwt import JWTConfig, JWTService
from src.api.services.billing import BillingService
from src.api.services.billing_errors import InsufficientBalanceError
from src.api.services.legal.errors import LegalAcceptanceRequiredError
from src.api.services.token_revocation import TokenRevocationService
from src.core.product_registry import VEX_CONFIG
from src.db.models.studio_chat import StudioArtistChat
from src.studio.chat.draft import MemoryDraftDraft
from src.studio.errors import StudioError
from src.studio.services.chat import ChatService
from src.studio.services.chat_context import ArtistChatContextAssembler
from src.studio.services.chat_events import StudioChatEvents
from src.studio.services.memory_draft import MemoryDraftService
from tests.legal_support import TEST_LEGAL_REGISTRY
from tests.llm_support import FakeLlm
from tests.studio_chat_support import reply
from tests.studio_runs_support import SECRET

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from litestar.types import ASGIApp, Receive, Scope, Send

    from src.studio.chat.jobs import ChatReplyJob, MemoryDraftJob
    from tests.studio_chat_support import ChatWorld
    from tests.studio_runs_support import Actor

S = "/v1/studio"
LONG = "I remember the rooftop, the neon sign and the rain on the asphalt that night."


class RecordingSupervisor:
    """What the routes hand their jobs to: it only records them."""

    def __init__(self) -> None:
        self.replies: list[ChatReplyJob] = []
        self.drafts: list[MemoryDraftJob] = []

    def start_reply(self, job: ChatReplyJob) -> None:
        self.replies.append(job)

    def start_memory_draft(self, job: MemoryDraftJob) -> None:
        self.drafts.append(job)


def _scope(app: ASGIApp) -> ASGIApp:
    async def middleware(scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] in ("http", "websocket"):
            state = scope.setdefault("state", {})
            state["product_config"] = VEX_CONFIG
            state["product_id"] = VEX_CONFIG.slug
        await app(scope, receive, send)

    return middleware


def _build_app(world: ChatWorld, recorder: RecordingSupervisor) -> Litestar:
    async def provide_session() -> AsyncGenerator[AsyncSession]:
        async with world.factory() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise

    def provide_chat_service(session: AsyncSession) -> ChatService:
        return ChatService(
            session,
            billing=BillingService(),
            settings=world.settings,
            events=StudioChatEvents(world.bus),  # type: ignore[arg-type]
            assembler=ArtistChatContextAssembler(),
            supervisor=recorder,  # type: ignore[arg-type]
        )

    def provide_draft_service(session: AsyncSession) -> MemoryDraftService:
        return MemoryDraftService(
            session,
            billing=BillingService(),
            settings=world.settings,
            events=StudioChatEvents(world.bus),  # type: ignore[arg-type]
            supervisor=recorder,  # type: ignore[arg-type]
        )

    app = Litestar(
        route_handlers=[StudioChatController, StudioMemoryController, StudioArtistController],
        dependencies={
            "session": Provide(provide_session),
            "product_id": Provide(get_product_id, sync_to_thread=False),
            "studio_chat_service": Provide(provide_chat_service, sync_to_thread=False),
            "studio_memory_draft_service": Provide(provide_draft_service, sync_to_thread=False),
            "studio_memory_service": Provide(get_studio_memory_service, sync_to_thread=False),
            "studio_artist_service": Provide(get_studio_artist_service, sync_to_thread=False),
        },
        exception_handlers={
            HTTPException: http_exception_handler,
            StudioError: studio_error_handler,
            InsufficientBalanceError: insufficient_balance_handler,
            LegalAcceptanceRequiredError: legal_acceptance_required_handler,
            Exception: global_exception_handler,
        },
        middleware=[_scope],
    )
    app.state["jwt_service"] = JWTService(JWTConfig(secret_key=SECRET))
    app.state["token_revocation"] = TokenRevocationService(None, max_token_ttl_seconds=0)
    app.state["legal_registry"] = TEST_LEGAL_REGISTRY
    return app


@pytest.fixture(autouse=True)
def chat_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "src.api.security.studio.get_settings",
        lambda: SimpleNamespace(studio_enabled=True, studio_chat_enabled=True),
    )


@pytest_asyncio.fixture
async def world(chat_world: ChatWorld) -> ChatWorld:
    return chat_world


@pytest.fixture
def recorder() -> RecordingSupervisor:
    return RecordingSupervisor()


@pytest_asyncio.fixture
async def client(
    world: ChatWorld, recorder: RecordingSupervisor
) -> AsyncGenerator[AsyncTestClient[Litestar]]:
    async with AsyncTestClient(app=_build_app(world, recorder)) as test_client:
        yield test_client


@pytest_asyncio.fixture
async def actor(world: ChatWorld) -> Actor:
    return await world.add_actor(balance=20, artist=True)


async def _new_chat(
    client: AsyncTestClient[Litestar], actor: Actor, title: str = "Interview"
) -> str:
    response = await client.post(
        f"{S}/artists/{actor.artist_id}/chats", headers=actor.headers, json={"title": title}
    )
    assert response.status_code == 201, response.text
    return str(response.json()["id"])


async def _send(
    client: AsyncTestClient[Litestar],
    actor: Actor,
    chat_id: str,
    content: str = "Where did you grow up?",
    client_message_id: str | None = None,
) -> Any:
    return await client.post(
        f"{S}/chats/{chat_id}/messages",
        headers=actor.headers,
        json={"client_message_id": client_message_id or str(uuid4()), "content": content},
    )


class TestChatRoutes:
    async def test_create_201_get_list_patch_delete_204(
        self, client: AsyncTestClient[Litestar], actor: Actor
    ) -> None:
        chat_id = await _new_chat(client, actor, "  First  ")

        got = await client.get(f"{S}/chats/{chat_id}", headers=actor.headers)
        assert got.status_code == 200
        assert got.json()["title"] == "First"
        assert set(got.json()) == {
            "id",
            "artist_id",
            "title",
            "message_count",
            "created_at",
            "updated_at",
        }

        listed = await client.get(f"{S}/artists/{actor.artist_id}/chats", headers=actor.headers)
        assert listed.status_code == 200
        assert [c["id"] for c in listed.json()["items"]] == [chat_id]
        assert listed.json()["has_more"] is False

        patched = await client.patch(
            f"{S}/chats/{chat_id}", headers=actor.headers, json={"title": "Renamed"}
        )
        assert patched.status_code == 200
        assert patched.json()["title"] == "Renamed"

        deleted = await client.delete(f"{S}/chats/{chat_id}", headers=actor.headers)
        assert deleted.status_code == 204
        assert (await client.get(f"{S}/chats/{chat_id}", headers=actor.headers)).status_code == 404

    @pytest.mark.parametrize(
        "body", [{}, {"title": ""}, {"title": "x" * 121}, {"title": "t", "x": 1}]
    )
    async def test_a_bad_title_is_a_client_error(
        self, client: AsyncTestClient[Litestar], actor: Actor, body: dict[str, Any]
    ) -> None:
        response = await client.post(
            f"{S}/artists/{actor.artist_id}/chats", headers=actor.headers, json=body
        )
        assert response.status_code in {400, 422}

    async def test_a_blank_title_is_document_invalid(
        self, client: AsyncTestClient[Litestar], actor: Actor
    ) -> None:
        response = await client.post(
            f"{S}/artists/{actor.artist_id}/chats", headers=actor.headers, json={"title": "   "}
        )
        assert response.status_code == 422
        assert response.json()["error"] == "document_invalid"

    async def test_404_for_another_users_chat_and_artist(
        self, client: AsyncTestClient[Litestar], world: ChatWorld, actor: Actor
    ) -> None:
        chat_id = await _new_chat(client, actor)
        stranger = await world.add_actor(balance=5, artist=True)
        for method, path, body in (
            ("GET", f"{S}/chats/{chat_id}", None),
            ("PATCH", f"{S}/chats/{chat_id}", {"title": "x"}),
            ("DELETE", f"{S}/chats/{chat_id}", None),
            ("GET", f"{S}/chats/{chat_id}/messages", None),
            ("GET", f"{S}/artists/{actor.artist_id}/chats", None),
        ):
            response = await client.request(method, path, headers=stranger.headers, json=body)
            assert response.status_code == 404, (method, path)
        response = await _send(client, stranger, chat_id)
        assert response.status_code == 404
        assert response.json()["error"] == "chat_not_found"

    async def test_401_without_a_token(
        self, client: AsyncTestClient[Litestar], actor: Actor
    ) -> None:
        assert (await client.get(f"{S}/artists/{actor.artist_id}/chats")).status_code == 401

    async def test_a_state_changing_route_needs_legal_acceptance(
        self, client: AsyncTestClient[Litestar], actor: Actor, world: ChatWorld
    ) -> None:
        token, _ = JWTService(JWTConfig(secret_key=SECRET)).create_access_token(
            actor.user_id, product_id="vex", legal_digest="stale-digest"
        )
        response = await client.post(
            f"{S}/artists/{actor.artist_id}/chats",
            headers={"Authorization": f"Bearer {token}"},
            json={"title": "t"},
        )
        assert response.status_code == 428
        assert await world.count(StudioArtistChat, actor) == 0


class TestSending:
    async def test_202_with_the_pair_and_the_job_is_handed_over(
        self,
        client: AsyncTestClient[Litestar],
        world: ChatWorld,
        recorder: RecordingSupervisor,
        actor: Actor,
    ) -> None:
        chat_id = await _new_chat(client, actor)
        response = await _send(client, actor, chat_id, "Where did you grow up?")

        assert response.status_code == 202, response.text
        body = response.json()
        user, artist = body["user_message"], body["artist_message"]
        assert (user["role"], user["status"], user["position"]) == ("user", "complete", 1)
        assert user["content"] == "Where did you grow up?"
        assert (artist["role"], artist["status"], artist["position"]) == ("artist", "pending", 2)
        assert artist["content"] == ""
        assert (artist["revision"], artist["truncated"], artist["error_code"]) == (1, False, None)
        assert set(artist) == {
            "id",
            "chat_id",
            "position",
            "role",
            "status",
            "content",
            "truncated",
            "memory_ids_used",
            "error_code",
            "revision",
            "created_at",
            "finished_at",
        }
        # durable before the job is handed over, and the pending event went out
        assert [j.message_id for j in recorder.replies] == [UUID(artist["id"])]
        assert [(u.status, u.revision) for u in world.updates(UUID(artist["id"]))] == [
            ("pending", 1)
        ]
        assert await world.refunds_and_debits(actor) == (1, 0)

    async def test_the_reply_is_read_with_get_after_the_task_finishes(
        self,
        client: AsyncTestClient[Litestar],
        world: ChatWorld,
        recorder: RecordingSupervisor,
        actor: Actor,
    ) -> None:
        chat_id = await _new_chat(client, actor)
        world.chat_llm.add(reply(LONG))
        artist = (await _send(client, actor, chat_id)).json()["artist_message"]

        world.supervisor.start_reply(recorder.replies[0])
        await world.settle()

        got = await client.get(f"{S}/chat-messages/{artist['id']}", headers=actor.headers)
        assert got.status_code == 200
        assert got.json()["status"] == "complete"
        assert got.json()["content"] == LONG
        assert got.json()["revision"] == 2
        page = await client.get(f"{S}/chats/{chat_id}/messages", headers=actor.headers)
        assert [m["position"] for m in page.json()["items"]] == [2, 1]
        assert (await client.get(f"{S}/chats/{chat_id}", headers=actor.headers)).json()[
            "message_count"
        ] == 2

    async def test_a_resent_message_is_202_with_the_same_pair_and_no_second_job(
        self,
        client: AsyncTestClient[Litestar],
        recorder: RecordingSupervisor,
        actor: Actor,
        world: ChatWorld,
    ) -> None:
        chat_id = await _new_chat(client, actor)
        cid = str(uuid4())
        first = await _send(client, actor, chat_id, "Hello", cid)
        again = await _send(client, actor, chat_id, "Hello", cid)

        assert (first.status_code, again.status_code) == (202, 202)
        assert again.json()["artist_message"]["id"] == first.json()["artist_message"]["id"]
        assert len(recorder.replies) == 1
        assert await world.refunds_and_debits(actor) == (1, 0)

        conflict = await _send(client, actor, chat_id, "Something else", cid)
        assert conflict.status_code == 409
        assert conflict.json()["error"] == "idempotency_conflict"

    async def test_409_while_a_reply_is_pending(
        self, client: AsyncTestClient[Litestar], actor: Actor
    ) -> None:
        chat_id = await _new_chat(client, actor)
        await _send(client, actor, chat_id)
        response = await _send(client, actor, chat_id, "and another")
        assert response.status_code == 409
        assert response.json()["error"] == "chat_reply_in_progress"

    async def test_402_leaves_nothing_behind(
        self, client: AsyncTestClient[Litestar], world: ChatWorld, recorder: RecordingSupervisor
    ) -> None:
        poor = await world.add_actor(balance=0, artist=True)
        chat_id = await _new_chat(client, poor)
        response = await _send(client, poor, chat_id)
        assert response.status_code == 402
        assert await world.messages(UUID(chat_id)) == []
        assert recorder.replies == []

    async def test_429_over_the_pending_limit(
        self, client: AsyncTestClient[Litestar], actor: Actor
    ) -> None:
        chats = [await _new_chat(client, actor, f"c{i}") for i in range(4)]
        for chat_id in chats[:3]:
            assert (await _send(client, actor, chat_id)).status_code == 202
        response = await _send(client, actor, chats[3])
        assert response.status_code == 429
        assert response.json()["error"] == "chat_rate_limited"

    @pytest.mark.parametrize(
        "body",
        [
            {},
            {"content": "hi"},
            {"client_message_id": "not-a-uuid", "content": "hi"},
            {"client_message_id": str(uuid4()), "content": ""},
            {"client_message_id": str(uuid4()), "content": "x" * 4_001},
            {"client_message_id": str(uuid4()), "content": "hi", "extra": 1},
        ],
    )
    async def test_a_bad_body_is_a_client_error_and_charges_nothing(
        self,
        client: AsyncTestClient[Litestar],
        actor: Actor,
        world: ChatWorld,
        body: dict[str, Any],
    ) -> None:
        chat_id = await _new_chat(client, actor)
        response = await client.post(
            f"{S}/chats/{chat_id}/messages", headers=actor.headers, json=body
        )
        assert response.status_code in {400, 422}
        assert await world.refunds_and_debits(actor) == (0, 0)

    async def test_a_message_of_exactly_four_thousand_characters_is_accepted(
        self, client: AsyncTestClient[Litestar], actor: Actor
    ) -> None:
        chat_id = await _new_chat(client, actor)
        assert (await _send(client, actor, chat_id, "x" * 4_000)).status_code == 202

    async def test_a_blank_message_is_document_invalid(
        self, client: AsyncTestClient[Litestar], actor: Actor
    ) -> None:
        chat_id = await _new_chat(client, actor)
        response = await _send(client, actor, chat_id, "   \n  ")
        assert response.status_code == 422
        assert response.json()["detail"]["issues"] == [
            {"path": "content", "code": "empty_after_normalization"}
        ]

    async def test_deleting_a_chat_with_a_pending_reply_is_409(
        self, client: AsyncTestClient[Litestar], actor: Actor
    ) -> None:
        chat_id = await _new_chat(client, actor)
        await _send(client, actor, chat_id)
        response = await client.delete(f"{S}/chats/{chat_id}", headers=actor.headers)
        assert response.status_code == 409
        assert response.json()["error"] == "chat_busy"

    async def test_deleting_an_artist_with_a_pending_reply_is_409_artist_busy(
        self, client: AsyncTestClient[Litestar], world: ChatWorld, actor: Actor
    ) -> None:
        """An artist no project uses, with a reply in flight: refused, nothing deleted or lost."""
        created = await client.post(f"{S}/artists", headers=actor.headers, json={"name": "Spare"})
        artist_id = created.json()["id"]
        chat = await client.post(
            f"{S}/artists/{artist_id}/chats", headers=actor.headers, json={"title": "Interview"}
        )
        chat_id = chat.json()["id"]
        assert (await _send(client, actor, chat_id)).status_code == 202  # pending, task not run

        refused = await client.delete(f"{S}/artists/{artist_id}", headers=actor.headers)
        assert refused.status_code == 409
        assert refused.json()["error"] == "artist_busy"
        assert (await client.get(f"{S}/chats/{chat_id}", headers=actor.headers)).status_code == 200
        assert await world.refunds_and_debits(actor) == (1, 0)  # the reservation is untouched


class TestMessages:
    async def test_the_page_limits_and_the_before_keyset(
        self,
        client: AsyncTestClient[Litestar],
        world: ChatWorld,
        actor: Actor,
        recorder: RecordingSupervisor,
    ) -> None:
        chat_id = await _new_chat(client, actor)
        world.chat_llm.add(reply(LONG), reply(LONG))
        for text_ in ("one", "two"):
            await _send(client, actor, chat_id, text_)
            world.supervisor.start_reply(recorder.replies[-1])
            await world.settle()

        page = await client.get(f"{S}/chats/{chat_id}/messages?limit=3", headers=actor.headers)
        assert [m["position"] for m in page.json()["items"]] == [4, 3, 2]
        assert page.json()["has_more"] is True
        rest = await client.get(
            f"{S}/chats/{chat_id}/messages?limit=3&before={page.json()['next_cursor']}",
            headers=actor.headers,
        )
        assert [m["position"] for m in rest.json()["items"]] == [1]
        assert rest.json()["has_more"] is False
        for bad in ("limit=0", "limit=51", "before=0", "before=abc"):
            response = await client.get(
                f"{S}/chats/{chat_id}/messages?{bad}", headers=actor.headers
            )
            assert response.status_code == 400, bad

    async def test_404_for_another_users_message(
        self, client: AsyncTestClient[Litestar], world: ChatWorld, actor: Actor
    ) -> None:
        chat_id = await _new_chat(client, actor)
        artist = (await _send(client, actor, chat_id)).json()["artist_message"]
        stranger = await world.add_actor(balance=1, artist=True)
        response = await client.get(f"{S}/chat-messages/{artist['id']}", headers=stranger.headers)
        assert response.status_code == 404
        assert response.json()["error"] == "message_not_found"


class TestMemoryDrafts:
    async def _finished_reply(
        self,
        client: AsyncTestClient[Litestar],
        world: ChatWorld,
        recorder: RecordingSupervisor,
        actor: Actor,
    ) -> str:
        chat_id = await _new_chat(client, actor)
        world.chat_llm.add(reply(LONG))
        artist = (await _send(client, actor, chat_id)).json()["artist_message"]
        world.supervisor.start_reply(recorder.replies[-1])
        await world.settle()
        return str(artist["id"])

    async def test_202_then_read_then_saved_as_a_memory_with_origin_chat(
        self,
        client: AsyncTestClient[Litestar],
        world: ChatWorld,
        recorder: RecordingSupervisor,
        actor: Actor,
    ) -> None:
        message_id = await self._finished_reply(client, world, recorder, actor)
        world.draft_llm = FakeLlm(
            MemoryDraftDraft(
                refusal=None,
                title="The rooftop",
                body="I learned to sing on a roof.",
                themes=["home"],
                imagery=["neon sign"],
            )
        )
        response = await client.post(
            f"{S}/chat-messages/{message_id}/memory-draft",
            headers={**actor.headers, "Idempotency-Key": "draft-1"},
        )
        assert response.status_code == 202, response.text
        draft = response.json()
        assert (draft["status"], draft["draft"], draft["revision"]) == ("pending", None, 1)
        assert draft["source_message_id"] == message_id
        assert set(draft) == {
            "id",
            "chat_id",
            "source_message_id",
            "status",
            "draft",
            "error_code",
            "revision",
            "created_at",
            "finished_at",
        }

        world.supervisor.start_memory_draft(recorder.drafts[0])
        await world.settle()
        done = await client.get(f"{S}/memory-drafts/{draft['id']}", headers=actor.headers)
        assert done.status_code == 200
        assert done.json()["status"] == "complete"
        assert done.json()["draft"] == {
            "title": "The rooftop",
            "body": "I learned to sing on a roof.",
            "themes": ["home"],
            "imagery": ["neon sign"],
        }

        saved = await client.post(
            f"{S}/artists/{actor.artist_id}/memories",
            headers=actor.headers,
            json={
                "title": "The rooftop, edited",
                "body": "I learned to sing on a roof.",
                "from_draft_id": draft["id"],
            },
        )
        assert saved.status_code == 201, saved.text
        assert saved.json()["origin"] == "chat"
        assert saved.json()["title"] == "The rooftop, edited"

        plain = await client.post(
            f"{S}/artists/{actor.artist_id}/memories",
            headers=actor.headers,
            json={"title": "Typed by hand", "body": "b"},
        )
        assert plain.json()["origin"] == "user"

    async def test_a_client_cannot_set_the_origin_directly(
        self, client: AsyncTestClient[Litestar], actor: Actor
    ) -> None:
        response = await client.post(
            f"{S}/artists/{actor.artist_id}/memories",
            headers=actor.headers,
            json={"title": "t", "body": "b", "origin": "chat"},
        )
        assert response.status_code in {400, 422}

    async def test_a_bad_from_draft_id_is_a_422_naming_the_field(
        self, client: AsyncTestClient[Litestar], actor: Actor
    ) -> None:
        response = await client.post(
            f"{S}/artists/{actor.artist_id}/memories",
            headers=actor.headers,
            json={"title": "t", "body": "b", "from_draft_id": str(uuid4())},
        )
        assert response.status_code == 422
        assert response.json()["detail"]["issues"] == [
            {"path": "from_draft_id", "code": "draft_not_found"}
        ]

    @pytest.mark.parametrize("key", [None, "", "k" * 129])
    async def test_400_without_a_usable_key(
        self,
        client: AsyncTestClient[Litestar],
        world: ChatWorld,
        recorder: RecordingSupervisor,
        actor: Actor,
        key: str | None,
    ) -> None:
        message_id = await self._finished_reply(client, world, recorder, actor)
        headers = dict(actor.headers)
        if key is not None:
            headers["Idempotency-Key"] = key
        response = await client.post(
            f"{S}/chat-messages/{message_id}/memory-draft", headers=headers
        )
        assert response.status_code == 400
        assert response.json()["error"] == "idempotency_key_required"

    async def test_422_for_a_message_that_is_not_complete_and_404_for_a_stranger(
        self, client: AsyncTestClient[Litestar], world: ChatWorld, actor: Actor
    ) -> None:
        chat_id = await _new_chat(client, actor)
        pending = (await _send(client, actor, chat_id)).json()["artist_message"]
        headers = {**actor.headers, "Idempotency-Key": "k"}
        response = await client.post(
            f"{S}/chat-messages/{pending['id']}/memory-draft", headers=headers
        )
        assert response.status_code == 422
        assert response.json()["error"] == "message_not_complete"

        stranger = await world.add_actor(balance=5, artist=True)
        response = await client.post(
            f"{S}/chat-messages/{pending['id']}/memory-draft",
            headers={**stranger.headers, "Idempotency-Key": "k2"},
        )
        assert response.status_code == 404
        assert (
            await client.get(f"{S}/memory-drafts/{uuid4()}", headers=actor.headers)
        ).status_code == 404


class TestTheSwitch:
    async def test_every_chat_route_is_404_when_the_chat_is_off(
        self, client: AsyncTestClient[Litestar], actor: Actor, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        chat_id = await _new_chat(client, actor)
        monkeypatch.setattr(
            "src.api.security.studio.get_settings",
            lambda: SimpleNamespace(studio_enabled=True, studio_chat_enabled=False),
        )
        for method, path in (
            ("GET", f"{S}/artists/{actor.artist_id}/chats"),
            ("POST", f"{S}/artists/{actor.artist_id}/chats"),
            ("GET", f"{S}/chats/{chat_id}"),
            ("POST", f"{S}/chats/{chat_id}/messages"),
            ("GET", f"{S}/chat-messages/{uuid4()}"),
            ("POST", f"{S}/chat-messages/{uuid4()}/memory-draft"),
            ("GET", f"{S}/memory-drafts/{uuid4()}"),
        ):
            response = await client.request(method, path, headers=actor.headers)
            assert response.status_code == 404, (method, path)

        # the rest of the studio is not affected
        memories = await client.get(
            f"{S}/artists/{actor.artist_id}/memories", headers=actor.headers
        )
        assert memories.status_code == 200
