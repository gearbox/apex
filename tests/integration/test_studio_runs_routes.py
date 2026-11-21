"""Studio planner runs over HTTP: real controllers, guards, DI, handlers and PostgreSQL.

Same shape as ``test_studio_routes``: committed rows, a session per request (``AsyncTestClient``
runs handlers on another event loop), a commit on success and a rollback when the handler raises.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, Any
from uuid import uuid4

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
    get_studio_track_service,
    get_studio_version_service,
)
from src.api.routes.studio_runs import StudioRunController
from src.api.routes.studio_tracks import StudioTrackController
from src.api.security.jwt import JWTConfig, JWTService
from src.api.services.billing import BillingService
from src.api.services.billing_errors import InsufficientBalanceError
from src.api.services.legal.errors import LegalAcceptanceRequiredError
from src.api.services.token_revocation import TokenRevocationService
from src.core.product_registry import VEX_CONFIG
from src.studio.errors import StudioError
from src.studio.services.run import StudioRunService
from src.studio.services.run_events import StudioRunEvents
from tests.legal_support import TEST_LEGAL_REGISTRY
from tests.llm_support import FakeLlm
from tests.studio_llm import make_draft
from tests.studio_runs_support import SECRET, RunWorld

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Callable

    from litestar.types import ASGIApp, Receive, Scope, Send

    from tests.studio_runs_support import Actor

S = "/v1/studio"
KEY = "Idempotency-Key"


def _scope(app: ASGIApp) -> ASGIApp:
    async def middleware(scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] in ("http", "websocket"):
            state = scope.setdefault("state", {})
            state["product_config"] = VEX_CONFIG
            state["product_id"] = VEX_CONFIG.slug
        await app(scope, receive, send)

    return middleware


def _build_app(world: RunWorld) -> Litestar:
    async def provide_session() -> AsyncGenerator[AsyncSession]:
        async with world.factory() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise

    def provide_run_service(session: AsyncSession) -> StudioRunService:
        return StudioRunService(
            session,
            billing=BillingService(),
            settings=world.settings,
            events=StudioRunEvents(world.bus),  # type: ignore[arg-type]
        )

    app = Litestar(
        route_handlers=[StudioRunController, StudioTrackController],
        dependencies={
            "session": Provide(provide_session),
            "product_id": Provide(get_product_id, sync_to_thread=False),
            "studio_run_service": Provide(provide_run_service, sync_to_thread=False),
            "studio_track_service": Provide(get_studio_track_service, sync_to_thread=False),
            "studio_version_service": Provide(get_studio_version_service, sync_to_thread=False),
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
def runs_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "src.api.security.studio.get_settings",
        lambda: SimpleNamespace(studio_enabled=True, studio_runs_enabled=True),
    )


@pytest_asyncio.fixture
async def world(run_world: RunWorld) -> RunWorld:
    return run_world


@pytest_asyncio.fixture
async def client(world: RunWorld) -> AsyncGenerator[AsyncTestClient[Litestar]]:
    async with AsyncTestClient(app=_build_app(world)) as test_client:
        yield test_client


def _body(**extra: Any) -> dict[str, Any]:
    return {"kind": "plan", "brief": "a song about rain", **extra}


async def _post(
    client: AsyncTestClient[Litestar],
    actor: Actor,
    track_id: Any,
    *,
    key: str | None = "k-1",
    body: dict[str, Any] | None = None,
) -> Any:
    headers = dict(actor.headers)
    if key is not None:
        headers[KEY] = key
    return await client.post(f"{S}/tracks/{track_id}/runs", headers=headers, json=body or _body())


class TestSubmit:
    async def test_202_with_the_queued_run(
        self, client: AsyncTestClient[Litestar], world: RunWorld
    ) -> None:
        actor = await world.add_actor()
        track_id = await world.new_track(actor)

        response = await _post(client, actor, track_id)

        assert response.status_code == 202, response.text
        body = response.json()
        assert body["track_id"] == str(track_id)
        assert (body["kind"], body["status"], body["stage"], body["revision"]) == (
            "plan",
            "queued",
            None,
            1,
        )
        assert (body["price_tokens"], body["model_key"], body["llm_calls"]) == (1, "grok", 0)
        assert body["warnings"] == []
        assert body["memory_ids_used"] == []  # the selection is made when a worker starts the run
        assert body["result_plan_version_id"] is None
        assert body["error_code"] is None
        assert body["finished_at"] is None
        # a client never sees the request text, the snapshot, token counts or the lease
        assert set(body) == {
            "id",
            "track_id",
            "kind",
            "status",
            "stage",
            "revision",
            "price_tokens",
            "model_key",
            "result_plan_version_id",
            "result_lyrics_version_id",
            "warnings",
            "memory_ids_used",
            "error_code",
            "error_detail",
            "llm_calls",
            "created_at",
            "started_at",
            "finished_at",
        }
        assert "rain" not in response.text
        assert [e.status for e in world.bus.run_events] == ["queued"]

    async def test_200_when_the_key_replays_the_same_body(
        self, client: AsyncTestClient[Litestar], world: RunWorld
    ) -> None:
        actor = await world.add_actor()
        track_id = await world.new_track(actor)
        first = await _post(client, actor, track_id)
        second = await _post(client, actor, track_id)

        assert (first.status_code, second.status_code) == (202, 200)
        assert second.json()["id"] == first.json()["id"]
        assert await world.refunds_and_debits(actor) == (1, 0)

    async def test_409_when_the_key_is_reused_with_another_body(
        self, client: AsyncTestClient[Litestar], world: RunWorld
    ) -> None:
        actor = await world.add_actor()
        track_id = await world.new_track(actor)
        await _post(client, actor, track_id)
        response = await _post(client, actor, track_id, body=_body(brief="something else"))
        assert response.status_code == 409
        assert response.json()["error"] == "idempotency_conflict"

    @pytest.mark.parametrize("key", [None, "", "   ", "k" * 129])
    async def test_400_without_a_usable_idempotency_key(
        self, client: AsyncTestClient[Litestar], world: RunWorld, key: str | None
    ) -> None:
        actor = await world.add_actor()
        track_id = await world.new_track(actor)
        response = await _post(client, actor, track_id, key=key)
        assert response.status_code == 400
        assert response.json()["error"] == "idempotency_key_required"
        assert await world.count_runs(actor) == 0

    async def test_402_when_the_balance_is_too_low_and_nothing_is_left_behind(
        self, client: AsyncTestClient[Litestar], world: RunWorld
    ) -> None:
        actor = await world.add_actor(balance=0)
        track_id = await world.new_track(actor)
        response = await _post(client, actor, track_id)
        assert response.status_code == 402
        assert await world.count_runs(actor) == 0
        assert await world.ledger(actor) == []

    async def test_404_for_another_users_track(
        self, client: AsyncTestClient[Litestar], world: RunWorld
    ) -> None:
        owner = await world.add_actor()
        intruder = await world.add_actor()
        track_id = await world.new_track(owner)
        response = await _post(client, intruder, track_id)
        assert response.status_code == 404
        assert await world.count_runs(owner) == 0

    async def test_404_for_an_unknown_track(
        self, client: AsyncTestClient[Litestar], world: RunWorld
    ) -> None:
        actor = await world.add_actor()
        assert (await _post(client, actor, uuid4())).status_code == 404

    async def test_401_without_a_token(
        self, client: AsyncTestClient[Litestar], world: RunWorld
    ) -> None:
        actor = await world.add_actor()
        track_id = await world.new_track(actor)
        response = await client.post(
            f"{S}/tracks/{track_id}/runs", headers={KEY: "k"}, json=_body()
        )
        assert response.status_code == 401

    async def test_409_run_in_progress(
        self, client: AsyncTestClient[Litestar], world: RunWorld
    ) -> None:
        actor = await world.add_actor()
        track_id = await world.new_track(actor)
        await _post(client, actor, track_id, key="a")
        response = await _post(client, actor, track_id, key="b")
        assert response.status_code == 409
        assert response.json()["error"] == "run_in_progress"
        assert await world.refunds_and_debits(actor) == (1, 0)

    async def test_409_run_limit_reached(
        self, client: AsyncTestClient[Litestar], world: RunWorld
    ) -> None:
        actor = await world.add_actor(balance=20)
        tracks = [await world.new_track(actor, f"T{i}") for i in range(4)]
        for index, track_id in enumerate(tracks[:3]):
            assert (await _post(client, actor, track_id, key=f"k{index}")).status_code == 202
        response = await _post(client, actor, tracks[3], key="k3")
        assert response.status_code == 409
        assert response.json()["error"] == "run_limit_reached"

    async def test_429_run_rate_limited(self, make_run_world: Callable[..., RunWorld]) -> None:
        world = make_run_world(studio_runs_per_user_per_hour=1)
        actor = await world.add_actor(balance=20)
        first = await world.new_track(actor, "A")
        second = await world.new_track(actor, "B")
        async with AsyncTestClient(app=_build_app(world)) as client:
            assert (await _post(client, actor, first, key="a")).status_code == 202
            response = await _post(client, actor, second, key="b")
        assert response.status_code == 429
        assert response.json()["error"] == "run_rate_limited"

    async def test_422_for_a_pin_when_the_project_has_no_artist(
        self, client: AsyncTestClient[Litestar], world: RunWorld
    ) -> None:
        actor = await world.add_actor()
        track_id = await world.new_track(actor)
        response = await _post(client, actor, track_id, body=_body(memory_ids=[str(uuid4())]))
        assert response.status_code == 422
        body = response.json()
        assert body["error"] == "document_invalid"
        assert body["detail"]["issues"] == [{"path": "memory_ids", "code": "no_artist"}]
        assert await world.count_runs(actor) == 0

    async def test_422_for_a_blank_brief(
        self, client: AsyncTestClient[Litestar], world: RunWorld
    ) -> None:
        actor = await world.add_actor()
        track_id = await world.new_track(actor)
        response = await _post(client, actor, track_id, body=_body(brief="  \n "))
        assert response.status_code == 422
        assert response.json()["error"] == "document_invalid"
        assert response.json()["detail"]["issues"] == [
            {"path": "brief", "code": "empty_after_normalization"}
        ]

    @pytest.mark.parametrize(
        "body",
        [
            {"kind": "plan"},
            {"kind": "plan", "brief": ""},
            {"kind": "plan", "brief": "x", "model": "pick-my-own"},
            {"kind": "plan", "brief": "x" * 4001},
            {"kind": "nope", "brief": "x"},
            {"brief": "x"},
        ],
    )
    async def test_400_for_a_malformed_body(
        self, client: AsyncTestClient[Litestar], world: RunWorld, body: dict[str, Any]
    ) -> None:
        actor = await world.add_actor()
        track_id = await world.new_track(actor)
        response = await _post(client, actor, track_id, body=body)
        assert response.status_code == 400
        assert await world.count_runs(actor) == 0


class TestRead:
    async def test_get_a_run_and_never_someone_elses(
        self, client: AsyncTestClient[Litestar], world: RunWorld
    ) -> None:
        owner = await world.add_actor()
        intruder = await world.add_actor()
        track_id = await world.new_track(owner)
        run_id = (await _post(client, owner, track_id)).json()["id"]

        mine = await client.get(f"{S}/runs/{run_id}", headers=owner.headers)
        theirs = await client.get(f"{S}/runs/{run_id}", headers=intruder.headers)
        unknown = await client.get(f"{S}/runs/{uuid4()}", headers=owner.headers)

        assert mine.status_code == 200
        assert mine.json()["id"] == run_id
        assert theirs.status_code == unknown.status_code == 404

    async def test_memories_are_pinned_over_http_and_reported_as_ids_only(
        self, client: AsyncTestClient[Litestar], world: RunWorld
    ) -> None:
        actor = await world.add_actor(artist=True)
        track_id = await world.new_track(actor)
        pin = await world.add_memory(actor, title="SECRET-MEMORY-TITLE", body="SECRET-MEMORY-BODY")
        other = await world.add_memory(actor, salience=5)

        refused = await _post(
            client, actor, track_id, key="bad", body=_body(memory_ids=[str(uuid4())])
        )
        assert (refused.status_code, refused.json()["error"]) == (422, "document_invalid")
        assert refused.json()["detail"]["issues"] == [
            {"path": "memory_ids[0]", "code": "memory_not_found"}
        ]
        assert await world.count_runs(actor) == 0  # nothing was written or charged

        queued = await _post(client, actor, track_id, body=_body(memory_ids=[str(pin)]))
        assert queued.status_code == 202
        assert queued.json()["memory_ids_used"] == []  # the selection happens when the run starts
        await world.drive(FakeLlm(make_draft()))

        done = await client.get(f"{S}/runs/{queued.json()['id']}", headers=actor.headers)
        assert done.json()["memory_ids_used"] == [str(pin), str(other)]  # pins first, then fill
        assert "SECRET-MEMORY" not in done.text  # ids only: the text stays in the memory API

    async def test_a_finished_run_reports_its_result_and_no_text(
        self, client: AsyncTestClient[Litestar], world: RunWorld
    ) -> None:
        actor = await world.add_actor()
        track_id = await world.new_track(actor)
        run_id = (await _post(client, actor, track_id)).json()["id"]
        await world.drive(
            FakeLlm(
                make_draft(tags=["120 BPM"], title="SECRET-TITLE"), make_draft(title="SECRET-TITLE")
            )
        )

        body = (await client.get(f"{S}/runs/{run_id}", headers=actor.headers)).json()

        assert (body["status"], body["stage"], body["error_code"]) == ("succeeded", None, None)
        assert body["llm_calls"] == 2
        assert body["finished_at"] is not None
        version = await client.get(
            f"{S}/versions/{body['result_plan_version_id']}", headers=actor.headers
        )
        assert version.status_code == 200
        assert version.json()["origin"] == "llm"
        assert version.json()["is_current"] is False
        assert "SECRET-TITLE" not in str(body)

    async def test_a_failed_run_reports_the_code_and_issue_paths(
        self, client: AsyncTestClient[Litestar], world: RunWorld
    ) -> None:
        actor = await world.add_actor()
        track_id = await world.new_track(actor)
        run_id = (await _post(client, actor, track_id)).json()["id"]
        bad = make_draft(tags=["120 BPM"], title="SECRET-TITLE")
        await world.drive(FakeLlm(bad, bad, bad))

        body = (await client.get(f"{S}/runs/{run_id}", headers=actor.headers)).json()

        assert (body["status"], body["error_code"]) == ("failed", "plan_invalid")
        assert body["error_detail"] == {
            "issues": [{"path": "global_style.tags[0]", "code": "reserved_tempo_tag"}]
        }
        assert "SECRET-TITLE" not in str(body)

    async def test_list_is_newest_first_and_paginated(
        self, client: AsyncTestClient[Litestar], world: RunWorld
    ) -> None:
        actor = await world.add_actor(balance=20)
        track_id = await world.new_track(actor)
        ids = []
        for index in range(3):
            ids.append((await _post(client, actor, track_id, key=f"k{index}")).json()["id"])
            await world.drive(FakeLlm(make_draft()))

        first = await client.get(f"{S}/tracks/{track_id}/runs?limit=2", headers=actor.headers)
        page = first.json()
        assert [item["id"] for item in page["items"]] == [ids[2], ids[1]]
        assert page["has_more"] is True

        second = await client.get(
            f"{S}/tracks/{track_id}/runs?limit=2&cursor={page['next_cursor']}",
            headers=actor.headers,
        )
        assert [item["id"] for item in second.json()["items"]] == [ids[0]]
        assert second.json()["has_more"] is False

    async def test_list_validates_its_inputs_and_scopes_the_track(
        self, client: AsyncTestClient[Litestar], world: RunWorld
    ) -> None:
        owner = await world.add_actor()
        intruder = await world.add_actor()
        track_id = await world.new_track(owner)
        assert (
            await client.get(f"{S}/tracks/{track_id}/runs?cursor=garbage", headers=owner.headers)
        ).status_code == 400
        assert (
            await client.get(f"{S}/tracks/{track_id}/runs?limit=0", headers=owner.headers)
        ).status_code == 400
        assert (
            await client.get(f"{S}/tracks/{track_id}/runs", headers=intruder.headers)
        ).status_code == 404


class TestCancel:
    async def test_cancel_a_queued_run(
        self, client: AsyncTestClient[Litestar], world: RunWorld
    ) -> None:
        actor = await world.add_actor()
        track_id = await world.new_track(actor)
        run_id = (await _post(client, actor, track_id)).json()["id"]

        response = await client.post(f"{S}/runs/{run_id}/cancel", headers=actor.headers)

        assert response.status_code == 200
        assert (response.json()["status"], response.json()["revision"]) == ("cancelled", 2)
        assert await world.balance(actor) == 10
        assert [e.status for e in world.bus.run_events] == ["queued", "cancelled"]

    async def test_409_for_a_running_or_finished_run(
        self, client: AsyncTestClient[Litestar], world: RunWorld
    ) -> None:
        actor = await world.add_actor(balance=20)
        running_track = await world.new_track(actor, "A")
        done_track = await world.new_track(actor, "B")
        running = (await _post(client, actor, running_track, key="a")).json()["id"]
        done = (await _post(client, actor, done_track, key="b")).json()["id"]
        executor = world.executor(FakeLlm(make_draft()))
        claimed = await executor.claim(2)
        await executor.execute(next(c for c in claimed if str(c.run_id) == done))

        for run_id in (running, done):
            response = await client.post(f"{S}/runs/{run_id}/cancel", headers=actor.headers)
            assert response.status_code == 409
            assert response.json()["error"] == "run_not_cancellable"

    async def test_404_for_another_users_run(
        self, client: AsyncTestClient[Litestar], world: RunWorld
    ) -> None:
        owner = await world.add_actor()
        intruder = await world.add_actor()
        track_id = await world.new_track(owner)
        run_id = (await _post(client, owner, track_id)).json()["id"]
        response = await client.post(f"{S}/runs/{run_id}/cancel", headers=intruder.headers)
        assert response.status_code == 404
        assert (await world.run(run_id)).status == "queued"  # type: ignore[arg-type]


class TestTrackDeletion:
    async def test_409_while_a_run_is_active_then_204_once_it_is_over(
        self, client: AsyncTestClient[Litestar], world: RunWorld
    ) -> None:
        actor = await world.add_actor()
        track_id = await world.new_track(actor)
        await _post(client, actor, track_id)

        refused = await client.delete(f"{S}/tracks/{track_id}", headers=actor.headers)
        assert refused.status_code == 409
        assert refused.json()["error"] == "run_in_progress"

        await world.drive(FakeLlm(make_draft()))
        allowed = await client.delete(f"{S}/tracks/{track_id}", headers=actor.headers)
        assert allowed.status_code == 204


class TestKillSwitch:
    async def test_every_route_is_404_when_runs_are_disabled(
        self,
        client: AsyncTestClient[Litestar],
        world: RunWorld,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        actor = await world.add_actor()
        track_id = await world.new_track(actor)
        run_id = (await _post(client, actor, track_id)).json()["id"]
        monkeypatch.setattr(
            "src.api.security.studio.get_settings",
            lambda: SimpleNamespace(studio_enabled=True, studio_runs_enabled=False),
        )

        responses = [
            await _post(client, actor, track_id, key="another"),
            await client.get(f"{S}/runs/{run_id}", headers=actor.headers),
            await client.get(f"{S}/tracks/{track_id}/runs", headers=actor.headers),
            await client.post(f"{S}/runs/{run_id}/cancel", headers=actor.headers),
        ]
        assert [r.status_code for r in responses] == [404, 404, 404, 404]
        # and the studio's own routes are unaffected
        assert (
            await client.get(f"{S}/tracks/{track_id}", headers=actor.headers)
        ).status_code == 200
