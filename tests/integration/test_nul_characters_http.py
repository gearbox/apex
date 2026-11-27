"""N2 end to end: the real app (``create_app``: every route, hook, guard and handler) over real
PostgreSQL. A request carrying U+0000 is ``400 invalid_character`` and **not one SQL statement**
reaches the database for it, so nothing can have been read, written or reserved.

The app runs in the test's event loop (``httpx.ASGITransport``), so it shares ``db_session``.
Only the request session, the JWT keys, the legal documents and the rate limiter's storage are
the test's own.
"""

from __future__ import annotations

import importlib
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Final
from uuid import uuid4

import httpx
import pytest
import pytest_asyncio
from limits.aio.storage import MemoryStorage
from litestar.di import Provide
from sqlalchemy import event, func, select
from sqlalchemy.ext.asyncio import AsyncSession  # noqa: TC002 - Litestar reads the provider's hints
from structlog.testing import capture_logs

from src.api.dependencies.common import dependencies
from src.api.schemas.studio import ArtistCreate, LyricsVersionCreate, PlanVersionCreate
from src.api.security.jwt import JWTConfig, JWTService
from src.api.services.token_revocation import TokenRevocationService
from src.core.enums import StudioSectionKind
from src.core.product_registry import VEX_CONFIG
from src.db.models.billing import TokenTransaction
from src.db.models.feedback import FeedbackReport
from src.db.models.idempotency import IdempotencyKey
from src.db.models.library import LibraryAssetMetadata
from src.db.models.storage import GenerationJob
from src.db.models.studio import StudioArtistMemory, StudioDocumentVersion
from tests.legal_support import make_legal_registry
from tests.studio_support import make_lyrics, make_plan, make_section, to_json
from tests.studio_world import StudioWorld

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, AsyncIterator
    from uuid import UUID

    from sqlalchemy.ext.asyncio import AsyncEngine

    from src.db.models.storage import UserImage

NUL: Final = "\x00"
MARKER: Final = "SECRET-MARKER-7f3a"
"""Other text of the offending field: it must be in no response and no log."""
TAINTED: Final = f"{MARKER}{NUL}tail"
MESSAGE: Final = "Text must not contain the NUL character (U+0000)."
_SECRET: Final = "integration-nul-secret-key-32-bytes-long"


@dataclass
class Api:
    client: httpx.AsyncClient
    engine: AsyncEngine
    session: AsyncSession
    world: StudioWorld
    headers: dict[str, str]
    sessions_opened: list[str]
    track_id: UUID
    plan_version_id: UUID
    section_id: UUID
    artist_id: UUID
    asset: UserImage
    account_id: UUID

    @asynccontextmanager
    async def statements(self) -> AsyncIterator[list[str]]:
        """Every SQL statement the engine executes inside the block."""
        executed: list[str] = []

        def record(_conn: Any, _cursor: Any, statement: str, *_rest: Any) -> None:
            executed.append(statement)

        event.listen(self.engine.sync_engine, "before_cursor_execute", record)
        try:
            yield executed
        finally:
            event.remove(self.engine.sync_engine, "before_cursor_execute", record)

    async def refused(self, method: str, url: str, location: str = "body", **kwargs: Any) -> None:
        """Send a request that must be refused, and check everything the refusal promises."""
        headers = {**self.headers, **kwargs.pop("headers", {})}
        self.sessions_opened.clear()
        with capture_logs() as logs:
            async with self.statements() as executed:
                response = await self.client.request(method, url, headers=headers, **kwargs)

        assert response.status_code == 400, response.text
        assert response.json() == {
            "error": "invalid_character",
            "message": MESSAGE,
            "status_code": 400,
            "detail": {"location": location},
        }
        assert executed == [], "a refused request must not reach the database"
        assert self.sessions_opened == []
        (refusal,) = [log for log in logs if log["event"] == "http.invalid_character"]
        assert set(refusal) == {"event", "log_level", "path", "method", "location"}
        assert (refusal["log_level"], refusal["method"], refusal["location"]) == (
            "warning",
            method,
            location,
        )
        assert MARKER not in response.text
        assert MARKER not in repr(logs)

    async def count(self, model: type[Any], *where: Any) -> int:
        query = select(func.count()).select_from(model).where(*where)
        return (await self.session.execute(query)).scalar_one()


@pytest_asyncio.fixture
async def api(
    db_engine: AsyncEngine,
    db_session: AsyncSession,
    make_user: Any,
    make_library_project: Any,
    make_user_image: Any,
    make_token_account: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> AsyncGenerator[Api]:
    user = await make_user(email=f"nul-{uuid4().hex[:10]}@example.com")
    project = await make_library_project(user=user)
    world = StudioWorld(session=db_session, user=user, library_project=project)
    await world.enable()
    track_id = await world.new_track()
    section = make_section(StudioSectionKind.VERSE)
    plan, _ = await world.plan(track_id, make_plan(section))
    artist = await world.artists.create(user.id, "vex", ArtistCreate(name=f"Nova {uuid4().hex}"))
    asset = await make_user_image(user=user)
    account = await make_token_account(user=user)
    db_session.add(
        TokenTransaction(
            id=uuid4(),
            account_id=account.id,
            transaction_type="credit",
            amount=100,
            balance_after=100,
            product_id="vex",
        )
    )
    await db_session.flush()

    sessions_opened: list[str] = []

    async def provide_session() -> AsyncGenerator[AsyncSession]:
        sessions_opened.append("session")
        yield db_session

    # The module, not the ``app`` instance that ``src.api`` re-exports under the same name.
    app_module = importlib.import_module("src.api.app")
    monkeypatch.setattr(app_module, "configure_logging", lambda _settings: None)
    monkeypatch.setattr(
        app_module, "dependencies", {**dependencies, "session": Provide(provide_session)}
    )
    monkeypatch.setattr(
        "src.api.security.studio.get_settings", lambda: SimpleNamespace(studio_enabled=True)
    )
    monkeypatch.setattr("src.api.middleware.rate_limit._limiter_storage", MemoryStorage())

    app = app_module.create_app()
    jwt = JWTService(JWTConfig(secret_key=_SECRET))
    registry = make_legal_registry()
    app.state["jwt_service"] = jwt
    app.state["token_revocation"] = TokenRevocationService(None, max_token_ttl_seconds=0)
    app.state["legal_registry"] = registry
    token, _ = jwt.create_access_token(
        user.id,
        product_id="vex",
        legal_digest=registry.required_digest(VEX_CONFIG, today=datetime.now(UTC).date()),
    )

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),  # type: ignore[arg-type]
        base_url="http://testserver.local",
    ) as client:
        yield Api(
            client=client,
            engine=db_engine,
            session=db_session,
            world=world,
            headers={"Authorization": f"Bearer {token}", "X-Product-Id": "vex"},
            sessions_opened=sessions_opened,
            track_id=track_id,
            plan_version_id=plan.version.id,
            section_id=section.id,
            artist_id=artist.id,
            asset=asset,
            account_id=account.id,
        )


def _lyrics_body(api: Api, line: str) -> dict[str, Any]:
    return to_json(
        LyricsVersionCreate(
            content=make_lyrics((api.section_id, [line])),
            based_on_plan_version_id=api.plan_version_id,
            parent_version_id=None,
            make_current=True,
        )
    )


def _versions_url(api: Api) -> str:
    return f"/v1/studio/tracks/{api.track_id}/versions"


# ---------------------------------------------------------------------------
# One route per area
# ---------------------------------------------------------------------------


async def test_a_studio_lyrics_version(api: Api) -> None:
    before = await api.world.version_count(api.track_id)

    await api.refused("POST", _versions_url(api), json=_lyrics_body(api, TAINTED))

    assert await api.world.version_count(api.track_id) == before


async def test_a_plan_section_label(api: Api) -> None:
    body = to_json(
        PlanVersionCreate(
            content=make_plan(make_section(StudioSectionKind.VERSE, label=TAINTED)),
            parent_version_id=api.plan_version_id,
            make_current=True,
        )
    )
    before = await api.count(StudioDocumentVersion, StudioDocumentVersion.track_id == api.track_id)

    await api.refused("POST", _versions_url(api), json=body)

    assert (
        await api.count(StudioDocumentVersion, StudioDocumentVersion.track_id == api.track_id)
        == before
    )


async def test_an_artist_memory(api: Api) -> None:
    await api.refused(
        "POST",
        f"/v1/studio/artists/{api.artist_id}/memories",
        json={"title": "The pier", "body": TAINTED},
    )

    assert await api.count(StudioArtistMemory, StudioArtistMemory.artist_id == api.artist_id) == 0


async def test_a_library_asset_title(api: Api) -> None:
    await api.refused(
        "PATCH", f"/v1/library/assets/upload:{api.asset.id}", json={"display_title": TAINTED}
    )

    assert await api.count(LibraryAssetMetadata, LibraryAssetMetadata.asset_id == api.asset.id) == 0


async def test_a_generation_prompt_reserves_no_tokens(api: Api) -> None:
    await api.refused(
        "POST",
        "/v1/generate/",
        headers={"Idempotency-Key": f"nul-{uuid4().hex}"},
        json={"prompt": TAINTED, "generation_type": "t2i", "model": "grok-imagine-image"},
    )

    user_id = api.world.user_id
    ledger = TokenTransaction.account_id == api.account_id
    assert await api.count(TokenTransaction, ledger) == 1, "only the seeded credit"
    assert await api.count(GenerationJob, GenerationJob.user_id == user_id) == 0
    assert await api.count(IdempotencyKey, IdempotencyKey.user_id == user_id) == 0


async def test_a_feedback_report(api: Api) -> None:
    await api.refused(
        "POST", "/v1/feedback", json={"category": "bug", "message": f"It broke: {TAINTED}"}
    )

    assert await api.count(FeedbackReport, FeedbackReport.user_id == api.world.user_id) == 0


async def test_an_unguarded_auth_route(api: Api) -> None:
    await api.refused(
        "POST", "/v1/auth/login", json={"email": "a@example.com", "password": TAINTED}
    )


async def test_a_query_parameter(api: Api) -> None:
    await api.refused("GET", f"/v1/library/?query={MARKER}%00", location="query")


async def test_a_path_parameter(api: Api) -> None:
    """No marker here: the log names the request path, which is where this NUL is."""
    await api.refused("GET", "/v1/library/assets/upload%3A%00", location="path")


@pytest.mark.parametrize("content_type", ["text/plain", None])
async def test_leaving_out_the_json_content_type_changes_nothing(
    api: Api, content_type: str | None
) -> None:
    encoded = httpx.Request("POST", "http://x", json=_lyrics_body(api, TAINTED)).read()
    headers = {"content-type": content_type} if content_type else {}

    await api.refused("POST", _versions_url(api), content=encoded, headers=headers)


# ---------------------------------------------------------------------------
# What is not refused
# ---------------------------------------------------------------------------


async def test_a_literal_backslash_u0000_is_stored_as_written(api: Api) -> None:
    """Six ordinary characters, which JSON writes with an escaped backslash. This is also the
    control for the refusals above: the same route does reach the database."""
    line = "type \\u0000 to see it"
    assert NUL not in line

    api.sessions_opened.clear()
    async with api.statements() as executed:
        response = await api.client.post(
            _versions_url(api), headers=api.headers, json=_lyrics_body(api, line)
        )

    assert response.status_code == 201, response.text
    assert executed
    assert api.sessions_opened == ["session"]
    row = await api.world.version_row(response.json()["version"]["id"])
    assert row.content["sections"][0]["lines"][0]["text"] == line


async def test_the_guards_still_answer_first(api: Api) -> None:
    response = await api.client.post(
        _versions_url(api), headers={"X-Product-Id": "vex"}, json=_lyrics_body(api, TAINTED)
    )

    assert response.status_code == 401, response.text
