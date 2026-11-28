"""Realized timelines over HTTP: real controllers, guards, DI, handlers and PostgreSQL (S3a W7).

Rows are committed and the app opens its own session per request (see ``build_app``): a 409 leaves
nothing behind, a 428 never reaches the service, the studio kill switch answers 404 first.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any
from uuid import uuid4

import pytest
import pytest_asyncio
from litestar.testing import AsyncTestClient
from sqlalchemy.ext.asyncio import async_sessionmaker
from structlog.testing import capture_logs

from src.api.routes.studio_timelines import StudioTimelineController
from src.api.routes.studio_track_audio import StudioTrackAudioController
from src.core.product_registry import VEX_CONFIG
from tests.library_holds_support import HoldWorld, bearer, build_app
from tests.studio_support import to_json
from tests.timeline_support import MARK, assert_timelines_consistent, committed_song
from tests.track_audio_support import assert_audio_reconciled

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator
    from uuid import UUID

    from httpx import Response
    from litestar import Litestar
    from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession

    from src.studio.timeline import TimelineDocument
    from tests.timeline_support import Song


@pytest.fixture(autouse=True)
def studio_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "src.api.security.studio.get_settings", lambda: SimpleNamespace(studio_enabled=True)
    )


@pytest.fixture
def factory(db_engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(bind=db_engine, expire_on_commit=False)


@pytest_asyncio.fixture
async def client(
    factory: async_sessionmaker[AsyncSession],
) -> AsyncGenerator[AsyncTestClient[Litestar]]:
    app = build_app(factory, [StudioTimelineController, StudioTrackAudioController])
    async with AsyncTestClient(app=app) as test_client:
        yield test_client


@pytest_asyncio.fixture
async def checked(hold_world: HoldWorld) -> AsyncGenerator[HoldWorld]:
    """The hold world; every test ends with the timeline and audio consistency checks."""
    yield hold_world
    async with hold_world.factory() as session:
        await assert_timelines_consistent(session)
        await assert_audio_reconciled(session)


@pytest_asyncio.fixture
async def song(checked: HoldWorld) -> Song:
    return await committed_song(checked, await checked.user())


async def _create(
    client: AsyncTestClient[Litestar],
    song: Song,
    document: TimelineDocument,
    **body: Any,
) -> Response:
    return await client.post(
        f"/v1/studio/audio/{song.audio_id}/timelines",
        headers=bearer(song.user.id),
        json={"content": to_json(document), **body},
    )


async def _created(
    client: AsyncTestClient[Litestar], song: Song, document: TimelineDocument, **body: Any
) -> dict[str, Any]:
    response = await _create(client, song, document, **body)
    assert response.status_code == 201, response.text
    return response.json()  # type: ignore[no-any-return]


class TestLifecycleOverHttp:
    async def test_create_replay_list_read_switch_export_and_delete(
        self, client: AsyncTestClient[Litestar], song: Song
    ) -> None:
        headers = bearer(song.user.id)

        first = await _created(client, song, song.complete())
        replay = await _create(client, song, song.complete())

        assert replay.status_code == 200
        assert replay.json()["id"] == first["id"]
        assert set(first) == {
            "id",
            "audio_version_id",
            "track_id",
            "version_number",
            "parent_id",
            "origin",
            "is_complete",
            "is_current",
            "content",
            "created_at",
        }
        assert (first["origin"], first["version_number"], first["parent_id"]) == ("user", 1, None)
        assert (first["is_complete"], first["is_current"]) == (True, False)
        assert first["content"] == to_json(song.complete())
        # defaults are omitted on the wire, exactly as stored
        assert "words" not in first["content"]["lines"][1]
        assert "inferred" not in first["content"]["lines"][0]["words"][0]

        child = await _created(
            client, song, song.sections_only(), parent_id=first["id"], make_current=True
        )
        assert "lines" not in child["content"]
        assert (child["is_current"], child["is_complete"]) == (True, False)

        listed = await client.get(
            f"/v1/studio/audio/{song.audio_id}/timelines?limit=1", headers=headers
        )
        assert listed.status_code == 200
        page = listed.json()
        assert [item["id"] for item in page["items"]] == [child["id"]]
        assert page["has_more"] is True
        (item,) = page["items"]
        assert set(item) == {
            "id",
            "audio_version_id",
            "track_id",
            "version_number",
            "parent_id",
            "origin",
            "is_complete",
            "is_current",
            "created_at",
        }
        assert item == {key: value for key, value in child.items() if key != "content"}
        older = await client.get(
            f"/v1/studio/audio/{song.audio_id}/timelines",
            headers=headers,
            params={"limit": 1, "cursor": page["next_cursor"]},
        )
        assert older.json()["items"] == [
            {key: value for key, value in first.items() if key != "content"} | {"is_current": False}
        ]

        read = await client.get(f"/v1/studio/timelines/{first['id']}", headers=headers)
        assert read.status_code == 200
        assert read.json() == {**first, "is_current": False}

        switched = await client.put(
            f"/v1/studio/audio/{song.audio_id}/current-timeline",
            headers=headers,
            json={"timeline_id": first["id"]},
        )
        assert switched.status_code == 200
        audio = switched.json()
        assert (audio["current_timeline_id"], audio["timeline_count"]) == (first["id"], 2)
        assert audio["timeline_complete"] is True
        fetched = await client.get(f"/v1/studio/audio/{song.audio_id}", headers=headers)
        assert fetched.json() == audio

        cleared = await client.put(
            f"/v1/studio/audio/{song.audio_id}/current-timeline",
            headers=headers,
            json={"timeline_id": None},
        )
        assert cleared.json()["current_timeline_id"] is None
        assert cleared.json()["timeline_complete"] is None

        assert (
            await client.delete(f"/v1/studio/timelines/{child['id']}", headers=headers)
        ).status_code == 204
        gone = await client.get(f"/v1/studio/timelines/{child['id']}", headers=headers)
        assert (gone.status_code, gone.json()["error"]) == (404, "timeline_not_found")


class TestExportsOverHttp:
    async def test_lrc_and_srt_come_in_the_json_envelope(
        self, client: AsyncTestClient[Litestar], song: Song
    ) -> None:
        timeline = await _created(client, song, song.complete())
        headers = bearer(song.user.id)

        lrc = await client.get(
            f"/v1/studio/timelines/{timeline['id']}/exports/lrc", headers=headers
        )
        srt = await client.get(
            f"/v1/studio/timelines/{timeline['id']}/exports/srt", headers=headers
        )

        assert (lrc.status_code, srt.status_code) == (200, 200)
        assert lrc.json() == {
            "timeline_id": timeline["id"],
            "format": "lrc",
            "text": (
                f"[00:01.00]<00:01.00>{MARK}-w1 <00:02.00>{MARK}-w2\n"
                f"[00:05.00]{MARK}-lyric verse two\n"
                f"[00:21.00]{MARK}-lyric hook one\n"
                f"[00:25.00]{MARK}-lyric hook two\n"
            ),
            "warnings": [{"code": "inferred_word_timings", "count": 1}],
        }
        body = srt.json()
        assert (body["format"], body["warnings"]) == ("srt", [])
        assert body["text"].startswith(
            f"1\n00:00:01,000 --> 00:00:05,000\n{MARK}-lyric verse one\n\n"
        )
        assert body["text"].endswith(f"4\n00:00:25,000 --> 00:00:30,000\n{MARK}-lyric hook two\n\n")

    async def test_untimed_lines_are_reported(
        self, client: AsyncTestClient[Litestar], song: Song
    ) -> None:
        timeline = await _created(client, song, song.sections_only())

        response = await client.get(
            f"/v1/studio/timelines/{timeline['id']}/exports/lrc", headers=bearer(song.user.id)
        )

        assert response.json()["text"] == ""
        assert response.json()["warnings"] == [{"code": "untimed_lines", "count": 4}]

    @pytest.mark.parametrize("export_format", ["vtt", "LRC", "suno", "json"])
    async def test_an_unknown_format_is_404(
        self, client: AsyncTestClient[Litestar], song: Song, export_format: str
    ) -> None:
        timeline = await _created(client, song, song.complete())

        response = await client.get(
            f"/v1/studio/timelines/{timeline['id']}/exports/{export_format}",
            headers=bearer(song.user.id),
        )

        assert (response.status_code, response.json()["error"]) == (404, "not_found")

    async def test_a_foreign_timeline_is_404_everywhere(
        self, client: AsyncTestClient[Litestar], checked: HoldWorld, song: Song
    ) -> None:
        timeline = await _created(client, song, song.complete())
        stranger = bearer((await checked.user()).id)
        base = f"/v1/studio/timelines/{timeline['id']}"

        responses = [
            await client.get(f"{base}/exports/lrc", headers=stranger),
            await client.get(base, headers=stranger),
            await client.delete(base, headers=stranger),
        ]
        audio_responses = [
            await client.get(f"/v1/studio/audio/{song.audio_id}/timelines", headers=stranger),
            await client.post(
                f"/v1/studio/audio/{song.audio_id}/timelines",
                headers=stranger,
                json={"content": {"sections": []}},
            ),
            await client.put(
                f"/v1/studio/audio/{song.audio_id}/current-timeline",
                headers=stranger,
                json={"timeline_id": None},
            ),
        ]

        assert [(r.status_code, r.json()["error"]) for r in responses] == [
            (404, "timeline_not_found")
        ] * 3
        assert [(r.status_code, r.json()["error"]) for r in audio_responses] == [
            (404, "audio_version_not_found")
        ] * 3


class TestErrorsOverHttp:
    async def test_the_422_envelope_lists_every_issue(
        self, client: AsyncTestClient[Litestar], song: Song
    ) -> None:
        response = await client.post(
            f"/v1/studio/audio/{song.audio_id}/timelines",
            headers=bearer(song.user.id),
            json={
                "content": {
                    "sections": [
                        {"section_id": str(uuid4()), "start_ms": 5, "end_ms": 5},
                    ]
                }
            },
        )

        assert response.status_code == 422
        assert response.json()["error"] == "document_invalid"
        assert response.json()["detail"] == {
            "issues": [
                {"path": "sections[0]", "code": "span_out_of_range"},
                {"path": "sections[0]", "code": "unknown_section"},
            ]
        }

    async def test_the_409s_carry_their_codes_and_details(
        self, client: AsyncTestClient[Litestar], checked: HoldWorld, song: Song
    ) -> None:
        headers = bearer(song.user.id)
        parent = await _created(client, song, song.complete(), make_current=True)
        await _created(client, song, song.sections_only(), parent_id=parent["id"])

        unchanged = await _create(client, song, song.complete(), parent_id=parent["id"])
        current = await client.delete(f"/v1/studio/timelines/{parent['id']}", headers=headers)
        frozen = await client.patch(
            f"/v1/studio/audio/{song.audio_id}",
            headers=headers,
            json={"lyrics_version_id": None},
        )
        label = await client.patch(
            f"/v1/studio/audio/{song.audio_id}", headers=headers, json={"label": "Mix"}
        )
        await client.put(
            f"/v1/studio/audio/{song.audio_id}/current-timeline",
            headers=headers,
            json={"timeline_id": None},
        )
        has_children = await client.delete(f"/v1/studio/timelines/{parent['id']}", headers=headers)
        no_plan_song = await committed_song(checked, song.user, with_plan=False)
        no_plan = await _create(client, no_plan_song, song.sections_only())

        assert [
            (r.status_code, r.json()["error"])
            for r in (unchanged, current, frozen, has_children, no_plan)
        ] == [
            (409, "timeline_unchanged"),
            (409, "timeline_is_current"),
            (409, "audio_has_timelines"),
            (409, "timeline_has_children"),
            (409, "audio_has_no_plan"),
        ]
        assert frozen.json()["detail"] == {"count": 2}
        assert label.status_code == 200

    @pytest.mark.parametrize(
        "body",
        [
            pytest.param({}, id="no-content"),
            pytest.param({"content": {"sections": []}, "origin": "aligner"}, id="client-origin"),
            pytest.param({"content": {"sections": [], "complete": True}}, id="client-completeness"),
            pytest.param(
                {
                    "content": {
                        "sections": [],
                        "lines": [
                            {
                                "section_id": str(uuid4()),
                                "line_index": 64,
                                "start_ms": 0,
                                "end_ms": 1,
                            }
                        ],
                    }
                },
                id="line-index-past-63",
            ),
            pytest.param(
                {
                    "content": {
                        "sections": [{"section_id": str(uuid4()), "start_ms": 0.5, "end_ms": 1}]
                    }
                },
                id="float-milliseconds",
            ),
        ],
    )
    async def test_a_wire_invalid_body_is_400_and_writes_nothing(
        self, client: AsyncTestClient[Litestar], song: Song, body: dict[str, Any]
    ) -> None:
        response = await client.post(
            f"/v1/studio/audio/{song.audio_id}/timelines",
            headers=bearer(song.user.id),
            json=body,
        )

        assert response.status_code == 400, response.text

    async def test_a_nul_in_a_word_is_a_422_never_a_500_that_logs_the_words(
        self, client: AsyncTestClient[Litestar], song: Song
    ) -> None:
        """PostgreSQL cannot store U+0000. The real app refuses the request at its boundary
        (``400 invalid_character``, ``test_nul_characters_http.py``). This app has no such hook,
        so this is the second line: the studio's walker, which also covers an aligner's output.
        Either way the statement (with every word in it) never reaches the database or a log."""
        content = to_json(song.complete())
        content["lines"][0]["words"][0]["text"] = f"{MARK}\x00"

        with capture_logs() as logs:
            response = await client.post(
                f"/v1/studio/audio/{song.audio_id}/timelines",
                headers=bearer(song.user.id),
                json={"content": content},
            )

        assert response.status_code == 422, response.text
        assert response.json()["detail"]["issues"] == [
            {"path": "lines[0].words[0].text", "code": "invalid_character"}
        ]
        assert MARK not in response.text
        assert not [log for log in logs if MARK in repr(log)]

    @pytest.mark.parametrize("body", [{}, {"timeline_id": "x"}, {"timeline_id": None, "y": 1}])
    async def test_the_current_timeline_body_needs_exactly_timeline_id(
        self, client: AsyncTestClient[Litestar], song: Song, body: dict[str, Any]
    ) -> None:
        response = await client.put(
            f"/v1/studio/audio/{song.audio_id}/current-timeline",
            headers=bearer(song.user.id),
            json=body,
        )

        assert response.status_code == 400, response.text

    @pytest.mark.parametrize(
        ("query", "status"),
        [("limit=0", 400), ("limit=51", 400), ("limit=50", 200), ("cursor=bad", 400)],
    )
    async def test_list_parameters_are_bounded(
        self, client: AsyncTestClient[Litestar], song: Song, query: str, status: int
    ) -> None:
        response = await client.get(
            f"/v1/studio/audio/{song.audio_id}/timelines?{query}", headers=bearer(song.user.id)
        )

        assert response.status_code == status, response.text


def _calls(audio_id: UUID, timeline_id: UUID) -> list[tuple[str, str, dict[str, Any]]]:
    """Every timeline route as ``(method, url, request kwargs)``."""
    return [
        ("post", f"/v1/studio/audio/{audio_id}/timelines", {"json": {"content": {"sections": []}}}),
        ("get", f"/v1/studio/audio/{audio_id}/timelines", {}),
        ("get", f"/v1/studio/timelines/{timeline_id}", {}),
        ("delete", f"/v1/studio/timelines/{timeline_id}", {}),
        (
            "put",
            f"/v1/studio/audio/{audio_id}/current-timeline",
            {"json": {"timeline_id": None}},
        ),
        ("get", f"/v1/studio/timelines/{timeline_id}/exports/lrc", {}),
    ]


class TestGuards:
    async def test_a_disabled_studio_answers_404_even_unauthenticated(
        self, factory: async_sessionmaker[AsyncSession], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "src.api.security.studio.get_settings", lambda: SimpleNamespace(studio_enabled=False)
        )
        app = build_app(factory, [StudioTimelineController])
        async with AsyncTestClient(app=app) as client:
            for method, url, kwargs in _calls(uuid4(), uuid4()):
                response = await getattr(client, method)(url, **kwargs)
                assert response.status_code == 404, (method, url)
                assert response.json()["error"] == "not_found"

    async def test_a_product_without_the_studio_feature_answers_404(
        self, factory: async_sessionmaker[AsyncSession], checked: HoldWorld
    ) -> None:
        app = build_app(
            factory, [StudioTimelineController], config=replace(VEX_CONFIG, features=frozenset())
        )
        headers = bearer((await checked.user()).id)
        async with AsyncTestClient(app=app) as client:
            for method, url, kwargs in _calls(uuid4(), uuid4()):
                response = await getattr(client, method)(url, headers=headers, **kwargs)
                assert response.status_code == 404, (method, url)

    async def test_every_route_needs_authentication(
        self, client: AsyncTestClient[Litestar]
    ) -> None:
        for method, url, kwargs in _calls(uuid4(), uuid4()):
            response = await getattr(client, method)(url, **kwargs)
            assert response.status_code in {401, 403}, (method, url)

    async def test_a_stale_legal_digest_blocks_mutations_with_428_but_not_reads(
        self, client: AsyncTestClient[Litestar], song: Song
    ) -> None:
        timeline = await _created(client, song, song.complete())
        stale = bearer(song.user.id, digest="stale")

        statuses = {}
        for method, url, kwargs in _calls(song.audio_id, timeline["id"]):
            response = await getattr(client, method)(url, headers=stale, **kwargs)
            statuses[(method, url)] = response.status_code

        for (method, url), status in statuses.items():
            assert status == (200 if method == "get" else 428), (method, url)
        # nothing a 428 refused happened
        listed = await client.get(
            f"/v1/studio/audio/{song.audio_id}/timelines", headers=bearer(song.user.id)
        )
        assert [item["id"] for item in listed.json()["items"]] == [timeline["id"]]


def test_the_app_registers_the_controller_and_wires_the_service() -> None:
    from src.api.dependencies.common import dependencies, get_studio_timeline_service
    from src.studio.services.timeline import TimelineService

    assert dependencies["studio_timeline_service"].dependency is get_studio_timeline_service
    assert isinstance(get_studio_timeline_service(SimpleNamespace()), TimelineService)  # type: ignore[arg-type]
    app_source = (Path(__file__).resolve().parents[2] / "src/api/app.py").read_text()
    assert "            StudioTimelineController,\n" in app_source
