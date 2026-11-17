"""``lyrics`` and ``revise_lyrics`` runs over HTTP: real controllers, guards, DI, handlers, PostgreSQL.

The worker is driven with ``drive`` (never a sleep); the response bodies are the contract (§11).
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, Any
from uuid import uuid4

import pytest
import pytest_asyncio
from litestar.testing import AsyncTestClient

from src.studio.planner.drafts import DraftRefusal
from tests.integration.test_studio_runs_routes import KEY, S, _build_app
from tests.llm_support import FakeLlm
from tests.studio_llm import (
    make_line_patch,
    make_lyrics_draft,
    make_lyrics_patch,
    make_lyrics_revision,
    refusal_lyrics_draft,
    section_lines,
)
from tests.studio_lyrics_support import PLAN, S1, S2, S3, S4, doc
from tests.studio_support import make_plan, make_section

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator
    from uuid import UUID

    from litestar import Litestar

    from tests.studio_runs_support import Actor, RunWorld

FOUR = "la la la la"
LONG = "la la la la la la la la la la"


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


def _good() -> Any:
    return make_lyrics_draft(
        section_lines("s1", FOUR, FOUR), section_lines("s2", FOUR), section_lines("s4", FOUR, FOUR)
    )


async def _track(
    world: RunWorld, actor: Actor, *, lyrics: bool = False
) -> tuple[UUID, UUID, UUID | None]:
    track_id = await world.new_track(actor, "Night Drive", PLAN)
    plan_id = (await world.track_row(track_id)).current_plan_version_id
    assert plan_id is not None
    lyrics_id = None
    if lyrics:
        lyrics_id = await world.create_lyrics(
            actor,
            track_id,
            doc((S1, [FOUR, LONG, FOUR]), (S2, [FOUR]), (S4, [FOUR])),
            plan_version_id=plan_id,
        )
    return track_id, plan_id, lyrics_id


async def _post(
    client: AsyncTestClient[Litestar],
    actor: Actor,
    track_id: Any,
    body: dict[str, Any],
    key: str = "k-1",
) -> Any:
    return await client.post(
        f"{S}/tracks/{track_id}/runs", headers={**actor.headers, KEY: key}, json=body
    )


def _revise(**targets: Any) -> dict[str, Any]:
    return {"kind": "revise_lyrics", "instruction": "make it warmer", "targets": targets}


class TestLyrics:
    async def test_202_then_the_result_is_an_ordinary_lyrics_version(
        self, client: AsyncTestClient[Litestar], world: RunWorld
    ) -> None:
        actor = await world.add_actor()
        track_id, plan_id, _ = await _track(world, actor)

        response = await _post(
            client, actor, track_id, {"kind": "lyrics", "language": "en", "avoid_words": ["baby"]}
        )

        assert response.status_code == 202, response.text
        body = response.json()
        assert (body["kind"], body["status"], body["result_lyrics_version_id"]) == (
            "lyrics",
            "queued",
            None,
        )
        assert "baby" not in response.text  # the request text is never echoed
        assert "inputs" not in body and "plan_version_id" not in body  # nor what it works on

        await world.drive(FakeLlm(_good()))
        done = (await client.get(f"{S}/runs/{body['id']}", headers=actor.headers)).json()
        assert (done["status"], done["warnings"], done["result_plan_version_id"]) == (
            "succeeded",
            [],
            None,
        )
        version = (
            await client.get(
                f"{S}/versions/{done['result_lyrics_version_id']}", headers=actor.headers
            )
        ).json()
        assert (version["kind"], version["origin"], version["is_current"]) == (
            "lyrics",
            "llm",
            False,
        )
        assert version["based_on_plan_version_id"] == str(plan_id)
        assert version["content"]["language"] == "en"

    async def test_a_miss_is_a_warning_with_a_path_and_a_count(
        self, client: AsyncTestClient[Litestar], world: RunWorld
    ) -> None:
        actor = await world.add_actor()
        track_id, _, _ = await _track(world, actor)
        body = {"kind": "lyrics", "language": "en", "line_syllables": {"min": 4, "max": 6}}
        run_id = (await _post(client, actor, track_id, body)).json()["id"]
        draft = make_lyrics_draft(
            section_lines("s1", FOUR, LONG), section_lines("s2", FOUR), section_lines("s4", FOUR)
        )
        keep = make_lyrics_patch(make_line_patch("s1", 2, LONG))
        await world.drive(FakeLlm(draft, keep, keep, keep))

        done = (await client.get(f"{S}/runs/{run_id}", headers=actor.headers)).json()
        assert done["status"] == "succeeded"
        assert done["warnings"] == [
            {"code": "syllable_range_missed", "path": "sections[0].lines[1]", "count": 10}
        ]

    async def test_a_failed_run_reports_plan_section_paths_and_no_text(
        self, client: AsyncTestClient[Litestar], world: RunWorld
    ) -> None:
        actor = await world.add_actor()
        track_id, _, _ = await _track(world, actor)
        run_id = (
            await _post(client, actor, track_id, {"kind": "lyrics", "language": "en"})
        ).json()["id"]
        bad = make_lyrics_draft(
            section_lines("s1", FOUR), section_lines("s2", "oh [no]"), section_lines("s4", FOUR)
        )
        keep = make_lyrics_patch(make_line_patch("s2", 1, "oh [no]"))
        await world.drive(FakeLlm(bad, keep, keep, keep))

        done = (await client.get(f"{S}/runs/{run_id}", headers=actor.headers)).json()
        assert (done["status"], done["error_code"]) == ("failed", "lyrics_invalid")
        assert done["error_detail"] == {
            "issues": [{"path": "sections[1].lines[0].text", "code": "reserved_characters"}]
        }
        assert "oh [no]" not in str(done)

    async def test_a_refusal_is_refunded(
        self, client: AsyncTestClient[Litestar], world: RunWorld
    ) -> None:
        actor = await world.add_actor()
        track_id, _, _ = await _track(world, actor)
        run_id = (
            await _post(client, actor, track_id, {"kind": "lyrics", "language": "en"})
        ).json()["id"]
        await world.drive(FakeLlm(refusal_lyrics_draft(DraftRefusal.CONTENT_POLICY)))
        done = (await client.get(f"{S}/runs/{run_id}", headers=actor.headers)).json()
        assert (done["status"], done["error_code"]) == ("failed", "content_policy_refused")
        assert await world.refunds_and_debits(actor) == (1, 1)

    async def test_the_errors_at_submit_leave_nothing_behind(
        self, client: AsyncTestClient[Litestar], world: RunWorld
    ) -> None:
        actor = await world.add_actor()  # no artist: no language to fall back on
        empty = await world.new_track(actor, "Empty")
        track_id, _, lyrics_id = await _track(world, actor, lyrics=True)
        instrumental = await world.new_track(
            actor, "Silent", make_plan(make_section(instrumental=True))
        )

        cases = [
            (empty, {"kind": "lyrics", "language": "en"}, 409, "plan_required"),
            (
                track_id,
                {"kind": "lyrics", "plan_version_id": str(uuid4())},
                404,
                "plan_version_not_found",
            ),
            (
                track_id,
                {"kind": "lyrics", "plan_version_id": str(lyrics_id), "language": "en"},
                404,
                "plan_version_not_found",
            ),
            (instrumental, {"kind": "lyrics", "language": "en"}, 422, "document_invalid"),
            (
                empty,
                {"kind": "lyrics", "memory_ids": [str(uuid4())]},
                422,
                "memories_not_available",
            ),
        ]
        for index, (target, body, status, error) in enumerate(cases):
            response = await _post(client, actor, target, body, key=f"k-{index}")
            assert (response.status_code, response.json()["error"]) == (status, error), (
                response.text
            )
        assert await world.count_runs(actor) == 0
        assert (await world.balance(actor), await world.refunds_and_debits(actor)) == (10, (0, 0))

    async def test_422_names_the_plan_problem_and_the_missing_language_together(
        self, client: AsyncTestClient[Litestar], world: RunWorld
    ) -> None:
        actor = await world.add_actor()
        track_id = await world.new_track(
            actor, "Silent", make_plan(make_section(instrumental=True))
        )
        response = await _post(client, actor, track_id, {"kind": "lyrics"})
        assert response.status_code == 422
        issues = response.json()["detail"]["issues"]
        assert {(i["path"], i["code"]) for i in issues} == {
            ("plan_version_id", "plan_has_no_vocal_sections"),
            ("language", "language_required"),
        }

    @pytest.mark.parametrize(
        "body",
        [
            {"kind": "lyrics", "whatever": 1},
            {"kind": "lyrics", "line_syllables": {"min": 0, "max": 5}},
            {"kind": "lyrics", "avoid_words": [""]},
            {"kind": "lyrics", "language": "NOT A TAG"},
        ],
    )
    async def test_400_for_a_malformed_body(
        self, client: AsyncTestClient[Litestar], world: RunWorld, body: dict[str, Any]
    ) -> None:
        actor = await world.add_actor()
        track_id, _, _ = await _track(world, actor)
        assert (await _post(client, actor, track_id, body)).status_code == 400
        assert await world.count_runs(actor) == 0

    async def test_422_for_an_inverted_syllable_range(
        self, client: AsyncTestClient[Litestar], world: RunWorld
    ) -> None:
        actor = await world.add_actor()
        track_id, _, _ = await _track(world, actor)
        response = await _post(
            client,
            actor,
            track_id,
            {"kind": "lyrics", "language": "en", "line_syllables": {"min": 9, "max": 3}},
        )
        assert response.status_code == 422
        assert response.json()["detail"]["issues"] == [
            {"path": "line_syllables", "code": "syllable_range_inverted"}
        ]

    async def test_409_when_the_track_already_has_an_active_run(
        self, client: AsyncTestClient[Litestar], world: RunWorld
    ) -> None:
        actor = await world.add_actor()
        track_id, _, _ = await _track(world, actor)
        assert (
            await _post(client, actor, track_id, {"kind": "lyrics", "language": "en"})
        ).status_code == 202
        response = await _post(
            client, actor, track_id, {"kind": "lyrics", "language": "ru"}, key="k-2"
        )
        assert (response.status_code, response.json()["error"]) == (409, "run_in_progress")


class TestRevise:
    async def test_202_and_the_revision_leaves_everything_else_alone(
        self, client: AsyncTestClient[Litestar], world: RunWorld
    ) -> None:
        actor = await world.add_actor()
        track_id, _, lyrics_id = await _track(world, actor, lyrics=True)
        body = _revise(sections=[str(S2)], lines=[{"section_id": str(S1), "line": 0}])

        response = await _post(client, actor, track_id, body)
        assert response.status_code == 202, response.text
        run = response.json()
        assert (run["kind"], run["status"]) == ("revise_lyrics", "queued")

        reply = make_lyrics_revision(
            sections=[section_lines("s2", "new chorus")],
            lines=[make_line_patch("s1", 1, "new opener")],
        )
        await world.drive(FakeLlm(reply))
        done = (await client.get(f"{S}/runs/{run['id']}", headers=actor.headers)).json()
        assert (done["status"], done["warnings"]) == ("succeeded", [])
        version = (
            await client.get(
                f"{S}/versions/{done['result_lyrics_version_id']}", headers=actor.headers
            )
        ).json()
        assert version["parent_version_id"] == str(lyrics_id)
        lines = [[line["text"] for line in s["lines"]] for s in version["content"]["sections"]]
        assert lines == [["new opener", LONG, FOUR], ["new chorus"], [FOUR]]

    async def test_an_unchanged_revision_returns_the_input_with_a_warning(
        self, client: AsyncTestClient[Litestar], world: RunWorld
    ) -> None:
        actor = await world.add_actor()
        track_id, _, lyrics_id = await _track(world, actor, lyrics=True)
        run_id = (
            await _post(
                client, actor, track_id, _revise(lines=[{"section_id": str(S2), "line": 0}])
            )
        ).json()["id"]
        await world.drive(FakeLlm(make_lyrics_revision(lines=[make_line_patch("s2", 1, FOUR)])))
        done = (await client.get(f"{S}/runs/{run_id}", headers=actor.headers)).json()
        assert (done["result_lyrics_version_id"], done["warnings"]) == (
            str(lyrics_id),
            [{"code": "revision_unchanged", "path": None, "count": None}],
        )

    async def test_the_errors_at_submit_leave_nothing_behind(
        self, client: AsyncTestClient[Litestar], world: RunWorld
    ) -> None:
        actor = await world.add_actor()
        empty = await world.new_track(actor, "Empty", PLAN)
        track_id, plan_id, _ = await _track(world, actor, lyrics=True)

        cases = [
            (empty, _revise(sections=[str(S2)]), 409, "lyrics_required"),
            (
                track_id,
                {**_revise(sections=[str(S2)]), "lyrics_version_id": str(uuid4())},
                404,
                "lyrics_version_not_found",
            ),
            (
                track_id,
                {**_revise(sections=[str(S2)]), "lyrics_version_id": str(plan_id)},
                404,
                "lyrics_version_not_found",
            ),
            (track_id, _revise(), 422, "document_invalid"),
            (track_id, _revise(sections=[str(S3)]), 422, "document_invalid"),
            (
                track_id,
                {**_revise(sections=[str(S2)]), "memory_ids": [str(uuid4())]},
                422,
                "memories_not_available",
            ),
        ]
        for index, (target, body, status, error) in enumerate(cases):
            response = await _post(client, actor, target, body, key=f"k-{index}")
            assert (response.status_code, response.json()["error"]) == (status, error), (
                response.text
            )
        assert await world.count_runs(actor) == 0
        assert (await world.balance(actor), await world.refunds_and_debits(actor)) == (10, (0, 0))

    async def test_the_target_issues_are_paths_into_the_request(
        self, client: AsyncTestClient[Litestar], world: RunWorld
    ) -> None:
        actor = await world.add_actor()
        track_id, _, _ = await _track(world, actor, lyrics=True)
        body = _revise(sections=[str(uuid4()), str(S3)], lines=[{"section_id": str(S1), "line": 9}])
        response = await _post(client, actor, track_id, body)
        assert response.status_code == 422
        assert response.json()["detail"]["issues"] == [
            {"path": "targets.sections[0]", "code": "unknown_section"},
            {"path": "targets.sections[1]", "code": "instrumental_section"},
            {"path": "targets.lines[0]", "code": "line_out_of_range"},
        ]

    @pytest.mark.parametrize(
        "body",
        [
            {"kind": "revise_lyrics", "targets": {"sections": []}},
            {"kind": "revise_lyrics", "instruction": "x"},
            {
                "kind": "revise_lyrics",
                "instruction": "x",
                "targets": {"lines": [{"section_id": str(S1), "line": -1}]},
            },
            {"kind": "revise_lyrics", "instruction": "x", "targets": {}, "extra": 1},
        ],
    )
    async def test_400_for_a_malformed_body(
        self, client: AsyncTestClient[Litestar], world: RunWorld, body: dict[str, Any]
    ) -> None:
        actor = await world.add_actor()
        track_id, _, _ = await _track(world, actor, lyrics=True)
        assert (await _post(client, actor, track_id, body)).status_code == 400

    async def test_a_failed_revision_reports_plan_section_paths(
        self, client: AsyncTestClient[Litestar], world: RunWorld
    ) -> None:
        actor = await world.add_actor()
        track_id, _, _ = await _track(world, actor, lyrics=True)
        run_id = (await _post(client, actor, track_id, _revise(sections=[str(S2)]))).json()["id"]
        bad = make_lyrics_revision(sections=[section_lines("s2", "oh [no]")])
        keep = make_lyrics_patch(make_line_patch("s2", 1, "oh [no]"))
        await world.drive(FakeLlm(bad, keep, keep))
        done = (await client.get(f"{S}/runs/{run_id}", headers=actor.headers)).json()
        assert (done["status"], done["error_code"]) == ("failed", "lyrics_invalid")
        assert done["error_detail"]["issues"] == [
            {"path": "sections[1].lines[0].text", "code": "reserved_characters"}
        ]


class TestDeleteGuardOverHttp:
    async def test_409_version_in_use_by_run_then_204_once_the_run_is_over(
        self, client: AsyncTestClient[Litestar], world: RunWorld
    ) -> None:
        actor = await world.add_actor()
        track_id, first, _ = await _track(world, actor)
        await world.create_plan(actor, track_id, make_plan(make_section(), title="B"))
        run_id = (
            await _post(
                client,
                actor,
                track_id,
                {"kind": "lyrics", "plan_version_id": str(first), "language": "en"},
            )
        ).json()["id"]

        blocked = await client.delete(f"{S}/versions/{first}", headers=actor.headers)
        assert (blocked.status_code, blocked.json()["error"]) == (409, "version_in_use_by_run")

        await world.drive(FakeLlm(refusal_lyrics_draft(DraftRefusal.CONTENT_POLICY)))
        assert (await client.get(f"{S}/runs/{run_id}", headers=actor.headers)).json()[
            "status"
        ] == "failed"
        assert (
            await client.delete(f"{S}/versions/{first}", headers=actor.headers)
        ).status_code == 204
