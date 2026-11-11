"""Constraint edges of every studio document Struct (msgspec wire validation)."""

from __future__ import annotations

from typing import Any
from uuid import uuid4

import msgspec
import pytest

from src.core.enums import StudioDocumentKind
from src.studio.documents import (
    CURRENT_SCHEMA_VERSION,
    LyricsDocument,
    PlanDocument,
    decode_stored,
)
from src.studio.errors import StudioDocumentSchemaError

pytestmark = pytest.mark.unit


def _section(**overrides: Any) -> dict[str, Any]:
    return {"id": str(uuid4()), "kind": "verse", **overrides}


def _plan(**overrides: Any) -> dict[str, Any]:
    return {"title": "T", "sections": [_section()], **overrides}


def _decode_plan(body: dict[str, Any]) -> PlanDocument:
    return msgspec.json.decode(msgspec.json.encode(body), type=PlanDocument)


def _decode_lyrics(body: dict[str, Any]) -> LyricsDocument:
    return msgspec.json.decode(msgspec.json.encode(body), type=LyricsDocument)


class TestPlanBounds:
    @pytest.mark.parametrize(
        "body",
        [
            _plan(title="x" * 200),
            _plan(idea="x" * 4000),
            _plan(tempo_bpm=40),
            _plan(tempo_bpm=240),
            _plan(global_style={"tags": ["t"] * 40, "avoid": ["a"] * 40}),
            _plan(global_style={"tags": ["x" * 60]}),
            _plan(references=["r" * 200] * 20),
            _plan(sections=[_section() for _ in range(64)]),
            _plan(sections=[_section(duration_seconds=1), _section(duration_seconds=600)]),
            _plan(sections=[_section(delivery=["d" * 40] * 12, label="l" * 60, notes="n" * 1000)]),
        ],
    )
    def test_accepts_boundary_values(self, body: dict[str, Any]) -> None:
        assert _decode_plan(body)

    @pytest.mark.parametrize(
        "body",
        [
            _plan(title=""),
            _plan(title="x" * 201),
            _plan(idea="x" * 4001),
            _plan(tempo_bpm=39),
            _plan(tempo_bpm=241),
            _plan(global_style={"tags": ["t"] * 41}),
            _plan(global_style={"avoid": ["a"] * 41}),
            _plan(global_style={"tags": ["x" * 61]}),
            _plan(global_style={"tags": [""]}),
            _plan(references=["r"] * 21),
            _plan(references=["r" * 201]),
            _plan(references=[""]),
            _plan(sections=[]),
            _plan(sections=[_section() for _ in range(65)]),
            _plan(sections=[_section(duration_seconds=0)]),
            _plan(sections=[_section(duration_seconds=601)]),
            _plan(sections=[_section(delivery=["d"] * 13)]),
            _plan(sections=[_section(delivery=["d" * 41])]),
            _plan(sections=[_section(label="")]),
            _plan(sections=[_section(label="l" * 61)]),
            _plan(sections=[_section(notes="n" * 1001)]),
            _plan(sections=[_section(kind="chorus2")]),
            _plan(sections=[{"kind": "verse"}]),
        ],
    )
    def test_rejects_out_of_bounds(self, body: dict[str, Any]) -> None:
        with pytest.raises(msgspec.ValidationError):
            _decode_plan(body)

    @pytest.mark.parametrize(
        "body",
        [
            _plan(surprise=1),
            _plan(global_style={"tags": [], "oops": 1}),
            _plan(sections=[_section(oops=1)]),
        ],
    )
    def test_unknown_fields_are_rejected_at_every_level(self, body: dict[str, Any]) -> None:
        with pytest.raises(msgspec.ValidationError, match="unknown field"):
            _decode_plan(body)

    def test_defaults_are_empty_immutable_values(self) -> None:
        plan = _decode_plan(_plan())
        assert plan.global_style.tags == ()
        assert plan.references == ()
        assert plan.sections[0].instrumental is False
        with pytest.raises(AttributeError):
            plan.title = "other"  # type: ignore[misc]


def _lyrics_section(**overrides: Any) -> dict[str, Any]:
    return {"section_id": str(uuid4()), "lines": [{"text": "la"}], **overrides}


class TestLyricsBounds:
    @pytest.mark.parametrize("language", ["en", "ru", "pt-BR", "zh-Hans-CN", "yue"])
    def test_language_accepts_bcp47_like_tags(self, language: str) -> None:
        assert _decode_lyrics({"language": language}).language == language

    @pytest.mark.parametrize("language", ["", "e", "EN", "english", "en_US", "en-", "x" * 36])
    def test_language_rejects_malformed(self, language: str) -> None:
        with pytest.raises(msgspec.ValidationError):
            _decode_lyrics({"language": language})

    @pytest.mark.parametrize(
        "body",
        [
            {"language": "en", "sections": [{"section_id": str(uuid4()), "lines": []}]},
            {"language": "en", "sections": [{"section_id": str(uuid4()), "lines": [{"text": ""}]}]},
            {
                "language": "en",
                "sections": [{"section_id": str(uuid4()), "lines": [{"text": "x" * 301}]}],
            },
            {
                "language": "en",
                "sections": [
                    {"section_id": str(uuid4()), "lines": [{"text": "a", "delivery": ["d"] * 7}]}
                ],
            },
            {
                "language": "en",
                "sections": [
                    {"section_id": str(uuid4()), "lines": [{"text": "a"}] * 65},
                ],
            },
            {"language": "en", "sections": [_lyrics_section() for _ in range(65)]},
            {"language": "en", "line_syllables": {"min": 0, "max": 5}},
            {"language": "en", "line_syllables": {"min": 1, "max": 33}},
            {"language": "en", "extra": True},
        ],
    )
    def test_rejects_out_of_bounds(self, body: dict[str, Any]) -> None:
        with pytest.raises(msgspec.ValidationError):
            _decode_lyrics(body)

    def test_accepts_boundaries(self) -> None:
        body = {
            "language": "en",
            "line_syllables": {"min": 1, "max": 32},
            "sections": [
                {
                    "section_id": str(uuid4()),
                    "lines": [{"text": "x" * 300, "delivery": ["d"] * 6}] * 64,
                }
                for _ in range(64)
            ],
        }
        assert len(_decode_lyrics(body).sections) == 64


class TestDecodeStored:
    def test_round_trips_builtins(self) -> None:
        plan = _decode_plan(_plan(tempo_bpm=100))
        again = decode_stored(StudioDocumentKind.PLAN, 1, msgspec.to_builtins(plan))
        assert again == plan

    def test_lyrics_round_trip(self) -> None:
        lyrics = _decode_lyrics({"language": "en"})
        assert decode_stored(StudioDocumentKind.LYRICS, 1, msgspec.to_builtins(lyrics)) == lyrics

    @pytest.mark.parametrize("kind", list(StudioDocumentKind))
    @pytest.mark.parametrize("version", [0, 2, 99])
    def test_unknown_schema_version_raises_loudly(
        self, kind: StudioDocumentKind, version: int
    ) -> None:
        assert version != CURRENT_SCHEMA_VERSION[kind]
        with pytest.raises(StudioDocumentSchemaError):
            decode_stored(kind, version, {})

    def test_mismatching_content_is_not_silently_repaired(self) -> None:
        with pytest.raises(msgspec.ValidationError):
            decode_stored(StudioDocumentKind.PLAN, 1, {"title": "x"})
