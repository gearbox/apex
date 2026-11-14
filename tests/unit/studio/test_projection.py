"""Renderers see projections that cannot carry non-exportable data (by type, not by grep)."""

from __future__ import annotations

import dataclasses
from types import SimpleNamespace
from typing import Any

import pytest

from src.core.enums import StudioSectionKind as K
from src.studio.renderers.base import (
    ArtistRenderInput,
    RenderablePlan,
    RenderableSection,
    RenderContext,
)
from src.studio.renderers.projection import to_artist_render_input, to_renderable_plan
from src.studio.renderers.suno import SunoRenderer
from src.studio.validation import normalize_plan
from tests.studio_support import make_plan, make_section

pytestmark = pytest.mark.unit


def _fields(cls: type) -> set[str]:
    return {f.name for f in dataclasses.fields(cls)}


def test_renderable_plan_has_exactly_the_allowed_fields() -> None:
    assert _fields(RenderablePlan) == {
        "title",
        "tempo_bpm",
        "instrumental",
        "global_style",
        "sections",
    }


def test_renderable_section_has_exactly_the_allowed_fields() -> None:
    assert _fields(RenderableSection) == {
        "id",
        "kind",
        "label",
        "duration_seconds",
        "style",
        "delivery",
        "instrumental",
    }


def test_artist_input_has_exactly_the_allowed_fields() -> None:
    assert _fields(ArtistRenderInput) == {"style_tags", "global_avoid"}


def test_render_context_carries_only_projections_and_lyrics() -> None:
    assert _fields(RenderContext) == {"plan", "lyrics", "artist"}


def test_marker_strings_in_non_exportable_fields_never_reach_the_output() -> None:
    plan = normalize_plan(
        make_plan(
            make_section(K.VERSE, notes="MARKER-NOTES", tags=("ok",)),
            idea="MARKER-IDEA",
            references=("MARKER-REFERENCE",),
            tags=("fine",),
        )
    )
    artist_row: Any = SimpleNamespace(
        style_tags=["vocal"],
        global_avoid=["bad"],
        persona="MARKER-PERSONA",
        voice_notes="MARKER-VOICE",
        aesthetic_references=["MARKER-AESTHETIC"],
        name="MARKER-NAME",
    )
    ctx = RenderContext(
        plan=to_renderable_plan(plan), lyrics=None, artist=to_artist_render_input(artist_row)
    )
    export = SunoRenderer().render(ctx)
    rendered = "\n".join(f.text for f in export.fields.values())
    assert "MARKER" not in rendered
    assert "MARKER" not in repr(ctx)


def test_projection_copies_exportable_data() -> None:
    plan = normalize_plan(
        make_plan(
            make_section(K.DROP, label="Boom", duration_seconds=7, delivery=("loud",), tags=("x",)),
            title="T",
            tempo_bpm=100,
            tags=("g",),
        )
    )
    projected = to_renderable_plan(plan)
    assert (projected.title, projected.tempo_bpm, projected.global_style.tags) == ("T", 100, ("g",))
    section = projected.sections[0]
    assert (section.kind, section.label, section.duration_seconds) == (K.DROP, "Boom", 7)
    assert section.delivery == ("loud",)
    assert section.style.tags == ("x",)


def test_no_artist_means_no_artist_input() -> None:
    assert to_artist_render_input(None) is None
