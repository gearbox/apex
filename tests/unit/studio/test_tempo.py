"""``src/studio/tempo.py``: the one definition of a tempo item in a Suno Style box."""

from __future__ import annotations

import msgspec
import pytest

from src.studio.documents import PlanDocument
from src.studio.tempo import TEMPO_MAX, TEMPO_MIN, tempo_of, tempo_token
from src.studio.validation import normalize_text

pytestmark = pytest.mark.unit


class TestTempoOf:
    @pytest.mark.parametrize(
        ("token", "expected"),
        [
            ("120 BPM", 120),
            ("120bpm", 120),
            ("120 bpm", 120),
            ("120 Bpm", 120),
            ("90BPM", 90),
            ("40 BPM", 40),  # the bounds are inclusive
            ("240 BPM", 240),
            ("  120 BPM", 120),  # surrounding whitespace is ignored
            ("120 BPM  ", 120),
            (normalize_text("  120   BPM "), 120),  # what the parser and the validators hand over
        ],
    )
    def test_a_tempo(self, token: str, expected: int) -> None:
        assert tempo_of(token) == expected

    @pytest.mark.parametrize(
        "token",
        [
            "39 BPM",  # below the range: an ordinary tag
            "241 BPM",
            "300 BPM",
            "1200 BPM",  # four digits
            "5 BPM",  # one digit
            "120 BPMs",
            "BPM 120",
            "120 BPM fast",
            "fast 120 BPM",
            "120",
            "bpm",
            "",
            "synthwave",
            "120 B P M",
        ],
    )
    def test_not_a_tempo(self, token: str) -> None:
        assert tempo_of(token) is None


class TestTempoToken:
    @pytest.mark.parametrize("bpm", [40, 90, 118, 240])
    def test_what_the_renderer_writes_is_read_back_as_the_same_tempo(self, bpm: int) -> None:
        assert tempo_token(bpm) == f"{bpm} BPM"
        assert tempo_of(tempo_token(bpm)) == bpm


class TestBoundsAreOneDefinition:
    def test_the_plan_documents_bounds_are_the_modules(self) -> None:
        """``PlanDocument.tempo_bpm`` cannot hold a tempo that the importer would not read."""
        info = msgspec.inspect.type_info(PlanDocument)
        assert isinstance(info, msgspec.inspect.StructType)
        field = next(f for f in info.fields if f.name == "tempo_bpm")
        assert isinstance(field.type, msgspec.inspect.UnionType)
        integer = next(t for t in field.type.types if isinstance(t, msgspec.inspect.IntType))
        assert (integer.ge, integer.le) == (TEMPO_MIN, TEMPO_MAX)
        assert (integer.gt, integer.lt) == (None, None)

    def test_the_bounds(self) -> None:
        assert (TEMPO_MIN, TEMPO_MAX) == (40, 240)
