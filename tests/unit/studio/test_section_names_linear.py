"""Invariant 30: ``resolve_kind`` is linear in the input length, and ``_base_name`` is exactly what the
former header pattern returned.

``resolve_kind`` runs for every bracket line of a Suno import and the Lyrics box accepts 20,000
characters, so a pattern that backtracks quadratically (the lazy ``.+?`` retrying an optional
``\\s+\\d{1,2}`` tail at every position) blocked the event loop for seconds on one header line.
The old pattern lives here, and only here, as the reference the replacement must agree with.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from hypothesis import HealthCheck, example, given, settings
from hypothesis import strategies as st

from src.core.enums import StudioSectionKind as K
from src.studio import section_names
from src.studio.section_names import _base_name, is_header_text, resolve_kind

pytestmark = pytest.mark.unit

_REFERENCE = re.compile(r"^(?P<name>.+?)(?:\s+(?P<n>\d{1,2}))?$")
"""The pattern ``_base_name`` replaced (group ``name`` is the only one that was ever used)."""

PROPERTY = settings(
    max_examples=3000,
    derandomize=True,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)

# Every class that decides a match: letters, digits (ASCII and other scripts' decimal digits, plus
# a digit-like character that is *not* decimal), the separators the pattern treats as whitespace
# (the ones ``str.isspace`` knows, with the line-break family), and a ``-``.
_ALPHABET = [
    "a",
    "Z",
    "-",
    "0",
    "1",
    "7",
    "٣",  # ARABIC-INDIC DIGIT THREE: decimal
    "\uff12",  # FULLWIDTH DIGIT TWO: decimal
    "²",  # SUPERSCRIPT TWO: a digit, but not decimal
    " ",
    "\t",
    "\n",
    "\r",
    "\x0b",
    "\x0c",
    "\x1c",
    "\x85",
    "\u00a0",
    "\u2003",
]


def _reference(name: str) -> str | None:
    match = _REFERENCE.match(name.strip())
    return None if match is None else match.group("name")


class TestEquivalence:
    @PROPERTY
    @given(st.text(alphabet=st.sampled_from(_ALPHABET), max_size=14))
    @example("1\n1")
    @example("a\nb 1")
    @example("Verse 123")
    @example("5")
    @example("")
    def test_it_returns_what_the_former_pattern_returned_over_the_decisive_classes(
        self, name: str
    ) -> None:
        assert _base_name(name) == _reference(name)

    @PROPERTY
    @given(st.text(max_size=40))
    def test_it_returns_what_the_former_pattern_returned_over_any_text(self, name: str) -> None:
        assert _base_name(name) == _reference(name)

    @PROPERTY
    @given(
        st.text(alphabet=st.sampled_from(["a", "b", " ", "\t", "-", "1", "2", "3"]), max_size=14)
    )
    def test_resolve_kind_agrees_end_to_end(self, name: str) -> None:
        reference = _reference(name)
        expected = (
            None if reference is None else section_names.ALIASES.get(section_names._key(reference))
        )
        assert resolve_kind(name) is expected


class TestPinnedCases:
    @pytest.mark.parametrize(
        ("name", "base"),
        [
            ("Verse", "Verse"),
            ("Verse 1", "Verse"),
            ("Verse 12", "Verse"),
            ("Verse 123", "Verse 123"),  # three digits are not a number
            ("Verse 1 2", "Verse 1"),
            ("12 3", "12"),
            ("5", "5"),  # nothing before the digits: the digits are the name
            ("a\t5", "a"),
            ("1\n1", "1"),  # \s+ may cross a line break, so the first line is the name
            ("a\nb 1", None),  # a name that spans lines is not a header name
            ("a\nb", None),
            ("Verse1", "Verse1"),  # no whitespace before the digit: part of the name
            ("  Verse   7  ", "Verse"),
            ("", None),
            ("   ", None),
        ],
    )
    def test_the_name_without_its_optional_number(self, name: str, base: str | None) -> None:
        assert _base_name(name) == base
        assert _reference(name) == base  # and that is what the former pattern said

    def test_a_non_decimal_digit_is_not_a_number(self) -> None:
        assert _base_name("Verse ²") == "Verse ²"
        assert _base_name("Verse ٣") == "Verse"


class TestLinearTime:
    """No wall-clock limits: the suite's own timeout is the guard, and the module has no pattern."""

    def test_no_regular_expression_is_left_in_the_module(self) -> None:
        assert not hasattr(section_names, "_HEADER")
        assert not hasattr(section_names, "re")
        source = Path(section_names.__file__).read_text(encoding="utf-8")
        assert "re.compile" not in source
        assert "import re" not in source

    @pytest.mark.parametrize(
        "name",
        [
            "a" + " " * 20_000 + "x",  # the measured worst case of the old pattern
            "a" + " " * 20_000,
            " " * 20_000 + "a",
            "Verse" + " " * 20_000 + "1",
            "1" * 20_000,
            "1 " * 10_000,
            "a\n" * 10_000,
            "Verse" + "\t" * 20_000 + "12",
        ],
    )
    def test_a_header_line_at_the_size_limit_resolves(self, name: str) -> None:
        resolve_kind(name)
        is_header_text(name)

    def test_the_answers_at_the_size_limit_are_the_right_ones(self) -> None:
        assert resolve_kind("a" + " " * 20_000 + "x") is None
        assert resolve_kind("Verse" + " " * 20_000 + "1") is K.VERSE
        assert resolve_kind("Verse" + " " * 20_000 + "123") is None
        assert is_header_text("Chorus" + " " * 20_000 + "2: whispered")
