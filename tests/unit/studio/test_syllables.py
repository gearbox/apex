"""Syllable counters (P2a-45): the table of the design, each alphabet's edge, tolerance, apostrophes."""

# ruff: noqa: RUF001, RUF003

from __future__ import annotations

import pytest

from src.studio.planner.syllables import counter_for

pytestmark = pytest.mark.unit

TABLE = [
    ("ru", "Я помню чудное мгновенье", 9),
    ("ru", "Передо мной явилась ты", 8),
    ("ru", "поэт аэропорт", 6),  # one per vowel LETTER: "поэт" is two syllables
    ("ru", "(о-о-о) ночь", 4),
    ("ru", "мой baby", None),
    ("ru", "24 часа", None),
    ("uk", "Реве та стогне Дніпр широкий", 9),
    ("bg", "Върви, народе възродени", 9),
    ("sr", "Тамо далеко, далеко од мора", 11),
    ("sr", "Tamo daleko, daleko od mora", 11),
    ("sr", "prst trg brdo rđa umro", 8),
    ("hr", "Lijepa naša domovino", 9),
    ("en", "Yesterday, all my troubles seemed so far away", 12),
    ("en", "smile while whole blue true eye people little apple", 12),
    ("en", "passed kissed wanted needed played", 7),
    ("en", "houses dances places horses", 8),
    ("en", "café", None),
]


@pytest.mark.parametrize(("language", "line", "expected"), TABLE)
def test_the_designs_table(language: str, line: str, expected: int | None) -> None:
    counter = counter_for(language)
    assert counter is not None
    assert counter.count(line) == expected


def test_an_unsupported_language_has_no_counter() -> None:
    assert counter_for("de") is None
    assert counter_for("zh-Hans") is None


@pytest.mark.parametrize("tag", ["en", "EN", "en-US", "en-gb", "ru-RU"])
def test_the_counter_is_chosen_by_primary_subtag_case_insensitively(tag: str) -> None:
    assert counter_for(tag) is not None


def test_only_english_has_a_tolerance() -> None:
    tolerances = {
        lang: counter.tolerance
        for lang in ("ru", "uk", "bg", "sr", "hr", "bs", "en")
        if (counter := counter_for(lang)) is not None
    }
    assert tolerances == {"ru": 0, "uk": 0, "bg": 0, "sr": 0, "hr": 0, "bs": 0, "en": 1}


def test_a_range_passes_inside_and_exact_languages_fail_one_outside() -> None:
    russian = counter_for("ru")
    english = counter_for("en")
    assert russian is not None
    assert english is not None
    assert russian.within(8, 8, 10)
    assert russian.within(10, 8, 10)
    assert not russian.within(7, 8, 10)
    assert not russian.within(11, 8, 10)
    # English: ±1, and no more
    assert english.within(7, 8, 10)
    assert english.within(11, 8, 10)
    assert not english.within(6, 8, 10)
    assert not english.within(12, 8, 10)


@pytest.mark.parametrize(
    ("language", "line", "countable"),
    [
        ("ru", "ёлка", True),  # ё is Russian
        ("ru", "їжак", False),  # ї is Ukrainian, not Russian
        ("uk", "ёлка", False),  # ё is not Ukrainian
        ("uk", "їжак", True),
        ("bg", "ёлка", False),
        ("bg", "ыэ", False),
        ("sr", "ђак", True),  # a Serbian Cyrillic letter
        ("sr", "wasser", False),  # w is not in the alphabet
        ("hr", "queen", False),
        ("en", "naïve", False),
        ("en", "übermensch", False),
    ],
)
def test_each_alphabets_edge(language: str, line: str, countable: bool) -> None:
    counter = counter_for(language)
    assert counter is not None
    assert (counter.count(line) is not None) is countable


def test_scripts_are_not_mixed_inside_one_word_but_may_be_mixed_between_words() -> None:
    serbian = counter_for("sr")
    assert serbian is not None
    assert serbian.count("moра") is None  # Latin "mo" + Cyrillic "ра"
    assert serbian.count("mora Тамо") == 4


def test_a_digit_anywhere_is_not_counted() -> None:
    assert counter_for("en") is not None
    assert counter_for("en").count("hello 2 you") is None  # type: ignore[union-attr]
    assert counter_for("ru").count("раз ٣ два") is None  # type: ignore[union-attr]  # an Arabic-Indic digit


def test_syllabic_r_needs_a_neighbour_and_no_vowel_next_to_it() -> None:
    serbian = counter_for("sr")
    assert serbian is not None
    assert serbian.count("r") == 0  # alone: no neighbour, not a syllable
    assert serbian.count("prst") == 1
    assert serbian.count("brdo") == 2
    assert serbian.count("rđa") == 2
    assert serbian.count("umro") == 2  # r between vowels is a consonant


def test_the_known_limit_of_the_syllabic_r_is_the_documented_one() -> None:
    """An r after a vowel-final prefix is read as a consonant ("zarđati": za-rđa-ti is 4)."""
    serbian = counter_for("sr")
    assert serbian is not None
    assert serbian.count("zarđati") == 3


@pytest.mark.parametrize("apostrophe", ["'", "\N{RIGHT SINGLE QUOTATION MARK}"])
def test_apostrophes_do_not_split_or_add_words(apostrophe: str) -> None:
    english = counter_for("en")
    assert english is not None
    assert english.count(f"don{apostrophe}t") == english.count("dont") == 1
    assert english.count(f"rock{apostrophe}n{apostrophe}roll") == english.count("rocknroll")


def test_case_and_unicode_form_do_not_matter() -> None:
    russian = counter_for("ru")
    assert russian is not None
    composed = "Ёлка"
    decomposed = "Е\N{COMBINING DIAERESIS}лка"  # NFD
    assert russian.count(composed) == russian.count(decomposed) == russian.count("ёлка") == 2


@pytest.mark.parametrize(
    ("word", "syllables"),
    [
        ("the", 1),  # a silent e after a consonant still leaves one syllable
        ("be", 1),
        ("red", 1),
        ("table", 2),
        ("smile", 1),
        ("loved", 1),
        ("wanted", 2),
        ("blue", 1),
        ("eye", 1),
        ("baby", 2),
        ("you", 1),
        ("its", 1),
        ("kiss", 1),
        ("roses", 2),  # "ses" keeps its s: the e is a syllable
    ],
)
def test_english_words(word: str, syllables: int) -> None:
    english = counter_for("en")
    assert english is not None
    assert english.count(word) == syllables


def test_an_empty_or_punctuation_only_line_counts_zero_syllables() -> None:
    for language in ("ru", "en", "sr"):
        counter = counter_for(language)
        assert counter is not None
        assert counter.count("... !?") == 0
