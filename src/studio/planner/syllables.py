"""Syllable counters for the lyric checks (P2a-35, P2a-45). Pure: no I/O, no model, no data files.

A counter is a *heuristic*, and its miss is only ever a warning (P2a-44):

* ``ru`` / ``uk`` / ``bg`` - one syllable per vowel **letter** (not per vowel group: "поэт" is two).
* ``sr`` / ``hr`` / ``bs`` (Cyrillic or Latin, one script per word) - vowel letters plus a syllabic
  *r* (an *r* with no vowel next to it: "prst", "brdo", "rđa"). Known limit: an *r* that follows a
  vowel-final prefix ("zarđati") is read as a plain consonant and its syllable is missed.
* ``en`` - vowel groups of ``aeiouy`` minus one for a silent ending, with a tolerance of ±1 because
  English spelling is not phonetic.

A line is **not counted** (``None``) when it has a digit or a word with a letter outside the
language's alphabet: a guess there would be noise, so the caller skips the line instead.
Any other language has no counter (``counter_for`` returns ``None``).
"""

# The alphabets below are the data: Cyrillic and Latin letters that look alike are the point.
# ruff: noqa: RUF001

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from collections.abc import Callable

__all__ = ["SyllableCounter", "counter_for", "fold_line", "tokenize_letters"]

_LETTERS: Final = re.compile(r"[^\W\d_]+(?:'[^\W\d_]+)*")
_DIGIT: Final = re.compile(r"\d")


def fold_line(text: str) -> str:
    """NFC, the typographic apostrophe folded to ``'``, then case-folded."""
    return (
        unicodedata.normalize("NFC", text)
        .replace("\N{RIGHT SINGLE QUOTATION MARK}", "'")
        .casefold()
    )


def tokenize_letters(text: str) -> list[str]:
    """The words of a line (letters, with inner apostrophes); the line must be folded already."""
    return _LETTERS.findall(text)


@dataclass(frozen=True, slots=True)
class SyllableCounter:
    """Counts the syllables of a line in one language."""

    count_word: Callable[[str], int | None]
    """Syllables of one folded word; ``None`` when the word is outside this counter's alphabet."""
    tolerance: int
    """How far a count may stray from the requested range and still pass (0 = exact)."""

    def count(self, text: str) -> int | None:
        """Syllables in a line; ``None`` when it has a digit or a word that cannot be counted."""
        folded = fold_line(text)
        if _DIGIT.search(folded):
            return None
        total = 0
        for word in tokenize_letters(folded):
            counted = self.count_word(word.replace("'", ""))
            if counted is None:
                return None
            total += counted
        return total

    def within(self, count: int, low: int, high: int) -> bool:
        """Whether ``count`` satisfies the range ``low``-``high`` under this counter's tolerance."""
        return low - self.tolerance <= count <= high + self.tolerance


# -- Slavic: one syllable per vowel letter ----------------------------------------------------

_RU_LETTERS: Final = frozenset("абвгдеёжзийклмнопрстуфхцчшщъыьэюя")
_UK_LETTERS: Final = frozenset("абвгґдеєжзиіїйклмнопрстуфхцчшщьюя")
_BG_LETTERS: Final = frozenset("абвгдежзийклмнопрстуфхцчшщъьюя")
_RU_VOWELS: Final = frozenset("аеёиоуыэюя")
_UK_VOWELS: Final = frozenset("аеєиіїоуюя")
_BG_VOWELS: Final = frozenset("аъоуеиюя")


def _vowel_letters(letters: frozenset[str], vowels: frozenset[str]) -> Callable[[str], int | None]:
    def count_word(word: str) -> int | None:
        if not word or not set(word) <= letters:
            return None
        return sum(ch in vowels for ch in word)

    return count_word


# -- Serbo-Croatian: vowel letters plus a syllabic r -------------------------------------------

_SH_CYRILLIC: Final = frozenset("абвгдђежзијклљмнњопрстћуфхцчџш")
_SH_LATIN: Final = frozenset("abcčćdđefghijklmnoprsštuvzž")
_SH_VOWELS: Final = frozenset("аеиоуaeiou")
_SH_R: Final = frozenset("рr")


def _serbo_croatian(word: str) -> int | None:
    letters = set(word)
    if not word or not (letters <= _SH_CYRILLIC or letters <= _SH_LATIN):
        return None  # outside the alphabet, or the two scripts mixed inside one word
    syllables = 0
    for index, ch in enumerate(word):
        if ch in _SH_VOWELS:
            syllables += 1
        elif ch in _SH_R:
            before = word[index - 1] if index > 0 else ""
            after = word[index + 1] if index + 1 < len(word) else ""
            if (before or after) and before not in _SH_VOWELS and after not in _SH_VOWELS:
                syllables += 1
    return syllables


# -- English: vowel groups minus a silent ending -----------------------------------------------

_EN_LETTERS: Final = frozenset("abcdefghijklmnopqrstuvwxyz")
_EN_VOWELS: Final = frozenset("aeiouy")
_VOWEL_GROUP: Final = re.compile(r"[aeiouy]+")
_SIBILANT_PLURALS: Final = ("ses", "zes", "xes", "ches", "shes", "ges", "ces")


def _english(word: str) -> int | None:
    if not word or not set(word) <= _EN_LETTERS:
        return None
    stem = word
    if len(stem) > 2 and stem.endswith("s") and not stem.endswith(_SIBILANT_PLURALS):
        stem = stem[:-1]  # a plural / third-person s adds no syllable ("troubles", "smiles")
    syllables = len(_VOWEL_GROUP.findall(stem))
    if stem.endswith("ed") and len(stem) > 2:
        # "loved" is one syllable, "wanted" and "needed" two: ``ed`` is silent after any other
        # consonant. (After a vowel it is part of a vowel group already.)
        before = stem[-3]
        if before not in _EN_VOWELS and before not in "td":
            syllables -= 1
    elif (
        stem.endswith("e")
        and len(stem) > 1
        and stem[-2] not in _EN_VOWELS
        and not (stem.endswith("le") and len(stem) > 2 and stem[-3] not in _EN_VOWELS)
    ):
        # A final e after a consonant is silent ("smile") unless it is consonant + "le" ("table").
        syllables -= 1
    return max(syllables, 1)


_COUNTERS: Final[dict[str, SyllableCounter]] = {
    "ru": SyllableCounter(_vowel_letters(_RU_LETTERS, _RU_VOWELS), tolerance=0),
    "uk": SyllableCounter(_vowel_letters(_UK_LETTERS, _UK_VOWELS), tolerance=0),
    "bg": SyllableCounter(_vowel_letters(_BG_LETTERS, _BG_VOWELS), tolerance=0),
    "sr": SyllableCounter(_serbo_croatian, tolerance=0),
    "hr": SyllableCounter(_serbo_croatian, tolerance=0),
    "bs": SyllableCounter(_serbo_croatian, tolerance=0),
    "en": SyllableCounter(_english, tolerance=1),
}


def counter_for(language: str) -> SyllableCounter | None:
    """The counter for a BCP-47-style tag (by its primary subtag, case-insensitive), if any."""
    return _COUNTERS.get(language.split("-", 1)[0].casefold())
