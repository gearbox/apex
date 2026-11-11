"""Hash determinism and sensitivity (the basis of natural idempotency)."""

from __future__ import annotations

import unicodedata

import pytest

from src.core.enums import StudioDocumentKind as D
from src.core.uid import new_id
from src.studio.validation import content_hash, normalize_lyrics, normalize_plan, to_content
from tests.studio_support import make_lyrics, make_plan, make_section

pytestmark = pytest.mark.unit


def test_hash_is_deterministic_hex_sha256() -> None:
    plan = normalize_plan(make_plan(title="T"))
    digest = content_hash(D.PLAN, None, plan)
    assert digest == content_hash(D.PLAN, None, plan)
    assert len(digest) == 64
    int(digest, 16)


def test_whitespace_variants_hash_identically_after_normalization() -> None:
    section = make_section()
    a = normalize_plan(make_plan(section, title="a  b", tags=("x  y",)))
    b = normalize_plan(make_plan(section, title=" a b ", tags=("x y",)))
    assert a == b
    assert content_hash(D.PLAN, None, a) == content_hash(D.PLAN, None, b)
    assert to_content(a) == to_content(b)


def test_nfc_and_nfd_hash_identically_after_normalization() -> None:
    section = make_section()
    composed = normalize_plan(make_plan(section, title="Café"))
    decomposed = normalize_plan(make_plan(section, title=unicodedata.normalize("NFD", "Café")))
    assert content_hash(D.PLAN, None, composed) == content_hash(D.PLAN, None, decomposed)


def test_content_changes_the_hash() -> None:
    section = make_section()
    a = normalize_plan(make_plan(section, title="a"))
    b = normalize_plan(make_plan(section, title="b"))
    assert content_hash(D.PLAN, None, a) != content_hash(D.PLAN, None, b)


def test_based_on_changes_the_hash() -> None:
    section = make_section()
    plan = normalize_plan(make_plan(section))
    lyrics = normalize_lyrics(make_lyrics((section.id, ["la"])), plan)
    assert content_hash(D.LYRICS, new_id(), lyrics) != content_hash(D.LYRICS, new_id(), lyrics)
    base = new_id()
    assert content_hash(D.LYRICS, base, lyrics) == content_hash(D.LYRICS, base, lyrics)


def test_kind_changes_the_hash() -> None:
    plan = normalize_plan(make_plan())
    assert content_hash(D.PLAN, None, plan) != content_hash(D.LYRICS, None, plan)


def test_stored_content_is_json_builtins() -> None:
    section = make_section()
    content = to_content(normalize_plan(make_plan(section)))
    assert content["sections"][0]["id"] == str(section.id)  # type: ignore[index]
    assert content["sections"][0]["kind"] == "verse"  # type: ignore[index]
