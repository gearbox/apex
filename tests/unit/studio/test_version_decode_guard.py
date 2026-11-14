"""RD7 / invariant 32: a version whose content would not decode is never written."""

from __future__ import annotations

import msgspec
import pytest

from src.core.enums import StudioDocumentKind, StudioSectionKind
from src.studio.services.version import DocumentVersionService
from src.studio.validation import to_content
from tests.studio_support import make_lyrics, make_plan, make_section

pytestmark = pytest.mark.unit

PLAN = StudioDocumentKind.PLAN
LYRICS = StudioDocumentKind.LYRICS


def test_content_that_decodes_passes_for_both_kinds() -> None:
    verse = make_section(StudioSectionKind.VERSE)
    DocumentVersionService._assert_decodes(PLAN, to_content(make_plan(verse)))
    DocumentVersionService._assert_decodes(LYRICS, to_content(make_lyrics((verse.id, ["la"]))))


def test_an_over_long_value_is_a_runtime_error_not_a_client_error() -> None:
    content = to_content(make_plan(title="x" * 201))  # past the 200 the wire allows
    with pytest.raises(RuntimeError, match="would not decode") as exc:
        DocumentVersionService._assert_decodes(PLAN, content)
    assert isinstance(exc.value.__cause__, msgspec.ValidationError)


def test_the_message_never_echoes_document_text() -> None:
    content = to_content(make_plan(title="SECRET-" + "x" * 300))
    with pytest.raises(RuntimeError) as exc:
        DocumentVersionService._assert_decodes(PLAN, content)
    assert "SECRET" not in str(exc.value)


def test_lyrics_content_is_decoded_as_lyrics() -> None:
    verse = make_section(StudioSectionKind.VERSE)
    content = to_content(make_lyrics((verse.id, ["x" * 301])))
    with pytest.raises(RuntimeError, match="would not decode"):
        DocumentVersionService._assert_decodes(LYRICS, content)


def test_content_of_the_wrong_shape_is_refused() -> None:
    with pytest.raises(RuntimeError, match="would not decode"):
        DocumentVersionService._assert_decodes(PLAN, {"sections": []})  # no title
