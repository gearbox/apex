"""``POST .../runs`` bodies: the tagged union and the brief's normalization (RD6)."""

from __future__ import annotations

from uuid import uuid4

import msgspec
import pytest

from src.studio.errors import StudioDocumentInvalid
from src.studio.run_requests import (
    LyricsRunRequest,
    PlanRunRequest,
    ReviseLyricsRunRequest,
    RunRequest,
    normalize_plan_run_request,
)

pytestmark = pytest.mark.unit


def _decode(body: str) -> RunRequest:
    return msgspec.json.decode(body, type=RunRequest)


def test_a_plan_request_decodes_with_defaults() -> None:
    request = _decode('{"kind": "plan", "brief": "a song"}')
    assert request == PlanRunRequest(brief="a song")
    assert request.language is None  # type: ignore[union-attr]
    assert request.memory_ids == ()  # type: ignore[union-attr]
    assert request.auto_fill is True  # type: ignore[union-attr]


def test_the_union_is_tagged_on_kind() -> None:
    assert isinstance(_decode('{"kind": "lyrics"}'), LyricsRunRequest)
    assert isinstance(_decode('{"kind": "revise_lyrics"}'), ReviseLyricsRunRequest)


def test_a_lyrics_body_with_any_fields_still_decodes_so_the_service_can_refuse_it() -> None:
    assert isinstance(_decode('{"kind": "lyrics", "brief": "x", "whatever": 1}'), LyricsRunRequest)


@pytest.mark.parametrize(
    "body",
    [
        '{"kind": "plan"}',
        '{"kind": "plan", "brief": ""}',
        '{"kind": "plan", "brief": "x", "extra": 1}',
        '{"kind": "plan", "brief": "x", "language": "NOT A TAG"}',
        '{"kind": "nope", "brief": "x"}',
        '{"brief": "x"}',
    ],
)
def test_malformed_plan_bodies_are_refused(body: str) -> None:
    with pytest.raises(msgspec.ValidationError):
        _decode(body)


def test_the_brief_is_limited_to_4000_characters() -> None:
    with pytest.raises(msgspec.ValidationError):
        _decode('{"kind": "plan", "brief": "' + "x" * 4001 + '"}')
    assert _decode('{"kind": "plan", "brief": "' + "x" * 4000 + '"}')


def test_memory_ids_are_decoded_as_uuids() -> None:
    memory_id = uuid4()
    request = _decode(f'{{"kind": "plan", "brief": "x", "memory_ids": ["{memory_id}"]}}')
    assert request.memory_ids == (memory_id,)  # type: ignore[union-attr]


def test_the_brief_is_normalized() -> None:
    normalized = normalize_plan_run_request(PlanRunRequest(brief="  Café\r\nbar  "))
    assert normalized.brief == "Café\nbar"


def test_a_blank_brief_is_refused() -> None:
    with pytest.raises(StudioDocumentInvalid) as caught:
        normalize_plan_run_request(PlanRunRequest(brief=" \n\t "))
    assert [(i.path, i.code) for i in caught.value.issues] == [
        ("brief", "empty_after_normalization")
    ]


def test_a_brief_that_grows_past_the_limit_under_nfc_is_refused() -> None:
    # U+0958 expands to two code points: 2001 of them are 4002 after NFC.
    request = PlanRunRequest(brief="क़" * 2001)
    with pytest.raises(StudioDocumentInvalid) as caught:
        normalize_plan_run_request(request)
    assert [(i.path, i.code) for i in caught.value.issues] == [
        ("brief", "too_long_after_normalization")
    ]
