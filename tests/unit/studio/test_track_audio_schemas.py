"""Track audio wire schemas, limits, normalization and wiring (S2a, W7 / W8 / W11 unit part)."""

from __future__ import annotations

import re
import unicodedata
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING
from unittest.mock import MagicMock
from uuid import uuid4

import msgspec
import pytest

from src.api.dependencies.common import get_studio_track_audio_service, get_studio_track_service
from src.api.schemas.pagination import decode_version_number_cursor, encode_version_number_cursor
from src.api.schemas.studio import CurrentPut, TrackAudioCreate, TrackAudioPatch, TrackDetail
from src.core.enums import StudioTrackAudioOrigin
from src.studio import constraints
from src.studio.constraints import collect_length_issues
from src.studio.services.track import TrackService
from src.studio.services.track_audio import TrackAudioService, _tidy

if TYPE_CHECKING:
    from collections.abc import Mapping

    from src.studio.errors import DocumentIssue

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[3]
ASSET = f"upload:{uuid4()}"
VERSION = uuid4()


def decode[T](payload: Mapping[str, object], type_: type[T]) -> T:
    return msgspec.json.decode(msgspec.json.encode(payload), type=type_)


class TestCreate:
    def test_only_the_asset_is_required(self) -> None:
        body = decode({"asset": ASSET}, TrackAudioCreate)

        assert body.asset == ASSET
        assert body.label is None
        assert body.plan_version_id is None
        assert body.lyrics_version_id is None

    def test_every_field_round_trips(self) -> None:
        plan, lyrics = uuid4(), uuid4()
        body = decode(
            {
                "asset": ASSET,
                "label": "Take 2",
                "plan_version_id": str(plan),
                "lyrics_version_id": str(lyrics),
            },
            TrackAudioCreate,
        )

        assert (body.label, body.plan_version_id, body.lyrics_version_id) == (
            "Take 2",
            plan,
            lyrics,
        )

    @pytest.mark.parametrize("payload", [{}, {"asset": ""}, {"asset": None}, {"label": "x"}])
    def test_the_asset_is_required_and_not_empty(self, payload: Mapping[str, object]) -> None:
        with pytest.raises(msgspec.ValidationError):
            decode(payload, TrackAudioCreate)

    @pytest.mark.parametrize(
        "unknown", ["asset_ref", "origin", "hold_id", "duration_ms", "version_number"]
    )
    def test_unknown_fields_are_refused(self, unknown: str) -> None:
        with pytest.raises(msgspec.ValidationError, match="unknown field"):
            decode({"asset": ASSET, unknown: 1}, TrackAudioCreate)

    def test_the_ids_must_be_uuids(self) -> None:
        with pytest.raises(msgspec.ValidationError):
            decode({"asset": ASSET, "plan_version_id": "nope"}, TrackAudioCreate)


class TestPatch:
    def test_an_omitted_field_is_unset_and_null_is_none(self) -> None:
        empty = decode({}, TrackAudioPatch)
        cleared = decode(
            {"label": None, "plan_version_id": None, "lyrics_version_id": None}, TrackAudioPatch
        )

        assert empty.label is msgspec.UNSET
        assert empty.plan_version_id is msgspec.UNSET
        assert empty.lyrics_version_id is msgspec.UNSET
        assert cleared.label is None
        assert cleared.plan_version_id is None
        assert cleared.lyrics_version_id is None

    def test_values_are_set(self) -> None:
        body = decode({"label": "Final", "plan_version_id": str(VERSION)}, TrackAudioPatch)

        assert body.label == "Final"
        assert body.plan_version_id == VERSION
        assert body.lyrics_version_id is msgspec.UNSET

    @pytest.mark.parametrize(
        "field", ["asset", "origin", "hold_id", "version_number", "duration_ms"]
    )
    def test_the_asset_is_immutable_an_asset_field_is_refused(self, field: str) -> None:
        with pytest.raises(msgspec.ValidationError, match="unknown field"):
            decode({field: ASSET}, TrackAudioPatch)

    def test_an_empty_label_is_refused_on_the_wire(self) -> None:
        with pytest.raises(msgspec.ValidationError):
            decode({"label": ""}, TrackAudioPatch)


class TestLabelLimit:
    """The limit lives on the Struct ``Meta`` only; each boundary holds exactly."""

    def test_the_constants_are_the_documented_limits(self) -> None:
        assert (constraints.AUDIO_LABEL_MAX, constraints.MAX_AUDIO_VERSIONS_PER_TRACK) == (60, 20)

    @pytest.mark.parametrize("struct", [TrackAudioCreate, TrackAudioPatch])
    def test_the_label_accepts_its_limit_and_rejects_one_more(
        self, struct: type[TrackAudioCreate | TrackAudioPatch]
    ) -> None:
        payload: dict[str, object] = {"asset": ASSET} if struct is TrackAudioCreate else {}

        label = "x" * constraints.AUDIO_LABEL_MAX
        ok = decode({**payload, "label": label}, struct)
        assert ok.label == label
        with pytest.raises(msgspec.ValidationError):
            decode({**payload, "label": "x" * (constraints.AUDIO_LABEL_MAX + 1)}, struct)

    def test_nothing_but_the_struct_meta_declares_the_length(self) -> None:
        """No literal copy of the limit in the schema line, the service or the model."""
        schemas = (ROOT / "src/api/schemas/studio.py").read_text()
        meta = [line for line in schemas.splitlines() if line.startswith("_AudioLabel = ")]
        assert len(meta) == 1
        assert "AUDIO_LABEL_MAX" in meta[0]
        assert not re.search(r"max_length=\d", meta[0])
        service = (ROOT / "src/studio/services/track_audio.py").read_text()
        assert not re.search(r"max_length|\b60\b", service)
        model = (ROOT / "src/db/models/studio.py").read_text()
        assert "String(AUDIO_LABEL_MAX)" in model

    def test_nfc_can_push_a_label_past_its_limit_and_the_walker_reports_it(self) -> None:
        # U+0958 is one code point as sent and two after NFC (U+0915 U+093C): 31 -> 62 > 60.
        sent = TrackAudioCreate(asset=ASSET, label="\u0958" * 31)
        msgspec.json.decode(msgspec.json.encode(sent), type=TrackAudioCreate)
        normalized = msgspec.structs.replace(
            sent, label=unicodedata.normalize("NFC", sent.label or "")
        )
        issues: list[DocumentIssue] = []

        collect_length_issues(normalized, issues)

        assert [(i.path, i.code) for i in issues] == [("label", "too_long_after_normalization")]


class TestTidy:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("  Take   2  ", "Take 2"),
            ("a\t\nb", "a b"),
            ("é", "é"),  # NFC
            ("   ", ""),
        ],
    )
    def test_nfc_trim_and_collapse(self, raw: str, expected: str) -> None:
        assert _tidy(raw) == expected


class TestCurrentPut:
    def test_the_audio_pointer_is_tri_state(self) -> None:
        omitted = decode({}, CurrentPut)
        cleared = decode({"audio_version_id": None}, CurrentPut)
        set_ = decode({"audio_version_id": str(VERSION)}, CurrentPut)

        assert omitted.audio_version_id is msgspec.UNSET
        assert cleared.audio_version_id is None
        assert set_.audio_version_id == VERSION

    def test_the_plan_and_lyrics_pointers_are_unchanged(self) -> None:
        body = decode({"plan_version_id": str(VERSION), "lyrics_version_id": None}, CurrentPut)

        assert body.plan_version_id == VERSION
        assert body.lyrics_version_id is None
        assert body.audio_version_id is msgspec.UNSET

    def test_unknown_fields_are_still_refused(self) -> None:
        with pytest.raises(msgspec.ValidationError, match="unknown field"):
            decode({"audio": str(VERSION)}, CurrentPut)


class TestTrackDetail:
    def test_it_carries_the_current_audio_pointer_and_the_item(self) -> None:
        fields = set(TrackDetail.__struct_fields__)

        assert {"current_audio_version_id", "current_audio"} <= fields

    def test_the_origin_is_upload_only_until_s2b(self) -> None:
        assert {member.value for member in StudioTrackAudioOrigin} == {"upload"}


class TestCursor:
    def test_it_round_trips_a_version_number(self) -> None:
        assert decode_version_number_cursor(encode_version_number_cursor(7)) == 7

    @pytest.mark.parametrize(
        "cursor",
        ["", "nope", "e30=", "eyJ2ZXJzaW9uX251bWJlciI6IDB9", "eyJ2ZXJzaW9uX251bWJlciI6ICJ4In0="],
    )
    def test_a_malformed_cursor_is_refused(self, cursor: str) -> None:
        with pytest.raises(ValueError, match="Invalid pagination cursor"):
            decode_version_number_cursor(cursor)

    def test_a_cursor_of_another_endpoint_is_refused(self) -> None:
        from src.api.schemas.pagination import encode_cursor

        other = encode_cursor(datetime.now(UTC), uuid4())
        with pytest.raises(ValueError, match="Invalid pagination cursor"):
            decode_version_number_cursor(other)

    def test_a_boolean_is_not_a_version_number(self) -> None:
        import base64
        import json

        token = base64.urlsafe_b64encode(json.dumps({"version_number": True}).encode()).decode()
        with pytest.raises(ValueError, match="positive integer"):
            decode_version_number_cursor(token)


class TestWiring:
    def test_the_audio_service_gets_a_hold_service_with_the_settings_retention_days(self) -> None:
        session = MagicMock()

        service = get_studio_track_audio_service(session, SimpleNamespace(retention_days=11))  # type: ignore[arg-type]

        assert isinstance(service, TrackAudioService)
        assert service._holds._retention_days == 11
        assert service._holds._session is session
        assert service._session is session  # one session: a hold and its version commit together

    def test_the_track_service_gets_one_too_for_the_delete_release(self) -> None:
        session = MagicMock()

        service = get_studio_track_service(session, SimpleNamespace(retention_days=5))  # type: ignore[arg-type]

        assert isinstance(service, TrackService)
        assert service._holds._retention_days == 5
        assert service._holds._session is session

    def test_the_audio_service_adds_no_setting_and_reads_none(self) -> None:
        text = (ROOT / "src/studio/services/track_audio.py").read_text()

        assert "Settings" not in text
        assert "retention_days" not in text  # the hold service is built with it by DI
