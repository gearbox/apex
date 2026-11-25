"""Cast wire schemas, limits, errors and wiring (S1-P3, W7 / W10 unit part)."""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING
from unittest.mock import MagicMock
from uuid import uuid4

import msgspec
import pytest

from src.api.app import asset_held_handler
from src.api.dependencies.common import get_studio_cast_service
from src.api.schemas.studio import (
    ArtistCreate,
    ArtistPatch,
    CastLookIn,
    CastLookPatch,
    CastMemberCreate,
    CastMemberPatch,
    CastReferenceIn,
    CastReferenceOrder,
    CastReferencePatch,
)
from src.api.services.library_holds import AssetHeldError, HolderCount
from src.studio import constraints
from src.studio.constraints import collect_length_issues
from src.studio.errors import (
    AssetNotEligible,
    AssetNotFound,
    CastLabelTaken,
    CastLimitReached,
    CastLookNameTaken,
    CastLookNotFound,
    CastMemberInUse,
    CastMemberNotFound,
    CastNameTaken,
    CastReferenceNotFound,
    DocumentIssue,
    InvalidAssetRef,
    InvalidReferenceOrder,
)
from src.studio.services.cast import CastService, _tidy

if TYPE_CHECKING:
    from collections.abc import Mapping

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[3]
ASSET = f"upload:{uuid4()}"


def decode[T](payload: Mapping[str, object], type_: type[T]) -> T:
    return msgspec.json.decode(msgspec.json.encode(payload), type=type_)


class TestLimitsOnTheWire:
    """Each limit holds exactly at its boundary: ``limit`` passes, ``limit + 1`` is rejected."""

    @pytest.mark.parametrize(
        ("build", "field", "limit"),
        [
            (lambda v: (CastMemberCreate, {"name": v}), "name", constraints.CAST_NAME_MAX),
            (
                lambda v: (CastMemberCreate, {"name": "ok", "description": v}),
                "description",
                constraints.CAST_DESCRIPTION_MAX,
            ),
            (
                lambda v: (CastMemberCreate, {"name": "ok", "appearance": v}),
                "appearance",
                constraints.CAST_APPEARANCE_MAX,
            ),
            (
                lambda v: (CastReferenceIn, {"label": v, "asset": ASSET}),
                "label",
                constraints.CAST_LABEL_MAX,
            ),
            (
                lambda v: (CastLookIn, {"name": v, "reference_ids": [str(uuid4())]}),
                "look name",
                constraints.CAST_LOOK_NAME_MAX,
            ),
            (lambda v: (CastMemberPatch, {"name": v}), "patch name", constraints.CAST_NAME_MAX),
            (
                lambda v: (CastMemberPatch, {"appearance": v}),
                "patch appearance",
                constraints.CAST_APPEARANCE_MAX,
            ),
            (
                lambda v: (CastReferencePatch, {"label": v}),
                "patch label",
                constraints.CAST_LABEL_MAX,
            ),
            (
                lambda v: (CastLookPatch, {"name": v}),
                "patch look name",
                constraints.CAST_LOOK_NAME_MAX,
            ),
        ],
    )
    def test_a_text_field_accepts_its_limit_and_rejects_one_more(
        self, build: object, field: str, limit: int
    ) -> None:
        type_, ok_payload = build("x" * limit)  # type: ignore[operator]
        decode(ok_payload, type_)
        _, bad_payload = build("x" * (limit + 1))  # type: ignore[operator]
        with pytest.raises(msgspec.ValidationError):
            decode(bad_payload, type_)
        assert field

    @pytest.mark.parametrize(
        ("type_", "payload"),
        [
            (CastMemberCreate, {"name": ""}),
            (CastReferenceIn, {"label": "", "asset": ASSET}),
            (CastReferenceIn, {"label": "x", "asset": ""}),
            (CastLookIn, {"name": "", "reference_ids": []}),
        ],
    )
    def test_required_text_must_not_be_empty_on_the_wire(
        self, type_: type[msgspec.Struct], payload: dict[str, object]
    ) -> None:
        with pytest.raises(msgspec.ValidationError):
            decode(payload, type_)

    def test_at_most_twelve_references_on_create_twelve_ids_in_a_look_or_an_order(self) -> None:
        refs = [{"label": f"l{i}", "asset": ASSET} for i in range(constraints.MAX_CAST_REFERENCES)]
        ids = [str(uuid4()) for _ in range(constraints.MAX_CAST_REFERENCES)]
        decode({"name": "ok", "references": refs}, CastMemberCreate)
        decode({"name": "ok", "reference_ids": ids}, CastLookIn)
        decode({"reference_ids": ids}, CastReferenceOrder)
        decode({"reference_ids": ids}, CastLookPatch)
        too_many_refs = [*refs, {"label": "13", "asset": ASSET}]
        too_many_ids = [*ids, str(uuid4())]
        for type_, payload in (
            (CastMemberCreate, {"name": "ok", "references": too_many_refs}),
            (CastLookIn, {"name": "ok", "reference_ids": too_many_ids}),
            (CastReferenceOrder, {"reference_ids": too_many_ids}),
            (CastLookPatch, {"reference_ids": too_many_ids}),
        ):
            with pytest.raises(msgspec.ValidationError):
                decode(payload, type_)

    def test_ids_must_be_uuids(self) -> None:
        with pytest.raises(msgspec.ValidationError):
            decode({"reference_ids": ["nope"]}, CastReferenceOrder)

    def test_the_constants_are_the_documented_limits(self) -> None:
        assert (
            constraints.MAX_CAST_MEMBERS_PER_OWNER,
            constraints.MAX_CAST_REFERENCES,
            constraints.MAX_CAST_LOOKS,
            constraints.CAST_NAME_MAX,
            constraints.CAST_LABEL_MAX,
            constraints.CAST_LOOK_NAME_MAX,
            constraints.CAST_APPEARANCE_MAX,
            constraints.CAST_DESCRIPTION_MAX,
        ) == (50, 12, 12, 100, 40, 60, 320, 2000)

    def test_the_wire_structs_and_the_models_use_the_constants_and_nothing_else_copies_them(
        self,
    ) -> None:
        """The limits live in ``constraints.py``; no literal copy of them anywhere else."""
        schemas = (ROOT / "src/api/schemas/studio.py").read_text()
        cast_meta = [line for line in schemas.splitlines() if re.match(r"_Cast\w+ = ", line)]
        assert cast_meta
        for line in cast_meta:
            assert not re.search(r"max_length=\d", line), line
        service = (ROOT / "src/studio/services/cast.py").read_text()
        assert not re.search(r"max_length|\b(?:100|320|2_?000)\b", service)
        models = (ROOT / "src/db/models/studio_cast.py").read_text()
        assert not re.search(r"String\((?:100|40|60|320)\)", models)  # from the constants


class TestNormalizationExpandsPastTheWireLimit:
    @pytest.mark.parametrize(
        ("struct", "path"),
        [
            (CastMemberCreate(name="क़" * 60), "name"),
            (CastMemberCreate(name="ok", appearance="क़" * 161), "appearance"),
            (CastMemberCreate(name="ok", description="क़" * 1001), "description"),
            (CastReferenceIn(label="क़" * 21, asset=ASSET), "label"),
            (CastLookIn(name="क़" * 31, reference_ids=[]), "name"),
        ],
    )
    def test_nfc_can_push_a_value_past_its_limit_and_the_walker_reports_it(
        self, struct: msgspec.Struct, path: str
    ) -> None:
        # as sent it is within the limit: the wire accepted it ...
        msgspec.json.decode(msgspec.json.encode(struct), type=type(struct))
        # ... normalized (NFC) it is not
        normalized = msgspec.structs.replace(
            struct,
            **{
                name: unicodedata.normalize("NFC", value)
                for name in struct.__struct_fields__
                if isinstance(value := getattr(struct, name), str) and name in {path, "name"}
            },
        )
        issues: list[DocumentIssue] = []

        collect_length_issues(normalized, issues)

        assert [(i.path, i.code) for i in issues] == [(path, "too_long_after_normalization")]


class TestTriStatePatches:
    def test_an_omitted_field_is_unset_and_null_is_none(self) -> None:
        empty = decode({}, CastMemberPatch)
        cleared = decode({"description": None, "appearance": None}, CastMemberPatch)

        assert empty.name is msgspec.UNSET
        assert empty.description is msgspec.UNSET
        assert empty.appearance is msgspec.UNSET
        assert cleared.description is None
        assert cleared.appearance is None
        assert cleared.name is msgspec.UNSET

    def test_a_name_cannot_be_cleared_with_null(self) -> None:
        with pytest.raises(msgspec.ValidationError):
            decode({"name": None}, CastMemberPatch)
        with pytest.raises(msgspec.ValidationError):
            decode({"label": None}, CastReferencePatch)
        with pytest.raises(msgspec.ValidationError):
            decode({"reference_ids": None}, CastLookPatch)

    def test_the_artist_performer_is_tri_state_too(self) -> None:
        member = uuid4()

        assert decode({}, ArtistPatch).performer_cast_member_id is msgspec.UNSET
        assert (
            decode({"performer_cast_member_id": None}, ArtistPatch).performer_cast_member_id is None
        )
        assert (
            decode({"performer_cast_member_id": str(member)}, ArtistPatch).performer_cast_member_id
            == member
        )
        assert decode({"name": "x"}, ArtistCreate).performer_cast_member_id is None
        with pytest.raises(msgspec.ValidationError):
            decode({"performer_cast_member_id": "nope"}, ArtistPatch)


class TestRequestsForbidUnknownFields:
    @pytest.mark.parametrize(
        ("type_", "payload"),
        [
            (CastMemberCreate, {"name": "x", "extra": 1}),
            (CastMemberPatch, {"extra": 1}),
            (CastReferenceIn, {"label": "x", "asset": ASSET, "extra": 1}),
            (CastReferencePatch, {"extra": 1}),
            (CastReferenceOrder, {"reference_ids": [], "extra": 1}),
            (CastLookIn, {"name": "x", "reference_ids": [], "extra": 1}),
            (CastLookPatch, {"extra": 1}),
        ],
    )
    def test_an_unknown_field_is_rejected(
        self, type_: type[msgspec.Struct], payload: dict[str, object]
    ) -> None:
        with pytest.raises(msgspec.ValidationError, match="unknown field"):
            decode(payload, type_)


class TestTidy:
    def test_it_composes_trims_and_collapses_whitespace(self) -> None:
        assert _tidy("  Mira \t\n  Vale ") == "Mira Vale"
        assert _tidy("é") == "é"  # NFC
        assert _tidy("   ") == ""


class TestErrors:
    @pytest.mark.parametrize(
        ("error", "code", "status"),
        [
            (CastMemberNotFound(), "cast_member_not_found", 404),
            (CastReferenceNotFound(), "cast_reference_not_found", 404),
            (CastLookNotFound(), "cast_look_not_found", 404),
            (CastNameTaken(), "cast_name_taken", 409),
            (CastLabelTaken(), "cast_label_taken", 409),
            (CastLookNameTaken(), "cast_look_name_taken", 409),
            (CastLimitReached("references", 12), "cast_limit_reached", 409),
            (CastMemberInUse(2), "cast_member_in_use", 409),
            (InvalidReferenceOrder(), "invalid_order", 422),
            (AssetNotFound(), "asset_not_found", 404),
            (AssetNotEligible("derivative"), "asset_not_eligible", 422),
            (InvalidAssetRef(), "invalid_asset_ref", 400),
        ],
    )
    def test_codes_and_statuses(self, error: Exception, code: str, status: int) -> None:
        assert (error.code, error.status_code) == (code, status)  # type: ignore[attr-defined]

    def test_details(self) -> None:
        assert CastLimitReached("looks", 12).detail == {"kind": "looks", "limit": 12}
        assert CastMemberInUse(3).detail == {"blockers": [{"code": "artists", "count": 3}]}
        assert AssetNotEligible("media_kind_not_accepted").detail == {
            "reason": "media_kind_not_accepted"
        }
        assert CastMemberNotFound().detail is None

    def test_messages_are_static(self) -> None:
        for error in (CastNameTaken(), CastMemberInUse(1), AssetNotEligible("derivative")):
            assert "derivative" not in error.message
            assert "1" not in error.message


class TestAssetHeldHandler:
    def test_it_answers_409_with_kinds_and_counts_only(self) -> None:
        request = MagicMock()
        request.url.path = "/v1/library/assets/upload:x"
        request.method = "DELETE"

        response = asset_held_handler(
            request,
            AssetHeldError([HolderCount("a.first", 1), HolderCount("studio.cast_reference", 2)]),
        )

        assert response.status_code == 409
        body = msgspec.to_builtins(response.content)
        assert body["error"] == "asset_held"
        assert body["detail"] == {
            "holders": [
                {"kind": "a.first", "count": 1},
                {"kind": "studio.cast_reference", "count": 2},
            ]
        }


class TestWiring:
    def test_the_cast_service_gets_a_hold_service_with_the_settings_retention_days(self) -> None:
        session = MagicMock()

        service = get_studio_cast_service(session, SimpleNamespace(retention_days=11))  # type: ignore[arg-type]

        assert isinstance(service, CastService)
        assert service._holds._retention_days == 11
        assert service._holds._session is session
        assert service._session is session  # one session: a hold and its reference commit together

    def test_the_cast_adds_no_setting_and_reads_none(self) -> None:
        text = (ROOT / "src/studio/services/cast.py").read_text()
        assert "get_settings" not in text
        assert "Settings" not in text
