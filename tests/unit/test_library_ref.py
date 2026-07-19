"""Unit tests for src/core/library_ref.py — asset reference parsing."""

from __future__ import annotations

from uuid import UUID, uuid4

import pytest

from src.core.library_ref import (
    AssetRef,
    LibraryAssetSource,
    format_asset_ref,
    parse_asset_ref,
)

pytestmark = pytest.mark.unit


class TestParseAssetRef:
    def test_parses_upload_ref(self) -> None:
        asset_id = uuid4()
        ref = parse_asset_ref(f"upload:{asset_id}")
        assert ref == AssetRef(source=LibraryAssetSource.UPLOAD, asset_id=asset_id)

    def test_parses_output_ref(self) -> None:
        asset_id = uuid4()
        ref = parse_asset_ref(f"output:{asset_id}")
        assert ref == AssetRef(source=LibraryAssetSource.OUTPUT, asset_id=asset_id)

    def test_rejects_unknown_source(self) -> None:
        with pytest.raises(ValueError, match="Invalid asset reference"):
            parse_asset_ref(f"generation:{uuid4()}")

    def test_rejects_bad_uuid(self) -> None:
        with pytest.raises(ValueError, match="Invalid asset reference"):
            parse_asset_ref("upload:not-a-uuid")

    def test_rejects_empty_source(self) -> None:
        with pytest.raises(ValueError, match="Invalid asset reference"):
            parse_asset_ref(f":{uuid4()}")

    def test_rejects_empty_uuid_segment(self) -> None:
        with pytest.raises(ValueError, match="Invalid asset reference"):
            parse_asset_ref("upload:")

    def test_rejects_no_colon(self) -> None:
        with pytest.raises(ValueError, match="Invalid asset reference"):
            parse_asset_ref("uploadwithoutcolon")

    def test_rejects_extra_colon_in_uuid_segment(self) -> None:
        with pytest.raises(ValueError, match="Invalid asset reference"):
            parse_asset_ref(f"upload:{uuid4()}:extra")

    def test_rejects_empty_string(self) -> None:
        with pytest.raises(ValueError, match="Invalid asset reference"):
            parse_asset_ref("")

    def test_uppercase_uuid_tolerated(self) -> None:
        asset_id = uuid4()
        ref = parse_asset_ref(f"upload:{str(asset_id).upper()}")
        assert ref.asset_id == asset_id


class TestFormatAssetRef:
    def test_formats_upload(self) -> None:
        asset_id = uuid4()
        assert format_asset_ref(LibraryAssetSource.UPLOAD, asset_id) == f"upload:{asset_id}"

    def test_formats_output(self) -> None:
        asset_id = uuid4()
        assert format_asset_ref(LibraryAssetSource.OUTPUT, asset_id) == f"output:{asset_id}"

    def test_round_trip(self) -> None:
        asset_id = uuid4()
        for source in LibraryAssetSource:
            raw = format_asset_ref(source, asset_id)
            assert parse_asset_ref(raw) == AssetRef(source=source, asset_id=asset_id)


def test_asset_ref_is_frozen() -> None:
    ref = AssetRef(source=LibraryAssetSource.UPLOAD, asset_id=UUID(int=0))
    with pytest.raises(AttributeError):
        ref.asset_id = uuid4()  # type: ignore[misc]
