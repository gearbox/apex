"""Unit tests for Library Phase 2 (projects/bulk) pure-logic pieces.

Covers: LIKE escaping, bulk tagged-union decoding, tri-state PATCH parsing,
project name normalization, and stale-cursor rejection across sorts —
all pure/decoding logic requiring no database.
"""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

import msgspec
import pytest

from src.api.schemas.library import (
    BulkDelete,
    BulkOperation,
    BulkSetFavorite,
    BulkSetProject,
    LibraryAssetPatch,
    LibraryProjectPatch,
)
from src.api.schemas.pagination import decode_library_cursor, encode_library_cursor
from src.api.services.library_project import LibraryProjectService, LibraryProjectValidationError
from src.db.repositories.library import _escape_like_term

pytestmark = pytest.mark.unit


class TestEscapeLikeTerm:
    def test_plain_text_unchanged(self) -> None:
        assert _escape_like_term("sunset") == "sunset"

    def test_percent_escaped(self) -> None:
        assert _escape_like_term("100%done") == "100\\%done"

    def test_underscore_escaped(self) -> None:
        assert _escape_like_term("a_b") == "a\\_b"

    def test_backslash_escaped_first(self) -> None:
        # A literal backslash must become \\ BEFORE % / _ escaping runs, or
        # a raw "\%" in the input would be mis-paired with the escaping below.
        assert _escape_like_term("a\\b") == "a\\\\b"

    def test_combined_metacharacters(self) -> None:
        assert _escape_like_term("50%_off\\sale") == "50\\%\\_off\\\\sale"

    def test_empty_string(self) -> None:
        assert _escape_like_term("") == ""


class TestBulkOperationDecoding:
    def test_decodes_set_favorite(self) -> None:
        ref = f"upload:{uuid4()}"
        raw = msgspec.json.encode({"type": "set_favorite", "asset_refs": [ref], "value": True})
        op = msgspec.json.decode(raw, type=BulkOperation)
        assert isinstance(op, BulkSetFavorite)
        assert op.asset_refs == [ref]
        assert op.value is True

    def test_decodes_set_project_with_null_project_id(self) -> None:
        ref = f"output:{uuid4()}"
        raw = msgspec.json.encode({"type": "set_project", "asset_refs": [ref], "project_id": None})
        op = msgspec.json.decode(raw, type=BulkOperation)
        assert isinstance(op, BulkSetProject)
        assert op.project_id is None

    def test_decodes_set_project_with_project_id(self) -> None:
        ref = f"output:{uuid4()}"
        project_id = uuid4()
        raw = msgspec.json.encode(
            {"type": "set_project", "asset_refs": [ref], "project_id": str(project_id)}
        )
        op = msgspec.json.decode(raw, type=BulkOperation)
        assert isinstance(op, BulkSetProject)
        assert op.project_id == project_id

    def test_decodes_delete(self) -> None:
        ref = f"upload:{uuid4()}"
        raw = msgspec.json.encode({"type": "delete", "asset_refs": [ref]})
        op = msgspec.json.decode(raw, type=BulkOperation)
        assert isinstance(op, BulkDelete)

    def test_unknown_tag_rejected(self) -> None:
        raw = msgspec.json.encode({"type": "bogus", "asset_refs": ["upload:x"]})
        with pytest.raises(msgspec.ValidationError):
            msgspec.json.decode(raw, type=BulkOperation)

    def test_empty_asset_refs_rejected(self) -> None:
        raw = msgspec.json.encode({"type": "set_favorite", "asset_refs": [], "value": True})
        with pytest.raises(msgspec.ValidationError):
            msgspec.json.decode(raw, type=BulkOperation)

    def test_over_100_asset_refs_rejected(self) -> None:
        refs = [f"upload:{uuid4()}" for _ in range(101)]
        raw = msgspec.json.encode({"type": "set_favorite", "asset_refs": refs, "value": True})
        with pytest.raises(msgspec.ValidationError):
            msgspec.json.decode(raw, type=BulkOperation)

    def test_exactly_100_asset_refs_accepted(self) -> None:
        refs = [f"upload:{uuid4()}" for _ in range(100)]
        raw = msgspec.json.encode({"type": "set_favorite", "asset_refs": refs, "value": True})
        op = msgspec.json.decode(raw, type=BulkOperation)
        assert isinstance(op, BulkSetFavorite)
        assert len(op.asset_refs) == 100


class TestTriStatePatchParsing:
    def test_asset_patch_absent_fields_are_unset(self) -> None:
        patch = msgspec.json.decode(b"{}", type=LibraryAssetPatch)
        assert patch.display_title is msgspec.UNSET
        assert patch.project_id is msgspec.UNSET

    def test_asset_patch_null_project_id_means_unassign(self) -> None:
        patch = msgspec.json.decode(b'{"project_id": null}', type=LibraryAssetPatch)
        assert patch.project_id is None

    def test_asset_patch_project_id_assign(self) -> None:
        project_id = uuid4()
        raw = msgspec.json.encode({"project_id": str(project_id)})
        patch = msgspec.json.decode(raw, type=LibraryAssetPatch)
        assert patch.project_id == project_id

    def test_project_patch_absent_fields_are_unset(self) -> None:
        patch = msgspec.json.decode(b"{}", type=LibraryProjectPatch)
        assert patch.name is msgspec.UNSET
        assert patch.description is msgspec.UNSET

    def test_project_patch_null_description_means_clear(self) -> None:
        patch = msgspec.json.decode(b'{"description": null}', type=LibraryProjectPatch)
        assert patch.description is None

    def test_project_patch_name_too_long_rejected(self) -> None:
        raw = msgspec.json.encode({"name": "x" * 101})
        with pytest.raises(msgspec.ValidationError):
            msgspec.json.decode(raw, type=LibraryProjectPatch)


class TestProjectNameNormalization:
    def test_trims_leading_trailing_whitespace(self) -> None:
        assert LibraryProjectService._normalize_name("  My Project  ") == "My Project"

    def test_collapses_inner_whitespace(self) -> None:
        assert LibraryProjectService._normalize_name("My   New  Project") == "My New Project"

    def test_all_whitespace_raises(self) -> None:
        with pytest.raises(LibraryProjectValidationError):
            LibraryProjectService._normalize_name("     ")

    def test_normal_name_unchanged(self) -> None:
        assert LibraryProjectService._normalize_name("Vacation Photos") == "Vacation Photos"


class TestStaleCursorRejectionAcrossSorts:
    def test_cursor_from_newest_rejected_under_expiring_soon(self) -> None:
        ts = datetime(2026, 1, 1, tzinfo=UTC)
        cursor = encode_library_cursor(ts, "upload", uuid4(), sort="newest")
        with pytest.raises(ValueError, match="Invalid pagination cursor"):
            decode_library_cursor(cursor, expected_sort="expiring_soon")

    def test_cursor_from_expiring_soon_rejected_under_oldest(self) -> None:
        ts = datetime(2026, 1, 1, tzinfo=UTC)
        cursor = encode_library_cursor(ts, "output", uuid4(), sort="expiring_soon")
        with pytest.raises(ValueError, match="Invalid pagination cursor"):
            decode_library_cursor(cursor, expected_sort="oldest")

    def test_matching_sort_accepted(self) -> None:
        ts = datetime(2026, 1, 1, tzinfo=UTC)
        id_ = uuid4()
        cursor = encode_library_cursor(ts, "upload", id_, sort="expiring_soon")
        decoded_ts, decoded_source, decoded_id = decode_library_cursor(
            cursor, expected_sort="expiring_soon"
        )
        assert decoded_ts == ts
        assert decoded_source == "upload"
        assert decoded_id == id_

    def test_no_expected_sort_skips_the_check(self) -> None:
        """Backward-compat: callers that don't pass expected_sort keep old behavior."""
        ts = datetime(2026, 1, 1, tzinfo=UTC)
        cursor = encode_library_cursor(ts, "upload", uuid4(), sort="newest")
        # No expected_sort passed — must not raise despite sort='newest' embedded.
        decode_library_cursor(cursor)

    def test_legacy_cursor_without_sort_field_defaults_to_newest(self) -> None:
        """A cursor JSON payload with no "sort" key (pre-Phase-2 shape) is treated as newest."""
        import base64
        import json

        payload = {
            "created_at": datetime(2026, 1, 1, tzinfo=UTC).isoformat(),
            "source": "upload",
            "id": str(uuid4()),
        }
        legacy_cursor = base64.urlsafe_b64encode(json.dumps(payload).encode()).decode()
        decode_library_cursor(legacy_cursor, expected_sort="newest")
        with pytest.raises(ValueError, match="Invalid pagination cursor"):
            decode_library_cursor(legacy_cursor, expected_sort="oldest")
