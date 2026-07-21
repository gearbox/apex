"""Unit tests for Library Phase 3 (tags/lineage) pure-logic pieces.

Covers: shared owner-scoped name normalization (both length profiles),
tag bulk-op tagged-union decoding, tri-state tag_ids patch parsing, and
lineage schema/enum shape — all pure/decoding logic requiring no database.
"""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

import msgspec
import pytest

from src.api.schemas.library import (
    BulkAddTags,
    BulkOperation,
    BulkRemoveTags,
    LibraryAssetPatch,
    LibraryLineageGraph,
    LibraryProjectPatch,
    LibraryTagPatch,
    LineageEdge,
    LineageNode,
    LineageRelation,
)
from src.api.schemas.media import MediaObject, MediaOriginal
from src.api.services.library_project import LibraryProjectService
from src.api.services.library_tag import LibraryTagService, LibraryTagValidationError
from src.api.services.owner_scoped_names import normalize_owner_scoped_name
from src.core.enums import OutputMediaType
from src.core.library_limits import MAX_TAGS_PER_ASSET
from src.core.library_ref import LibraryAssetSource

pytestmark = pytest.mark.unit


class TestNormalizeOwnerScopedName:
    def test_trims_leading_trailing_whitespace(self) -> None:
        assert normalize_owner_scoped_name("  My Tag  ", max_length=50) == "My Tag"

    def test_collapses_inner_whitespace(self) -> None:
        assert normalize_owner_scoped_name("My   New  Tag", max_length=50) == "My New Tag"

    def test_all_whitespace_raises(self) -> None:
        with pytest.raises(ValueError, match="1-50 characters"):
            normalize_owner_scoped_name("     ", max_length=50)

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="1-50 characters"):
            normalize_owner_scoped_name("", max_length=50)

    def test_over_max_length_raises(self) -> None:
        with pytest.raises(ValueError, match="1-10 characters"):
            normalize_owner_scoped_name("way too long", max_length=10)

    def test_normal_name_unchanged(self) -> None:
        assert normalize_owner_scoped_name("Sunset", max_length=50) == "Sunset"

    def test_different_max_length_profiles_independent(self) -> None:
        """Same raw input, different bound — project (100) vs tag (50)."""
        raw = "x" * 60
        # Fits under the project bound...
        assert normalize_owner_scoped_name(raw, max_length=100) == raw
        # ...but not the tag bound.
        with pytest.raises(ValueError, match="1-50 characters"):
            normalize_owner_scoped_name(raw, max_length=50)


class TestProjectServiceStillUsesSharedHelper:
    """Existing project tests must pass unchanged after the extraction (per prompt 06)."""

    def test_trims_leading_trailing_whitespace(self) -> None:
        assert LibraryProjectService._normalize_name("  My Project  ") == "My Project"

    def test_collapses_inner_whitespace(self) -> None:
        assert LibraryProjectService._normalize_name("My   New  Project") == "My New Project"

    def test_all_whitespace_raises(self) -> None:
        from src.api.services.library_project import LibraryProjectValidationError

        with pytest.raises(LibraryProjectValidationError):
            LibraryProjectService._normalize_name("     ")

    def test_normal_name_unchanged(self) -> None:
        assert LibraryProjectService._normalize_name("Vacation Photos") == "Vacation Photos"


class TestTagNameNormalization:
    def test_trims_leading_trailing_whitespace(self) -> None:
        assert LibraryTagService._normalize_name("  sunset  ") == "sunset"

    def test_collapses_inner_whitespace(self) -> None:
        assert LibraryTagService._normalize_name("golden   hour") == "golden hour"

    def test_all_whitespace_raises(self) -> None:
        with pytest.raises(LibraryTagValidationError):
            LibraryTagService._normalize_name("     ")

    def test_over_50_chars_raises(self) -> None:
        with pytest.raises(LibraryTagValidationError):
            LibraryTagService._normalize_name("x" * 51)

    def test_normal_name_unchanged(self) -> None:
        assert LibraryTagService._normalize_name("portrait") == "portrait"


class TestBulkTagOperationDecoding:
    def test_decodes_add_tags(self) -> None:
        ref = f"upload:{uuid4()}"
        tag_id = uuid4()
        raw = msgspec.json.encode(
            {"type": "add_tags", "asset_refs": [ref], "tag_ids": [str(tag_id)]}
        )
        op = msgspec.json.decode(raw, type=BulkOperation)
        assert isinstance(op, BulkAddTags)
        assert op.asset_refs == [ref]
        assert op.tag_ids == [tag_id]

    def test_decodes_remove_tags(self) -> None:
        ref = f"output:{uuid4()}"
        tag_id = uuid4()
        raw = msgspec.json.encode(
            {"type": "remove_tags", "asset_refs": [ref], "tag_ids": [str(tag_id)]}
        )
        op = msgspec.json.decode(raw, type=BulkOperation)
        assert isinstance(op, BulkRemoveTags)
        assert op.tag_ids == [tag_id]

    def test_empty_tag_ids_rejected(self) -> None:
        ref = f"upload:{uuid4()}"
        raw = msgspec.json.encode({"type": "add_tags", "asset_refs": [ref], "tag_ids": []})
        with pytest.raises(msgspec.ValidationError):
            msgspec.json.decode(raw, type=BulkOperation)

    def test_over_10_tag_ids_rejected(self) -> None:
        ref = f"upload:{uuid4()}"
        tag_ids = [str(uuid4()) for _ in range(11)]
        raw = msgspec.json.encode({"type": "add_tags", "asset_refs": [ref], "tag_ids": tag_ids})
        with pytest.raises(msgspec.ValidationError):
            msgspec.json.decode(raw, type=BulkOperation)

    def test_exactly_10_tag_ids_accepted(self) -> None:
        ref = f"upload:{uuid4()}"
        tag_ids = [str(uuid4()) for _ in range(10)]
        raw = msgspec.json.encode({"type": "remove_tags", "asset_refs": [ref], "tag_ids": tag_ids})
        op = msgspec.json.decode(raw, type=BulkOperation)
        assert isinstance(op, BulkRemoveTags)
        assert len(op.tag_ids) == 10


class TestTagIdsPatchTriState:
    def test_absent_is_unset(self) -> None:
        patch = msgspec.json.decode(b"{}", type=LibraryAssetPatch)
        assert patch.tag_ids is msgspec.UNSET

    def test_empty_list_means_clear(self) -> None:
        patch = msgspec.json.decode(b'{"tag_ids": []}', type=LibraryAssetPatch)
        assert patch.tag_ids == []

    def test_list_of_ids_sets_exact_set(self) -> None:
        ids = [uuid4(), uuid4()]
        raw = msgspec.json.encode({"tag_ids": [str(i) for i in ids]})
        patch = msgspec.json.decode(raw, type=LibraryAssetPatch)
        assert patch.tag_ids == ids

    def test_null_is_rejected(self) -> None:
        """No None arm — clearing is expressed as [] (T4), not null."""
        with pytest.raises(msgspec.ValidationError):
            msgspec.json.decode(b'{"tag_ids": null}', type=LibraryAssetPatch)

    def test_exactly_20_tag_ids_accepted(self) -> None:
        """S1: the wire bound matches _MAX_TAGS_PER_ASSET exactly."""
        ids = [uuid4() for _ in range(MAX_TAGS_PER_ASSET)]
        raw = msgspec.json.encode({"tag_ids": [str(i) for i in ids]})
        patch = msgspec.json.decode(raw, type=LibraryAssetPatch)
        assert patch.tag_ids == ids

    def test_21_tag_ids_rejected_at_schema_boundary(self) -> None:
        """S1: one over the cap is rejected by msgspec, before the service
        ever sees it — the schema bound, not just the service-layer check."""
        ids = [uuid4() for _ in range(MAX_TAGS_PER_ASSET + 1)]
        raw = msgspec.json.encode({"tag_ids": [str(i) for i in ids]})
        with pytest.raises(msgspec.ValidationError):
            msgspec.json.decode(raw, type=LibraryAssetPatch)


def _openapi_property_schema(struct_type: type, property_name: str) -> dict[str, object]:
    """Build the Litestar-generated OpenAPI property schema for a single
    msgspec.Struct field, via a throwaway app — independent of the real
    ``create_app()``/``Settings`` wiring (docs are off by default there),
    and isolated so this test only exercises struct -> OpenAPI generation.
    """
    from litestar import Litestar, post
    from litestar.openapi.config import OpenAPIConfig

    async def _handler(data):  # type: ignore[no-untyped-def]
        return data

    # Set the annotations dict directly (real type objects, not strings) —
    # this module's `from __future__ import annotations` would otherwise
    # stringify `struct_type` in the signature, and Litestar's
    # `get_type_hints()` can't resolve a closure variable by name.
    _handler.__annotations__ = {"data": struct_type, "return": struct_type}
    handler = post("/x")(_handler)
    app = Litestar([handler], openapi_config=OpenAPIConfig(title="t", version="1"))
    schema = app.openapi_schema.to_schema()
    return schema["components"]["schemas"][struct_type.__name__]["properties"][property_name]  # type: ignore[no-any-return]


class TestOpenApiSchemaBounds:
    """S1/S2: tri-state (``X | UnsetType``) constrained fields must expose
    their bounds in the generated OpenAPI schema — not just enforce them at
    decode time — including on the ``oneOf`` branch a oneOf-strict
    validator/codegen tool actually inspects, not only as a sibling of
    ``oneOf`` on the wrapper."""

    def test_asset_patch_tag_ids_declares_max_items(self) -> None:
        prop = _openapi_property_schema(LibraryAssetPatch, "tag_ids")
        assert prop["maxItems"] == MAX_TAGS_PER_ASSET
        member = prop["oneOf"][0]  # type: ignore[index]
        assert member["maxItems"] == MAX_TAGS_PER_ASSET

    def test_tag_patch_name_declares_length_bounds_on_oneof_member(self) -> None:
        prop = _openapi_property_schema(LibraryTagPatch, "name")
        assert prop["minLength"] == 1
        assert prop["maxLength"] == 50
        member = prop["oneOf"][0]  # type: ignore[index]
        assert member["type"] == "string"
        assert member["minLength"] == 1
        assert member["maxLength"] == 50

    def test_project_patch_name_declares_length_bounds_on_oneof_member(self) -> None:
        prop = _openapi_property_schema(LibraryProjectPatch, "name")
        assert prop["minLength"] == 1
        assert prop["maxLength"] == 100
        member = prop["oneOf"][0]  # type: ignore[index]
        assert member["type"] == "string"
        assert member["minLength"] == 1
        assert member["maxLength"] == 100

    def test_project_patch_description_remains_nullable_oneof(self) -> None:
        """description is a genuine nullable (str | None | UnsetType) — unlike
        name, it must NOT carry length bounds, and its oneOf branches are
        exactly [string, null]."""
        prop = _openapi_property_schema(LibraryProjectPatch, "description")
        one_of = prop["oneOf"]
        assert isinstance(one_of, list)
        types = {member["type"] for member in one_of}
        assert types == {"string", "null"}


class TestLineageSchemaShape:
    def test_relation_enum_values(self) -> None:
        assert LineageRelation.GENERATED_FROM_UPLOAD.value == "generated_from_upload"
        assert LineageRelation.GENERATED_FROM_OUTPUT.value == "generated_from_output"
        assert LineageRelation.FRAME_OF_OUTPUT.value == "frame_of_output"
        assert LineageRelation.FRAME_OF_UPLOAD.value == "frame_of_upload"

    def test_graph_roundtrips_through_json(self) -> None:
        media = MediaObject(
            media_type=OutputMediaType.IMAGE,
            original=MediaOriginal(
                url="/v1/content/outputs/x", content_type="image/png", size_bytes=10
            ),
            variants=[],
        )
        node = LineageNode(
            asset_ref=f"output:{uuid4()}",
            source=LibraryAssetSource.OUTPUT,
            media=media,
            created_at=datetime.now(UTC),
        )
        from src.api.schemas.library import LibraryDescendants

        graph = LibraryLineageGraph(
            focus=node,
            ancestors=(
                LineageEdge(
                    relation=LineageRelation.GENERATED_FROM_UPLOAD,
                    node=node,
                    source_timestamp_ms=None,
                ),
            ),
            descendants=(),
            descendant_totals=LibraryDescendants(job_count=0, frame_count=0),
            ancestors_truncated=False,
            descendants_truncated=False,
        )
        encoded = msgspec.json.encode(graph)
        decoded = msgspec.json.decode(encoded, type=LibraryLineageGraph)
        assert decoded.focus.asset_ref == node.asset_ref
        assert len(decoded.ancestors) == 1
        assert decoded.ancestors[0].relation == LineageRelation.GENERATED_FROM_UPLOAD
