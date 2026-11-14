"""The post-normalization length walker (RD6, invariant 31).

``collect_length_issues`` must visit every ``max_length`` that the wire Structs declare, report
one issue per violating value at the wire path, and nothing else.
"""

from __future__ import annotations

from typing import Annotated, Any
from uuid import UUID

import msgspec
import msgspec.inspect as mi
import pytest
from msgspec import Meta, Struct

from src.api.schemas.studio import (
    ArtistCreate,
    ArtistPatch,
    ProjectProfilePut,
    TrackCreate,
    TrackPatch,
)
from src.studio.constraints import TOO_LONG, TOO_MANY, collect_count_issues, collect_length_issues
from src.studio.documents import LyricsDocument, PlanDocument
from src.studio.errors import DocumentIssue

pytestmark = pytest.mark.unit


# -- local shapes -----------------------------------------------------------------------------


class _Leaf(Struct, frozen=True, kw_only=True):
    word: Annotated[str, Meta(max_length=3)]
    count: int = 0


class _Shape(Struct, frozen=True, kw_only=True):
    name: Annotated[str, Meta(max_length=5)]
    nickname: Annotated[str, Meta(max_length=4)] | None = None
    unbounded: str = ""
    leaf: _Leaf | None = None
    leaves: Annotated[tuple[_Leaf, ...], Meta(max_length=4)] = ()
    words: list[Annotated[str, Meta(max_length=2)]] = msgspec.field(default_factory=list)
    either: Annotated[str, Meta(max_length=2)] | list[Annotated[str, Meta(max_length=2)]] | None = (
        None
    )
    ratio: float = 0.0


class _Patch(Struct, kw_only=True):
    name: Annotated[str, Meta(max_length=5)] | msgspec.UnsetType = msgspec.UNSET
    note: Annotated[str, Meta(max_length=5)] | msgspec.UnsetType | None = msgspec.UNSET
    tags: list[Annotated[str, Meta(max_length=2)]] | msgspec.UnsetType = msgspec.UNSET


def _issues(value: msgspec.Struct) -> list[DocumentIssue]:
    issues: list[DocumentIssue] = []
    collect_length_issues(value, issues)
    return issues


def _at(*paths: str) -> list[DocumentIssue]:
    return [DocumentIssue(path, TOO_LONG) for path in paths]


# -- shapes -----------------------------------------------------------------------------------


class TestWalkerShapes:
    def test_the_code_is_stable(self) -> None:
        assert TOO_LONG == "too_long_after_normalization"

    def test_a_struct_within_every_limit_has_no_issues(self) -> None:
        assert _issues(_Shape(name="abcde", leaf=_Leaf(word="abc"), words=["ab"])) == []

    def test_top_level_field(self) -> None:
        assert _issues(_Shape(name="abcdef")) == _at("name")

    def test_optional_field_is_checked_when_present_and_skipped_when_none(self) -> None:
        assert _issues(_Shape(name="a", nickname="abcde")) == _at("nickname")
        assert _issues(_Shape(name="a", nickname=None)) == []

    def test_a_field_without_a_limit_is_never_flagged(self) -> None:
        assert _issues(_Shape(name="a", unbounded="x" * 100_000)) == []

    def test_nested_struct_uses_a_dotted_path(self) -> None:
        assert _issues(_Shape(name="a", leaf=_Leaf(word="abcd"))) == _at("leaf.word")

    def test_optional_struct_is_walked_as_itself_not_as_none(self) -> None:
        # Walking every union branch would also visit the ``None`` member; exactly one issue.
        assert _issues(_Shape(name="a", leaf=_Leaf(word="abcd"))) == _at("leaf.word")
        assert _issues(_Shape(name="a", leaf=None)) == []

    def test_tuple_of_structs_indexes_every_item(self) -> None:
        shape = _Shape(
            name="a", leaves=(_Leaf(word="ok"), _Leaf(word="toolong"), _Leaf(word="alsolong"))
        )
        assert _issues(shape) == _at("leaves[1].word", "leaves[2].word")

    def test_list_of_str_indexes_every_item(self) -> None:
        assert _issues(_Shape(name="a", words=["ab", "abc", "ab", "abcd"])) == _at(
            "words[1]", "words[3]"
        )

    def test_union_picks_the_branch_the_value_is(self) -> None:
        assert _issues(_Shape(name="a", either="abc")) == _at("either")
        assert _issues(_Shape(name="a", either=["ab", "abc"])) == _at("either[1]")
        assert _issues(_Shape(name="a", either="ab")) == []

    def test_every_violation_is_reported_in_field_order(self) -> None:
        shape = _Shape(
            name="abcdef",
            nickname="abcde",
            leaf=_Leaf(word="abcd"),
            leaves=(_Leaf(word="abcd"),),
            words=["abc"],
        )
        assert _issues(shape) == _at("name", "nickname", "leaf.word", "leaves[0].word", "words[0]")

    def test_values_that_normalization_never_touches_are_skipped(self) -> None:
        assert _issues(_Shape(name="a", leaf=_Leaf(word="a", count=10**9), ratio=1e300)) == []

    def test_a_path_prefix_is_prepended(self) -> None:
        issues: list[DocumentIssue] = []
        collect_length_issues(_Shape(name="abcdef", words=["abc"]), issues, "content")
        assert issues == _at("content.name", "content.words[0]")

    def test_appends_to_existing_issues(self) -> None:
        issues = [DocumentIssue("other", "duplicate_tag")]
        collect_length_issues(_Shape(name="abcdef"), issues)
        assert issues == [DocumentIssue("other", "duplicate_tag"), *_at("name")]


class TestUnsetFields:
    """PATCH structs: ``type_info`` drops ``UnsetType``, so UNSET is skipped by identity."""

    def test_type_info_reports_a_bare_type_for_an_unset_capable_field(self) -> None:
        info = mi.type_info(_Patch)
        assert isinstance(info, mi.StructType)
        by_name = {f.name: f.type for f in info.fields}
        assert isinstance(by_name["name"], mi.StrType)  # no UnsetType in sight
        assert isinstance(by_name["tags"], mi.ListType)

    def test_unset_fields_are_skipped(self) -> None:
        assert _issues(_Patch()) == []

    def test_set_fields_are_checked_and_unset_neighbours_ignored(self) -> None:
        assert _issues(_Patch(tags=["abc", "ab"])) == _at("tags[0]")
        assert _issues(_Patch(name="abcdef")) == _at("name")

    def test_none_clears_and_is_not_a_violation(self) -> None:
        assert _issues(_Patch(note=None)) == []
        assert _issues(_Patch(note="abcdef")) == _at("note")


# -- boundaries -------------------------------------------------------------------------------


class TestLimitBoundary:
    @pytest.mark.parametrize("length", [0, 1, 199, 200])
    def test_a_value_at_or_under_the_limit_passes(self, length: int) -> None:
        assert _issues(TrackCreate(title="x" * length)) == []

    @pytest.mark.parametrize("length", [201, 202, 10_000])
    def test_a_value_over_the_limit_fails(self, length: int) -> None:
        assert _issues(TrackCreate(title="x" * length)) == _at("title")

    def test_limits_count_code_points_not_bytes_or_graphemes(self) -> None:
        # 100 two-byte, one-code-point letters; and 100 combining sequences of 2 code points.
        assert _issues(TrackCreate(title="я" * 200)) == []
        assert _issues(TrackCreate(title="é" * 100)) == []  # 200 code points
        assert _issues(TrackCreate(title="é" * 101)) == _at("title")  # 202 code points


# -- the real wire Structs --------------------------------------------------------------------

_ROOTS: tuple[type[msgspec.Struct], ...] = (
    PlanDocument,
    LyricsDocument,
    ArtistCreate,
    ArtistPatch,
    TrackCreate,
    TrackPatch,
    ProjectProfilePut,
)

_SUPPORTED = (
    mi.StrType,
    mi.IntType,
    mi.BoolType,
    mi.UUIDType,
    mi.EnumType,
    mi.NoneType,
    mi.UnionType,
    mi.StructType,
    mi.ListType,
    mi.VarTupleType,
)


def _nodes(info: mi.Type) -> list[mi.Type]:
    """Every type node reachable from ``info`` (``info`` included)."""
    found = [info]
    match info:
        case mi.UnionType(types=members):
            for member in members:
                found.extend(_nodes(member))
        case mi.StructType(fields=fields):
            for field in fields:
                found.extend(_nodes(field.type))
        case mi.ListType(item_type=item) | mi.VarTupleType(item_type=item):
            found.extend(_nodes(item))
    return found


def _filled(info: mi.Type, path: str, over: list[str]) -> Any:
    """A value for ``info`` with every capped ``str`` one over its limit.

    Appends the wire path of each such string to ``over`` — the oracle of the guard test. Lists
    get two items so that indexes ``[0]`` and ``[1]`` are both exercised.
    """
    match info:
        case mi.StrType(max_length=limit):
            if limit is not None:
                over.append(path)
                return "x" * (limit + 1)
            return "x"
        case mi.UnionType(types=members):
            first = next(m for m in members if not isinstance(m, mi.NoneType))
            return _filled(first, path, over)
        case mi.StructType(cls=cls, fields=fields):
            return cls(
                **{
                    f.name: _filled(
                        f.type, f"{path}.{f.encode_name}" if path else f.encode_name, over
                    )
                    for f in fields
                }
            )
        case mi.ListType(item_type=item):
            return [_filled(item, f"{path}[{i}]", over) for i in range(2)]
        case mi.VarTupleType(item_type=item):
            return tuple(_filled(item, f"{path}[{i}]", over) for i in range(2))
        case mi.EnumType(cls=enum_cls):
            return next(iter(enum_cls))
        case mi.UUIDType():
            return UUID(int=1)
        case mi.IntType():
            return 1
        case mi.BoolType():
            return False
    raise AssertionError(f"no filler for {info!r}")  # pragma: no cover


class TestEveryDeclaredLimitIsVisited:
    """Guard: a limit added to a wire Struct is checked after normalization without extra code."""

    @pytest.mark.parametrize("root", _ROOTS, ids=lambda r: r.__name__)
    def test_every_reachable_node_type_is_one_the_walker_descends_into(
        self, root: type[msgspec.Struct]
    ) -> None:
        # A dict / set / fixed tuple holding capped strings would be invisible to the walker.
        unsupported = [n for n in _nodes(mi.type_info(root)) if not isinstance(n, _SUPPORTED)]
        assert not unsupported, f"extend constraints._walk for {unsupported!r}"

    @pytest.mark.parametrize("root", _ROOTS, ids=lambda r: r.__name__)
    def test_an_over_long_value_in_every_capped_str_yields_one_issue_at_its_path(
        self, root: type[msgspec.Struct]
    ) -> None:
        over: list[str] = []
        info = mi.type_info(root)
        value = _filled(info, "", over)

        assert over, "the guard itself must find capped strings in every root"
        assert [(i.path, i.code) for i in _issues(value)] == [(path, TOO_LONG) for path in over]

    def test_the_guard_finds_the_expected_capped_strings(self) -> None:
        # Spot-check the oracle against hand-written wire paths so it cannot drift silently.
        over: list[str] = []
        _filled(mi.type_info(PlanDocument), "", over)
        assert {
            "title",
            "idea",
            "references[1]",
            "global_style.tags[0]",
            "global_style.avoid[1]",
            "sections[0].label",
            "sections[1].style.tags[1]",
            "sections[0].delivery[0]",
            "sections[1].notes",
        } <= set(over)

        over = []
        _filled(mi.type_info(LyricsDocument), "", over)
        assert {
            "language",
            "sections[0].lines[0].text",
            "sections[1].lines[1].delivery[0]",
        } <= set(over)

        over = []
        _filled(mi.type_info(ArtistPatch), "", over)
        assert {
            "name",
            "persona",
            "voice_notes",
            "aesthetic_references[1]",
            "style_tags[0]",
            "global_avoid[1]",
        } <= set(over)

        over = []
        _filled(mi.type_info(ProjectProfilePut), "", over)
        assert over == ["concept"]

    @pytest.mark.parametrize("root", [ArtistPatch, TrackPatch])
    def test_a_patch_with_every_field_unset_has_nothing_to_check(
        self, root: type[msgspec.Struct]
    ) -> None:
        assert _issues(root()) == []


class TestItemCounts:
    """``collect_count_issues``: the array ``max_length`` of a document built in code."""

    def test_the_code_is_stable(self) -> None:
        assert TOO_MANY == "too_many_items"

    def test_an_array_within_its_limit_has_no_issue(self) -> None:
        leaves = tuple(_Leaf(word="a") for _ in range(4))
        issues: list[DocumentIssue] = []
        collect_count_issues(_Shape(name="a", leaves=leaves), issues)
        assert issues == []

    def test_an_array_past_its_limit_is_reported_at_its_own_path(self) -> None:
        issues: list[DocumentIssue] = []
        collect_count_issues(
            _Shape(name="a", leaves=tuple(_Leaf(word="a") for _ in range(5))), issues
        )
        assert issues == [DocumentIssue("leaves", TOO_MANY)]

    def test_strings_are_not_this_walker_business(self) -> None:
        issues: list[DocumentIssue] = []
        collect_count_issues(_Shape(name="far too long"), issues)
        assert issues == []

    def test_nested_arrays_get_indexed_paths_and_a_prefix(self) -> None:
        class Outer(Struct, frozen=True, kw_only=True):
            groups: tuple[_Shape, ...] = ()

        over = tuple(_Leaf(word="a") for _ in range(5))
        issues: list[DocumentIssue] = []
        collect_count_issues(
            Outer(groups=(_Shape(name="a"), _Shape(name="b", leaves=over))), issues, "content"
        )
        assert issues == [DocumentIssue("content.groups[1].leaves", TOO_MANY)]

    def test_the_real_plan_and_lyrics_limits_are_read_from_the_wire_structs(self) -> None:
        from src.studio.documents import PlanDocument, StyleSpec

        plan = PlanDocument(
            title="x", global_style=StyleSpec(tags=tuple(str(i) for i in range(41)))
        )
        issues: list[DocumentIssue] = []
        collect_count_issues(plan, issues)
        assert issues == [DocumentIssue("global_style.tags", TOO_MANY)]
