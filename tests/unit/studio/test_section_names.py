"""Section names: the one vocabulary shared by the renderer, the validators and the importer."""

from __future__ import annotations

import pytest

from src.core.enums import StudioSectionKind as K
from src.studio.section_names import (
    ALIASES,
    _key,
    default_labels,
    display_name,
    is_header_text,
    rendered_labels,
    resolve_kind,
    split_header,
)

pytestmark = pytest.mark.unit


class TestKey:
    @pytest.mark.parametrize(
        ("name", "key"),
        [
            ("Verse", "verse"),
            ("  PRE-CHORUS ", "pre chorus"),
            ("Fade-Out", "fade out"),
            ("Build   Up", "build up"),
            ("Straße", "strasse"),  # casefold, not lower
        ],
    )
    def test_casefold_dash_as_space_and_collapsed_whitespace(self, name: str, key: str) -> None:
        assert _key(name) == key

    def test_every_alias_key_is_already_normalized(self) -> None:
        assert all(_key(key) == key for key in ALIASES)


class TestAliasTable:
    @pytest.mark.parametrize(
        ("name", "kind"),
        [
            ("intro", K.INTRO),
            ("verse", K.VERSE),
            ("pre-chorus", K.PRE_CHORUS),
            ("prechorus", K.PRE_CHORUS),
            ("pre chorus", K.PRE_CHORUS),
            ("chorus", K.CHORUS),
            ("refrain", K.CHORUS),
            ("post-chorus", K.POST_CHORUS),
            ("postchorus", K.POST_CHORUS),
            ("post chorus", K.POST_CHORUS),
            ("hook", K.HOOK),
            ("bridge", K.BRIDGE),
            ("build", K.BUILD),
            ("build-up", K.BUILD),
            ("buildup", K.BUILD),
            ("build up", K.BUILD),
            ("drop", K.DROP),
            ("breakdown", K.BREAKDOWN),
            ("break", K.BREAKDOWN),
            ("instrumental", K.INSTRUMENTAL),
            ("solo", K.INSTRUMENTAL),
            ("interlude", K.INTERLUDE),
            ("outro", K.OUTRO),
            ("end", K.OUTRO),
            ("fade out", K.OUTRO),
            ("fade-out", K.OUTRO),
        ],
    )
    def test_each_documented_name(self, name: str, kind: K) -> None:
        assert resolve_kind(name) is kind

    def test_every_kind_is_reachable(self) -> None:
        assert set(ALIASES.values()) == set(K)

    def test_every_name_the_renderer_writes_resolves_to_its_own_kind(self) -> None:
        for kind in K:
            assert resolve_kind(display_name(kind)) is kind

    def test_every_rendered_default_label_reads_back_as_a_header(self) -> None:
        for kind in K:
            for label in default_labels([kind, kind, kind]):
                assert is_header_text(label), label


class TestResolveKind:
    @pytest.mark.parametrize(
        ("name", "kind"),
        [
            ("Verse 1", K.VERSE),
            ("VERSE 12", K.VERSE),
            ("Pre-Chorus 2", K.PRE_CHORUS),
            ("  chorus  3 ", K.CHORUS),
            ("Fade Out 1", K.OUTRO),
        ],
    )
    def test_an_optional_one_or_two_digit_number_is_ignored(self, name: str, kind: K) -> None:
        assert resolve_kind(name) is kind

    @pytest.mark.parametrize(
        "name",
        ["", "   ", "1", "Verse 100", "Verse 1a", "Verse one", "whispered", "Chorus!", "Verses"],
    )
    def test_anything_else_is_unknown(self, name: str) -> None:
        assert resolve_kind(name) is None


class TestSplitHeader:
    def test_splits_name_from_descriptors(self) -> None:
        assert split_header("Verse 1: whispered, cello") == ("Verse 1", ["whispered", "cello"])

    def test_no_colon_is_all_name(self) -> None:
        assert split_header("Chorus") == ("Chorus", [])

    def test_splits_at_the_first_colon_only(self) -> None:
        assert split_header("Bridge: key: C minor, slow") == ("Bridge", ["key: C minor", "slow"])

    def test_trims_and_drops_empty_descriptors(self) -> None:
        assert split_header("  Drop :  a ,, b , ") == ("Drop", ["a", "b"])

    def test_empty_name_is_kept_empty(self) -> None:
        assert split_header(": x") == ("", ["x"])


class TestIsHeaderText:
    @pytest.mark.parametrize(
        "content",
        [
            "Verse",
            "Verse 2",
            "verse: whispered",
            "Pre-Chorus: lift, strings",
            "Refrain",
            "Fade Out",
            "BREAK",
            "End",
            "Instrumental: solo guitar",
            "Verse :",
        ],
    )
    def test_true_when_the_name_part_resolves(self, content: str) -> None:
        assert is_header_text(content)

    @pytest.mark.parametrize(
        "content",
        [
            "whispered",
            "whispered, breathy",
            "soft: verse",  # the name part is what is *before* the first colon
            "",
            "Verse 100",
            "choruses",
            ": Verse",
        ],
    )
    def test_false_otherwise(self, content: str) -> None:
        assert not is_header_text(content)


class TestLabels:
    def test_default_labels_number_a_repeated_kind(self) -> None:
        assert default_labels([K.VERSE, K.CHORUS, K.VERSE, K.CHORUS, K.BRIDGE]) == [
            "Verse 1",
            "Chorus",
            "Verse 2",
            "Chorus",
            "Bridge",
        ]

    def test_unnumbered_kinds_never_get_a_number(self) -> None:
        kinds = [K.CHORUS, K.PRE_CHORUS, K.POST_CHORUS, K.HOOK] * 2
        assert default_labels(kinds) == ["Chorus", "Pre-Chorus", "Post-Chorus", "Hook"] * 2

    def test_display_names(self) -> None:
        assert display_name(K.PRE_CHORUS) == "Pre-Chorus"
        assert display_name(K.POST_CHORUS) == "Post-Chorus"
        assert display_name(K.BREAKDOWN) == "Breakdown"

    def test_rendered_labels_count_every_section_of_a_kind_labelled_or_not(self) -> None:
        assert rendered_labels([(K.VERSE, None), (K.VERSE, "Opening"), (K.VERSE, None)]) == [
            "Verse 1",
            "Opening",
            "Verse 3",
        ]

    def test_an_empty_plan_has_no_labels(self) -> None:
        assert default_labels([]) == []
        assert rendered_labels([]) == []
