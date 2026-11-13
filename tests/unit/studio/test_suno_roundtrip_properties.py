"""Properties of render -> import -> render (invariants 1 and 2), over generated songs.

The strategies build *documents that pass the P1 validators* (anything the validators reject is
discarded with ``assume``), cover every section kind, labels equal to a default label or holding a
``:``, delivery, tags, section- and track-level instrumental, zero sections, parentheses, Cyrillic
and artist tags overlapping the plan's. ``derandomize`` keeps CI stable.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import pytest
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

from src.core.enums import StudioSectionKind as K
from src.studio.documents import (
    LyricLine,
    LyricsDocument,
    LyricsSection,
    PlanDocument,
    PlanSection,
    StyleSpec,
    SyllableRange,
)
from src.studio.errors import ImportInvalid, StudioDocumentInvalid
from src.studio.importers.suno import parse_suno
from src.studio.renderers.base import ArtistRenderInput, dedupe_ci
from src.studio.validation import normalize_lyrics, normalize_plan, normalize_text
from tests.studio_import import import_text, reimport, render_song, texts

if TYPE_CHECKING:
    from collections.abc import Callable

pytestmark = pytest.mark.unit

PROPERTY = settings(
    max_examples=300,
    deadline=None,
    derandomize=True,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much],
)

_BPM_LIKE = re.compile(r"^\d{2,3}\s*bpm$")
_CYRILLIC = "".join(chr(code) for code in range(0x0430, 0x043C))  # a handful of lowercase letters
_ALPHABET = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ" + _CYRILLIC + "0123456789 -'."

# Words that are also section names, so labels / tags / deliveries brush against the vocabulary.
_NAMEY = [
    "Verse",
    "verse 2",
    "Chorus",
    "Chorus 2",
    "Pre-Chorus",
    "Hook",
    "Intro",
    "Outro",
    "End",
    "break",
]


def _tidy(value: str) -> str:
    return normalize_text(value)


word = (
    st.text(alphabet=_ALPHABET, min_size=1, max_size=10)
    .map(_tidy)
    .filter(lambda w: bool(w) and not _BPM_LIKE.match(w.casefold()))
)
descriptor = st.one_of(word, st.sampled_from(["soft", "cello", "Dreamy", "key: C minor", "(live)"]))
label_word = st.one_of(word, st.sampled_from(_NAMEY))
colon_label = st.builds(lambda a, b: f"{a}: {b}", label_word, word)

# A small pool, so plan and artist tags often collide (different spellings included).
pool_tag = st.sampled_from(["a", "A", "b", "B", "c", "dreamy", "Dreamy", "warm", "lofi"])


@st.composite
def plan_sections(draw: st.DrawFn, *, colon_labels: bool) -> PlanSection:
    labels = [st.none(), label_word, st.sampled_from(["Verse", "Verse 2", "Chorus", "Pre-Chorus"])]
    if colon_labels:
        labels.append(colon_label)
    return PlanSection(
        id=draw(st.uuids(version=4)),
        kind=draw(st.sampled_from(list(K))),
        label=draw(st.one_of(*labels)),
        duration_seconds=draw(st.one_of(st.none(), st.integers(1, 90))),
        style=StyleSpec(
            tags=tuple(dedupe_ci(draw(st.lists(descriptor, max_size=3)))),
            avoid=tuple(dedupe_ci(draw(st.lists(word, max_size=2)))),
        ),
        delivery=tuple(dedupe_ci(draw(st.lists(descriptor, max_size=3)))),
        instrumental=draw(st.booleans()),
        notes=draw(st.one_of(st.none(), word)),
    )


@st.composite
def songs(
    draw: st.DrawFn, *, colon_labels: bool
) -> tuple[PlanDocument, LyricsDocument | None, ArtistRenderInput | None]:
    """A valid (plan, lyrics, artist) triple: normalized, and compatible as a pair."""
    sections = tuple(draw(st.lists(plan_sections(colon_labels=colon_labels), max_size=6)))
    instrumental = draw(st.booleans()) and draw(st.booleans())  # about a quarter
    # Half of the songs draw every tag from the small pool the artist also draws from, so plan
    # and artist tags collide often — same tag, other spelling, other position.
    tag_source = pool_tag if draw(st.booleans()) else st.one_of(pool_tag, word)
    tags = dedupe_ci(draw(st.lists(tag_source, max_size=4)))
    folded = {t.casefold() for t in tags}
    avoid = [
        t for t in dedupe_ci(draw(st.lists(tag_source, max_size=3))) if t.casefold() not in folded
    ]
    plan = PlanDocument(
        title=draw(word),
        idea=draw(st.one_of(st.none(), word)),
        tempo_bpm=draw(st.one_of(st.none(), st.integers(40, 240))),
        instrumental=instrumental,
        global_style=StyleSpec(tags=tuple(tags), avoid=tuple(avoid)),
        references=tuple(draw(st.lists(word, max_size=2))),
        sections=sections,
    )
    try:
        plan = normalize_plan(plan)
    except StudioDocumentInvalid:
        assume(False)

    lyrics: LyricsDocument | None = None
    if not instrumental and draw(st.booleans()):
        entries: list[LyricsSection] = []
        for section in plan.sections:
            if not draw(st.booleans()):
                continue
            lines = draw(
                st.lists(
                    st.builds(
                        LyricLine,
                        text=st.one_of(
                            word,
                            st.builds(lambda a, b: f"{a} ({b})", word, word),
                            st.sampled_from(["la la (la la)", "(ooh)", "Привет, мир"]),
                        ),
                        delivery=st.lists(descriptor, max_size=2).map(
                            lambda items: tuple(dedupe_ci(items))
                        ),
                    ),
                    min_size=1,
                    max_size=3,
                )
            )
            entries.append(LyricsSection(section_id=section.id, lines=tuple(lines)))
        if entries:
            lyrics = LyricsDocument(
                language=draw(st.sampled_from(["en", "ru"])),
                line_syllables=draw(st.one_of(st.none(), st.just(SyllableRange(min=2, max=8)))),
                sections=tuple(entries),
            )
            try:
                lyrics = normalize_lyrics(lyrics, plan)
            except StudioDocumentInvalid:
                assume(False)

    artist: ArtistRenderInput | None = None
    if draw(st.booleans()):
        artist = ArtistRenderInput(
            style_tags=tuple(dedupe_ci(draw(st.lists(pool_tag, max_size=4)))),
            global_avoid=tuple(dedupe_ci(draw(st.lists(pool_tag, max_size=3)))),
        )
    return plan, lyrics, artist


def _same_text(
    plan: PlanDocument, lyrics: LyricsDocument | None, artist: ArtistRenderInput | None
) -> None:
    plan2, lyrics2, _ = reimport(plan, lyrics, artist, parent=False)
    assert texts(render_song(plan2, lyrics2, artist)) == texts(render_song(plan, lyrics, artist))
    assert render_song(plan2, lyrics2, artist).instrumental == plan.instrumental


@PROPERTY
@given(songs(colon_labels=True))
def test_invariant_1_the_text_round_trips_byte_identically(
    song: tuple[PlanDocument, LyricsDocument | None, ArtistRenderInput | None],
) -> None:
    """``render(import(render(x))) == render(x)`` for all four fields and the flag."""
    _same_text(*song)


@PROPERTY
@given(songs(colon_labels=False))
def test_invariant_2_over_the_song_itself_nothing_changes(
    song: tuple[PlanDocument, LyricsDocument | None, ArtistRenderInput | None],
) -> None:
    """Importing ``render(x)`` over a track whose current pair is ``x`` yields ``x`` again."""
    plan, lyrics, artist = song
    plan2, lyrics2, built = reimport(plan, lyrics, artist, parent=True)
    assert plan2 == plan
    assert lyrics2 == lyrics
    assert not any(
        w.code in {"text_before_first_header", "no_section_headers"} for w in built.warnings
    )


@PROPERTY
@given(songs(colon_labels=True))
def test_importing_twice_over_the_first_import_changes_nothing(
    song: tuple[PlanDocument, LyricsDocument | None, ArtistRenderInput | None],
) -> None:
    """Invariant 3's pure core: the second import of the same text has the first as its parent."""
    plan, lyrics, artist = song
    shown = texts(render_song(plan, lyrics, artist))
    kwargs = {
        "style": shown["style"],
        "exclude": shown["exclude"],
        "lyrics": shown["lyrics"],
        "title": plan.title,
        "instrumental": plan.instrumental,
        "language": lyrics.language if lyrics else "en",
        "artist": artist,
    }
    first_plan, first_lyrics, _ = import_text(**kwargs)
    second_plan, second_lyrics, _ = import_text(
        **kwargs, parent_plan=first_plan, parent_lyrics=first_lyrics
    )
    assert second_plan == first_plan
    assert second_lyrics == first_lyrics


@PROPERTY
@given(st.text(max_size=300), st.text(max_size=80), st.text(max_size=80))
def test_the_parser_is_deterministic_and_only_ever_refuses_with_import_invalid(
    lyrics: str, style: str, exclude: str
) -> None:
    def run() -> object:
        try:
            return parse_suno(style=style, exclude=exclude, lyrics=lyrics)
        except ImportInvalid as exc:
            return ("refused", exc.issues, exc.source_lines)

    assert run() == run()


_BRACKETY = st.text(alphabet="[]abc ,:\n\r-", max_size=120)


@PROPERTY
@given(_BRACKETY)
def test_any_pasted_lyrics_become_valid_documents_or_a_precise_refusal(lyrics: str) -> None:
    """Whatever is pasted either imports cleanly or fails with ``ImportInvalid`` — never a crash."""
    run: Callable[[], object] = lambda: import_text(lyrics=lyrics)  # noqa: E731
    try:
        plan, lyrics_doc, _ = run()  # type: ignore[misc]
    except ImportInvalid as exc:
        assert exc.issues
        return
    # what was stored is itself importable text: render it and paste it back
    shown = texts(render_song(plan, lyrics_doc, None))
    again = import_text(style=shown["style"], lyrics=shown["lyrics"])
    assert texts(render_song(again[0], again[1], None)) == shown
