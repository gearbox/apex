"""Properties of render -> import -> render (invariants 1, 2, 16, 18 and 19), over generated songs.

**A round-trip property generates every class of value the validators accept**, and the
validators decide (``assume``): the strategies never filter a class out themselves. That is the
rule that was broken twice before — a filter that kept tempo-like words out of every list, and a
flag that kept ``:`` out of labels, hid exactly the two bugs they were guarding against.

So the songs cover every section kind, labels equal to a default label or holding a ``:``,
delivery, tags (including ``80 BPM`` / ``300 BPM`` spellings, which the validators accept or refuse
by themselves), section- and track-level instrumental, zero sections, parentheses, Cyrillic, and
artist tags built by the artist validator that overlap the plan's. ``derandomize`` keeps CI stable;
the ``@example`` pins run on every CI run regardless.
"""


from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple
from uuid import UUID

import pytest
from hypothesis import HealthCheck, assume, example, given, settings
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
from src.studio.errors import DocumentIssue, ImportInvalid, StudioDocumentInvalid
from src.studio.importers.suno import parse_suno
from src.studio.renderers.base import ArtistRenderInput, dedupe_ci
from src.studio.validation import (
    collect_artist_style,
    normalize_lyrics,
    normalize_plan,
    normalize_text,
)
from tests.studio_import import import_text, reimport, render_song, texts
from tests.studio_support import make_lyrics, make_plan, make_section

if TYPE_CHECKING:
    from collections.abc import Callable

pytestmark = pytest.mark.unit

PROPERTY = settings(
    max_examples=300,
    deadline=None,
    derandomize=True,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much],
)

_CYRILLIC = "".join(chr(code) for code in range(0x0430, 0x043C))  # a handful of lowercase letters
_ALPHABET = f"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ{_CYRILLIC}0123456789 -'."

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


# Nothing is filtered out of ``word``: a word that reads as a tempo ("88 bpm") is a valid label,
# descriptor, delivery, title... and the validators refuse it only where it is reserved.
word = st.text(alphabet=_ALPHABET, min_size=1, max_size=10).map(_tidy).filter(bool)
descriptor = st.one_of(word, st.sampled_from(["soft", "cello", "Dreamy", "key: C minor", "(live)"]))
label_word = st.one_of(word, st.sampled_from(_NAMEY))
colon_label = st.builds(lambda a, b: f"{a}: {b}", label_word, word)

# A small pool, so plan and artist tags often collide (different spellings included). The tempo
# spellings are reserved in the plan's global tags and an artist's style tags (the validators
# discard those songs); "300 BPM" is out of range and an ordinary tag.
_POOL = ["a", "A", "b", "B", "c", "dreamy", "Dreamy", "warm", "lofi"]
pool_tag = st.sampled_from([*_POOL, "80 BPM", "120 bpm", "120 BPM", "300 BPM"])


class RawSong(NamedTuple):
    """Generated (or pinned) input, *before* the validators have seen it."""

    plan: PlanDocument
    lyrics: LyricsDocument | None
    artist_tags: tuple[str, ...] | None
    """``None``: the project has no artist."""
    artist_avoid: tuple[str, ...] = ()


Song = tuple[PlanDocument, LyricsDocument | None, ArtistRenderInput | None]


class Refused(Exception):
    """The studio would not store this song (a validator said no)."""


def store(raw: RawSong) -> Song:
    """What the studio would store: the validators decide.

    The plan and lyrics go through ``normalize_*``; the artist through ``collect_artist_style``
    (the artist service's own validator), never a hand-made ``ArtistRenderInput``.

    Raises:
        Refused: A validator rejected the plan, the lyrics or the artist.
    """
    try:
        plan = normalize_plan(raw.plan)
        lyrics = normalize_lyrics(raw.lyrics, plan) if raw.lyrics is not None else None
    except StudioDocumentInvalid as exc:
        raise Refused from exc
    artist: ArtistRenderInput | None = None
    if raw.artist_tags is not None:
        issues: list[DocumentIssue] = []
        style = collect_artist_style(
            style_tags=raw.artist_tags, global_avoid=raw.artist_avoid, issues=issues
        )
        if issues:
            raise Refused
        artist = ArtistRenderInput(style_tags=style.style_tags, global_avoid=style.global_avoid)
    return plan, lyrics, artist


def accept(raw: RawSong) -> Song:
    """:func:`store` inside a property: a song the studio would refuse is discarded."""
    try:
        return store(raw)
    except Refused:
        assume(False)
        raise  # pragma: no cover — assume(False) already stopped the example


@st.composite
def plan_sections(draw: st.DrawFn) -> PlanSection:
    return PlanSection(
        id=draw(st.uuids(version=4)),
        kind=draw(st.sampled_from(list(K))),
        label=draw(
            st.one_of(
                st.none(),
                label_word,
                st.sampled_from(["Verse", "Verse 2", "Chorus", "Pre-Chorus"]),
                colon_label,
            )
        ),
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
def songs(draw: st.DrawFn) -> RawSong:
    """A (plan, lyrics, artist) triple; ``accept`` decides whether the studio would store it."""
    sections = tuple(draw(st.lists(plan_sections(), max_size=6)))
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

    artist_tags: tuple[str, ...] | None = None
    artist_avoid: tuple[str, ...] = ()
    if draw(st.booleans()):
        artist_tags = tuple(dedupe_ci(draw(st.lists(pool_tag, max_size=4))))
        artist_folded = {t.casefold() for t in artist_tags}
        artist_avoid = tuple(
            t
            for t in dedupe_ci(draw(st.lists(pool_tag, max_size=3)))
            if t.casefold() not in artist_folded
        )
    return RawSong(plan, lyrics, artist_tags, artist_avoid)


# -- pinned inputs: they run on every CI run, whatever ``derandomize`` generates ---------------------

_SECTION_ID = UUID("00000000-0000-7000-8000-0000000000b1")

# R2: a label holding ":" — the parser reads it as a name plus a descriptor.
COLON_LABEL_SONG = RawSong(
    make_plan(make_section(K.VERSE, section_id=_SECTION_ID, label="Verse 1: The Return")),
    make_lyrics((_SECTION_ID, ["la la"])),
    None,
)
# R2: the intro labelled "0: 0", the first failure the widened strategy found.
ZERO_COLON_ZERO_SONG = RawSong(
    make_plan(make_section(K.INTRO, section_id=_SECTION_ID, label="0: 0")), None, None
)
# R3: the plan's title differs from the track's, and the plan is instrumental.
RENAMED_INSTRUMENTAL_SONG = RawSong(
    make_plan(
        make_section(K.DROP, section_id=_SECTION_ID),
        title="Not The Track Title",
        instrumental=True,
    ),
    None,
    None,
)
# R1: a tempo-like tag next to a tempo, and the artist-side spellings. The validators refuse each
# of these at write time, so they never become a song; pinning them runs the gate on every CI run
# (``test_the_tempo_like_reproduction_rows_are_refused_by_the_validators`` asserts the refusal).
REJECTED_TEMPO_ROWS = [
    RawSong(make_plan(tags=("80 BPM",), tempo_bpm=120), None, None),  # row 1
    RawSong(make_plan(tags=("rock",), tempo_bpm=120), None, ("80 BPM",)),  # row 2
    RawSong(make_plan(tags=("rock",)), None, ("90 BPM",)),  # row 3
    RawSong(make_plan(tags=("120 bpm",), tempo_bpm=120), None, None),  # row 4
]


def _same_text(song: Song) -> None:
    plan, lyrics, artist = song
    plan2, lyrics2, _ = reimport(plan, lyrics, artist, parent=False)
    assert texts(render_song(plan2, lyrics2, artist)) == texts(render_song(plan, lyrics, artist))
    assert render_song(plan2, lyrics2, artist).instrumental == plan.instrumental


@PROPERTY
@given(songs())
@example(raw=COLON_LABEL_SONG)
@example(raw=ZERO_COLON_ZERO_SONG)
@example(raw=RENAMED_INSTRUMENTAL_SONG)
@example(raw=REJECTED_TEMPO_ROWS[0])
@example(raw=REJECTED_TEMPO_ROWS[1])
@example(raw=REJECTED_TEMPO_ROWS[2])
@example(raw=REJECTED_TEMPO_ROWS[3])
def test_invariant_1_the_text_round_trips_byte_identically(raw: RawSong) -> None:
    """``render(import(render(x))) == render(x)`` for all four fields and the flag."""
    _same_text(accept(raw))


@pytest.mark.parametrize("send_title_and_flag", [True, False], ids=["sent", "omitted"])
@PROPERTY
@given(raw=songs())
@example(raw=COLON_LABEL_SONG)
@example(raw=ZERO_COLON_ZERO_SONG)
@example(raw=RENAMED_INSTRUMENTAL_SONG)
@example(raw=REJECTED_TEMPO_ROWS[0])
@example(raw=REJECTED_TEMPO_ROWS[1])
@example(raw=REJECTED_TEMPO_ROWS[2])
@example(raw=REJECTED_TEMPO_ROWS[3])
def test_invariant_2_over_the_song_itself_nothing_changes(
    raw: RawSong, send_title_and_flag: bool
) -> None:
    """Importing ``render(x)`` over a track whose current pair is ``x`` yields ``x`` again.

    Both with the request carrying ``title`` / ``instrumental`` and without them (R3: a client
    that pastes only the three boxes must not reset the plan's title or flag).
    """
    plan, lyrics, artist = accept(raw)
    plan2, lyrics2, built = reimport(
        plan, lyrics, artist, parent=True, send_title_and_flag=send_title_and_flag
    )
    assert plan2 == plan
    assert lyrics2 == lyrics
    assert all(
        w.code not in {"text_before_first_header", "no_section_headers"}
        for w in built.warnings
    )


@PROPERTY
@given(songs())
@example(raw=COLON_LABEL_SONG)
@example(raw=RENAMED_INSTRUMENTAL_SONG)
@example(raw=REJECTED_TEMPO_ROWS[0])
def test_importing_twice_over_the_first_import_changes_nothing(raw: RawSong) -> None:
    """Invariant 3's pure core: the second import of the same text has the first as its parent.

    The second request omits ``title`` and ``instrumental``: the first import's values carry over.
    """
    plan, lyrics, artist = accept(raw)
    shown = texts(render_song(plan, lyrics, artist))
    kwargs = {
        "style": shown["style"],
        "exclude": shown["exclude"],
        "lyrics": shown["lyrics"],
        "language": lyrics.language if lyrics else "en",
        "artist": artist,
    }
    first_plan, first_lyrics, _ = import_text(
        **kwargs, title=plan.title, instrumental=plan.instrumental
    )
    second_plan, second_lyrics, _ = import_text(
        **kwargs, parent_plan=first_plan, parent_lyrics=first_lyrics
    )
    assert second_plan == first_plan
    assert second_lyrics == first_lyrics


@pytest.mark.parametrize("row", range(len(REJECTED_TEMPO_ROWS)))
def test_the_tempo_like_reproduction_rows_are_refused_by_the_validators(row: int) -> None:
    """R1, rows 1-4 of the reproduction table: no such plan or artist can be stored any more."""
    raw = REJECTED_TEMPO_ROWS[row]
    issues: list[DocumentIssue] = []
    try:
        normalize_plan(raw.plan)
    except StudioDocumentInvalid as exc:
        issues.extend(exc.issues)
    if raw.artist_tags is not None:
        collect_artist_style(
            style_tags=raw.artist_tags, global_avoid=raw.artist_avoid, issues=issues
        )
    assert [issue.code for issue in issues] == ["reserved_tempo_tag"], row


def test_a_spelling_that_is_not_a_tempo_survives_the_whole_round_trip() -> None:
    """``300 BPM`` is an ordinary tag, in the plan and in the artist's tags alike."""
    raw = RawSong(make_plan(tags=("300 BPM", "rock"), tempo_bpm=120), None, ("39 BPM",))
    plan, lyrics, artist = store(raw)
    plan2, _, _ = reimport(plan, lyrics, artist, parent=True)
    assert plan2 == plan
    _same_text((plan, lyrics, artist))


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
