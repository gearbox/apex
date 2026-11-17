"""Opt-in smoke tests of the lyrics prompts: ``pytest -m live_llm tests/live``.

CI never calls a model (the default run excludes ``live_llm``). For **every** ``LLM_MODELS`` entry
whose API-key environment variable is set, this sends the real ``lyrics-v1`` and
``revise-lyrics-v1`` prompts with a fixed plan over the real adapter and asserts that the reply
decodes into the draft Struct - which is what proves a model honours the entry's
``structured_output`` mode for these schemas. It also prints how many lines missed the syllable
range in ``ru`` and ``en``, and how many lines the model changed outside the targets (none must).
Run it for ``grok`` and ``qwen`` before merging and put the output (model ids, calls, tokens, the
miss counts) in the PR description. It spends a few thousand tokens per entry.

    LLM_MODELS='{...}' XAI_API_KEY=... MODEL_STUDIO_API_KEY=... \\
        uv run pytest -m live_llm tests/live -s
"""

from __future__ import annotations

import os
from uuid import UUID

import httpx
import pytest
from pydantic import SecretStr

from src.core.config import Settings
from src.core.enums import StudioSectionKind as K
from src.core.llm_models import LlmModelEntry, parse_model_entries
from src.llm.openai_compatible import OpenAICompatibleLlmAdapter
from src.studio.documents import SyllableRange
from src.studio.planner.context import RunContextSnapshot
from src.studio.planner.drafts import LyricsDraft, LyricsRevisionDraft
from src.studio.planner.graphs.lyrics import LYRICS_SCHEMA_NAME
from src.studio.planner.graphs.revise_lyrics import REVISION_SCHEMA_NAME
from src.studio.planner.lyric_checks import check_lines
from src.studio.planner.lyrics_mapping import apply_revision, lyrics_from_draft
from src.studio.planner.prompts.lyrics_v1 import build_lyrics_messages
from src.studio.planner.prompts.revise_lyrics_v1 import build_revise_messages
from src.studio.planner.syllables import counter_for
from src.studio.run_requests import (
    LineTarget,
    LyricsRunRequest,
    ReviseLyricsRunRequest,
    ReviseTargets,
)
from src.studio.validation import normalize_lyrics, normalize_plan
from tests.studio_support import make_lyrics, make_plan, make_section

pytestmark = pytest.mark.live_llm

VERSE = UUID(int=0x3001)
CHORUS = UUID(int=0x3002)
PLAN = normalize_plan(
    make_plan(
        make_section(K.VERSE, section_id=VERSE, notes="quiet, a little tired"),
        make_section(K.CHORUS, section_id=CHORUS, tags=["anthemic"]),
        title="Night Drive",
        idea="driving home through the rain",
        tempo_bpm=100,
    )
)
RANGE = SyllableRange(min=6, max=10)
LANGUAGES = ("en", "ru")


def _configured() -> dict[str, LlmModelEntry]:
    try:
        return parse_model_entries(Settings().llm_models)  # type: ignore[call-arg]
    except Exception:  # an unusable local environment simply means nothing is configured
        return {}


ENTRIES = _configured()
KEYS = sorted(ENTRIES) or [pytest.param("none", marks=pytest.mark.skip(reason="no LLM_MODELS"))]


def _context() -> RunContextSnapshot:
    return RunContextSnapshot(
        track_title="Night Drive",
        artist=None,
        concept=None,
        content_rating="sfw",
        plan_version_id=None,
        plan_content_hash=None,
        current_plan=None,
        lyrics_version_id=None,
        lyrics_content_hash=None,
    )


def _adapter(key: str, client: httpx.AsyncClient) -> OpenAICompatibleLlmAdapter:
    entry = ENTRIES[key]
    api_key = os.environ.get(entry.api_key_env, "")
    if not api_key:
        pytest.skip(f"{entry.api_key_env} is not set")
    return OpenAICompatibleLlmAdapter(entry, SecretStr(api_key), client, model_key=key)


@pytest.mark.parametrize("language", LANGUAGES)
@pytest.mark.parametrize("key", KEYS)
async def test_the_model_answers_the_lyrics_prompt_with_a_valid_draft(
    key: str, language: str
) -> None:
    messages = build_lyrics_messages(
        _context(),
        LyricsRunRequest(language=language, brief="a quiet song about getting home"),
        PLAN,
        language=language,
        line_syllables=RANGE,
    )
    async with httpx.AsyncClient() as client:
        result = await _adapter(key, client).complete(
            messages, schema=LyricsDraft, schema_name=LYRICS_SCHEMA_NAME
        )

    draft = result.value
    lyrics, structural = lyrics_from_draft(draft, PLAN, language=language, line_syllables=RANGE)
    misses = check_lines(lyrics, counter=counter_for(language), syllables=RANGE, avoid=())
    lines = sum(len(s.lines) for s in lyrics.sections)
    print(  # noqa: T201 - the PR description quotes this line
        f"live_llm lyrics key={key} lang={language} model={result.model_id!r} "
        f"prompt_tokens={result.usage.prompt_tokens} "
        f"completion_tokens={result.usage.completion_tokens} "
        f"lines={lines} syllable_misses={len(misses.issues)} skipped={misses.skipped_lines} "
        f"structural={[(i.path, i.code) for i in structural]}"
    )
    assert isinstance(draft, LyricsDraft)
    assert draft.refusal is None
    assert lines, "the model wrote no lines"


@pytest.mark.parametrize("language", LANGUAGES)
@pytest.mark.parametrize("key", KEYS)
async def test_the_model_answers_the_revise_prompt_and_leaves_the_rest_alone(
    key: str, language: str
) -> None:
    source = normalize_lyrics(
        make_lyrics(
            (VERSE, ["the rain keeps falling down", "I follow the red lights home"]),
            (CHORUS, ["take me home tonight", "take me home"]),
            language=language,
        ),
        PLAN,
    )
    targets = ReviseTargets(sections=(CHORUS,), lines=(LineTarget(section_id=VERSE, line=0),))
    request = ReviseLyricsRunRequest(instruction="make the mood more hopeful", targets=targets)
    messages = build_revise_messages(_context(), request, PLAN, source, line_syllables=RANGE)
    async with httpx.AsyncClient() as client:
        result = await _adapter(key, client).complete(
            messages, schema=LyricsRevisionDraft, schema_name=REVISION_SCHEMA_NAME
        )

    merged, touched, issues = apply_revision(source, result.value, targets, PLAN)
    changed_outside = [
        (s.section_id, i)
        for s, before in zip(merged.sections, source.sections, strict=False)
        for i, (new, old) in enumerate(zip(s.lines, before.lines, strict=False))
        if new != old and (s.section_id, i) not in {(r.section_id, r.line) for r in touched}
    ]
    print(  # noqa: T201 - the PR description quotes this line
        f"live_llm revise key={key} lang={language} model={result.model_id!r} "
        f"prompt_tokens={result.usage.prompt_tokens} "
        f"completion_tokens={result.usage.completion_tokens} touched={len(touched)} "
        f"issues={[(i.path, i.code) for i in issues]} changed_outside={len(changed_outside)}"
    )
    assert isinstance(result.value, LyricsRevisionDraft)
    assert result.value.refusal is None
    assert changed_outside == []  # holds by construction; asserted so a regression is loud
