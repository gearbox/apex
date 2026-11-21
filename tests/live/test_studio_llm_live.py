"""Opt-in smoke test of the configured LLM models: ``pytest -m live_llm tests/live``.

CI never calls a model (the default run excludes ``live_llm``). This sends the studio's real
``plan-v2`` prompt with a fixed brief to **every** entry of ``LLM_MODELS`` whose API-key environment
variable is set, over the real adapter, and asserts that the reply decodes into ``PlanDraft`` -
which is what proves a model really honours the entry's ``structured_output`` mode. Run it for the
``grok`` and ``qwen`` entries before merging and put the result (pass / fail, model ids, token
counts) in the PR description. It spends a few tokens per entry.

    LLM_MODELS='{...}' XAI_API_KEY=... MODEL_STUDIO_API_KEY=... \\
        uv run pytest -m live_llm tests/live -s
"""

from __future__ import annotations

import os

import httpx
import pytest
from pydantic import SecretStr

from src.core.config import Settings
from src.core.llm_models import LlmModelEntry, parse_model_entries
from src.llm.openai_compatible import OpenAICompatibleLlmAdapter
from src.studio.planner.context import RunContextSnapshot
from src.studio.planner.drafts import PlanDraft
from src.studio.planner.graphs.plan import PLAN_SCHEMA_NAME
from src.studio.planner.mapping import plan_from_draft
from src.studio.planner.prompts.plan_v2 import build_plan_messages
from src.studio.planner.validate import validate_plan
from src.studio.run_requests import PlanRunRequest

pytestmark = pytest.mark.live_llm

BRIEF = "A mellow synthwave track about driving home through the rain at night."


def _configured() -> dict[str, LlmModelEntry]:
    try:
        return parse_model_entries(Settings().llm_models)  # type: ignore[call-arg]
    except Exception:  # an unusable local environment simply means nothing is configured
        return {}


ENTRIES = _configured()


@pytest.mark.parametrize(
    "key",
    sorted(ENTRIES) or [pytest.param("none", marks=pytest.mark.skip(reason="no LLM_MODELS"))],
)
async def test_the_model_answers_the_plan_prompt_with_a_valid_draft(key: str) -> None:
    entry = ENTRIES[key]
    api_key = os.environ.get(entry.api_key_env, "")
    if not api_key:
        pytest.skip(f"{entry.api_key_env} is not set")

    context = RunContextSnapshot(
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
    messages = build_plan_messages(context, PlanRunRequest(brief=BRIEF))
    async with httpx.AsyncClient() as client:
        adapter = OpenAICompatibleLlmAdapter(entry, SecretStr(api_key), client, model_key=key)
        result = await adapter.complete(messages, schema=PlanDraft, schema_name=PLAN_SCHEMA_NAME)

    draft = result.value
    plan, issues = validate_plan(plan_from_draft(draft, brief=BRIEF))
    print(  # noqa: T201 - the PR description quotes this line
        f"live_llm key={key} mode={entry.structured_output} model={result.model_id!r} "
        f"prompt_tokens={result.usage.prompt_tokens} "
        f"completion_tokens={result.usage.completion_tokens} "
        f"sections={len(draft.sections)} valid={plan is not None} "
        f"issues={[(i.path, i.code) for i in issues]}"
    )
    assert isinstance(draft, PlanDraft)
    assert draft.refusal is None
    assert draft.sections, "the model returned no sections"
