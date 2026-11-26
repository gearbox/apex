"""The studio / LLM settings reach the API container, and compose defaults are safe.

Compose passes only the variables an ``environment:`` block names, so a setting that is
missing there can never be switched on in a deployment.

A compose default must not contain JSON: older Compose releases (seen on 2.17; fixed by
2.24) end ``${VAR:-{}}`` at the first ``}`` and append a stray ``}`` to a *set* value
(staging failed to boot with ``LLM_MODELS`` = valid JSON + ``}``). JSON settings are passed
as ``${VAR:-}`` and ``Settings`` reads a blank value as the field's default (``JsonFromEnv``).

Deployment defaults keep the artist chat on ``grok`` (memory drafts follow the chat model).
Plan runs go to ``qwen`` through ``STUDIO_LLM_MODEL_BY_KIND``, which the env templates set.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
from pydantic import ValidationError

from src.core.enums import StudioRunKind
from tests.env_isolation import compose_defaults
from tests.unit.helpers import hermetic_settings

_REPO_ROOT = Path(__file__).parent.parent.parent

# Files that define the api service's environment block themselves: the base file (dev,
# prod and staging extend it) and the standalone Portainer stack.
_COMPOSE_FILES = (
    _REPO_ROOT / "docker-compose.yml",
    _REPO_ROOT / "docker-compose.portainer-staging.yml",
)
_ALL_COMPOSE_FILES = tuple(sorted(_REPO_ROOT.glob("docker-compose*.yml")))
_ENV_TEMPLATES = (_REPO_ROOT / ".env.example", _REPO_ROOT / ".env.staging.example")

_STUDIO_VARS = (
    "STUDIO_ENABLED",
    "STUDIO_RUNS_ENABLED",
    "STUDIO_CHAT_ENABLED",
    "LLM_MODELS",
    "LLM_DEFAULT_MODEL",
    "STUDIO_LLM_MODEL_BY_KIND",
    "STUDIO_CHAT_MODEL",
    "STUDIO_CHAT_MODEL_BY_PRODUCT",
    "STUDIO_MEMORY_DRAFT_MODEL",
    # The keys LLM_MODELS names by api_key_env must be process env vars too.
    "XAI_API_KEY",
    "MODEL_STUDIO_API_KEY",
)
_JSON_VARS = ("LLM_MODELS", "STUDIO_LLM_MODEL_BY_KIND", "STUDIO_CHAT_MODEL_BY_PRODUCT")
_KILL_SWITCHES = ("STUDIO_ENABLED", "STUDIO_RUNS_ENABLED", "STUDIO_CHAT_ENABLED")

_NESTED = re.compile(r"^\$\{([A-Z][A-Z0-9_]*):-(.*)\}$")
# An innermost `${NAME}` / `${NAME:-x}` / `${NAME-x}` / `${NAME:?x}` with no braces inside.
_INNER_REFERENCE = re.compile(r"\$\{[A-Za-z_][A-Za-z0-9_]*(?::?[-?+][^{}]*)?\}")
_DEFAULT_START = re.compile(r"\$\{[A-Za-z_][A-Za-z0-9_]*:?-")


def _resolve(value: str, defaults: dict[str, str]) -> str:
    """Resolve a nested `${OTHER:-x}` default the way compose does with nothing set."""
    nested = _NESTED.match(value)
    if nested is None:
        return value
    name, fallback = nested.groups()
    return _resolve(defaults.get(name, fallback), defaults)


def _default_values(text: str) -> list[str]:
    """Every `${VAR:-default}` / `${VAR-default}` default, read with balanced braces."""
    found: list[str] = []
    for match in _DEFAULT_START.finditer(text):
        depth, start = 1, match.end()
        for index in range(start, len(text)):
            if text[index] == "{":
                depth += 1
            elif text[index] == "}":
                depth -= 1
                if depth == 0:
                    found.append(text[start:index])
                    break
    return found


def _yaml_without_comments(text: str) -> str:
    """Comment lines are not interpolated (and the compose files quote the bad pattern)."""
    return "\n".join(line for line in text.splitlines() if not line.lstrip().startswith("#"))


def _without_references(value: str) -> str:
    previous = None
    while previous != value:
        previous, value = value, _INNER_REFERENCE.sub("", value)
    return value


def _template_values(path: Path) -> dict[str, str]:
    return dict(re.findall(r"^([A-Z][A-Z0-9_]*)=(.*)$", path.read_text(), re.MULTILINE))


@pytest.mark.parametrize("compose_path", _ALL_COMPOSE_FILES, ids=lambda p: p.name)
def test_no_compose_default_contains_a_literal_brace(compose_path: Path) -> None:
    offending = [
        default
        for default in _default_values(_yaml_without_comments(compose_path.read_text()))
        if {"{", "}"} & set(_without_references(default))
    ]

    assert not offending, (
        f"{compose_path.name}: a default holds a literal brace, which older Compose "
        f"truncates at the first '}}' (pass ${{VAR:-}} + JsonFromEnv instead): {offending}"
    )


def test_the_brace_check_catches_the_staging_failure() -> None:
    """Guards the check above against matching nothing."""
    bad = '      LLM_MODELS: ${LLM_MODELS:-{}}\n      B: ${B:-{"plan":"qwen"}}\n'
    good = "      N: ${N:-${M:-grok}}\n      R: ${R:-redis://:${P}@redis:6379/0}\n"

    assert [_without_references(d) for d in _default_values(bad)] == ["{}", '{"plan":"qwen"}']
    assert [_without_references(d) for d in _default_values(good)] == [
        "",  # ${M:-grok} removed from N's default
        "grok",  # M's own default, found inside N
        "redis://:@redis:6379/0",
    ]
    assert _default_values(_yaml_without_comments(f"      # ${{X:-{{}}}}\n{good}")) == (
        _default_values(good)
    )


@pytest.mark.parametrize("compose_path", _COMPOSE_FILES, ids=lambda p: p.name)
def test_every_studio_var_is_forwarded(compose_path: Path) -> None:
    defaults = compose_defaults(compose_path)

    missing = [name for name in _STUDIO_VARS if name not in defaults]

    assert not missing, f"{compose_path.name} does not forward {missing}"


@pytest.mark.parametrize("compose_path", _COMPOSE_FILES, ids=lambda p: p.name)
def test_kill_switches_default_off_and_json_settings_default_blank(compose_path: Path) -> None:
    defaults = compose_defaults(compose_path)

    assert {name: defaults[name] for name in _KILL_SWITCHES} == dict.fromkeys(
        _KILL_SWITCHES, "false"
    )
    assert {name: defaults[name] for name in _JSON_VARS} == dict.fromkeys(_JSON_VARS, "")


@pytest.mark.parametrize("compose_path", _COMPOSE_FILES, ids=lambda p: p.name)
def test_compose_defaults_load_and_keep_the_chat_on_grok(
    compose_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    defaults = compose_defaults(compose_path)
    for name in _STUDIO_VARS:
        monkeypatch.setenv(name, _resolve(defaults[name], defaults))

    settings = hermetic_settings()

    assert settings.llm_models == {}
    assert not dict(settings.studio_llm_model_by_kind)
    assert settings.studio_chat_model_by_product == {}
    assert settings.studio_chat_model_key_default == "grok"
    assert settings.studio_memory_draft_model == "grok"
    assert settings.llm_default_model == "grok"
    assert not settings.studio_enabled


def test_memory_draft_model_follows_the_chat_model() -> None:
    defaults = compose_defaults(_COMPOSE_FILES[0])
    switched = {**defaults, "STUDIO_CHAT_MODEL": "qwen"}

    assert _resolve(defaults["STUDIO_MEMORY_DRAFT_MODEL"], defaults) == "grok"
    assert _resolve(switched["STUDIO_MEMORY_DRAFT_MODEL"], switched) == "qwen"


@pytest.mark.parametrize("template", _ENV_TEMPLATES, ids=lambda p: p.name)
def test_env_templates_route_plan_to_qwen_and_chat_like_compose(template: Path) -> None:
    compose = compose_defaults(_COMPOSE_FILES[0])
    values = _template_values(template)

    assert json.loads(values["STUDIO_LLM_MODEL_BY_KIND"]) == {"plan": "qwen"}
    assert "qwen" in json.loads(values["LLM_MODELS"])
    for name in ("STUDIO_CHAT_MODEL", "LLM_DEFAULT_MODEL"):
        assert values[name] == compose[name], name


@pytest.mark.parametrize("name", _JSON_VARS)
@pytest.mark.parametrize("blank", ["", "   "])
def test_a_blank_json_setting_means_its_default(
    name: str, blank: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(name, blank)

    settings = hermetic_settings()

    assert getattr(settings, name.lower()) == {}


def test_json_settings_still_decode(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("STUDIO_LLM_MODEL_BY_KIND", '{"plan":"qwen"}')
    monkeypatch.setenv("STUDIO_CHAT_MODEL_BY_PRODUCT", '{"synthara":"qwen"}')

    settings = hermetic_settings()

    assert dict(settings.studio_llm_model_by_kind) == {StudioRunKind.PLAN: "qwen"}
    assert settings.studio_chat_model_by_product == {"synthara": "qwen"}


@pytest.mark.parametrize("value", ['{"plan":"qwen"}}', "not json"])
def test_malformed_json_names_the_setting_not_the_value(
    value: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("STUDIO_LLM_MODEL_BY_KIND", value)

    with pytest.raises(ValidationError) as caught:
        hermetic_settings()

    message = str(caught.value)
    assert "studio_llm_model_by_kind" in message
    assert "must be a JSON object" in message
    assert value not in message


def test_init_kwargs_bypass_the_env_decoding() -> None:
    settings = hermetic_settings(studio_llm_model_by_kind={StudioRunKind.PLAN: "qwen"})

    assert dict(settings.studio_llm_model_by_kind) == {StudioRunKind.PLAN: "qwen"}


def test_the_entry_pattern_matches_something() -> None:
    """Guards against the pattern silently matching nothing after a syntax change."""
    for compose_path in _COMPOSE_FILES:
        assert len(compose_defaults(compose_path)) > 20, compose_path.name
