"""The studio / LLM settings reach the API container, with defaults Settings accepts.

Compose passes only the variables an ``environment:`` block names, so a setting that is
missing there can never be switched on in a deployment. JSON settings additionally need a
JSON default: an empty string is a pydantic-settings startup error, not "unset".

The deployment defaults route plan runs to ``qwen`` and keep the artist chat on ``grok``
(memory drafts follow the chat model); the env templates must say the same.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from src.core.config import Settings
from src.core.enums import StudioRunKind

_REPO_ROOT = Path(__file__).parent.parent.parent

# Files that define the api service's environment block themselves: the base file (dev,
# prod and staging extend it) and the standalone Portainer stack.
_COMPOSE_FILES = (
    _REPO_ROOT / "docker-compose.yml",
    _REPO_ROOT / "docker-compose.portainer-staging.yml",
)
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
_JSON_VARS = frozenset({"LLM_MODELS", "STUDIO_LLM_MODEL_BY_KIND", "STUDIO_CHAT_MODEL_BY_PRODUCT"})
_KILL_SWITCHES = ("STUDIO_ENABLED", "STUDIO_RUNS_ENABLED", "STUDIO_CHAT_ENABLED")

# `      NAME: ${NAME:-default}` - the default may itself be `${OTHER:-value}`.
_ENTRY = re.compile(r"^ +([A-Z][A-Z0-9_]*): \$\{\1:-(.*)\}\s*$", re.MULTILINE)
_NESTED = re.compile(r"^\$\{([A-Z][A-Z0-9_]*):-(.*)\}$")


def _compose_defaults(path: Path) -> dict[str, str]:
    return dict(_ENTRY.findall(path.read_text()))


def _resolve(value: str, defaults: dict[str, str]) -> str:
    """Resolve a nested `${OTHER:-x}` default the way compose does with nothing set."""
    nested = _NESTED.match(value)
    if nested is None:
        return value
    name, fallback = nested.groups()
    return _resolve(defaults.get(name, fallback), defaults)


def _template_values(path: Path) -> dict[str, str]:
    return dict(re.findall(r"^([A-Z][A-Z0-9_]*)=(.*)$", path.read_text(), re.MULTILINE))


@pytest.mark.parametrize("compose_path", _COMPOSE_FILES, ids=lambda p: p.name)
def test_every_studio_var_is_forwarded(compose_path: Path) -> None:
    defaults = _compose_defaults(compose_path)

    missing = [name for name in _STUDIO_VARS if name not in defaults]

    assert not missing, f"{compose_path.name} does not forward {missing}"


@pytest.mark.parametrize("compose_path", _COMPOSE_FILES, ids=lambda p: p.name)
def test_kill_switches_default_off_and_json_defaults_are_json(compose_path: Path) -> None:
    defaults = _compose_defaults(compose_path)

    assert {name: defaults[name] for name in _KILL_SWITCHES} == dict.fromkeys(
        _KILL_SWITCHES, "false"
    )
    for name in _JSON_VARS:
        assert isinstance(json.loads(defaults[name]), dict), name


@pytest.mark.parametrize("compose_path", _COMPOSE_FILES, ids=lambda p: p.name)
def test_compose_defaults_route_plan_to_qwen_and_chat_to_grok(
    compose_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    defaults = _compose_defaults(compose_path)
    for name in _STUDIO_VARS:
        monkeypatch.setenv(name, _resolve(defaults[name], defaults))

    settings = Settings(_env_file=None)

    assert dict(settings.studio_llm_model_by_kind) == {StudioRunKind.PLAN: "qwen"}
    assert settings.studio_chat_model_key_default == "grok"
    assert settings.studio_memory_draft_model == "grok"
    assert settings.studio_chat_model_by_product == {}
    assert settings.llm_default_model == "grok"
    assert not settings.studio_enabled


def test_memory_draft_model_follows_the_chat_model() -> None:
    defaults = _compose_defaults(_COMPOSE_FILES[0])
    switched = {**defaults, "STUDIO_CHAT_MODEL": "qwen"}

    assert _resolve(defaults["STUDIO_MEMORY_DRAFT_MODEL"], defaults) == "grok"
    assert _resolve(switched["STUDIO_MEMORY_DRAFT_MODEL"], switched) == "qwen"


@pytest.mark.parametrize("template", _ENV_TEMPLATES, ids=lambda p: p.name)
def test_env_templates_match_the_compose_routing(template: Path) -> None:
    compose = _compose_defaults(_COMPOSE_FILES[0])
    values = _template_values(template)

    for name in ("STUDIO_LLM_MODEL_BY_KIND", "STUDIO_CHAT_MODEL", "LLM_DEFAULT_MODEL"):
        assert name in values, f"{template.name} does not set {name}"
        if name in _JSON_VARS:
            assert json.loads(values[name]) == json.loads(compose[name]), name
        else:
            assert values[name] == compose[name], name


def test_the_entry_pattern_matches_something() -> None:
    """Guards against the pattern silently matching nothing after a syntax change."""
    for compose_path in _COMPOSE_FILES:
        assert len(_compose_defaults(compose_path)) > 20, compose_path.name
