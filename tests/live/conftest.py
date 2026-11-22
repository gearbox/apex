"""Environment of the opt-in live-LLM tests: make the API keys of ``.env`` visible to them.

Every live test (and ``LlmModelRegistry``) reads an entry's API key with
``os.environ[entry.api_key_env]``, but ``Settings`` loads ``.env`` into its own fields only and
exports nothing - so a key that sits in ``.env`` was reported as "is not set". This session
fixture exports exactly the variables the configured ``LLM_MODELS`` entries name, read from the
file the way ``Settings`` reads it (same file, encoding and case rules), for the length of the
run. A variable that is already set in the shell always wins.

Do not ``uv run --env-file .env``: uv's parser strips the quotes of the unquoted JSON in
``LLM_MODELS``, so ``Settings`` cannot parse it and the whole suite fails to import.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import pytest
from pydantic_settings import DotEnvSettingsSource

from src.core.config import Settings

if TYPE_CHECKING:
    from collections.abc import Iterator


def _dotenv_api_keys() -> dict[str, str]:
    """API keys of the configured entries that ``.env`` holds and the process environment lacks."""
    try:
        names = {entry.api_key_env for entry in Settings().llm_model_entries().values()}
        # Lower-cased names: ``Settings`` is ``case_sensitive=False``, and so is this source.
        dotenv = DotEnvSettingsSource(Settings).env_vars
    except Exception:  # an unusable local environment: the tests skip with "no LLM_MODELS"
        return {}
    keys: dict[str, str] = {}
    for name in names:
        value = dotenv.get(name.lower()) or dotenv.get(name)
        if value and not os.environ.get(name):
            keys[name] = value
    return keys


@pytest.fixture(scope="session", autouse=True)
def _dotenv_api_keys_in_environ() -> Iterator[None]:
    with pytest.MonkeyPatch.context() as patch:
        for name, value in _dotenv_api_keys().items():
            patch.setenv(name, value)
        yield
