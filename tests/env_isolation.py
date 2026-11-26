"""Keeps the tests independent of what the dev compose puts in the process environment.

The ``api`` service forwards the studio / LLM settings into the container, interpolated from the
developer's ``.env``. ``Settings(_env_file=None)`` skips the file but not the process env, so a
test run inside that container would build ``Settings`` from the developer's registry and kill
switches. The root ``tests/conftest.py`` removes them before every test; a test that needs one
sets it with ``monkeypatch.setenv``.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    import pytest

# Every ``STUDIO_*`` / ``LLM_*`` name a compose file forwards (tests/unit/test_env_isolation.py
# holds it to that). All of them are removed together: removing the registry but not a kill switch
# that requires it makes ``Settings`` refuse to start.
COMPOSE_FORWARDED_ENV: tuple[str, ...] = (
    "STUDIO_ENABLED",
    "STUDIO_RUNS_ENABLED",
    "STUDIO_CHAT_ENABLED",
    "LLM_MODELS",
    "LLM_DEFAULT_MODEL",
    "STUDIO_LLM_MODEL_BY_KIND",
    "STUDIO_CHAT_MODEL",
    "STUDIO_CHAT_MODEL_BY_PRODUCT",
    "STUDIO_MEMORY_DRAFT_MODEL",
)

# `      NAME: ${NAME:-default}` - the default may itself be `${OTHER:-value}`.
_ENTRY = re.compile(r"^ +([A-Z][A-Z0-9_]*): \$\{\1:-(.*)\}\s*$", re.MULTILINE)


def compose_defaults(path: Path) -> dict[str, str]:
    """``{NAME: default}`` for every ``NAME: ${NAME:-default}`` line of a compose file."""
    return dict(_ENTRY.findall(path.read_text()))


def scrub_compose_forwarded_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Remove what the dev compose forwards from the developer's .env.

    (``_env_file=None`` skips the file, not the process env.)
    """
    for name in COMPOSE_FORWARDED_ENV:
        monkeypatch.delenv(name, raising=False)
