"""The test-wide environment scrub (tests/env_isolation.py) covers what compose forwards.

A studio / LLM setting added to a compose file but not to the scrub list would let the
developer's value back into every test.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from tests.env_isolation import COMPOSE_FORWARDED_ENV, compose_defaults

if TYPE_CHECKING:
    from collections.abc import Iterator

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).parent.parent.parent
# The files that define the api service's environment block themselves.
_COMPOSE_FILES = (
    _REPO_ROOT / "docker-compose.yml",
    _REPO_ROOT / "docker-compose.portainer-staging.yml",
)
_STUDIO_OR_LLM = re.compile(r"^(STUDIO|LLM)_")


def test_the_scrub_list_is_what_the_compose_files_forward() -> None:
    forwarded = {
        name
        for path in _COMPOSE_FILES
        for name in compose_defaults(path)
        if _STUDIO_OR_LLM.match(name)
    }

    assert forwarded == set(COMPOSE_FORWARDED_ENV)


@pytest.fixture(scope="module")
def _developer_env() -> Iterator[None]:
    """Every forwarded name set before the test starts, as inside the dev container.

    Module scope: pytest sets it up before any function-scoped fixture, the scrub included.
    """
    with pytest.MonkeyPatch.context() as patch:
        for name in COMPOSE_FORWARDED_ENV:
            patch.setenv(name, "from-the-developer")
        yield


@pytest.mark.usefixtures("_developer_env")
def test_every_test_starts_without_the_forwarded_names() -> None:
    """Without this, removing the root fixture would still pass wherever nothing is exported."""
    assert not set(COMPOSE_FORWARDED_ENV) & os.environ.keys()
