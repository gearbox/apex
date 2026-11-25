"""``AUDIO_UPLOAD_MAX_MB`` reaches the API container with a plain default (S2a-4).

Compose passes only the variables an ``environment:`` block names, so a setting missing there can
never be changed in a deployment. The default is an int (never JSON: the compose rule of #229) and
equals the ``Settings`` default, so a deployment that sets nothing gets the documented 80 MB.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from src.core.config import Settings
from tests.unit.helpers import hermetic_settings

_REPO_ROOT = Path(__file__).parent.parent.parent
# Files that define the api service's environment block themselves: the base file (dev, prod and
# staging extend it) and the standalone Portainer stack.
_COMPOSE_FILES = (
    _REPO_ROOT / "docker-compose.yml",
    _REPO_ROOT / "docker-compose.portainer-staging.yml",
)
_ENTRY = re.compile(r"^ +AUDIO_UPLOAD_MAX_MB: \$\{AUDIO_UPLOAD_MAX_MB:-([^}]*)\}\s*$", re.MULTILINE)


@pytest.mark.parametrize("compose_path", _COMPOSE_FILES, ids=lambda p: p.name)
def test_audio_upload_limit_is_forwarded_with_a_plain_integer_default(compose_path: Path) -> None:
    found = _ENTRY.findall(compose_path.read_text())

    assert found == ["80"], (
        f"{compose_path.name}: expected one forward with default 80, got {found}"
    )


def test_the_compose_default_is_the_settings_default() -> None:
    assert Settings.model_fields["audio_upload_max_mb"].default == 80
    assert hermetic_settings().audio_upload_max_mb == 80


def test_the_env_template_documents_the_setting() -> None:
    template = (_REPO_ROOT / ".env.example").read_text()

    assert re.search(r"^AUDIO_UPLOAD_MAX_MB=80$", template, re.MULTILINE)
