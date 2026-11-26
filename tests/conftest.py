"""Fixtures every test tree shares: the environment scrub and the media ingest writer fixture."""

import os
from pathlib import Path

import pytest

# Point every Settings() / init_services() at hermetic, past-dated legal
# documents so app startup never depends on the real legal/ calendar or text.
# The real directory is validated in tests/unit/test_legal_real_documents.py.
os.environ.setdefault("LEGAL_DOCUMENTS_DIR", str(Path(__file__).parent / "fixtures" / "legal"))

from src.api.services.media_ingest import MediaIngestService
from tests.audio_support import audio_files  # noqa: F401 - session fixture of the audio tests
from tests.env_isolation import scrub_compose_forwarded_env
from tests.media_ingest_support import make_media_ingestor


@pytest.fixture(autouse=True)
def _no_compose_forwarded_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """No test sees the developer's studio / LLM settings (see tests/env_isolation.py)."""
    scrub_compose_forwarded_env(monkeypatch)


@pytest.fixture
def media_ingestor() -> MediaIngestService:
    return make_media_ingestor()
