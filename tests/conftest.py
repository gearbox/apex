"""Shared media ingest fixture for writer tests."""

import os
from pathlib import Path

import pytest

# Point every Settings() / init_services() at hermetic, past-dated legal
# documents so app startup never depends on the real legal/ calendar or text.
# The real directory is validated in tests/unit/test_legal_real_documents.py.
os.environ.setdefault("LEGAL_DOCUMENTS_DIR", str(Path(__file__).parent / "fixtures" / "legal"))

from src.api.services.media_ingest import MediaIngestService
from tests.media_ingest_support import make_media_ingestor


@pytest.fixture
def media_ingestor() -> MediaIngestService:
    return make_media_ingestor()
