"""Shared media ingest fixture for writer tests."""

import pytest

from src.api.services.media_ingest import MediaIngestService
from tests.media_ingest_support import make_media_ingestor


@pytest.fixture
def media_ingestor() -> MediaIngestService:
    return make_media_ingestor()
