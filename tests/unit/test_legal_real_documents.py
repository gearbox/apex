"""The real ``legal/`` directory must load — catches manifest/file drift in CI.

Runs with ``environment="development"`` because the real vex documents still
contain template placeholders (which would, by design, fail a production
start). ``today`` is pinned to the first effective date so this test doesn't
depend on the calendar.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from src.api.services.legal.registry import LegalDocumentRegistry
from src.core.product_registry import PRODUCT_REGISTRY

pytestmark = pytest.mark.unit

REPO_LEGAL_DIR = Path(__file__).resolve().parents[2] / "legal"
FIRST_EFFECTIVE = date(2026, 10, 1)


def test_real_legal_directory_loads_for_every_product() -> None:
    registry = LegalDocumentRegistry.load(
        REPO_LEGAL_DIR,
        products=PRODUCT_REGISTRY.values(),
        environment="development",
        today=FIRST_EFFECTIVE,
    )
    for config in PRODUCT_REGISTRY.values():
        required = registry.required_versions(config, today=FIRST_EFFECTIVE)
        assert set(required) == config.required_legal_documents
        for doc in registry.list_current(config.product, today=FIRST_EFFECTIVE):
            assert doc.content_md.strip()
            assert "\r" not in doc.content_md


def test_dockerfile_copies_legal_directory() -> None:
    dockerfile = (REPO_LEGAL_DIR.parent / "Dockerfile").read_text(encoding="utf-8")
    assert "COPY --chown=appuser:appuser legal/ ./legal/" in dockerfile
