"""The repository's own ``legal/`` directory loads (the test suite otherwise uses fixtures).

``LegalDocumentRegistry.load`` runs at startup: a manifest hash that does not match its
file, or versions out of order, would stop the API from booting. Nothing else loads the
real directory in CI, so a new document version is checked here.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

from src.api.services.legal.registry import LegalDocumentRegistry
from src.core.enums import LegalDocumentType, Product
from src.core.product_registry import PRODUCT_REGISTRY

_LEGAL_DIR = Path(__file__).parent.parent.parent / "legal"


def _load(today: date) -> LegalDocumentRegistry:
    return LegalDocumentRegistry.load(
        _LEGAL_DIR, products=PRODUCT_REGISTRY.values(), environment="test", today=today
    )


def test_published_documents_load_on_every_version_date() -> None:
    for today in (date(2026, 9, 25), date(2026, 11, 30), date(2026, 12, 1)):
        assert _load(today) is not None


def test_synthara_privacy_names_alibaba_cloud_from_2026_12_01() -> None:
    before = _load(date(2026, 11, 30)).current(
        Product.SYNTHARA, LegalDocumentType.PRIVACY, today=date(2026, 11, 30)
    )
    after = _load(date(2026, 12, 1)).current(
        Product.SYNTHARA, LegalDocumentType.PRIVACY, today=date(2026, 12, 1)
    )

    assert before.version == date(2026, 9, 25)
    assert "Alibaba Cloud" not in before.content_md
    assert after.version == date(2026, 12, 1)
    assert "Alibaba Cloud (Singapore) Private Limited" in after.content_md
    assert "Studio content" in after.content_md
